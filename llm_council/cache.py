"""Per-participant on-disk result cache keyed on (participant, prompt+config).

Caching is intentionally isolated from adapter logic so the read/write
surface is small and easy to test. Hits return immediately and never
touch the network or spawn a subprocess; misses run normally and write
the successful payload through. Failed runs (ok=False) are never cached
to avoid amplifying a transient failure across reruns.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


CACHE_SUBDIR = ".llm-council/cache"
DEFAULT_TTL_SECONDS = 24 * 3600
PROMPT_PREVIEW_CHARS = 200
CACHE_SCHEMA_VERSION = 3  # v3 = structured evidence shape (list[{text, tag}]), prompt_chars, recovered_after_timeout, section_repair_attempted. evidence_verification_failures is optional with a `[]` default on rehydrate, so we deliberately do NOT bump for it — keeps old v3 caches readable.

_MODES_THAT_SKIP_CACHE = frozenset({"consensus"})


def is_caching_disabled_for_mode(mode: str | None) -> bool:
    if not mode:
        return False
    return mode in _MODES_THAT_SKIP_CACHE


def _canonical_config(participant_cfg: dict[str, Any]) -> str:
    return json.dumps(participant_cfg, sort_keys=True, default=str, ensure_ascii=False)


def _canonical_image_manifest(
    image_manifest: list[dict[str, Any]] | None,
) -> str:
    if not image_manifest:
        return ""
    canonical: list[dict[str, Any]] = []
    for entry in image_manifest:
        if not isinstance(entry, dict):
            continue
        canonical.append(
            {
                "sha256": entry.get("sha256"),
                "mime": entry.get("mime"),
                "size": entry.get("size"),
                "relative_path": entry.get("relative_path"),
            }
        )
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False)


def compute_key(
    participant_name: str,
    participant_cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"v{CACHE_SCHEMA_VERSION}".encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(participant_name.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(_canonical_config(participant_cfg or {}).encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(prompt.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(_canonical_image_manifest(image_manifest).encode("utf-8"))
    return hasher.hexdigest()


def cache_dir(working_dir: Path) -> Path:
    return Path(working_dir) / CACHE_SUBDIR


def cache_path(working_dir: Path, participant_name: str, key: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in participant_name)
    return cache_dir(working_dir) / f"{safe_name}__{key}.json"


def read_cache(path: Path, *, expected_key: str | None = None) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        _safe_unlink(path)
        return None
    if not isinstance(payload, dict):
        _safe_unlink(path)
        return None
    if expected_key is not None and payload.get("prompt_sha256") != expected_key:
        _safe_unlink(path)
        return None
    cached_at = payload.get("cached_at_unix")
    ttl_seconds = payload.get("ttl_seconds")
    try:
        cached_at_f = float(cached_at)
        ttl_f = float(ttl_seconds)
    except (TypeError, ValueError):
        _safe_unlink(path)
        return None
    if time.time() - cached_at_f > ttl_f:
        _safe_unlink(path)
        return None
    return payload


def write_cache(path: Path, payload: dict[str, Any], ttl_seconds: int) -> None:
    enriched = dict(payload)
    enriched["cached_at_unix"] = time.time()
    enriched["ttl_seconds"] = int(ttl_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(enriched, fp, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_path, path)
    except OSError:
        _safe_unlink(tmp_path)
        raise


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def build_payload(
    *,
    participant_name: str,
    prompt: str,
    key: str,
    output: str,
    recommendation_label: str | None,
    elapsed_seconds: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost_usd: float | None,
    model: str | None,
    command: list[str] | None,
    recovered_after_timeout: bool = False,
    prompt_chars: int | None = None,
    section_repair_attempted: bool = False,
    terse_retry_attempted: bool = False,
    evidence_verification_failures: list[str] | None = None,
    continue_debate: str | None = None,
    tool_call_status: str | None = None,
    is_ranking_round: bool = False,
    model_fallback_used: str | None = None,
    recovered_after_quota: bool = False,
) -> dict[str, Any]:
    preview = prompt[:PROMPT_PREVIEW_CHARS]
    payload: dict[str, Any] = {
        "participant_name": participant_name,
        "prompt_sha256": key,
        "prompt_preview": preview,
        "output": output,
        "recommendation_label": recommendation_label,
        "elapsed_seconds": float(elapsed_seconds),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "model": model,
        "command": list(command) if command else None,
        # v0.7.0 receipts. `recovered_after_timeout` records that the
        # original call timed out and the terse-retry rescued the
        # response; persisting it through the cache means repeat runs
        # still surface the recovery in transcripts and bump the
        # `timeout_recoveries` stat. `prompt_chars` carries the
        # cached run's prompt size for the timeout-by-prompt-size
        # bucket. Old payloads written before v3 are evicted by the
        # CACHE_SCHEMA_VERSION bump; readers also default-on-missing
        # so a stale schema-3 payload (e.g. one written before this
        # field landed within the v3 window) rehydrates cleanly.
        "recovered_after_timeout": bool(recovered_after_timeout),
        "prompt_chars": prompt_chars,
        # Pass-9 fix: persist whether the section-repair retry path fired
        # for this run. Only relevant on successful results (failed runs
        # aren't cached), but threading it through means a cache hit on a
        # sections-recovered result still sees the flag, which keeps the
        # strict-evidence wrapper's guard correct across cache hits.
        "section_repair_attempted": bool(section_repair_attempted),
    }
    # A timeout-recovered result is merged with BOTH recovered_after_timeout
    # AND terse_retry_attempted set. Persisting recovered_after_timeout but
    # not terse_retry_attempted made a cache hit rehydrate to a contradictory
    # state (recovered but "no retry attempted"). Persist it too; omit-when-
    # False keeps the common payload tight and old payloads default to False.
    if terse_retry_attempted:
        payload["terse_retry_attempted"] = True
    # v4: only include the key when there's something to record so payloads
    # stay tight for the overwhelming majority of runs (no VERIFIED tags
    # cited, or all VERIFIED refs verified). Readers default to `[]` on
    # missing, so the absence is semantically identical to an empty list.
    if evidence_verification_failures:
        payload["evidence_verification_failures"] = list(evidence_verification_failures)
    # v0.8.1: persist the peer's CONTINUE_DEBATE vote so cached rehydrates
    # still drive the unanimity gate. Only stored when present — readers
    # default to None on absence, so absence is semantically identical to
    # "peer did not emit the tag". Schema version is NOT bumped because
    # the default is None (no behavioral change for old payloads).
    if continue_debate is not None:
        payload["continue_debate"] = str(continue_debate)
    # v0.9.0 Feature 3: persist the tool-call telemetry distinction so cache
    # hits still surface the absent/ok/malformed bucket in stats. Only
    # written when not None — `None` means "extraction did not run for
    # this peer", which is the default state and is semantically
    # identical to absence. Schema version is NOT bumped because the
    # default-on-missing read is None (no behavioral change for old
    # payloads).
    if tool_call_status is not None:
        payload["tool_call_status"] = str(tool_call_status)
    # v0.9.0 Feature 2: persist the ranking-round flag so cache hits still
    # surface as ranking-round and stay filtered from the deliberation
    # round-2 prompt builder. Only written when True (default False);
    # readers default-on-missing to False, so absence is semantically
    # identical to "primary response, not a ranking pass".
    if is_ranking_round:
        payload["is_ranking_round"] = True
    # v0.11.6 Phase 2: persist quota-fallback receipts so a cache hit
    # surfaces the same fallback context as the original run. Only
    # written when the fallback fired (default state on a non-fallback
    # call is None/False), keeping payloads tight for the common case.
    # Readers default-on-missing to None/False; absence is semantically
    # identical to "no fallback fired". Schema version NOT bumped.
    if model_fallback_used:
        payload["model_fallback_used"] = str(model_fallback_used)
    if recovered_after_quota:
        payload["recovered_after_quota"] = True
    return payload


def resolve_ttl_seconds(
    config: dict[str, Any] | None,
    mode: str | None,
) -> int:
    if not isinstance(config, dict):
        return DEFAULT_TTL_SECONDS
    defaults = config.get("defaults") or {}
    base = _coerce_hours(defaults.get("cache_ttl_hours"))
    override = None
    if mode:
        modes = config.get("modes") or {}
        mode_cfg = modes.get(mode) or {}
        if isinstance(mode_cfg, dict):
            override = _coerce_hours(mode_cfg.get("cache_ttl_hours"))
    if override is not None:
        return int(override * 3600)
    if base is not None:
        return int(base * 3600)
    return DEFAULT_TTL_SECONDS


def _coerce_hours(value: Any) -> float | None:
    if value is None:
        return None
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return None
    if hours <= 0:
        return None
    return hours
