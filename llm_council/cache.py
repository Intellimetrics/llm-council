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
CACHE_SCHEMA_VERSION = 1

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
) -> dict[str, Any]:
    preview = prompt[:PROMPT_PREVIEW_CHARS]
    return {
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
    }


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
