"""Council run orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from llm_council.adapters import (
    CacheContext,
    ParticipantResult,
    PREFLIGHT_FAILED_PREFIX,
    is_timeout_error,
    run_participants,
)
from llm_council.cache import (
    is_caching_disabled_for_mode,
    resolve_ttl_seconds,
)
from llm_council.config import is_local_participant, is_loopback_base_url
from llm_council.convergence import (
    MIN_TOKENS_FOR_CLASSIFICATION,
    classify,
    jaccard_similarity,
    resolve_thresholds,
    tokenize,
)
from llm_council.deliberation import (
    build_deliberation_prompt,
    default_min_quorum,
    has_disagreement,
    labeled_quorum_count,
)


PREFLIGHT_TIMEOUT_SECONDS = 1.0


# Embedded-credential regex shared by `_redact_base_url` (for the rendered
# base_url) and `_redact_credentials_in_text` (defense-in-depth pass over
# arbitrary strings — e.g. an httpx exception that echoes the URL it tried
# to reach). Matches `<scheme>://<userinfo>@<host>` and replaces userinfo
# with `***`. Conservative: the scheme/host shapes stay intact.
_EMBEDDED_CRED_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)[^/@\s]+@")


def _redact_base_url(base_url: str) -> str:
    """Strip embedded credentials (user:pass@host) from a URL before
    rendering it into user-facing error messages or transcripts.

    `allow_private: true` skips the embedded-credentials validator (so
    that local participants with `http://user:pass@127.0.0.1` are
    permitted), which means a careless config could otherwise leak the
    creds into transcripts via the preflight error message.
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return base_url
    if not parsed.username and not parsed.password:
        return base_url
    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username or parsed.password:
        netloc = f"***@{netloc}"
    return parsed._replace(netloc=netloc).geturl()


def _redact_credentials_in_text(text: str) -> str:
    """Defense-in-depth scrub of an arbitrary string for `scheme://user:pass@host`
    patterns. Used on the raw exception text from httpx, which may echo
    back the URL it tried to reach. `_redact_base_url` already handles
    the rendered `base_url`; this pass catches anything else that might
    quote the user-info portion.
    """
    return _EMBEDDED_CRED_RE.sub(lambda m: f"{m.group('scheme')}***@", text)


async def _preflight_one(name: str, cfg: dict[str, Any]) -> str | None:
    """Probe one local participant. Returns an error message on failure, None on success.

    For `type: ollama` we hit `/api/tags` (matches the doctor probe). For
    `type: openai_compatible` we hit `/v1/models` (matches the
    --probe-local-openai probe). 1-second timeout — anything slower than
    that on a loopback endpoint indicates a hung server, not a slow one.
    """
    ptype = cfg.get("type")
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        return None  # nothing to probe; let the run-time path produce the real error
    if ptype == "ollama":
        url = f"{base_url}/api/tags"
    elif ptype == "openai_compatible":
        # `/v1/models` lives under whatever path the user configured. The
        # base_url canonical form ends at `/v1`; tolerate both shapes.
        if base_url.endswith("/v1"):
            url = f"{base_url}/models"
        else:
            url = f"{base_url}/v1/models"
    else:
        return None
    redacted = _redact_base_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=PREFLIGHT_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
    except Exception as exc:  # noqa: BLE001 — surface every failure mode legibly
        # Defense-in-depth: httpx errors sometimes quote the URL they tried
        # to reach, which would re-introduce embedded creds even after the
        # base_url is redacted. Run the same scrub over the exception text.
        exc_text = _redact_credentials_in_text(f"{type(exc).__name__}: {exc}")
        return (
            f"{PREFLIGHT_FAILED_PREFIX} local endpoint unreachable for "
            f"{name!r} (base_url={redacted!r}): {exc_text}"
        )
    if response.status_code >= 500:
        return (
            f"{PREFLIGHT_FAILED_PREFIX} local endpoint at {redacted!r} returned "
            f"HTTP {response.status_code} for {name!r}"
        )
    # 2xx, 3xx, 4xx: server is up enough for the run path to make progress
    # or produce its own meaningful error. Don't pre-judge 4xx — some
    # llama.cpp builds 404 on /v1/models but serve /v1/chat/completions fine.
    return None


async def preflight_local_participants(
    participants: list[str],
    participant_cfg: dict[str, Any],
) -> dict[str, str]:
    """Quick probe of every selected local participant.

    Returns a mapping from participant name to a `PreflightFailed:` error
    string for every participant whose endpoint is unreachable. Hosted
    participants and any participant with `pre_flight_check: false` are
    skipped silently.

    Probes run concurrently — total wall time is bounded by the single-
    probe timeout, not the participant count. Pre-flight is best-effort:
    if the probe library itself raises an unexpected error, we let the
    real run path report it.
    """
    candidates = [
        (name, participant_cfg.get(name) or {})
        for name in participants
    ]
    # Default-on for loopback (`127.0.0.1`, `localhost`, `[::1]`, `0.0.0.0`)
    # where a 1s timeout is reasonable. Default-off for RFC1918 (`10.x`,
    # `192.168.x`, `172.16-31.x`) where a homelab/VPN endpoint might
    # legitimately take longer to respond. Users wanting to ping their LAN
    # vLLM can opt in with `pre_flight_check: true`. Users wanting to skip
    # an unreliable loopback endpoint can opt out with `pre_flight_check:
    # false`.
    todo = []
    for name, cfg in candidates:
        if not is_local_participant(cfg):
            continue
        opted_in = cfg.get("pre_flight_check")  # tri-state: True / False / None
        is_loopback = is_loopback_base_url(str(cfg.get("base_url") or ""))
        # Ollama's default base_url is loopback; treat type:ollama as loopback
        # for preflight purposes when its base_url is omitted/local.
        if cfg.get("type") == "ollama" and not cfg.get("base_url"):
            is_loopback = True
        if opted_in is False:
            continue  # explicit opt-out always wins
        if opted_in is True:
            todo.append((name, cfg))
            continue
        # opted_in is None — use the default policy
        if is_loopback:
            todo.append((name, cfg))
    if not todo:
        return {}
    results = await asyncio.gather(
        *(_preflight_one(name, cfg) for name, cfg in todo),
        return_exceptions=False,
    )
    return {
        name: error
        for (name, _cfg), error in zip(todo, results)
        if error is not None
    }


def _synth_preflight_failure(
    name: str, error: str, *, model: str | None = None
) -> ParticipantResult:
    """Construct a ParticipantResult that mirrors what a failed run looks like.

    Synthesizing this here (rather than letting the participant attempt to
    run and fail at the timeout) keeps the failure visible early and
    uses our explicit `preflight_failed` error_kind instead of the
    catch-all `downstream_error`. Carrying `model` through means
    transcripts and the summary table still identify which model was
    targeted, even though no call was made.
    """
    return ParticipantResult(
        name=name,
        ok=False,
        output="",
        error=error,
        elapsed_seconds=0.0,
        model=model,
    )


def _resolve_convergence_thresholds(
    config: dict[str, Any], mode: str | None
) -> dict[str, float]:
    defaults = config.get("defaults", {}) or {}
    base = defaults.get("convergence_thresholds") if isinstance(defaults, dict) else None
    override: dict[str, float] | None = None
    if mode:
        modes = config.get("modes", {}) or {}
        mode_cfg = modes.get(mode) or {}
        if isinstance(mode_cfg, dict):
            override = mode_cfg.get("convergence_thresholds")
    merged: dict[str, float] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return resolve_thresholds(merged or None)


def _base_name(name: str) -> str:
    return name.split(":round")[0]


def _index_by_base_name(
    results: list[ParticipantResult],
) -> dict[str, ParticipantResult]:
    return {_base_name(result.name): result for result in results}


def _compute_round_convergence(
    prior: list[ParticipantResult],
    current: list[ParticipantResult],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    prior_index = _index_by_base_name(prior)
    records: list[dict[str, Any]] = []
    for result in current:
        base = _base_name(result.name)
        prior_result = prior_index.get(base)
        if prior_result is None or not prior_result.ok or not result.ok:
            continue
        prior_tokens = tokenize(prior_result.output or "")
        current_tokens = tokenize(result.output or "")
        token_floor = min(len(prior_tokens), len(current_tokens))
        if token_floor < MIN_TOKENS_FOR_CLASSIFICATION:
            records.append(
                {
                    "participant": base,
                    "similarity": None,
                    "state": "insufficient",
                    "prior_tokens": len(prior_tokens),
                    "current_tokens": len(current_tokens),
                }
            )
            continue
        similarity = jaccard_similarity(prior_tokens, current_tokens)
        state = classify(similarity, thresholds)
        records.append(
            {
                "participant": base,
                "similarity": round(similarity, 4),
                "state": state,
            }
        )
    return records


def _failed_for_deliberation(results: list[ParticipantResult]) -> set[str]:
    excluded: set[str] = set()
    for result in results:
        if result.ok:
            continue
        if is_timeout_error(result.error) or result.error.startswith("PromptTooLarge:"):
            excluded.add(result.name.split(":round")[0])
    return excluded


async def execute_council(
    participants: list[str],
    participant_cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    config: dict[str, Any],
    *,
    deliberate: bool = False,
    max_rounds: int = 2,
    progress: Callable[[dict[str, Any]], None] | None = None,
    image_manifest: list[dict[str, Any]] | None = None,
    min_quorum: int | None = None,
    mode: str | None = None,
    cache_mode: str = "on",
    stances: dict[str, str] | None = None,
) -> tuple[list[ParticipantResult], dict[str, Any]]:
    max_concurrency = int(config.get("defaults", {}).get("max_concurrency") or 4)
    convergence_thresholds = _resolve_convergence_thresholds(config, mode)

    cache_disabled_for_mode = is_caching_disabled_for_mode(mode)
    cache_ctx_round1 = CacheContext(
        cwd=cwd,
        cache_mode=cache_mode,
        ttl_seconds=resolve_ttl_seconds(config, mode),
        cache_disabled=cache_disabled_for_mode,
    )
    cache_ctx_deliberation = CacheContext(
        cwd=cwd,
        cache_mode=cache_mode,
        ttl_seconds=resolve_ttl_seconds(config, mode),
        cache_disabled=True,
    )

    progress_events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        progress_events.append(event)
        if progress:
            progress(event)

    emit(
        {
            "event": "council_start",
            "participants": participants,
            "round": 1,
            "max_rounds": max_rounds,
            "deliberate": deliberate,
            "image_count": len(image_manifest or []),
        }
    )
    if image_manifest:
        for name in participants:
            cfg = participant_cfg.get(name, {})
            ptype = cfg.get("type")
            if ptype == "cli":
                continue
            if not cfg.get("vision"):
                emit(
                    {
                        "event": "images_skipped",
                        "participant": name,
                        "round": 1,
                        "reason": "non_vision",
                        "image_count": len(image_manifest),
                    }
                )
    # Pre-flight ping for local participants. Turns opaque "downstream_error"
    # at full timeout into a fast, legible "PreflightFailed: …" with the
    # base_url named, so the user sees what's actually wrong (server not
    # running, port wrong, model not loaded) without waiting through the
    # participant timeout.
    preflight_failures = await preflight_local_participants(
        participants, participant_cfg
    )
    if preflight_failures:
        for name, error in preflight_failures.items():
            emit(
                {
                    "event": "preflight_failed",
                    "participant": name,
                    "round": 1,
                    "error": error,
                }
            )
        run_targets = [name for name in participants if name not in preflight_failures]
    else:
        run_targets = participants

    if run_targets:
        run_results = await run_participants(
            run_targets,
            participant_cfg,
            prompt,
            cwd,
            max_concurrency=max_concurrency,
            progress=emit,
            round_number=1,
            image_manifest=image_manifest,
            cache_ctx=cache_ctx_round1,
        )
    else:
        run_results = []
    # Merge pre-flight failures back in, preserving the original participant
    # order so transcripts and the summary table stay deterministic.
    by_name: dict[str, ParticipantResult] = {result.name: result for result in run_results}
    for name, error in preflight_failures.items():
        cfg = participant_cfg.get(name) or {}
        by_name[name] = _synth_preflight_failure(
            name, error, model=cfg.get("model")
        )
    results = [by_name[name] for name in participants if name in by_name]
    round_results = results
    round_number = 1
    initial_disagreement = has_disagreement(round_results)
    metadata = {
        "rounds": round_number,
        "max_rounds": max_rounds,
        "deliberation_requested": deliberate,
        "deliberated": False,
        "disagreement_detected": initial_disagreement,
        "final_disagreement_detected": initial_disagreement,
        "deliberation_status": "not_requested",
        "progress_events": progress_events,
    }
    if deliberate:
        if not initial_disagreement:
            metadata["deliberation_status"] = "skipped_no_labeled_disagreement"
            emit(
                {
                    "event": "deliberation_skip",
                    "reason": "no_labeled_disagreement",
                    "round": round_number,
                }
            )
        elif max_rounds <= 1:
            metadata["deliberation_status"] = "skipped_max_rounds"
            emit({"event": "deliberation_skip", "reason": "max_rounds", "round": round_number})
        else:
            metadata["deliberation_status"] = "pending"
            emit({"event": "deliberation_pending", "round": round_number + 1})

    cumulative_excluded: set[str] = set()
    aborted_all_excluded = False
    convergence_by_round: dict[int, list[dict[str, Any]]] = {}
    deliberation_prompts: dict[int, str] = {}
    while deliberate and max_rounds > round_number and has_disagreement(round_results):
        cumulative_excluded.update(_failed_for_deliberation(round_results))
        deliberation_participants = [
            name for name in participants if name not in cumulative_excluded
        ]
        if cumulative_excluded:
            emit(
                {
                    "event": "deliberation_skip_participants",
                    "round": round_number + 1,
                    "skipped": sorted(cumulative_excluded),
                    "reason": "timed_out_or_prompt_too_large",
                }
            )
        if not deliberation_participants:
            metadata["deliberation_status"] = "skipped_all_excluded"
            aborted_all_excluded = True
            emit(
                {
                    "event": "deliberation_skip",
                    "reason": "no_remaining_participants",
                    "round": round_number + 1,
                }
            )
            break
        next_prompt, truncated_peers = build_deliberation_prompt(prompt, round_results)
        deliberation_prompts[round_number + 1] = next_prompt
        for peer_name in truncated_peers:
            emit(
                {
                    "event": "truncated_for_deliberation",
                    "round": round_number + 1,
                    "participant": peer_name,
                }
            )
        emit({"event": "deliberation_round_start", "round": round_number + 1})
        next_results = await run_participants(
            deliberation_participants,
            participant_cfg,
            next_prompt,
            cwd,
            max_concurrency=max_concurrency,
            progress=emit,
            round_number=round_number + 1,
            image_manifest=image_manifest,
            cache_ctx=cache_ctx_deliberation,
        )
        prior_round_results = list(round_results)
        round_number += 1
        round_results = [
            replace(result, name=f"{result.name}:round{round_number}")
            for result in next_results
        ]
        results.extend(round_results)
        metadata["rounds"] = round_number
        metadata["deliberated"] = True

        round_convergence = _compute_round_convergence(
            prior_round_results, round_results, convergence_thresholds
        )
        if round_convergence:
            convergence_by_round[round_number] = round_convergence
            for record in round_convergence:
                emit(
                    {
                        "event": "convergence",
                        "round": round_number,
                        "participant": record["participant"],
                        "state": record["state"],
                        "similarity": record["similarity"],
                    }
                )

    if metadata["deliberated"] and not aborted_all_excluded:
        final_disagreement = has_disagreement(round_results)
        metadata["final_disagreement_detected"] = final_disagreement
        metadata["deliberation_status"] = (
            "ran_max_rounds_unresolved"
            if final_disagreement
            else "ran_no_labeled_disagreement"
        )
        emit(
            {
                "event": "deliberation_finish",
                "rounds": metadata["rounds"],
                "status": metadata["deliberation_status"],
            }
        )
    elif aborted_all_excluded:
        emit(
            {
                "event": "deliberation_finish",
                "rounds": metadata["rounds"],
                "status": metadata["deliberation_status"],
            }
        )

    if convergence_by_round:
        metadata["convergence"] = {
            str(round_no): records for round_no, records in sorted(convergence_by_round.items())
        }
        metadata["convergence_thresholds"] = convergence_thresholds

    if deliberation_prompts:
        metadata["deliberation_prompts"] = {
            str(round_no): text
            for round_no, text in sorted(deliberation_prompts.items())
        }

    effective_min_quorum = (
        int(min_quorum) if min_quorum is not None else default_min_quorum(len(participants))
    )
    effective_min_quorum = max(1, effective_min_quorum)
    final_labeled = labeled_quorum_count(round_results)
    degraded = final_labeled < effective_min_quorum
    metadata["min_quorum"] = effective_min_quorum
    metadata["labeled_quorum"] = final_labeled
    metadata["degraded"] = degraded
    if degraded:
        emit(
            {
                "event": "degraded_consensus",
                "labeled_quorum": final_labeled,
                "min_quorum": effective_min_quorum,
                "participant_count": len(participants),
                "round": metadata["rounds"],
            }
        )

    if stances:
        metadata["stances"] = dict(stances)
        for idx, result in enumerate(results):
            base_name = result.name.split(":round", 1)[0]
            assigned = stances.get(base_name)
            if assigned is not None:
                results[idx] = replace(result, stance=assigned)

    emit(
        {
            "event": "council_finish",
            "rounds": metadata["rounds"],
            "ok": sum(1 for result in results if result.ok),
            "total": len(results),
        }
    )

    return results, metadata
