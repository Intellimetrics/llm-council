"""Council run orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from llm_council.adapters import (
    CacheContext,
    ParticipantResult,
    is_timeout_error,
    run_participants,
)
from llm_council.cache import (
    is_caching_disabled_for_mode,
    resolve_ttl_seconds,
)
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
    results = await run_participants(
        participants,
        participant_cfg,
        prompt,
        cwd,
        max_concurrency=max_concurrency,
        progress=emit,
        round_number=1,
        image_manifest=image_manifest,
        cache_ctx=cache_ctx_round1,
    )
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

    emit(
        {
            "event": "council_finish",
            "rounds": metadata["rounds"],
            "ok": sum(1 for result in results if result.ok),
            "total": len(results),
        }
    )

    return results, metadata
