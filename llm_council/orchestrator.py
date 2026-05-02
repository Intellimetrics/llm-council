"""Council run orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from llm_council.adapters import (
    ParticipantResult,
    is_timeout_error,
    run_participants,
)
from llm_council.deliberation import build_deliberation_prompt, has_disagreement


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
) -> tuple[list[ParticipantResult], dict[str, Any]]:
    max_concurrency = int(config.get("defaults", {}).get("max_concurrency") or 4)

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
        )
        round_number += 1
        round_results = [
            replace(result, name=f"{result.name}:round{round_number}")
            for result in next_results
        ]
        results.extend(round_results)
        metadata["rounds"] = round_number
        metadata["deliberated"] = True

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

    emit(
        {
            "event": "council_finish",
            "rounds": metadata["rounds"],
            "ok": sum(1 for result in results if result.ok),
            "total": len(results),
        }
    )

    return results, metadata
