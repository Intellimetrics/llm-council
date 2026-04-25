"""Council run orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_council.adapters import ParticipantResult, run_participants
from llm_council.deliberation import build_deliberation_prompt, has_disagreement


async def execute_council(
    participants: list[str],
    participant_cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    config: dict[str, Any],
    *,
    deliberate: bool = False,
    max_rounds: int = 2,
) -> tuple[list[ParticipantResult], dict[str, Any]]:
    max_concurrency = int(config.get("defaults", {}).get("max_concurrency") or 4)
    results = await run_participants(
        participants,
        participant_cfg,
        prompt,
        cwd,
        max_concurrency=max_concurrency,
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
    }
    if deliberate:
        if not initial_disagreement:
            metadata["deliberation_status"] = "skipped_no_labeled_disagreement"
        elif max_rounds <= 1:
            metadata["deliberation_status"] = "skipped_max_rounds"
        else:
            metadata["deliberation_status"] = "pending"

    while deliberate and max_rounds > round_number and has_disagreement(round_results):
        next_prompt = build_deliberation_prompt(prompt, round_results)
        next_results = await run_participants(
            participants,
            participant_cfg,
            next_prompt,
            cwd,
            max_concurrency=max_concurrency,
        )
        round_number += 1
        round_results = [
            replace(result, name=f"{result.name}:round{round_number}")
            for result in next_results
        ]
        results.extend(round_results)
        metadata["rounds"] = round_number
        metadata["deliberated"] = True

    if metadata["deliberated"]:
        final_disagreement = has_disagreement(round_results)
        metadata["final_disagreement_detected"] = final_disagreement
        metadata["deliberation_status"] = (
            "ran_max_rounds_unresolved"
            if final_disagreement
            else "ran_no_labeled_disagreement"
        )

    return results, metadata
