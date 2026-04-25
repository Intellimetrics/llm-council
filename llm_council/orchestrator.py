"""Council run orchestration."""

from __future__ import annotations

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
    metadata = {
        "rounds": 1,
        "deliberation_requested": deliberate,
        "deliberated": False,
        "disagreement_detected": has_disagreement(results),
    }

    if deliberate and max_rounds > 1 and metadata["disagreement_detected"]:
        second_prompt = build_deliberation_prompt(prompt, results)
        second_results = await run_participants(
            participants,
            participant_cfg,
            second_prompt,
            cwd,
            max_concurrency=max_concurrency,
        )
        for result in second_results:
            result.name = f"{result.name}:round2"
        results.extend(second_results)
        metadata["rounds"] = 2
        metadata["deliberated"] = True

    return results, metadata
