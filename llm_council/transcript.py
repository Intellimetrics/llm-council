"""Transcript writing."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_council.adapters import ParticipantResult, command_for_display
from llm_council.deliberation import model_comparison, recommendation_counts

ROUND_SUFFIX_RE = re.compile(r":round(\d+)$")


def safe_slug(text: str, max_len: int = 60) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned or "council")[:max_len].strip("-")


def transcript_paths(base_dir: Path, question: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{safe_slug(question)}"
    return base_dir / f"{stem}.md", base_dir / f"{stem}.json"


def latest_transcript(base_dir: Path, *, suffix: str = ".md") -> Path | None:
    matches = sorted(
        _existing_paths(base_dir.glob(f"*{suffix}")), key=lambda item: item[1]
    )
    return matches[-1][0] if matches else None


def _existing_paths(paths) -> list[tuple[Path, float]]:
    existing = []
    for path in paths:
        try:
            existing.append((path, path.stat().st_mtime))
        except FileNotFoundError:
            continue
    return existing


def transcript_records(base_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path, mtime in sorted(
        _existing_paths(base_dir.glob("*.json")), key=lambda item: item[1]
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = data.get("results") or []
        records.append(
            {
                "path": str(path),
                "markdown": str(path.with_suffix(".md")),
                "question": data.get("question", ""),
                "mode": data.get("mode", ""),
                "current": data.get("current"),
                "participants": data.get("participants", []),
                "ok": sum(1 for result in results if result.get("ok")),
                "total": len(results),
                "tokens": sum(result.get("total_tokens") or 0 for result in results),
                "cost_usd": sum(result.get("cost_usd") or 0 for result in results),
                "mtime": mtime,
            }
        )
    return records


def result_to_dict(result: ParticipantResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "ok": result.ok,
        "model": result.model,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "command": result.command,
        "output": result.output,
        "error": result.error,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
    }


def deliberation_summary(metadata: dict[str, Any]) -> str:
    status = metadata.get("deliberation_status")
    if status == "ran_no_labeled_disagreement":
        return "ran; no labeled disagreement remained"
    if status == "ran_max_rounds_unresolved":
        return "ran; max rounds reached with labeled disagreement"
    if status == "skipped_no_labeled_disagreement":
        return "skipped, no labeled disagreement detected"
    if status == "skipped_max_rounds":
        return "skipped, max rounds is 1"
    if status == "pending":
        return "pending"
    if metadata.get("deliberated"):
        return "ran"
    if metadata.get("deliberation_requested"):
        return "skipped"
    return "not requested"


def result_round(name: str) -> int:
    match = ROUND_SUFFIX_RE.search(name)
    return int(match.group(1)) if match else 1


def final_round_results(results: list[ParticipantResult]) -> list[ParticipantResult]:
    if not results:
        return []
    final_round = max(result_round(result.name) for result in results)
    return [result for result in results if result_round(result.name) == final_round]


def write_transcript(
    markdown_path: Path,
    json_path: Path,
    *,
    question: str,
    mode: str,
    current: str | None,
    participants: list[str],
    prompt: str,
    results: list[ParticipantResult],
    transparent: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = metadata or {}
    ok_count = sum(1 for result in results if result.ok)
    final_results = final_round_results(results)
    final_ok_count = sum(1 for result in final_results if result.ok)
    elapsed_total = sum(result.elapsed_seconds for result in results)
    token_total = sum(result.total_tokens or 0 for result in results)
    cost_total = sum(result.cost_usd or 0 for result in results)
    recommendations = recommendation_counts(final_results)
    lines = [
        "# LLM Council Transcript",
        "",
        f"- Mode: `{mode}`",
        f"- Current agent: `{current or 'unknown'}`",
        f"- Participants: {', '.join(f'`{name}`' for name in participants)}",
        f"- Successful responses: {ok_count}/{len(results)} total",
        f"- Final-round successful responses: {final_ok_count}/{len(final_results)}",
        f"- Participant elapsed total: `{elapsed_total:.1f}s`",
        f"- Tokens reported: `{token_total}`",
        f"- Cost reported: `${cost_total:.6f}`",
        f"- Rounds: `{metadata.get('rounds', 1)}`",
        f"- Deliberation: {deliberation_summary(metadata)}",
        "- Recommendations (final round): "
        f"`{recommendations['yes']} yes / {recommendations['no']} no / "
        f"{recommendations['tradeoff']} tradeoff / {recommendations['unknown']} unknown`",
        "",
        "## Question",
        "",
        question.strip(),
        "",
    ]

    if transparent:
        lines.extend(["## Model Comparison", ""])
        lines.extend(model_comparison(results))
        lines.append("")

    lines.extend(["## Participant Responses", ""])

    for result in results:
        status = "ok" if result.ok else "error"
        lines.extend(
            [
                f"### {result.name} ({status})",
                "",
                f"- Model: `{result.model or 'cli default'}`",
                f"- Elapsed: `{result.elapsed_seconds:.1f}s`",
            ]
        )
        if result.total_tokens is not None:
            lines.append(f"- Tokens: `{result.total_tokens}`")
        if result.cost_usd is not None:
            lines.append(f"- Cost: `${result.cost_usd:.6f}`")
        if result.command:
            lines.append(f"- Command: `{command_for_display(result.command)}`")
        lines.append("")
        if result.ok:
            lines.extend([result.output.strip() or "[empty response]", ""])
        else:
            lines.extend(["```", result.error.strip() or "[unknown error]", "```", ""])

    lines.extend(["## Prompt Sent", "", "```text", prompt, "```", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    json_path.write_text(
        json.dumps(
            {
                "question": question,
                "mode": mode,
                "current": current,
                "participants": participants,
                "prompt": prompt,
                "metadata": metadata,
                "results": [result_to_dict(result) for result in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
