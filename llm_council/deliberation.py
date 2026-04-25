"""Lightweight opt-in deliberation helpers."""

from __future__ import annotations

import re

from llm_council.adapters import ParticipantResult

MAX_DELIBERATION_PROMPT_CHARS = 80_000
RECOMMENDATION_RE = re.compile(
    r"""
    ^\s*
    (?:>\s*)?
    (?:[-*]\s+)?
    (?:\#{1,6}\s*)?
    (?:\*\*)?
    recommendation
    (?:\*\*)?
    \s*[:\-]\s*
    (?:\*\*)?
    \s*
    (yes|no|tradeoff)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip().strip(">*-_ ")
        if cleaned:
            return cleaned
    return ""


def model_comparison(results: list[ParticipantResult]) -> list[str]:
    lines: list[str] = []
    for result in results:
        if not result.ok:
            lines.append(f"- {result.name}: error - {result.error}")
            continue
        usage = []
        if result.total_tokens is not None:
            usage.append(f"{result.total_tokens} tokens")
        if result.cost_usd is not None:
            usage.append(f"${result.cost_usd:.6f}")
        suffix = f" ({', '.join(usage)})" if usage else ""
        lines.append(f"- {result.name}: {first_nonempty_line(result.output)}{suffix}")
    return lines


def recommendation_label(text: str) -> str:
    fenced_label = "unknown"
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        match = RECOMMENDATION_RE.match(line)
        if not match:
            continue
        label = match.group(1).lower()
        if not in_fence:
            return label
        if fenced_label == "unknown":
            fenced_label = label
    return fenced_label


def recommendation_counts(results: list[ParticipantResult]) -> dict[str, int]:
    counts = {"yes": 0, "no": 0, "tradeoff": 0, "unknown": 0}
    for result in results:
        if not result.ok:
            continue
        counts[recommendation_label(result.output)] += 1
    return counts


def has_disagreement(results: list[ParticipantResult]) -> bool:
    counts = recommendation_counts(results)
    labeled_positions = [label for label in ("yes", "no", "tradeoff") if counts[label]]
    labeled_total = sum(counts[label] for label in labeled_positions)
    return labeled_total >= 2 and len(labeled_positions) > 1


def build_deliberation_prompt(original_prompt: str, results: list[ParticipantResult]) -> str:
    excerpts = []
    for result in results:
        if not result.ok:
            continue
        excerpts.append(f"## {result.name}\n\n{result.output.strip()[:4000]}")
    prompt = (
        f"{original_prompt}\n\n"
        "Second-round deliberation:\n"
        "The first-round participants may disagree. Read the peer responses below, "
        "identify what you still disagree with, and try to converge on a practical "
        "recommendation. If consensus is impossible, state the remaining split clearly.\n\n"
        + "\n\n".join(excerpts)
    )
    if len(prompt) <= MAX_DELIBERATION_PROMPT_CHARS:
        return prompt
    return (
        prompt[:MAX_DELIBERATION_PROMPT_CHARS]
        + "\n\n[deliberation prompt truncated by llm-council]\n"
    )
