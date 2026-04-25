"""Lightweight opt-in deliberation helpers."""

from __future__ import annotations

from llm_council.adapters import ParticipantResult


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip().strip("*-_ ")
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


def has_disagreement(results: list[ParticipantResult]) -> bool:
    lines = [first_nonempty_line(result.output).lower() for result in results if result.ok]
    if len(lines) < 2:
        return False

    yes_count = sum(_contains_word(line, "yes") for line in lines)
    no_count = sum(_contains_word(line, "no") for line in lines)
    if yes_count and no_count:
        return True

    positive = sum(
        any(word in line for word in ("recommend", "should", "use", "proceed"))
        for line in lines
    )
    negative = sum(
        any(word in line for word in ("avoid", "defer", "do not", "should not"))
        for line in lines
    )
    return bool(positive and negative)


def build_deliberation_prompt(original_prompt: str, results: list[ParticipantResult]) -> str:
    excerpts = []
    for result in results:
        if not result.ok:
            continue
        excerpts.append(f"## {result.name}\n\n{result.output.strip()[:4000]}")
    return (
        f"{original_prompt}\n\n"
        "Second-round deliberation:\n"
        "The first-round participants may disagree. Read the peer responses below, "
        "identify what you still disagree with, and try to converge on a practical "
        "recommendation. If consensus is impossible, state the remaining split clearly.\n\n"
        + "\n\n".join(excerpts)
    )


def _contains_word(text: str, word: str) -> bool:
    return f" {word} " in f" {text} "
