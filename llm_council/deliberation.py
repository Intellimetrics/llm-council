"""Lightweight opt-in deliberation helpers."""

from __future__ import annotations

from llm_council.adapters import RECOMMENDATION_RE, ParticipantResult

MAX_DELIBERATION_PROMPT_CHARS = 80_000
# Per-peer excerpt cap for deliberation rounds. Sized so a 3-peer council
# fits inside MAX_DELIBERATION_PROMPT_CHARS alongside the question text and
# pointer preamble (the bulky `Context:` payload from round 1 is stripped);
# raise if peer responses are getting cut off in the second round.
MAX_DELIBERATION_PEER_EXCERPT_CHARS = 20_000


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip().strip(">*-_ ")
        if cleaned:
            return cleaned
    return ""


def model_comparison(results: list[ParticipantResult]) -> list[str]:
    from llm_council.adapters import is_timeout_error

    lines: list[str] = []
    for result in results:
        if not result.ok:
            label = "timeout" if is_timeout_error(result.error) else "error"
            lines.append(f"- {result.name}: {label} - {result.error}")
            continue
        usage = []
        if result.total_tokens is not None:
            usage.append(f"{result.total_tokens} tokens")
        if result.cost_usd is not None:
            usage.append(f"${result.cost_usd:.6f}")
        suffix = f" ({', '.join(usage)})" if usage else ""
        lines.append(f"- {result.name}: {recommendation_line(result.output)}{suffix}")
    return lines


def recommendation_line(text: str) -> str:
    fenced_line = ""
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not RECOMMENDATION_RE.match(line):
            continue
        cleaned = line.strip()
        if not in_fence:
            return cleaned
        if not fenced_line:
            fenced_line = cleaned
    return fenced_line or first_nonempty_line(text)


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


MAX_RECOMMENDATION_LABEL_CHARS = 240


def _truncate_at_line_boundary(text: str, limit: int) -> tuple[str, bool]:
    """Truncate ``text`` so it ends at a newline at or before ``limit``.

    Returns ``(maybe_truncated_text, was_truncated)``. Falls back to the
    hard character limit when the last newline within ``limit`` is so early
    that snapping to it would discard most of the budget (e.g., a header
    plus a single monolithic body) — preserving more useful content.
    """
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped, False
    head = stripped[:limit]
    boundary = head.rfind("\n")
    if boundary >= limit // 2:
        return head[:boundary].rstrip(), True
    return head, True


def _strip_context_payload(original_prompt: str) -> str:
    """Drop the ``Context:`` section (diff, files, stdin) but keep the question.

    Round-2 peers need the task wording — output constraints, persona, etc. —
    but not the bulky diff/file blobs that are paid for in round 1. The
    prompt builder always emits a literal ``\\nContext:\\n`` line before
    these sections; if it is absent (no context attached), return the prompt
    unchanged.
    """
    marker = "\n\nContext:\n"
    # rfind: the real `Context:` block is always the last section, so
    # searching from the end avoids stripping a question that quotes the
    # marker text itself.
    idx = original_prompt.rfind(marker)
    if idx == -1:
        return original_prompt
    return original_prompt[:idx]


def build_deliberation_prompt(
    original_prompt: str, results: list[ParticipantResult]
) -> tuple[str, list[str]]:
    """Build a slim round-2 prompt + list of peers whose excerpts were truncated.

    Earlier versions re-sent ``original_prompt`` (including any ``--diff``
    payload) on every round, paying for the same tokens repeatedly. We now
    keep the question/instructions but drop the bulky ``Context:`` section
    (diff, files, stdin) since peers reasoned over it in round 1 and their
    excerpts carry forward the relevant findings.
    """
    truncated_peers: list[str] = []
    excerpts = []
    label_lines = []
    for result in results:
        if not result.ok:
            continue
        excerpt, was_truncated = _truncate_at_line_boundary(
            result.output, MAX_DELIBERATION_PEER_EXCERPT_CHARS
        )
        if was_truncated:
            truncated_peers.append(result.name)
        excerpts.append(f"## {result.name}\n\n{excerpt}")
        label = recommendation_line(result.output)
        if len(label) > MAX_RECOMMENDATION_LABEL_CHARS:
            label = label[:MAX_RECOMMENDATION_LABEL_CHARS].rstrip() + "..."
        label_lines.append(f"- {result.name}: {label}")

    task_capsule = _strip_context_payload(original_prompt).rstrip()

    pointer_lines = [
        "Second-round deliberation:",
        "",
        "You answered the question below in an earlier round of an llm-council, "
        "alongside the peers listed. The original code context (diff/files) is "
        "not repeated here to save tokens; rely on the peer excerpts for any "
        "specifics that matter. Peer RECOMMENDATION labels from the prior round:",
        "",
        *label_lines,
        "",
        "Original task:",
        "",
        task_capsule,
        "",
        "Now read the peer responses below, identify what you still disagree "
        "with, and try to converge on a practical recommendation. If consensus "
        "is impossible, state the remaining split clearly. "
        "Start your reply with `RECOMMENDATION: yes - ...`, "
        "`RECOMMENDATION: no - ...`, or `RECOMMENDATION: tradeoff - ...`.",
    ]
    prompt = "\n".join(pointer_lines) + "\n\n" + "\n\n".join(excerpts)
    if len(prompt) <= MAX_DELIBERATION_PROMPT_CHARS:
        return prompt, truncated_peers
    return (
        prompt[:MAX_DELIBERATION_PROMPT_CHARS]
        + "\n\n[deliberation prompt truncated by llm-council]\n",
        truncated_peers,
    )
