"""Lightweight opt-in deliberation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from llm_council.adapters import RECOMMENDATION_RE, ParticipantResult

MAX_DELIBERATION_PROMPT_CHARS = 80_000
DELIBERATION_TRUNCATION_SUFFIX = (
    "\n\n[deliberation prompt truncated by llm-council]\n"
)


def deliberation_body_budget(
    effective_prompt_cap: int | None, directive_suffix_chars: int
) -> int:
    """Derive the round-2 body budget from the cap that governs the run.

    ``run_participants`` appends per-peer directive suffixes AFTER the
    deliberation body is built, so the final prompt a peer receives is
    ``body + suffix``. For that sum to respect both the builder ceiling
    (``MAX_DELIBERATION_PROMPT_CHARS``) and the surface's effective prompt
    cap (``defaults.max_prompt_chars`` / ``mcp_max_prompt_chars``), the
    body budget must be the smaller of the two minus the largest suffix
    that will be appended. Derived here — never tuned per peer or per cap —
    so the invariant holds for any directive length and any configured cap.
    """

    ceiling = MAX_DELIBERATION_PROMPT_CHARS
    if effective_prompt_cap is not None:
        ceiling = min(ceiling, int(effective_prompt_cap))
    return max(
        len(DELIBERATION_TRUNCATION_SUFFIX),
        ceiling - max(0, int(directive_suffix_chars)),
    )
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


_NO_LABEL_PLACEHOLDER = "(no RECOMMENDATION label emitted)"


def recommendation_line(text: str) -> str:
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if RECOMMENDATION_RE.match(line):
            return line.strip()
    # No out-of-fence label = no usable vote. Return an explicit
    # placeholder so the next-round prompt clearly shows the peer had no
    # position, instead of injecting arbitrary intro prose (the previous
    # `first_nonempty_line` fallback would echo "Here is my analysis:"
    # into the round-2 summary as if it were a recommendation).
    return _NO_LABEL_PLACEHOLDER


def recommendation_label(text: str) -> str:
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = RECOMMENDATION_RE.match(line)
        if match:
            return match.group(1).lower()
    return "unknown"


def recommendation_counts(results: list[ParticipantResult]) -> dict[str, int]:
    counts = {"yes": 0, "no": 0, "tradeoff": 0, "unknown": 0}
    for result in results:
        if not result.ok:
            continue
        counts[recommendation_label(result.output)] += 1
    return counts


@dataclass(frozen=True)
class RecommendationSummary:
    """Machine-facing summary of a set of peer votes.

    ``recommendation`` is emitted for a unique trinary leader. A tie
    between a definite label and ``tradeoff`` — with ZERO votes for the
    opposing definite label — reports the direction as ``leaning-yes`` /
    ``leaning-no``: the peers agree on posture and differ only in label
    strength, and ``unknown`` undersells that (2026-08 field issue #4).
    Any tie involving true yes/no opposition (or a three-way tie) stays
    ``unknown`` rather than relying on iteration order (which previously
    biased unresolved yes/no ties toward ``yes``). ``tied`` remains True
    for leaning outcomes — there is still no unique leader, and
    ``agreement_count`` stays 0 for the same reason.
    """

    recommendation: str
    agreement_count: int
    total_labeled: int
    counts: dict[str, int]
    tied: bool


def summarize_recommendations(
    results: list[ParticipantResult],
) -> RecommendationSummary:
    """Summarize votes, requiring a unique leader for a recommendation.

    Callers holding cumulative multi-round results should pass only their
    final-round view (for example ``transcript.final_round_results(results)``).
    """

    counts = recommendation_counts(results)
    labels = ("yes", "no", "tradeoff")
    total_labeled = sum(counts[label] for label in labels)
    if total_labeled == 0:
        return RecommendationSummary(
            recommendation="unknown",
            agreement_count=0,
            total_labeled=0,
            counts=counts,
            tied=False,
        )

    top_count = max(counts[label] for label in labels)
    leaders = [label for label in labels if counts[label] == top_count]
    if len(leaders) != 1:
        recommendation = "unknown"
        definite_leaders = [label for label in leaders if label != "tradeoff"]
        if len(leaders) == 2 and len(definite_leaders) == 1:
            # {yes, tradeoff} or {no, tradeoff} tie. Directional ONLY when
            # the opposing definite label drew zero votes — every labeled
            # peer sits on the same side of the yes/no axis.
            direction = definite_leaders[0]
            opposition = "no" if direction == "yes" else "yes"
            if counts[opposition] == 0:
                recommendation = f"leaning-{direction}"
        return RecommendationSummary(
            recommendation=recommendation,
            agreement_count=0,
            total_labeled=total_labeled,
            counts=counts,
            tied=True,
        )

    recommendation = leaders[0]
    return RecommendationSummary(
        recommendation=recommendation,
        agreement_count=top_count,
        total_labeled=total_labeled,
        counts=counts,
        tied=False,
    )


def labeled_quorum_count(results: list[ParticipantResult]) -> int:
    """Number of results that produced a usable trinary label."""
    counts = recommendation_counts(results)
    return counts["yes"] + counts["no"] + counts["tradeoff"]


def default_min_quorum(participant_count: int) -> int:
    """Default trust threshold: 2 if 3+ peers, else clamp to peer count."""
    if participant_count >= 3:
        return 2
    return max(1, participant_count)


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

    Round-2 peers need the task wording — output constraints, stance, etc. —
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
    original_prompt: str,
    results: list[ParticipantResult],
    *,
    max_chars: int | None = None,
) -> tuple[str, list[str]]:
    """Build a slim round-2 prompt + list of peers whose excerpts were truncated.

    Earlier versions re-sent ``original_prompt`` (including any ``--diff``
    payload) on every round, paying for the same tokens repeatedly. We now
    keep the question/instructions but drop the bulky ``Context:`` section
    (diff, files, stdin) since peers reasoned over it in round 1 and their
    excerpts carry forward the relevant findings.

    ``max_chars`` is the body budget — callers pass
    :func:`deliberation_body_budget` so the per-peer directive suffixes
    appended downstream still fit inside the run's prompt cap. ``None``
    falls back to the bare builder ceiling.
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
        "Now read the peer responses below and identify what, if anything, "
        "you still disagree with. Converge toward what is actually correct, "
        "not toward agreement for its own sake. Do not change your "
        "recommendation merely to side with the group, and do not hold your "
        "position merely to stay consistent with your earlier answer — move "
        "only toward the truth. If you change your recommendation, name the "
        "specific peer point that convinced you; if you hold your position, "
        "name the strongest peer argument against you and say why it does not "
        "move you. "
        "Focus your critique on the OTHER peers' responses rather than "
        "re-justifying your own prior answer. If consensus is impossible, "
        "state the remaining split clearly. "
        "Start your reply with `RECOMMENDATION: yes - ...`, "
        "`RECOMMENDATION: no - ...`, or `RECOMMENDATION: tradeoff - ...`. "
        "The optional envelope fields (`EFFORT:`, `CONFIDENCE:`, `RISK:`, "
        "`BLOCKERS:`, `EVIDENCE:`, `TESTS_TO_RUN:`, `ASSUMPTIONS:`) apply "
        "here too. If you cannot evaluate, emit `EFFORT: blocked` plus a "
        "non-empty `BLOCKERS:` list naming what is missing — blocked "
        "without blockers is treated as abdication and dropped from quorum.",
    ]
    prompt = "\n".join(pointer_lines) + "\n\n" + "\n\n".join(excerpts)
    budget = (
        MAX_DELIBERATION_PROMPT_CHARS
        if max_chars is None
        else max(len(DELIBERATION_TRUNCATION_SUFFIX), int(max_chars))
    )
    if len(prompt) <= budget:
        return prompt, truncated_peers
    return (
        prompt[: budget - len(DELIBERATION_TRUNCATION_SUFFIX)].rstrip()
        + DELIBERATION_TRUNCATION_SUFFIX,
        truncated_peers,
    )
