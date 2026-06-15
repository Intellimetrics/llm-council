"""Council usage policy heuristics."""

from __future__ import annotations

from typing import Any

# Module-level trigger keyword list, shared by `should_use_council`
# (the legacy single-prose-reason heuristic) and `recommend` (the richer
# advisory enrichment). Hoisted to a constant so the two surfaces can never
# drift apart and so `recommend` can report which keywords actually matched.
COUNCIL_TRIGGER_KEYWORDS = [
    "architecture",
    "design",
    "refactor",
    "security",
    "auth",
    "database",
    "schema",
    "migration",
    "api",
    "mcp",
    "strategy",
    "tradeoff",
]

# Mechanical difficulty-class thresholds (M10). Documented rule, all
# zero-cost — no LLM call. `matched` is the count of `COUNCIL_TRIGGER_KEYWORDS`
# found in the (lowercased) task text.
#
#   hard    if risk=="high"  OR failed_attempts >= HARD_FAILED_ATTEMPTS
#                            OR files_touched   >= HARD_FILES_TOUCHED
#                            OR matched         >= HARD_MATCHED_KEYWORDS
#   trivial if risk=="low"   AND failed_attempts == 0
#                            AND files_touched  <= TRIVIAL_MAX_FILES_TOUCHED
#                            AND matched        == 0
#   moderate otherwise
HARD_FAILED_ATTEMPTS = 2
HARD_FILES_TOUCHED = 5
HARD_MATCHED_KEYWORDS = 3
TRIVIAL_MAX_FILES_TOUCHED = 1


def should_use_council(
    task: str, *, failed_attempts: int = 0, files_touched: int = 0, risk: str = "medium"
) -> tuple[bool, str, str]:
    text = task.lower()
    if risk == "high":
        return True, "plan", "High-risk task."
    if failed_attempts >= 2:
        return True, "review", "Multiple failed attempts."
    if files_touched >= 5:
        return True, "review", "Cross-file change."
    if any(trigger in text for trigger in COUNCIL_TRIGGER_KEYWORDS):
        return True, "plan", "Task contains architectural or cross-cutting keywords."
    return False, "quick", "Likely small or well-scoped enough to handle directly."


def _matched_trigger_keywords(task: str) -> list[str]:
    """Return the trigger keywords present in `task`, in catalog order."""
    text = task.lower()
    return [kw for kw in COUNCIL_TRIGGER_KEYWORDS if kw in text]


def _classify_difficulty(
    *, risk: str, failed_attempts: int, files_touched: int, matched: int
) -> str:
    """Mechanical difficulty classification — see module-level threshold doc."""
    if (
        risk == "high"
        or failed_attempts >= HARD_FAILED_ATTEMPTS
        or files_touched >= HARD_FILES_TOUCHED
        or matched >= HARD_MATCHED_KEYWORDS
    ):
        return "hard"
    if (
        risk == "low"
        and failed_attempts == 0
        and files_touched <= TRIVIAL_MAX_FILES_TOUCHED
        and matched == 0
    ):
        return "trivial"
    return "moderate"


def recommend(
    task: str,
    *,
    failed_attempts: int = 0,
    files_touched: int = 0,
    risk: str = "medium",
) -> dict[str, Any]:
    """Richer, always-on, zero-cost advisory enrichment around
    `should_use_council`.

    Returns the legacy `use_council`/`mode`/`reason` verbatim from
    `should_use_council`, plus:
      - `difficulty_class`: "trivial" | "moderate" | "hard" (mechanical rule
        combining risk, failed_attempts, files_touched, matched-keyword count).
      - `suggested_mode_reason_codes`: the LIST of trigger keywords actually
        matched in the task text (machine-actionable, vs the prose `reason`).
        Empty list when none matched.

    Advisory only: this does not change any council run or participant
    selection.
    """
    use, mode, reason = should_use_council(
        task,
        failed_attempts=failed_attempts,
        files_touched=files_touched,
        risk=risk,
    )
    matched = _matched_trigger_keywords(task)
    difficulty_class = _classify_difficulty(
        risk=risk,
        failed_attempts=failed_attempts,
        files_touched=files_touched,
        matched=len(matched),
    )
    return {
        "use_council": use,
        "mode": mode,
        "reason": reason,
        "difficulty_class": difficulty_class,
        "suggested_mode_reason_codes": matched,
    }


# ---------------------------------------------------------------------------
# L6 — reliability-based "consider dropping" advisory.
# ---------------------------------------------------------------------------

# A peer is flagged for "consider dropping" only when it has emitted more
# false blockers than useful votes AND its verified-citation rate is low.
# Peers with no verified-citation signal (`verified_citation_rate is None`)
# are skipped — absence of signal is not evidence of low quality.
LOW_VERIFIED_CITATION_RATE = 0.5


def peers_to_consider_dropping(reliability: dict[str, Any]) -> list[str]:
    """Return peer names worth considering for removal, given the output of
    `stats.aggregate_reliability`.

    A peer qualifies when:
      - `false_blocker_count > useful_count`, AND
      - `verified_citation_rate` is not None AND `< LOW_VERIFIED_CITATION_RATE`.

    Advisory only: callers must NOT drop these peers automatically. Returns
    an empty list (never raises) when there is no data or nothing qualifies.
    """
    flagged: list[str] = []
    if not isinstance(reliability, dict):
        return flagged
    for row in reliability.get("peers") or []:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not name:
            continue
        rate = row.get("verified_citation_rate")
        # No verified-citation signal => no evidence to flag on.
        if rate is None:
            continue
        try:
            false_blockers = int(row.get("false_blocker_count") or 0)
            useful = int(row.get("useful_count") or 0)
            rate_val = float(rate)
        except (TypeError, ValueError):
            continue
        if false_blockers > useful and rate_val < LOW_VERIFIED_CITATION_RATE:
            flagged.append(str(name))
    return flagged
