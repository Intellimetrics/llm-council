"""Council usage policy heuristics."""

from __future__ import annotations


def should_use_council(
    task: str, *, failed_attempts: int = 0, files_touched: int = 0, risk: str = "medium"
) -> tuple[bool, str, str]:
    text = task.lower()
    triggers = [
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
    if risk == "high":
        return True, "plan", "High-risk task."
    if failed_attempts >= 2:
        return True, "review", "Multiple failed attempts."
    if files_touched >= 5:
        return True, "review", "Cross-file change."
    if any(trigger in text for trigger in triggers):
        return True, "plan", "Task contains architectural or cross-cutting keywords."
    return False, "quick", "Likely small or well-scoped enough to handle directly."
