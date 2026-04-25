"""Prompt and context helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


MAX_CONTEXT_FILE_CHARS = 120_000


def read_context_file(path: str | Path) -> str:
    source = Path(path)
    text = source.read_text(errors="replace")
    if len(text) > MAX_CONTEXT_FILE_CHARS:
        text = text[:MAX_CONTEXT_FILE_CHARS] + "\n\n[truncated]\n"
    return f"## File: {source}\n\n```\n{text}\n```"


def read_git_diff(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--"],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    diff = result.stdout.strip()
    return "## Git Diff\n\n```diff\n" + (diff or "[no diff]") + "\n```"


def build_prompt(
    question: str,
    *,
    mode: str,
    cwd: Path,
    context_paths: list[str],
    include_diff: bool,
    stdin_text: str | None,
) -> str:
    """Build the read-only prompt sent to each participant."""

    sections = [
        "You are a read-only participant in an LLM council for a coding project.",
        "Do not edit files. Do not run write operations. If you need code changes, propose them as recommendations only.",
        f"Working directory: {cwd}",
        f"Council mode: {mode}",
        "",
        "User question:",
        question.strip(),
    ]

    context_sections: list[str] = []
    if include_diff:
        context_sections.append(read_git_diff(cwd))
    for item in context_paths:
        context_sections.append(read_context_file(item))
    if stdin_text:
        context_sections.append("## Stdin Context\n\n```\n" + stdin_text + "\n```")

    if context_sections:
        sections.extend(["", "Context:", *context_sections])

    sections.extend(
        [
            "",
            "Response format:",
            "- Start with a one-line recommendation.",
            "- List the strongest reasons.",
            "- List concrete risks or things to verify.",
            "- Keep implementation suggestions read-only unless explicitly asked to write code.",
        ]
    )
    return "\n".join(sections)
