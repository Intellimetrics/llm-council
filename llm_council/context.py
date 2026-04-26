"""Prompt and context helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


MAX_CONTEXT_FILE_CHARS = 120_000
MAX_PROMPT_CHARS = 200_000
GIT_DIFF_TIMEOUT_SECONDS = 15


def ensure_inside_cwd(path: Path, cwd: Path) -> None:
    try:
        path.resolve().relative_to(cwd.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Context file is outside working directory: {path}. "
            "Use --allow-outside-cwd only when this is intentional."
        ) from exc


def read_context_file(
    path: str | Path, *, cwd: Path, allow_outside_cwd: bool = False
) -> str:
    source = Path(path)
    if not source.is_absolute():
        source = cwd / source
    if not allow_outside_cwd:
        ensure_inside_cwd(source, cwd)
    text = source.read_text(errors="replace")
    if len(text) > MAX_CONTEXT_FILE_CHARS:
        text = text[:MAX_CONTEXT_FILE_CHARS] + "\n\n[truncated]\n"
    label = source.resolve().relative_to(cwd.resolve()) if source.resolve().is_relative_to(cwd.resolve()) else source
    return f"## File: {label}\n\n```\n{text}\n```"


def read_git_diff(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--"],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        timeout=GIT_DIFF_TIMEOUT_SECONDS,
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
    allow_outside_cwd: bool = False,
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
        "",
        "Response format:",
        "- Start with `RECOMMENDATION: yes - ...`, `RECOMMENDATION: no - ...`, or `RECOMMENDATION: tradeoff - ...`.",
        "- List the strongest reasons.",
        "- List concrete risks or things to verify.",
        "- Keep implementation suggestions read-only unless explicitly asked to write code.",
    ]

    context_sections: list[str] = []
    if include_diff:
        context_sections.append(read_git_diff(cwd))
    for item in context_paths:
        context_sections.append(
            read_context_file(item, cwd=cwd, allow_outside_cwd=allow_outside_cwd)
        )
    if stdin_text:
        context_sections.append("## Stdin Context\n\n```\n" + stdin_text + "\n```")

    if context_sections:
        sections.extend(["", "Context:", *context_sections])

    prompt = "\n".join(sections)
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = (
            prompt[:MAX_PROMPT_CHARS]
            + "\n\n[llm-council prompt truncated at "
            + str(MAX_PROMPT_CHARS)
            + " characters]\n"
        )
    return prompt
