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
    if not source.exists():
        raise ValueError(f"Context file does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Context path is not a file: {source}")
    text = source.read_text(errors="replace")
    if len(text) > MAX_CONTEXT_FILE_CHARS:
        text = text[:MAX_CONTEXT_FILE_CHARS] + "\n\n[truncated]\n"
    label = (
        source.resolve().relative_to(cwd.resolve())
        if source.resolve().is_relative_to(cwd.resolve())
        else source
    )
    return f"## File: {label}\n\n```\n{text}\n```"


def read_git_diff(cwd: Path) -> str:
    if not _git_ok(cwd, ["rev-parse", "--is-inside-work-tree"]):
        return _git_diff_unavailable("not a git repository")

    staged = _git_output(cwd, ["diff", "--cached", "--"])
    unstaged = _git_output(cwd, ["diff", "--"])
    if staged is None or unstaged is None:
        return _git_diff_unavailable("git diff failed")

    sections = ["## Git Diff"]
    if staged.strip():
        sections.extend(["", "### Staged Changes", "", "```diff", staged.strip(), "```"])
    if unstaged.strip():
        sections.extend(
            ["", "### Unstaged Changes", "", "```diff", unstaged.strip(), "```"]
        )
    if len(sections) == 1:
        sections.extend(["", "```diff", "[no diff]", "```"])
    return "\n".join(sections)


def _git_ok(cwd: Path, args: list[str]) -> bool:
    return _run_git(cwd, args).returncode == 0


def _git_output(cwd: Path, args: list[str]) -> str | None:
    result = _run_git(cwd, args)
    return result.stdout if result.returncode == 0 else None


def _run_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=GIT_DIFF_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(["git", *args], 1, "", str(exc))


def _git_diff_unavailable(reason: str) -> str:
    return f"## Git Diff\n\n```text\n[git diff unavailable: {reason}]\n```"


def build_prompt(
    question: str,
    *,
    mode: str,
    cwd: Path,
    context_paths: list[str],
    include_diff: bool,
    stdin_text: str | None,
    allow_outside_cwd: bool = False,
    max_prompt_chars: int | None = MAX_PROMPT_CHARS,
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
    if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
        prompt = (
            prompt[:max_prompt_chars]
            + "\n\n[llm-council prompt truncated at "
            + str(max_prompt_chars)
            + " characters]\n"
        )
    return prompt
