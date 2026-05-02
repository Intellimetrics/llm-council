"""Prompt and context helpers."""

from __future__ import annotations

import hashlib
import mimetypes
import subprocess
from pathlib import Path
from typing import Any, Callable

from llm_council.defaults import (
    DEFAULT_STANCE_PROMPTS,
    STANCE_INVARIANT_SUFFIX,
    VALID_STANCES,
)
from llm_council.diff_chunking import VALID_STRATEGIES, chunk_diff


MAX_CONTEXT_FILE_CHARS = 120_000
MAX_PROMPT_CHARS = 200_000
GIT_DIFF_TIMEOUT_SECONDS = 15
IMAGE_MIME_ALLOWLIST = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)


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


def resolve_image_path(
    path: str | Path, *, cwd: Path, allow_outside_cwd: bool = False
) -> tuple[Path, str, int]:
    """Validate an image path and return (resolved path, mime, size)."""

    source = Path(path)
    if not source.is_absolute():
        source = cwd / source
    if not allow_outside_cwd:
        ensure_inside_cwd(source, cwd)
    if not source.exists():
        raise ValueError(f"Image path does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Image path is not a file: {source}")
    mime, _ = mimetypes.guess_type(str(source))
    if mime is None:
        raise ValueError(f"Unable to detect mime type for image: {source}")
    if mime not in IMAGE_MIME_ALLOWLIST:
        raise ValueError(
            f"Image mime '{mime}' is not allowed for {source}. "
            f"Allowed: {', '.join(sorted(IMAGE_MIME_ALLOWLIST))}."
        )
    return source, mime, source.stat().st_size


def _hash_file_streaming(path: Path, *, chunk_size: int = 1 << 16) -> str:
    """Compute sha256 without loading the whole file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_image_manifest(
    image_paths: list[str], *, cwd: Path, allow_outside_cwd: bool = False
) -> list[dict[str, Any]]:
    """Resolve and hash each image path so adapters and transcripts see the same view."""

    manifest: list[dict[str, Any]] = []
    for item in image_paths:
        source, mime, size = resolve_image_path(
            item, cwd=cwd, allow_outside_cwd=allow_outside_cwd
        )
        try:
            label = source.resolve().relative_to(cwd.resolve())
        except ValueError:
            label = source
        sha256 = _hash_file_streaming(source)
        manifest.append(
            {
                "path": str(source),
                "relative_path": str(label),
                "mime": mime,
                "size": size,
                "sha256": sha256,
            }
        )
    return manifest


def render_image_section(manifest: list[dict[str, Any]]) -> str:
    if not manifest:
        return ""
    lines = [
        "## Images",
        "",
        (
            "The host has staged the following images for council review. "
            "If you are running as a CLI subprocess on the project filesystem, "
            "open these paths with your file-read tool. If you received the "
            "same images attached to this message (vision-capable hosted "
            "models), refer to them by their relative path in your response."
        ),
        "",
    ]
    for entry in manifest:
        label = entry.get("relative_path") or entry.get("path") or "?"
        mime = entry.get("mime") or "?"
        size = entry.get("size")
        size_str = f"{size} bytes" if size is not None else "?"
        lines.append(f"- `{label}` ({mime}, {size_str})")
    return "\n".join(lines)


def read_git_diff(cwd: Path) -> str:
    sections, _raw = _read_git_diff_sections(cwd)
    return "\n".join(sections)


def _read_git_diff_sections(cwd: Path) -> tuple[list[str], str]:
    """Return (markdown sections, raw concatenated diff text).

    The raw text is the union of staged and unstaged diff bodies (separated
    by a blank line) so callers can apply chunking on the underlying unified
    diff before it is re-wrapped in markdown.
    """

    if not _git_ok(cwd, ["rev-parse", "--is-inside-work-tree"]):
        return [_git_diff_unavailable("not a git repository")], ""

    staged = _git_output(cwd, ["diff", "--cached", "--"])
    unstaged = _git_output(cwd, ["diff", "--"])
    if staged is None or unstaged is None:
        return [_git_diff_unavailable("git diff failed")], ""

    sections = ["## Git Diff"]
    if staged.strip():
        sections.extend(["", "### Staged Changes", "", "```diff", staged.strip(), "```"])
    if unstaged.strip():
        sections.extend(
            ["", "### Unstaged Changes", "", "```diff", unstaged.strip(), "```"]
        )
    if len(sections) == 1:
        sections.extend(["", "```diff", "[no diff]", "```"])
    raw_parts = [text.strip() for text in (staged, unstaged) if text and text.strip()]
    raw = "\n\n".join(raw_parts)
    return sections, raw


def _wrap_chunked_diff(chunked_text: str) -> str:
    return "\n".join(["## Git Diff", "", "```diff", chunked_text.strip("\n"), "```"])


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


def _resolve_stance_inputs(
    *,
    mode: str,
    cwd: Path,
    stances: dict[str, str] | None,
    participants: dict[str, dict[str, Any]] | None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Resolve per-participant stances, optionally loading from project config.

    Explicit kwargs win. When `stances` is None we attempt a best-effort lookup
    in the project config under cwd; failures fall through silently so the
    existing prompt structure is preserved.
    """

    if stances is not None:
        return dict(stances), dict(participants or {})
    try:
        from llm_council.config import find_config, load_config

        config_path = find_config(cwd)
        if not config_path:
            return {}, {}
        loaded = load_config(config_path)
    except (OSError, ValueError):
        return {}, {}
    mode_cfg = loaded.get("modes", {}).get(mode, {}) if isinstance(loaded, dict) else {}
    if not isinstance(mode_cfg, dict):
        return {}, {}
    raw_stances = mode_cfg.get("stances")
    if not isinstance(raw_stances, dict) or not raw_stances:
        return {}, {}
    raw_participants = loaded.get("participants", {})
    if not isinstance(raw_participants, dict):
        raw_participants = {}
    return dict(raw_stances), dict(raw_participants)


def resolve_stance_prompt(
    stance: str, *, override: str | None = None
) -> str:
    """Return the stance paragraph, preferring an explicit override."""

    if override is not None:
        text = override.strip()
        if text:
            return text
    if stance in DEFAULT_STANCE_PROMPTS:
        return DEFAULT_STANCE_PROMPTS[stance]
    raise ValueError(
        f"Unknown stance '{stance}'. Expected one of: "
        f"{', '.join(VALID_STANCES)}"
    )


def _sanitize_identifier(value: str, *, max_chars: int = 64) -> str:
    forbidden = {"`", "\n", "\r", "\t", "#", "*", "_", "[", "]"}
    cleaned = "".join(
        ch for ch in str(value) if ch.isprintable() and ch not in forbidden
    ).strip()
    if not cleaned:
        return "unknown"
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def render_stance_section(
    stances: dict[str, str],
    *,
    participants: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the per-participant Stance Assignments block."""

    if not stances:
        return ""
    participants = participants or {}
    lines = [
        "## Stance Assignments",
        "",
        (
            "This is a consensus-mode council where each participant has been "
            "assigned a stance to attack groupthink and sycophancy. Find the "
            "row matching your CLI / model identity below and adopt that "
            "stance for this response. If you cannot identify which row "
            "applies to you, default to the `neutral` stance."
        ),
        "",
    ]
    for name, stance in stances.items():
        if stance not in VALID_STANCES:
            raise ValueError(
                f"Stance for '{name}' must be one of {', '.join(VALID_STANCES)}"
            )
        cfg = participants.get(name) or {}
        family = cfg.get("family") or name
        override = cfg.get("stance_prompt")
        paragraph = resolve_stance_prompt(stance, override=override)
        safe_name = _sanitize_identifier(name)
        safe_family = _sanitize_identifier(family)
        lines.append(
            f"### Participant `{safe_name}` (family: {safe_family}) — Stance: {stance}"
        )
        lines.append("")
        lines.append(paragraph)
        lines.append("")
        lines.append(STANCE_INVARIANT_SUFFIX)
        lines.append("")
    return "\n".join(lines).rstrip()


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
    image_paths: list[str] | None = None,
    image_manifest: list[dict[str, Any]] | None = None,
    stances: dict[str, str] | None = None,
    participants: dict[str, dict[str, Any]] | None = None,
    prior_context: str | None = None,
    chunk_strategy: str = "fail",
    chunk_progress: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Build the read-only prompt sent to each participant."""

    if chunk_strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown chunk_strategy '{chunk_strategy}'. "
            f"Expected one of: {', '.join(VALID_STRATEGIES)}"
        )

    resolved_stances, resolved_participants = _resolve_stance_inputs(
        mode=mode,
        cwd=cwd,
        stances=stances,
        participants=participants,
    )

    head_sections = [
        "You are a read-only participant in an LLM council for a coding project.",
        "Do not edit files. Do not run write operations. If you need code changes, propose them as recommendations only.",
        f"Working directory: {cwd}",
        f"Council mode: {mode}",
    ]
    if prior_context:
        head_sections.extend(["", prior_context.strip()])
    head_sections.extend(
        [
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
    )

    context_sections: list[str] = []
    diff_section_index: int | None = None
    diff_raw: str = ""
    diff_default_section: str = ""
    manifest_for_render = image_manifest
    if manifest_for_render is None and image_paths:
        manifest_for_render = build_image_manifest(
            image_paths, cwd=cwd, allow_outside_cwd=allow_outside_cwd
        )
    if manifest_for_render:
        context_sections.append(render_image_section(manifest_for_render))
    if include_diff:
        diff_lines, diff_raw = _read_git_diff_sections(cwd)
        diff_default_section = "\n".join(diff_lines)
        diff_section_index = len(context_sections)
        context_sections.append(diff_default_section)
    for item in context_paths:
        context_sections.append(
            read_context_file(item, cwd=cwd, allow_outside_cwd=allow_outside_cwd)
        )
    if stdin_text:
        context_sections.append("## Stdin Context\n\n```\n" + stdin_text + "\n```")

    stance_tail: list[str] = []
    if resolved_stances:
        stance_block = render_stance_section(
            resolved_stances, participants=resolved_participants
        )
        if stance_block:
            stance_tail = ["", stance_block]

    def assemble(ctx: list[str]) -> str:
        sections = list(head_sections)
        # Stance must precede Context: so that (a) the round-2 deliberation
        # `_strip_context_payload` (rfind on `\n\nContext:\n`) does not also
        # strip the stance block, and (b) hard end-truncation falls on
        # context, not on the stance assignments.
        sections.extend(stance_tail)
        if ctx:
            sections.extend(["", "Context:", *ctx])
        return "\n".join(sections)

    prompt = assemble(context_sections)
    if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
        if (
            chunk_strategy != "fail"
            and include_diff
            and diff_section_index is not None
            and diff_raw
        ):
            empty_wrapper = _wrap_chunked_diff("")
            rest_chars = len(prompt) - len(diff_default_section)
            wrapper_overhead = len(empty_wrapper)
            budget = max_prompt_chars - rest_chars - wrapper_overhead
            if budget > 0:
                chunk = chunk_diff(
                    diff_raw,
                    strategy=chunk_strategy,
                    budget=budget,
                    question=question,
                )
                if chunk.triggered:
                    rebuilt = list(context_sections)
                    rebuilt[diff_section_index] = _wrap_chunked_diff(chunk.text)
                    rebuilt_prompt = assemble(rebuilt)
                    if len(rebuilt_prompt) <= max_prompt_chars:
                        if chunk_progress is not None:
                            chunk_progress(
                                {
                                    "event": "diff_chunked",
                                    "strategy": chunk.strategy,
                                    "original_chars": chunk.original_chars,
                                    "chunked_chars": chunk.chunked_chars,
                                    "dropped_chars": chunk.dropped_chars,
                                    "dropped_files": list(chunk.dropped_files),
                                }
                            )
                        return rebuilt_prompt
        if prior_context:
            raise ValueError(
                "Continuation prompt exceeds max_prompt_chars: "
                f"{len(prompt)} > {max_prompt_chars}. The prior council "
                "context is preserved verbatim by design. Drop --context/--diff, "
                "shorten the new question, or run without --continue / "
                "continuation_id."
            )
        # Fail-fast on overflow. Silently truncating the tail used to drop
        # stance assignments and let the council answer from a partial diff
        # without surfacing it; both modes need the caller to know.
        if chunk_strategy == "fail":
            raise ValueError(
                f"Prompt exceeds max_prompt_chars: {len(prompt)} > "
                f"{max_prompt_chars}. Either pass --chunk-strategy "
                "{head|tail|hash-aware} to attempt chunking, raise "
                "max_prompt_chars in config, or shorten --diff/--context."
            )
        raise ValueError(
            f"Prompt exceeds max_prompt_chars: {len(prompt)} > "
            f"{max_prompt_chars}. Chunk strategy '{chunk_strategy}' "
            "could not produce a fitting prompt — non-diff context "
            "(files, stdin, prior council context) alone exceeds the "
            "budget. Raise max_prompt_chars or drop --context/--diff."
        )
    return prompt
