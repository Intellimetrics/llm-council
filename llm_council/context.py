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
from llm_council.diff_chunking import (
    VALID_STRATEGIES,
    chunk_context_files,
    chunk_diff,
)


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
            f"Context file is outside working directory: {path}.\n"
            "  Pass --allow-outside-cwd if this is intentional."
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
            "",
            "Optional structured envelope (helps the council aggregate and",
            "cuts abdication noise). Emit any fields that apply, each on",
            "its own line, OUTSIDE any code fence:",
            "- `EFFORT: full|limited|blocked` — how thoroughly you analysed.",
            "- `CONFIDENCE: low|medium|high`",
            "- `RISK: low|medium|high|critical`",
            "- `BLOCKERS:` then bullet lines for each concrete missing artifact",
            "  (file, command output, policy doc) that prevented full analysis.",
            "- `EVIDENCE:` bullet lines, each tagged `[PUBLISHED]`/`[OBSERVABLE]`/`[INFERRED]`/`[SPECULATIVE]`,",
            "  or `[VERIFIED:path:start-end]` for code claims (orchestrator mechanically verifies file/range). Report findings broadly; a downstream step filters.",
            "- `FINDINGS:` optional bullets (`id`, `severity`, `claim`, `evidence`) for cross-peer dedup; not fed to round 2.",
            "- `TESTS_TO_RUN:` then bullet lines of verification commands.",
            "- `ASSUMPTIONS:` then bullet lines of stated assumptions.",
            "- `CONTINUE_DEBATE: yes|no` — unanimous `no` skips round-2.",
            "If you cannot evaluate, emit `EFFORT: blocked` AND a non-empty",
            "`BLOCKERS:` list naming what is missing. `EFFORT: blocked`",
            "without `BLOCKERS:` is treated as abdication and dropped from quorum.",
        ]
    )

    # Surface REQUIRED section markers found in the question body so peers
    # know up-front that section coverage will be enforced. The validator
    # is no-op when no markers are present, so this block stays silent for
    # ordinary prompts.
    from llm_council.sections import required_sections as _required_sections
    _detected = _required_sections(question)
    if _detected:
        head_sections.extend(
            [
                "",
                "REQUIRED sections detected in your prompt:",
                *[f"- {req['label']}" for req in _detected],
                "Each must appear in your response (literal `PART N` token or",
                "all salient title tokens within a few lines). Responses missing",
                "any required section will be retried once, then marked",
                "`error_kind: incomplete_response` and dropped from quorum.",
            ]
        )

    context_sections: list[str] = []
    diff_section_index: int | None = None
    diff_raw: str = ""
    diff_default_section: str = ""
    # Track context-file sections so we can route them through the hash-aware
    # chunker on overflow. `context_file_indices` maps section index ->
    # path label (the same label rendered into the section header).
    context_file_indices: dict[int, str] = {}
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
        rendered = read_context_file(
            item, cwd=cwd, allow_outside_cwd=allow_outside_cwd
        )
        # Derive the label the same way read_context_file did so the chunker
        # can match path-mentions in the question.
        source = Path(item)
        if not source.is_absolute():
            source = cwd / source
        try:
            label = str(source.resolve().relative_to(cwd.resolve()))
        except ValueError:
            label = str(source)
        context_file_indices[len(context_sections)] = label
        context_sections.append(rendered)
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
        # Context-files chunking is automatic (no `chunk_strategy` opt-in
        # required) because context_files are explicitly-requested attachments
        # — the user's intent is "include as much of this as fits", not "fail
        # if it doesn't". Hash-aware scoring preserves files mentioned by name
        # in the question; files larger than the available budget on their
        # own are dropped entirely with a `config_warning`-shaped progress
        # event so the operator can see what was lost.
        if context_file_indices:
            file_items: list[tuple[str, str]] = [
                (path_label, context_sections[idx])
                for idx, path_label in sorted(context_file_indices.items())
            ]
            # Compute framing exactly: assemble the prompt with each
            # context-file slot replaced by an empty string, then subtract
            # from `max_prompt_chars`. This accounts for the separators
            # added by `assemble`'s "\n".join — empty slots still take 1
            # char (the separator) each, which exactly matches the chunker's
            # accounting where M surviving sections contribute (M-1)
            # internal separators.
            scratch = list(context_sections)
            for idx in context_file_indices:
                scratch[idx] = ""
            framing_chars = len(assemble(scratch))
            file_budget = max_prompt_chars - framing_chars
            chunked = chunk_context_files(
                file_items, budget=file_budget, question=question
            )
            if chunked.triggered:
                # Rebuild context_sections with the surviving file sections
                # in their original positions. Files dropped entirely are
                # removed; other sections (images/diff/stdin) stay put.
                # Map original path_label -> surviving section text using
                # exact text equality (rendered sections are unique per file
                # since they contain the file's path in the header).
                text_to_label = {text: label for label, text in file_items}
                surviving_label_to_text: dict[str, str] = {}
                for section_text in chunked.sections:
                    label = text_to_label.get(section_text, "<unknown>")
                    surviving_label_to_text[label] = section_text
                rebuilt: list[str] = []
                for idx, section in enumerate(context_sections):
                    label = context_file_indices.get(idx)
                    if label is None:
                        rebuilt.append(section)
                        continue
                    if label in surviving_label_to_text:
                        rebuilt.append(surviving_label_to_text[label])
                    # else: file was dropped — omit entirely
                rebuilt_prompt = assemble(rebuilt)
                if chunk_progress is not None:
                    chunk_progress(
                        {
                            "event": "context_files_chunked",
                            "strategy": "hash-aware",
                            "original_chars": chunked.original_chars,
                            "chunked_chars": chunked.chunked_chars,
                            "dropped_chars": chunked.dropped_chars,
                            "dropped_files": list(chunked.dropped_files),
                            "oversize_files": list(chunked.oversize_files),
                        }
                    )
                # If the rebuilt prompt now fits, return immediately. If not,
                # fall through to diff-chunking / fail-fast paths below with
                # the updated context list.
                if len(rebuilt_prompt) <= max_prompt_chars:
                    return rebuilt_prompt
                context_sections = rebuilt
                # Recompute diff_section_index against the rebuilt list since
                # files may have been removed before the diff section.
                if diff_section_index is not None and diff_default_section:
                    try:
                        diff_section_index = context_sections.index(
                            diff_default_section
                        )
                    except ValueError:
                        diff_section_index = None
                prompt = rebuilt_prompt
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


# Families that ship with file-read / grep / glob tools enabled by the
# read-only sandbox flags baked into their CLI baseline args
# (`defaults.py:DEFAULT_CONFIG["participants"]`). Hosted families
# (openrouter / openai_compatible) and local Ollama peers never see tool
# access at the council layer, even if they're forcibly routed into a
# tool-mode run via `--include`.
_TOOL_CAPABLE_CLI_FAMILIES = frozenset({"claude", "codex", "gemini"})

REVIEW_WITH_TOOLS_DIRECTIVE = (
    "You have file-read, grep, and glob tools available via your CLI "
    "sandbox. Use them when the diff alone is insufficient to verify a "
    "claim — open the cited files, search for related callers, check "
    "surrounding context. Cite specific findings with "
    "`[VERIFIED:path:start-end]` so the orchestrator can mechanically "
    "validate them. Keep tool use proportional: investigate suspected "
    "issues, do not enumerate the entire repo. Hosted/local peers do "
    "not see this instruction."
)


def apply_per_peer_directives(
    prompt: str,
    *,
    mode: str | None,
    family: str | None,
) -> str:
    """Append per-peer prompt directives based on mode + peer family.

    Returns the prompt unchanged when no directive applies (the default
    path for `mode=None`, hosted/local peers, and every mode except
    `review-with-tools`). Backward-compatible by design — callers that
    do not need per-peer variation can ignore this function.

    Currently the only directive is the `review-with-tools` tool-use
    block, appended when:
    - `mode == "review-with-tools"`, AND
    - `family` is one of `_TOOL_CAPABLE_CLI_FAMILIES`.

    Hosted/local participants get the unchanged base prompt even when
    routed into a `review-with-tools` run (defensive — the mode SHOULD
    only route to CLI peers per the mode config, but `--include` can
    override).
    """
    if mode != "review-with-tools":
        return prompt
    if family is None or family not in _TOOL_CAPABLE_CLI_FAMILIES:
        return prompt
    return prompt + "\n\n" + REVIEW_WITH_TOOLS_DIRECTIVE
