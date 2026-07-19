"""Prompt and context helpers."""

from __future__ import annotations

import errno
import hashlib
import locale
import mimetypes
import os
import re as _re_cross_rank
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Callable

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


MAX_PROMPT_CHARS = 200_000
MAX_CONTEXT_FILES = 32
MAX_CONTEXT_PATH_CHARS = 4_096
MAX_CONTEXT_FILE_CHARS = 120_000
MAX_CONTEXT_TOTAL_CHARS = 480_000
MAX_ACCEPTANCE_CONTRACT_CHARS = 120_000
MAX_GIT_DIFF_CAPTURE_BYTES = 2_000_000
MAX_GIT_DIFF_TOTAL_CAPTURE_BYTES = MAX_GIT_DIFF_CAPTURE_BYTES * 2
MAX_GIT_STDERR_CAPTURE_BYTES = 64_000
GIT_DIFF_TIMEOUT_SECONDS = 15
TEXT_TRUNCATION_SUFFIX = "\n\n[truncated]\n"
GIT_DIFF_TRUNCATION_PREFIX = "[git diff output truncated after "

# Defensive-review "safe context" framing (opt-in via a mode's
# `safe_context: true`; the built-in `fable` mode sets it). Motivated by
# Claude Fable 5's request-side safety classifiers, which false-positive on
# benign security-adjacent review and — on the Claude Code surface — silently
# fall back to Opus. This block states only facts the tool can actually
# vouch for: the review itself is operator-invoked, read-only, and
# analysis-only. It deliberately does NOT claim the reviewed content is the
# operator's own work or otherwise safe — the content may be a third-party
# patch or untrusted code under evaluation, and a true-positive refusal on
# genuinely malicious material must stay possible. Instead of suppressing
# suspicion it redirects it: flagging malicious code as a finding IS the
# requested output, so a peer never has to choose between engaging and
# refusing. The reasoning-visibility line also heads off Fable's
# `reasoning_extraction` refusal category — the structured response format is
# all the council consumes, so peers never need to expose raw
# chain-of-thought. Used by `build_prompt` (round 1 / deliberation task
# capsule) and by the orchestrator's ranking-pass prompt so the framing
# persists across rounds.
SAFE_CONTEXT_DIRECTIVE = (
    "Context and authorization: This is an operator-invoked, read-only "
    "second-opinion code review running inside a local developer tool "
    "(llm-council). Your role is analysis only — surfacing bugs, risks, "
    "and security weaknesses so they can be fixed; you are not being asked "
    "to build, extend, or operate anything. The material under review may "
    "be the operator's own changes or third-party code the operator is "
    "evaluating; judge it on its merits. If something appears malicious, "
    "backdoored, or dangerous, say so plainly as a finding — identifying "
    "it is the requested output, and a clear warning is more useful here "
    "than declining to look. Report findings in the requested response "
    "format; you do not need to expose your internal reasoning steps."
)
IMAGE_MIME_ALLOWLIST = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)


def _relative_label(source: Path, cwd: Path) -> str:
    """Return ``source`` relative to ``cwd`` when inside it, else the path itself.

    Centralizes the "relative-to-cwd label, else absolute path" derivation used
    when rendering context files and image manifests. The returned string is the
    same one the prior inline f-string / ``str(...)`` call sites produced.
    """

    try:
        return str(source.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(source)


def ensure_inside_cwd(path: Path, cwd: Path) -> None:
    try:
        path.resolve().relative_to(cwd.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Context file is outside working directory: {path}.\n"
            "  Pass --allow-outside-cwd if this is intentional."
        ) from exc


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Return a character-bounded prefix using the historical marker."""

    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + TEXT_TRUNCATION_SUFFIX, True


def _read_text_bounded(path: Path, max_chars: int) -> tuple[str, int, bool]:
    """Read at most ``max_chars + 1`` decoded characters from ``path``.

    The extra character is only a truncation sentinel. Text-mode reading keeps
    the limit character-based (matching the old post-read behavior) while
    avoiding an unbounded ``Path.read_text`` allocation.
    """

    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    # Council-authored/project text inputs are UTF-8 everywhere else in the
    # CLI.  Spell the encoding out here as well: relying on the process locale
    # makes a character budget become a byte/code-page budget on Windows (for
    # example, one UTF-8 ``é`` is decoded as two cp1252 characters).
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(max_chars + 1)
    text, truncated = _truncate_text(sample, max_chars)
    return text, min(len(sample), max_chars), truncated


def _render_context_file(
    path: str | Path,
    *,
    cwd: Path,
    allow_outside_cwd: bool,
    max_chars: int,
) -> tuple[str, int, bool]:
    raw_path = str(path)
    if len(raw_path) > MAX_CONTEXT_PATH_CHARS:
        raise ValueError(
            f"Context file path exceeds {MAX_CONTEXT_PATH_CHARS} characters"
        )
    source = Path(path)
    if not source.is_absolute():
        source = cwd / source
    if not allow_outside_cwd:
        ensure_inside_cwd(source, cwd)
    if not source.exists():
        raise ValueError(f"Context file does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Context path is not a file: {source}")
    text, chars_read, truncated = _read_text_bounded(source, max_chars)
    label = _relative_label(source, cwd)
    return f"## File: {label}\n\n```\n{text}\n```", chars_read, truncated


def read_context_file(
    path: str | Path, *, cwd: Path, allow_outside_cwd: bool = False
) -> str:
    rendered, _chars_read, _truncated = _render_context_file(
        path,
        cwd=cwd,
        allow_outside_cwd=allow_outside_cwd,
        max_chars=MAX_CONTEXT_FILE_CHARS,
    )
    return rendered


def resolve_acceptance_contract(
    value: str | None, *, cwd: Path, allow_outside_cwd: bool = False
) -> str | None:
    """Resolve an acceptance-contract <text|path> value to its contract text.

    Prefer file-read only when the value resolves to an existing regular file
    (inside cwd unless ``allow_outside_cwd``); otherwise treat it as literal
    contract text. So an arbitrary sentence is never misread as a path. A value
    that looks like a path but fails the cwd safety check still raises (via
    ``ensure_inside_cwd``) rather than being silently reinterpreted as text.
    """

    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    source = Path(value)
    if not source.is_absolute():
        source = cwd / source
    # Only attempt a file-read when the value resolves to an existing regular
    # file. A multi-line contract or an arbitrary sentence won't, so it falls
    # through to literal text below.
    try:
        source_is_file = source.is_file()
    except OSError as exc:
        # A long inline contract is text, not a path. POSIX stat(2) raises
        # ENAMETOOLONG before ``is_file`` can return False; treating that one
        # path-shape error as literal preserves the advertised <text|path>
        # contract without masking permission or I/O failures for real paths.
        if exc.errno != errno.ENAMETOOLONG:
            raise
        source_is_file = False
    if source_is_file:
        if not allow_outside_cwd:
            ensure_inside_cwd(source, cwd)
        text, _chars_read, _truncated = _read_text_bounded(
            source, MAX_ACCEPTANCE_CONTRACT_CHARS
        )
        return text.strip() or None
    bounded, _truncated = _truncate_text(stripped, MAX_ACCEPTANCE_CONTRACT_CHARS)
    return bounded.strip()


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
        label = _relative_label(source, cwd)
        sha256 = _hash_file_streaming(source)
        manifest.append(
            {
                "path": str(source),
                "relative_path": label,
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


def _filter_semantic_diff(diff_text: str | None) -> str:
    """Filter out lockfiles and binary/asset files from the diff to save tokens."""
    if not diff_text:
        return ""
    ignored_extensions = (
        ".lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
        ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3"
    )
    truncation_notices = [
        line
        for line in diff_text.splitlines()
        if line.startswith(GIT_DIFF_TRUNCATION_PREFIX)
    ]
    blocks = diff_text.split("diff --git ")
    filtered_blocks = []
    # The first block might be empty or preamble
    if blocks[0].strip():
        filtered_blocks.append(blocks[0])
        
    for block in blocks[1:]:
        lines = block.splitlines()
        header_line = lines[0] if lines else ""
        # Check if the block refers to an ignored file
        should_ignore = False
        for pattern in ignored_extensions:
            if pattern in header_line:
                should_ignore = True
                break
        if not should_ignore:
            filtered_blocks.append("diff --git " + block)
            
    filtered = "".join(filtered_blocks)
    for notice in truncation_notices:
        if notice not in filtered:
            filtered = filtered.rstrip() + "\n" + notice + "\n"
    return filtered


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

    staged = _filter_semantic_diff(staged)
    unstaged = _filter_semantic_diff(unstaged)

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
    if not _has_git_metadata(cwd):
        return False
    return _run_git(cwd, args).returncode == 0


def _git_output(cwd: Path, args: list[str]) -> str | None:
    if not _has_git_metadata(cwd):
        return None
    result = _run_git(cwd, args)
    return result.stdout if result.returncode == 0 else None


def _has_git_metadata(cwd: Path) -> bool:
    """Cheaply reject non-repositories before launching Git.

    Several run-planning features inspect the same diff. On Windows, spawning
    Git from a non-repository directory with inherited MCP stdio can consume
    the full subprocess timeout for every probe. A worktree always has a
    ``.git`` file or directory at its root; explicit Git environment overrides
    remain eligible and are left for Git itself to validate.
    """

    if os.environ.get("GIT_DIR") or os.environ.get("GIT_WORK_TREE"):
        return True
    try:
        resolved = cwd.resolve()
        for directory in (resolved, *resolved.parents):
            if (directory / ".git").exists():
                return True
    except OSError:
        # Fail open on an unusual filesystem and preserve Git's own diagnostics.
        return True
    return False


def _read_git_capture(
    handle: BinaryIO,
    *,
    max_bytes: int,
    stream_name: str,
) -> str:
    """Decode a bounded prefix from a temporary git output stream."""

    handle.seek(0)
    sample = handle.read(max_bytes + 1)
    truncated = len(sample) > max_bytes
    text = sample[:max_bytes].decode(
        locale.getpreferredencoding(False), errors="replace"
    )
    if truncated:
        text += (
            f"\n[git {stream_name} truncated after {max_bytes} bytes; "
            "narrow the diff before review]\n"
        )
    return text


def _run_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    git_env["GIT_PAGER"] = "cat"
    git_env["PAGER"] = "cat"
    try:
        # Direct child output to disk-backed temporary files. `capture_output`
        # buffers the entire diff in memory before this module gets a chance to
        # truncate it; temporary streams keep peak memory bounded even when a
        # repository has a multi-gigabyte generated diff. We then decode only a
        # fixed prefix for prompt construction.
        with (
            tempfile.TemporaryFile(mode="w+b") as stdout_file,
            tempfile.TemporaryFile(mode="w+b") as stderr_file,
        ):
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=git_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                timeout=GIT_DIFF_TIMEOUT_SECONDS,
            )
            stdout = _read_git_capture(
                stdout_file,
                max_bytes=MAX_GIT_DIFF_CAPTURE_BYTES,
                stream_name="diff output",
            )
            stderr = _read_git_capture(
                stderr_file,
                max_bytes=MAX_GIT_STDERR_CAPTURE_BYTES,
                stream_name="stderr",
            )
            return subprocess.CompletedProcess(
                command, completed.returncode, stdout, stderr
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 1, "", str(exc))


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
    stance: str, *, override: str | None = None, mode: str | None = None
) -> str:
    """Return the stance paragraph, preferring an explicit override."""

    if override is not None:
        text = override.strip()
        if text:
            return text
    if mode in ("adversarial-red-team", "deep-audit"):
        if stance == "against":
            return (
                "Stance: ATTACKER (Red Team). Your sole objective is to find security vulnerabilities, "
                "logic flaws, race conditions, edge-case crashes, performance regressions, or test gaps "
                "in the proposed changes. Do not be agreeable. Act as a critical adversary trying to "
                "break the code. You MUST emit `RECOMMENDATION: no` if you find any potential issues, "
                "or `RECOMMENDATION: tradeoff` if the risks are present but manageable."
            )
        elif stance == "for":
            return (
                "Stance: DEFENDER (Blue Team). Your objective is to defend the implementation. Explain "
                "why the changes are robust, how they handle edge cases, and why the proposed approach "
                "is correct and safe. However, do not blindly defend if there is a critical exploit "
                "or safety issue — invariants always apply."
            )
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
    mode: str | None = None,
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
        paragraph = resolve_stance_prompt(stance, override=override, mode=mode)
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
    acceptance_contract: str | None = None,
    safe_context: bool = False,
    chunk_strategy: str = "fail",
    chunk_progress: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Build the read-only prompt sent to each participant."""

    if len(context_paths) > MAX_CONTEXT_FILES:
        raise ValueError(
            f"Too many context files: {len(context_paths)} > {MAX_CONTEXT_FILES}"
        )
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
    # Defensive-review "safe context" framing (opt-in; the `fable` mode sets
    # it). See SAFE_CONTEXT_DIRECTIVE for the rationale. The orchestrator
    # reuses the same constant for the --cross-rank ranking prompt — the most
    # refusal-prone request of the run (it quotes peers' security findings
    # verbatim) — so the framing persists across rounds like focus directives.
    if safe_context:
        head_sections.append(SAFE_CONTEXT_DIRECTIVE)
    if prior_context:
        head_sections.extend(["", prior_context.strip()])
    head_sections.extend(
        [
            "",
            "User question:",
            question.strip(),
        ]
    )
    # Acceptance-contract gate (advisory-only). When present, peers review
    # the change ONLY against the numbered criteria below: a finding blocks
    # (RECOMMENDATION: no) only when it violates a criterion; everything else
    # is surfaced as a non-blocking concern. Counted toward max_prompt_chars
    # like the rest of head_sections (same length guard); kept small by
    # design, so no separate chunking path. Placed after the user question
    # and before the response-format block.
    if acceptance_contract and acceptance_contract.strip():
        contract_text, _contract_truncated = _truncate_text(
            acceptance_contract.strip(), MAX_ACCEPTANCE_CONTRACT_CHARS
        )
        head_sections.extend(
            [
                "",
                "ACCEPTANCE CONTRACT — Review the change ONLY against the "
                "numbered criteria below. Treat a finding as a blocker (and "
                "vote `RECOMMENDATION: no`) only when it violates one of these "
                "criteria; surface anything else as a non-blocking concern or "
                "suggestion, not a blocker.",
                "Criteria:",
                contract_text,
            ]
        )
    head_sections.extend(
        [
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

    if mode == "test-gap-analysis":
        head_sections.extend(
            [
                "",
                "TEST GAP ANALYSIS INSTRUCTION:",
                "Analyze the proposed changes and identify any missing test cases or gaps in test coverage.",
                "If logic is modified but no tests are added or updated, you should point this out.",
                "If the code change lacks tests, you MUST vote `RECOMMENDATION: no` or `RECOMMENDATION: tradeoff`",
                "and list the missing tests under the `TESTS_TO_RUN:` section.",
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
    context_chars_read = 0
    for item in context_paths:
        remaining = MAX_CONTEXT_TOTAL_CHARS - context_chars_read
        if remaining <= 0:
            raise ValueError(
                "Context files exceed aggregate character limit: "
                f"maximum is {MAX_CONTEXT_TOTAL_CHARS}"
            )
        read_limit = min(MAX_CONTEXT_FILE_CHARS, remaining)
        rendered, chars_read, truncated = _render_context_file(
            item,
            cwd=cwd,
            allow_outside_cwd=allow_outside_cwd,
            max_chars=read_limit,
        )
        if truncated and read_limit < MAX_CONTEXT_FILE_CHARS:
            raise ValueError(
                "Context files exceed aggregate character limit: "
                f"maximum is {MAX_CONTEXT_TOTAL_CHARS}"
            )
        context_chars_read += chars_read
        # Derive the label the same way read_context_file did so the chunker
        # can match path-mentions in the question.
        source = Path(item)
        if not source.is_absolute():
            source = cwd / source
        label = _relative_label(source, cwd)
        context_file_indices[len(context_sections)] = label
        context_sections.append(rendered)
    if stdin_text:
        context_sections.append("## Stdin Context\n\n```\n" + stdin_text + "\n```")

    stance_tail: list[str] = []
    if resolved_stances:
        stance_block = render_stance_section(
            resolved_stances, participants=resolved_participants, mode=mode
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

# v0.9.0 Feature 3 — additional directive appended on top of
# REVIEW_WITH_TOOLS_DIRECTIVE when `modes.review-with-tools.tool_call_voting`
# is true AND the peer is a tool-capable CLI family. `record_recommendation`
# is NOT a real MCP tool the orchestrator hosts; we rely on the CLI peer's
# own tool-calling machinery to emit a structured artifact in stdout
# (claude `tool_use` content blocks, codex JSON function-calls, gemini
# Vertex-AI flavored). `adapters._extract_tool_call_recommendation`
# detects + parses after the fact, falling back to regex
# `RECOMMENDATION:` parsing when the tool call is absent or malformed.
TOOL_CALL_VOTING_DIRECTIVE = (
    "You may additionally invoke a `record_recommendation` tool to "
    "submit your verdict as a structured payload instead of (or in "
    "addition to) the `RECOMMENDATION:` label. Schema:\n\n"
    "  record_recommendation({\n"
    '    "verdict": "yes" | "no" | "tradeoff",\n'
    '    "blockers": ["concrete missing artifact or hard requirement", ...],\n'
    '    "evidence": [{"text": "...", "tag": '
    '"verified|published|observable|inferred|speculative", '
    '"path": "...", "start_line": 0, "end_line": 0}, ...]\n'
    "  })\n\n"
    "If you emit this tool call, the orchestrator parses your "
    "recommendation from the structured payload rather than from the "
    "`RECOMMENDATION:` label. The label is still accepted as a "
    "fallback when no tool call is emitted or the payload is malformed."
)


def apply_per_peer_directives(
    prompt: str,
    *,
    mode: str | None,
    family: str | None,
    tool_call_voting: bool = False,
    stance: str | None = None,
    persona: str | None = None,
    persona_prompt: str | None = None,
    focus_directive: str | None = None,
) -> str:
    """Append per-peer prompt directives based on mode + peer family + stance + persona.

    Returns the prompt unchanged when no directive applies. Backward-compatible
    by design.

    ``focus_directive`` is the optional, operator-authored "review focus"
    block (see ``review_skills.render_focus_directive``). It is INERT PROMPT
    TEXT only — it shapes WHAT peers scrutinize and grants no tool or
    write/exec capability. It composes additively with (does not replace)
    every existing mode/stance/persona block and is appended LAST.

    # TODO(focus): existing mode-specific branches (review-with-tools /
    # stance / persona / the test-gap-analysis mode prose) could later
    # migrate to bundles; leave them wired today to avoid regressing voting.
    """
    result = prompt
    if mode == "review-with-tools" and family in _TOOL_CAPABLE_CLI_FAMILIES:
        result = result + "\n\n" + REVIEW_WITH_TOOLS_DIRECTIVE
        if tool_call_voting:
            result = result + "\n\n" + TOOL_CALL_VOTING_DIRECTIVE
            
    if stance:
        # Resolve the specific stance prompt
        stance_desc = resolve_stance_prompt(stance, mode=mode)
        result = (
            result +
            f"\n\n=== INDIVIDUAL ASSIGNMENT ===\n"
            f"You are participating under the identity representing stance: {stance.upper()}.\n"
            f"Your specific stance instructions for this run:\n{stance_desc}\n"
        )
        
    if persona_prompt:
        result = (
            result +
            f"\n\n=== CONTEXTUAL ROLE ASSIGNMENT ===\n"
            f"You have been recruited for this run due to the nature of the files changed.\n"
            f"{persona_prompt}\n"
        )

    # Operator-authored review focus, appended AFTER the existing
    # review-with-tools / stance / persona blocks so it composes with them.
    if focus_directive:
        result = result + "\n\n" + focus_directive
    return result


# --- v0.9.0 Feature 2: Anonymized cross-ranking helpers --------------------
#
# Opt-in `--cross-rank` flag (composable with ANY existing mode) runs an
# extra stage between round 1 and the optional deliberation. Each peer
# receives the OTHER peers' round-1 responses relabeled per a stable
# anonymization map ("Response A" / "Response B" / ...) and is asked to
# emit a `FINAL RANKING:` line ordering the labels from best to worst.
#
# Critical MAD-literature constraint (council risk #2): the ranking-round
# outputs MUST NOT leak into round-2 deliberation. Each ranking-round
# `ParticipantResult` is tagged `is_ranking_round=True`; the round-2
# deliberation builder filters those out. We DO NOT feed ranking
# results back to peers in-round — they are post-deliberation telemetry
# only, mirroring how the v0.8 finding-matrix is handled.

CROSS_RANK_MIN_PEERS = 2


def build_anonymization_map(peer_names: list[str]) -> dict[str, str]:
    """Build a stable name -> "Response A|B|C|..." map.

    Sort the peer names alphabetically before assigning letters so the
    map is deterministic across runs; persisting it into transcript
    metadata makes the ranking replayable / de-anonymizable by the
    operator. Labels go A, B, ..., Z, AA, AB, ... (zero peers returns
    `{}`; the orchestrator guard prevents the >26-peer case from
    arising in practice but the helper degrades gracefully).
    """
    sorted_names = sorted({n for n in peer_names if isinstance(n, str) and n})
    out: dict[str, str] = {}
    for idx, name in enumerate(sorted_names):
        out[name] = f"Response {_anonymization_label(idx)}"
    return out


def _anonymization_label(idx: int) -> str:
    # 0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, ... (excel-column style).
    out = ""
    n = idx
    while True:
        out = chr(ord("A") + (n % 26)) + out
        n = n // 26 - 1
        if n < 0:
            break
    return out


def build_ranking_prompt(
    peer_name: str,
    own_response: str,
    other_peers: dict[str, str],
    anonymization_map: dict[str, str],
    question: str,
) -> str:
    """Build the stage-2 ranking prompt sent to `peer_name`.

    Shows each OTHER peer's round-1 output relabeled per the
    `anonymization_map` (the peer's own response is excluded — no
    self-rank). Asks for a single `FINAL RANKING:` line followed by
    labels from best to worst. `own_response` is currently unused
    (peers do not self-rank) but accepted for API symmetry so future
    variants can inject the peer's prior position for context if
    needed.

    The prompt is intentionally terse: ranking is a small structural
    task, not a re-review. Long preambles risk the peer re-arguing the
    underlying question instead of ranking. We also explicitly forbid
    chain-of-thought spillage into the response body — only the
    `FINAL RANKING:` line is needed.
    """
    del own_response  # reserved for future variants; see docstring.
    if not other_peers:
        # Defensive: orchestrator already gates on `>= 2 labeled peers`,
        # but be permissive at the helper boundary.
        other_peers = {}

    lines: list[str] = [
        "You are participating in a council ranking pass.",
        "",
        "The original question was:",
        question.strip() if isinstance(question, str) else "",
        "",
        "Below are the other council members' anonymized responses.",
        "Read each one, then emit a single `FINAL RANKING:` line ranking",
        "them from best (most accurate and insightful) to worst.",
        "",
    ]
    for original_name, label in sorted(
        anonymization_map.items(), key=lambda kv: kv[1]
    ):
        if original_name == peer_name:
            continue
        body = other_peers.get(original_name, "")
        lines.append(f"{label}:")
        lines.append("```")
        lines.append(body.strip() or "(empty response)")
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "Respond with exactly one line in this format (no preamble,",
            "no analysis, no closing remarks):",
            "",
            "FINAL RANKING: <best> <next> ... <worst>",
            "",
            "Example: `FINAL RANKING: B A C` (best is B, worst is C).",
            "Use only the response letters (the part after `Response `).",
            "Keep your reply short — a single line is sufficient.",
        ]
    )
    return "\n".join(lines)


_FINAL_RANKING_LINE_RE = _re_cross_rank.compile(
    r"(?im)^[ \t]*[*_]*[ \t]*final\s+ranking[*_]*[ \t]*:[ \t]*(.+?)[ \t]*$"
)
# Inline / colon variants — captures "FINAL RANKING: B A C" or "FINAL RANKING: B, A, C"
_FINAL_RANKING_BLOCK_HEAD_RE = _re_cross_rank.compile(
    r"(?im)^[ \t]*[*_]*[ \t]*final\s+ranking[*_]*[ \t]*:[ \t]*$"
)
_NUMBERED_TOKEN_RE = _re_cross_rank.compile(
    r"^\s*(?:\d+[\.\)]\s*)?\*?\*?([A-Za-z]{1,4})\*?\*?\s*$"
)


def parse_final_ranking(
    output: str, valid_labels: set[str]
) -> list[str] | None:
    """Parse a `FINAL RANKING:` line from a peer's stage-2 output.

    Accepts (tolerant of markdown bold + bullets + commas + numbered):
    - ``FINAL RANKING: B A C``
    - ``FINAL RANKING: B, A, C``
    - ``**FINAL RANKING:** B A C``
    - ``FINAL RANKING:\n1. B\n2. A\n3. C``

    `valid_labels` is the set of bare letter labels ("A", "B", ...)
    derived from the anonymization map MINUS the responding peer's
    own label (the ranking prompt asks for n-1 entries). Returns the
    ordered list of labels (best first) or ``None`` when:
    - No `FINAL RANKING:` line is found, or
    - Extracted tokens contain duplicates, OR
    - Extracted tokens are not a permutation/subset of `valid_labels`
    """
    if not isinstance(output, str) or not output.strip():
        return None
    if not valid_labels:
        return None

    candidate_tokens: list[str] = []
    text = output

    inline_match = _FINAL_RANKING_LINE_RE.search(text)
    if inline_match:
        payload = inline_match.group(1)
        # Strip trailing markdown / punctuation noise.
        payload = payload.strip().strip("*_`")
        candidate_tokens = _split_ranking_tokens(payload)

    if not candidate_tokens:
        # Try the numbered-block form: "FINAL RANKING:\n1. B\n2. A\n3. C"
        head = _FINAL_RANKING_BLOCK_HEAD_RE.search(text)
        if head is not None:
            tail = text[head.end():].splitlines()
            for raw_line in tail:
                stripped = raw_line.strip()
                if not stripped:
                    if candidate_tokens:
                        break
                    continue
                m = _NUMBERED_TOKEN_RE.match(stripped)
                if not m:
                    if candidate_tokens:
                        break
                    continue
                candidate_tokens.append(m.group(1).upper())

    if not candidate_tokens:
        return None

    # Reject duplicates — a coherent ranking has no repeats.
    if len(set(candidate_tokens)) != len(candidate_tokens):
        return None
    # Subset/permutation check: every emitted token must be in
    # valid_labels. Missing entries are NOT auto-filled — we surface
    # the partial-rank decision up to the caller.
    upper_valid = {label.upper() for label in valid_labels}
    if not set(candidate_tokens).issubset(upper_valid):
        return None
    return candidate_tokens


def _split_ranking_tokens(payload: str) -> list[str]:
    """Tokenize a `FINAL RANKING:` payload tolerantly.

    Accepts space-separated, comma-separated, ``>`` / ``→`` separated,
    or numbered (``1. B  2. A``) forms. Returns uppercase labels with
    markdown noise stripped.
    """
    # Normalize separators to whitespace.
    normalized = payload
    for sep in (",", "→", "->", "->", ">", ";"):
        normalized = normalized.replace(sep, " ")
    raw_parts = [p for p in normalized.split() if p]
    out: list[str] = []
    for part in raw_parts:
        # Strip leading numbering ("1.", "2)") and markdown noise.
        stripped = part.strip().lstrip("0123456789.()").strip()
        stripped = stripped.strip("*_`").strip()
        if not stripped:
            continue
        # A valid ranking token is 1-4 ASCII letters.
        if not stripped.isalpha() or len(stripped) > 4:
            return []
        out.append(stripped.upper())
    return out


def compute_rank_position_means(
    anonymization_map: dict[str, str],
    rankings_by_peer: dict[str, list[str]],
) -> dict[str, float]:
    """Aggregate per-peer mean rank position from individual rankings.

    `rankings_by_peer[name]` is the ordered list of labels returned by
    that peer (best first). Position 1 = best. Each peer's score is
    the average rank position assigned to them across all OTHER peers'
    rankings. Lower is better (1.0 = unanimously ranked first).

    Returns `{peer_name: mean_position}` for every peer the anonymization
    map names that received at least one ranking from another peer. Peers
    that received zero rankings (every other peer failed to emit a parsable
    `FINAL RANKING:` line) are omitted — they have no signal to score.
    """
    # Build the reverse map ("Response A" -> "claude") and the bare-label
    # lookup ("A" -> "claude") used to translate token sequences back.
    bare_to_name: dict[str, str] = {}
    for name, full_label in anonymization_map.items():
        bare = full_label.replace("Response ", "").strip()
        if bare:
            bare_to_name[bare.upper()] = name

    accumulators: dict[str, list[int]] = {name: [] for name in anonymization_map}
    for ranker, ordered_labels in rankings_by_peer.items():
        if not isinstance(ordered_labels, list):
            continue
        for position, label in enumerate(ordered_labels, start=1):
            target_name = bare_to_name.get(str(label).upper())
            if target_name is None:
                continue
            if target_name == ranker:
                # Self-rank should not happen (prompt excludes own
                # response), but guard against malformed peer output.
                continue
            accumulators[target_name].append(position)

    means: dict[str, float] = {}
    for name, positions in accumulators.items():
        if not positions:
            continue
        means[name] = round(sum(positions) / len(positions), 4)
    return means
