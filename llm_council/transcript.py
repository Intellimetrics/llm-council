"""Transcript writing."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_council.adapters import (
    ParticipantResult,
    command_for_display,
    is_context_overflow_error,
    is_timeout_error,
)
from llm_council.convergence import tally_states
from llm_council.deliberation import (
    default_min_quorum,
    labeled_quorum_count,
    model_comparison,
    recommendation_counts,
    recommendation_label,
    recommendation_line,
    summarize_recommendations,
)

ROUND_SUFFIX_RE = re.compile(r":round(\d+)$")


def transcript_paths(base_dir: Path, question: str) -> tuple[Path, Path]:
    # Keep the timestamp prefix for sorting and continuation-prefix lookup, but
    # never place question text in a filename. Questions can contain source,
    # incident details, or credentials; directory listings must not disclose
    # them even when the transcript files themselves are private.
    _ = question  # retained for API compatibility; intentionally not used in paths
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{stamp}_{uuid.uuid4().hex}"
    return base_dir / f"{stem}.md", base_dir / f"{stem}.json"


def _current_euid() -> int | None:
    get_euid = getattr(os, "geteuid", None)
    return int(get_euid()) if get_euid is not None else None


def _uses_windows_path_fallback() -> bool:
    """Return whether directory descriptors are unavailable on this platform."""

    return os.name == "nt"


def _is_link_or_reparse_point(path: Path, info: os.stat_result | None = None) -> bool:
    """Recognize symlinks and Windows junction/reparse-point entries."""

    try:
        observed = info if info is not None else os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(observed.st_mode):
        return True
    attributes = getattr(observed, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _prepare_transcript_directory_path(
    path: Path,
    *,
    create: bool,
    make_private: bool,
) -> os.stat_result:
    """Windows-safe path fallback for transcript-directory validation.

    Windows does not support opening a directory with :func:`os.open` or
    passing that descriptor to :func:`os.scandir`. Reject reparse points,
    validate the entry before and after chmod, and return the final identity so
    callers can detect a directory swap before committing an atomic replace.
    POSIX callers never use this helper and retain descriptor-relative,
    ``O_NOFOLLOW`` operations.
    """

    path = Path(path)
    if _is_link_or_reparse_point(path):
        raise OSError(f"Refusing symlink or reparse-point transcript directory: {path}")
    if create:
        try:
            path.mkdir(parents=True, mode=0o700, exist_ok=True)
        except FileExistsError:
            pass

    listed = os.lstat(path)
    if _is_link_or_reparse_point(path, listed):
        raise OSError(f"Refusing symlink or reparse-point transcript directory: {path}")
    if not stat.S_ISDIR(listed.st_mode):
        raise NotADirectoryError(f"Transcript path is not a directory: {path}")
    if make_private:
        # Windows' chmod surface is narrower than POSIX ACLs, but this keeps the
        # path compatible with restrictive parent ACLs and is the strongest
        # stdlib-only permission request available. The no-follow guarantees in
        # this branch come from repeated lstat/reparse checks and identity
        # validation; POSIX continues to use fchmod on a pinned descriptor.
        os.chmod(path, 0o700)

    current = os.lstat(path)
    if _is_link_or_reparse_point(path, current):
        raise OSError(f"Refusing symlink or reparse-point transcript directory: {path}")
    if not stat.S_ISDIR(current.st_mode):
        raise NotADirectoryError(f"Transcript path is not a directory: {path}")
    if (listed.st_dev, listed.st_ino) != (current.st_dev, current.st_ino):
        raise OSError(f"Transcript directory changed while preparing it: {path}")
    return current


def _open_owned_transcript_directory(
    path: Path,
    *,
    create: bool,
    make_private: bool,
) -> int:
    """Open ``path`` as an owned directory without following a leaf symlink.

    The returned descriptor pins subsequent operations to the directory that
    was inspected, so replacing the pathname after this check cannot redirect a
    transcript write. On POSIX, ownership is required before permissions are
    changed; silently chmod'ing a shared or foreign-owned directory would be a
    dangerous repair policy.
    """

    path = Path(path)
    if path.is_symlink():
        raise OSError(f"Refusing symlink transcript directory: {path}")
    if create:
        try:
            path.mkdir(parents=True, mode=0o700, exist_ok=True)
        except FileExistsError:
            # A concurrent creator won the race. The no-follow open and owner
            # validation below decide whether its directory is safe to use.
            pass

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise OSError(f"Refusing symlink transcript directory: {path}") from exc
        raise

    try:
        opened = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise NotADirectoryError(f"Transcript path is not a directory: {path}")

        # Defense for platforms without O_NOFOLLOW and a consistency check for
        # the normal path: lstat must describe the same non-symlink inode that
        # the descriptor pins.
        listed = os.lstat(path)
        if stat.S_ISLNK(listed.st_mode):
            raise OSError(f"Refusing symlink transcript directory: {path}")
        if (listed.st_dev, listed.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(f"Transcript directory changed while opening it: {path}")

        euid = _current_euid()
        if euid is not None and opened.st_uid != euid:
            raise PermissionError(
                f"Refusing transcript directory not owned by uid {euid}: {path} "
                f"(owner uid {opened.st_uid})"
            )
        if make_private:
            if hasattr(os, "fchmod"):
                os.fchmod(directory_fd, 0o700)
            else:  # pragma: no cover - Windows fallback
                os.chmod(path, 0o700, follow_symlinks=False)
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def ensure_private_transcript_dir(path: Path) -> Path:
    """Create or tighten an owned transcript directory to mode 0700.

    Existing directories are repaired even when they were created under a
    permissive umask. Leaf symlinks and foreign-owned directories are refused.
    """

    if _uses_windows_path_fallback():
        _prepare_transcript_directory_path(
            Path(path), create=True, make_private=True
        )
        return Path(path)

    directory_fd = _open_owned_transcript_directory(
        Path(path), create=True, make_private=True
    )
    os.close(directory_fd)
    return Path(path)


def transcript_dir_within_root(cwd: Path, config: dict, *, root: Path) -> Path:
    """Resolve ``transcripts_dir`` and require it to remain inside ``root``.

    This is the confinement variant intended for MCP callers. The ordinary
    :func:`transcript_dir` remains permissive for explicit CLI configurations
    that intentionally store transcripts elsewhere. Resolving before the
    containment check also rejects an existing symlink that escapes the root.
    """

    resolved_root = Path(root).resolve()
    resolved_cwd = Path(cwd).resolve()
    try:
        resolved_cwd.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"working directory must be inside transcript root: {resolved_root}"
        ) from exc
    resolved = transcript_dir(resolved_cwd, config).resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"transcripts_dir must be inside MCP project root: {resolved_root}"
        ) from exc
    return resolved


_TRANSCRIPT_ARTIFACT_SUFFIXES = frozenset({".md", ".json", ".html"})


def _is_transcript_artifact_name(name: str) -> bool:
    candidate = Path(name)
    return (
        candidate.suffix.lower() in _TRANSCRIPT_ARTIFACT_SUFFIXES
        and _RUN_ID_RE.match(candidate.stem) is not None
    )


def _new_permission_report(base_dir: Path) -> dict[str, Any]:
    return {
        "directory": str(base_dir),
        "directory_mode_before": None,
        "directory_repaired": False,
        "eligible_files": 0,
        "already_private_files": [],
        "repaired_files": [],
        "would_repair_files": [],
        "skipped_symlinks": [],
        "skipped_hardlinks": [],
        "skipped_unowned": [],
        "skipped_non_regular": [],
        "skipped_changed": [],
    }


def _sort_permission_report(report: dict[str, Any]) -> dict[str, Any]:
    for value in report.values():
        if isinstance(value, list):
            value.sort()
    return report


def _inspect_transcript_permissions_path(
    base_dir: Path,
    *,
    repair: bool,
) -> dict[str, Any]:
    """Windows path-based counterpart to the POSIX descriptor audit."""

    directory_stat = _prepare_transcript_directory_path(
        base_dir, create=False, make_private=False
    )
    report = _new_permission_report(base_dir)
    directory_mode = stat.S_IMODE(directory_stat.st_mode)
    report["directory_mode_before"] = directory_mode
    if repair and directory_mode != 0o700:
        _prepare_transcript_directory_path(
            base_dir, create=False, make_private=True
        )
        report["directory_repaired"] = True

    euid = _current_euid()
    with os.scandir(base_dir) as entries:
        for entry in entries:
            name = entry.name
            if not _is_transcript_artifact_name(name):
                continue
            candidate = base_dir / name
            try:
                # On Windows, DirEntry.stat() deliberately reports zero for
                # st_dev, st_ino, and st_nlink.  Those fields are security
                # inputs below: a zero link count made every normal file look
                # multiply linked, while zero identities could not be compared
                # with the subsequently opened file.  A path stat performs the
                # real system call and, with lstat, still refuses to follow a
                # symlink/reparse-point leaf.
                observed = os.lstat(candidate)
            except OSError:
                report["skipped_changed"].append(name)
                continue
            if _is_link_or_reparse_point(candidate, observed):
                report["skipped_symlinks"].append(name)
                continue
            if not stat.S_ISREG(observed.st_mode):
                report["skipped_non_regular"].append(name)
                continue
            if observed.st_nlink != 1:
                report["skipped_hardlinks"].append(name)
                continue
            if euid is not None and observed.st_uid != euid:
                report["skipped_unowned"].append(name)
                continue

            report["eligible_files"] += 1
            if stat.S_IMODE(observed.st_mode) == 0o600:
                report["already_private_files"].append(name)
                continue
            if not repair:
                report["would_repair_files"].append(name)
                continue

            # Open only to compare identity before changing the pathname. On
            # Windows there is no dir_fd/O_NOFOLLOW equivalent in os.open; a
            # swapped link is followed read-only, detected by the identity
            # mismatch, and never chmod'd.
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
            try:
                file_fd = os.open(candidate, flags)
            except OSError:
                report["skipped_changed"].append(name)
                continue
            try:
                current = os.fstat(file_fd)
            finally:
                os.close(file_fd)
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino)
                != (observed.st_dev, observed.st_ino)
                or (euid is not None and current.st_uid != euid)
            ):
                report["skipped_changed"].append(name)
                continue
            try:
                os.chmod(candidate, 0o600)
                after = os.lstat(candidate)
            except OSError:
                report["skipped_changed"].append(name)
                continue
            if (
                _is_link_or_reparse_point(candidate, after)
                or (after.st_dev, after.st_ino)
                != (observed.st_dev, observed.st_ino)
            ):
                report["skipped_changed"].append(name)
                continue
            report["repaired_files"].append(name)

    return _sort_permission_report(report)


def inspect_transcript_permissions(
    base_dir: Path,
    *,
    repair: bool = False,
) -> dict[str, Any]:
    """Inspect, and optionally repair, council-owned transcript permissions.

    Only timestamp-prefixed ``.md``, ``.json``, and ``.html`` regular files
    owned by the current effective user are eligible. Symlinks, multiply-linked
    files, foreign-owned files, non-regular entries, and entries replaced during
    inspection are reported and never followed. With ``repair=True`` the
    directory becomes 0700 and eligible files become 0600.
    """

    base_dir = Path(base_dir)
    if _uses_windows_path_fallback():
        return _inspect_transcript_permissions_path(base_dir, repair=repair)

    directory_fd = _open_owned_transcript_directory(
        base_dir, create=False, make_private=False
    )
    report = _new_permission_report(base_dir)
    try:
        directory_stat = os.fstat(directory_fd)
        directory_mode = stat.S_IMODE(directory_stat.st_mode)
        report["directory_mode_before"] = directory_mode
        if repair and directory_mode != 0o700:
            if hasattr(os, "fchmod"):
                os.fchmod(directory_fd, 0o700)
            else:  # pragma: no cover - Windows fallback
                os.chmod(base_dir, 0o700, follow_symlinks=False)
            report["directory_repaired"] = True

        euid = _current_euid()
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                name = entry.name
                if not _is_transcript_artifact_name(name):
                    continue
                try:
                    observed = entry.stat(follow_symlinks=False)
                except OSError:
                    report["skipped_changed"].append(name)
                    continue
                if stat.S_ISLNK(observed.st_mode):
                    report["skipped_symlinks"].append(name)
                    continue
                if not stat.S_ISREG(observed.st_mode):
                    report["skipped_non_regular"].append(name)
                    continue
                if observed.st_nlink != 1:
                    report["skipped_hardlinks"].append(name)
                    continue
                if euid is not None and observed.st_uid != euid:
                    report["skipped_unowned"].append(name)
                    continue

                report["eligible_files"] += 1
                if stat.S_IMODE(observed.st_mode) == 0o600:
                    report["already_private_files"].append(name)
                    continue
                if not repair:
                    report["would_repair_files"].append(name)
                    continue

                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    file_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError:
                    report["skipped_changed"].append(name)
                    continue
                try:
                    current = os.fstat(file_fd)
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or (current.st_dev, current.st_ino)
                        != (observed.st_dev, observed.st_ino)
                        or (euid is not None and current.st_uid != euid)
                    ):
                        report["skipped_changed"].append(name)
                        continue
                    if hasattr(os, "fchmod"):
                        os.fchmod(file_fd, 0o600)
                    else:  # pragma: no cover - Windows fallback
                        os.chmod(base_dir / name, 0o600, follow_symlinks=False)
                    report["repaired_files"].append(name)
                finally:
                    os.close(file_fd)
    finally:
        os.close(directory_fd)

    return _sort_permission_report(report)


def _atomic_write_private_path(path: Path, content: str) -> None:
    """Windows path-based atomic writer with reparse and identity checks."""

    directory_stat = _prepare_transcript_directory_path(
        path.parent, create=True, make_private=True
    )
    file_fd, raw_temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(raw_temporary)
    try:
        os.chmod(temporary_path, 0o600)
        with os.fdopen(file_fd, "w", encoding="utf-8", newline="") as handle:
            file_fd = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        current_directory = os.lstat(path.parent)
        if (
            _is_link_or_reparse_point(path.parent, current_directory)
            or not stat.S_ISDIR(current_directory.st_mode)
            or (current_directory.st_dev, current_directory.st_ino)
            != (directory_stat.st_dev, directory_stat.st_ino)
        ):
            raise OSError(
                f"Transcript directory changed before atomic replace: {path.parent}"
            )
        # os.replace replaces the directory entry itself rather than opening
        # the destination, so an existing file symlink/reparse point is not
        # followed. The random same-directory temporary name prevents partial
        # readers and keeps the rename on one filesystem.
        os.replace(temporary_path, path)
    except BaseException:
        if file_fd >= 0:
            os.close(file_fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_private(path: Path, content: str) -> None:
    """Atomically replace ``path`` with a private UTF-8 text file.

    Transcripts can contain source, prompts, and model output. Create the
    temporary file in the destination directory with mode 0600, flush it, and
    then replace the destination atomically so readers never observe a partial
    document. The descriptor-relative rename replaces a destination symlink
    itself rather than following it, avoiding writes through a pre-planted link.
    """

    if _uses_windows_path_fallback():
        _atomic_write_private_path(path, content)
        return

    directory_fd = _open_owned_transcript_directory(
        path.parent, create=True, make_private=True
    )
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    file_fd = -1
    temporary_path: Path | None = None
    supports_dir_fd = (
        os.open in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
    )
    try:
        if supports_dir_fd:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            file_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        else:  # pragma: no cover - Windows fallback
            file_fd, raw_temporary = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(raw_temporary)
        if hasattr(os, "fchmod"):
            os.fchmod(file_fd, 0o600)
        with os.fdopen(file_fd, "w", encoding="utf-8", newline="") as handle:
            file_fd = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if supports_dir_fd:
            os.rename(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:  # pragma: no cover - Windows fallback
            assert temporary_path is not None
            os.replace(temporary_path, path)
    except BaseException:
        if file_fd >= 0:
            os.close(file_fd)
        try:
            if supports_dir_fd:
                os.unlink(temporary_name, dir_fd=directory_fd)
            elif temporary_path is not None:  # pragma: no cover - Windows fallback
                temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory_fd)


def transcript_dir(cwd: Path, config: dict) -> Path:
    """Resolve the transcripts directory from config, anchored at ``cwd``.

    Single source of truth for the previously-inlined copies across
    ``mcp_server.py`` and ``cli.py`` (the
    ``Path(config.get("transcripts_dir", ".llm-council/runs"))`` +
    relative-to-cwd resolution pattern).
    """

    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    return out_dir if out_dir.is_absolute() else cwd / out_dir


_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}")


def normalize_run_id(value: str) -> str:
    """Strip directory and known suffixes; require the timestamp prefix."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("run id is empty")
    raw = Path(raw).name
    for suffix in (".json", ".md"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    match = _RUN_ID_RE.match(raw)
    if not match:
        raise ValueError(
            f"run id '{value}' does not start with a YYYYMMDD_HHMMSS prefix"
        )
    return raw


def find_transcript_by_id(base_dir: Path, run_id: str) -> dict[str, Any]:
    """Locate a JSON transcript by run-id prefix or filename and load it.

    Accepts either the bare timestamp prefix (``20260502_062608``) or a
    full filename (``20260502_062608_question.json`` / ``.md``). Raises
    ``FileNotFoundError`` if no JSON transcript matches.
    """

    normalized = normalize_run_id(run_id)
    candidates = sorted(base_dir.glob(f"{normalized}*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No council transcript matching run id '{run_id}' was found in "
            f"{base_dir}. Run `llm-council transcripts list` to see available ids."
        )
    if len(candidates) > 1:
        for candidate in candidates:
            if candidate.stem == normalized:
                return _load_transcript_json(candidate)
        names = ", ".join(c.name for c in candidates)
        raise ValueError(
            f"Run id '{run_id}' matches multiple transcripts ({names}); "
            "supply the full filename or a longer prefix."
        )
    return _load_transcript_json(candidates[0])


DEFAULT_MAX_CONTINUATION_DEPTH = 5


def count_continuation_depth(base_dir: Path, run_id: str, *, max_depth: int = 32) -> int:
    """Walk parent_run_id chain backwards and return the depth.

    Depth 1 means "this run has one parent" (i.e., it would be the second
    link in the chain when resumed). The traversal is bounded by
    ``max_depth`` so a corrupt cycle can't hang the caller.

    Callers that want to enforce a configured cap should pass
    ``max_depth=cap + 1`` so the walker can return a value strictly
    greater than the cap when the chain exceeds it. A cycle in the
    transcripts is always treated as corruption and surfaced via
    ``ValueError`` rather than silently truncating, since under-counting
    in that case would mistakenly approve a chain that should be
    rejected.
    """

    visited: set[str] = set()
    current = run_id
    depth = 0
    while current and depth < max_depth:
        normalized = normalize_run_id(current)
        if normalized in visited:
            raise ValueError(
                f"Continuation chain contains a cycle: '{normalized}' "
                "appears more than once. Inspect the affected transcript "
                "JSON files' parent_run_id fields and remove the loop."
            )
        visited.add(normalized)
        try:
            transcript = find_transcript_by_id(base_dir, normalized)
        except (FileNotFoundError, ValueError):
            break
        parent = transcript.get("parent_run_id")
        if not parent:
            break
        depth += 1
        current = str(parent)
    return depth


def continuation_depth_limit_error(
    config: dict[str, Any], transcripts_dir: Path, run_id: str
) -> str | None:
    """Return an error message if continuing ``run_id`` would exceed the
    configured ``defaults.max_continuation_depth``, else None.

    Shared by the CLI (`cmd_run_async`) and MCP (`run_council`) run pipelines so
    the cap computation and the message can't drift between them (they had
    already drifted slightly before this was extracted). Each caller raises its
    own exception type (SystemExit / ValueError) with the returned message.
    Passes ``max_depth + 1`` to the walker so it can count strictly past the cap
    even when the user-configured cap exceeds the walker's internal ceiling.
    """
    max_depth = int(
        config.get("defaults", {}).get(
            "max_continuation_depth", DEFAULT_MAX_CONTINUATION_DEPTH
        )
    )
    depth = count_continuation_depth(transcripts_dir, run_id, max_depth=max_depth + 1)
    if depth >= max_depth:
        return (
            f"Continuation chain depth ({depth} parents) reaches the configured "
            f"limit of {max_depth}. Each link summarizes its predecessor, so deep "
            "chains eat into MAX_PROMPT_CHARS without adding new signal. Start a "
            "fresh run, or raise `defaults.max_continuation_depth` in "
            "`.llm-council.yaml`."
        )
    return None


def _load_transcript_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read transcript JSON at {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Transcript JSON at {path} is not an object")
    data.setdefault("_path", str(path))
    return data


_PRIOR_QUESTION_MAX_CHARS = 2000
_PRIOR_PEER_SUMMARY_MAX_CHARS = 240


def _final_round_label(name: str) -> str:
    return ROUND_SUFFIX_RE.sub("", name)


def _strip_recommendation_from_summary(summary: str) -> str:
    return _strip_recommendation_prefix(summary or "").strip()


def _select_final_round_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    rounds = [result_round(str(r.get("name") or "")) for r in results]
    final_round = max(rounds)
    return [
        result
        for result, round_no in zip(results, rounds)
        if round_no == final_round
    ]


def _summarize_record_label(record: dict[str, Any]) -> tuple[str, str, bool]:
    output = str(record.get("output") or "")
    if record.get("ok") and output:
        label = recommendation_label(output)
        line = recommendation_line(output)
        summary = _strip_recommendation_from_summary(line)
        return label, _cap_peer_summary(summary), False
    error = str(record.get("error") or "")
    summary = _first_nonempty_line(error)
    return "unknown", _cap_peer_summary(summary), True


def _cap_peer_summary(text: str) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= _PRIOR_PEER_SUMMARY_MAX_CHARS:
        return cleaned
    return cleaned[: _PRIOR_PEER_SUMMARY_MAX_CHARS - 3].rstrip() + "..."


def format_prior_council_context(
    transcript: dict[str, Any],
    *,
    run_id: str | None = None,
) -> str:
    """Render a compact 'Prior council context' block for prompt prepending.

    The block summarizes the prior question, the final-round labels and a
    one-line rationale per peer, plus notes pulled from
    ``remaining_disagreement`` and ``degraded_consensus`` payloads when the
    prior run recorded them.
    """

    if not isinstance(transcript, dict):
        raise ValueError("transcript must be a dict loaded from JSON")
    if run_id is None:
        path = transcript.get("_path")
        if path:
            run_id = Path(str(path)).stem
        else:
            run_id = "unknown"
    question = str(transcript.get("question") or "").strip()
    truncated_question = question
    if len(truncated_question) > _PRIOR_QUESTION_MAX_CHARS:
        truncated_question = (
            truncated_question[:_PRIOR_QUESTION_MAX_CHARS].rstrip()
            + "...[truncated]"
        )

    results = transcript.get("results") or []
    if not isinstance(results, list):
        results = []
    final_records = _select_final_round_records(results)

    counts = {"yes": 0, "no": 0, "tradeoff": 0, "unknown": 0}
    peer_lines: list[str] = []
    for record in final_records:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "?")
        display_name = _final_round_label(name)
        label, summary, is_error = _summarize_record_label(record)
        counts[label if label in counts else "unknown"] += 1
        if is_error:
            rendered_summary = (
                f"error: {summary}" if summary else "error: no detail recorded"
            )
        else:
            rendered_summary = summary or "no rationale recorded"
        peer_lines.append(f"- {display_name}: {label} — {rendered_summary}")

    remaining = transcript.get("remaining_disagreement")
    if isinstance(remaining, dict):
        rem_participants = remaining.get("participants") or []
        for entry in rem_participants:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "?")
            display_name = _final_round_label(name)
            already = any(
                line.startswith(f"- {display_name}: ") for line in peer_lines
            )
            if already:
                continue
            label = entry.get("label") or "unknown"
            summary = (entry.get("summary") or "").strip() or "no rationale recorded"
            peer_lines.append(f"- {display_name}: {label} — {summary}")

    degraded = transcript.get("degraded_consensus")
    is_degraded = isinstance(degraded, dict)

    lines: list[str] = [f"Prior council context (run {run_id}):", ""]
    if truncated_question:
        lines.append(f"Question: {truncated_question}")
        lines.append("")
    summary_line = (
        "Recommendations (final round): "
        f"{counts['yes']} yes / {counts['no']} no / "
        f"{counts['tradeoff']} tradeoff / {counts['unknown']} unknown"
    )
    lines.append(summary_line)
    if peer_lines:
        lines.extend(peer_lines)
    else:
        lines.append("- (no participant responses recorded)")
    if is_degraded:
        labeled = degraded.get("labeled_quorum")
        threshold = degraded.get("min_quorum")
        lines.extend(
            [
                "",
                "[Note: prior run was degraded — "
                f"{labeled} of {threshold} required peers labeled.]",
            ]
        )
    if isinstance(remaining, dict) and remaining.get("ran_max_rounds_unresolved"):
        lines.extend(
            [
                "",
                "[Note: prior run reached max deliberation rounds without convergence.]",
            ]
        )
    return "\n".join(lines).rstrip()


def latest_transcript(base_dir: Path, *, suffix: str = ".md") -> Path | None:
    matches = sorted(
        _existing_paths(base_dir.glob(f"*{suffix}")), key=lambda item: item[1]
    )
    return matches[-1][0] if matches else None


def _existing_paths(paths) -> list[tuple[Path, float]]:
    existing = []
    for path in paths:
        try:
            existing.append((path, path.stat().st_mtime))
        except FileNotFoundError:
            continue
    return existing


def iter_run_json(base_dir: Path) -> list[tuple[Path, float, dict]]:
    """Mtime-sorted ``(path, mtime, data)`` for every readable run JSON.

    Shared scan over ``base_dir/*.json`` (mirrors
    ``stats.load_transcript_files``): each file is stat'd via
    ``_existing_paths``, then read with ``json.loads``; unreadable or
    malformed files are skipped. Unlike ``_load_transcript_json`` this
    never raises and does not enforce ``dict`` shape or set ``_path`` —
    those guarantees are intentionally reserved for the by-id loader.
    """

    rows: list[tuple[Path, float, dict]] = []
    for path, mtime in sorted(
        _existing_paths(base_dir.glob("*.json")), key=lambda item: item[1]
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append((path, mtime, data))
    return rows


def transcript_records(base_dir: Path) -> list[dict[str, Any]]:
    # Mirrors stats.load_transcript_files's scan via the shared iter_run_json.
    records: list[dict[str, Any]] = []
    for path, mtime, data in iter_run_json(base_dir):
        results = data.get("results") or []
        records.append(
            {
                "path": str(path),
                "markdown": str(path.with_suffix(".md")),
                "question": data.get("question", ""),
                "mode": data.get("mode", ""),
                "current": data.get("current"),
                "participants": data.get("participants", []),
                "ok": sum(1 for result in results if result.get("ok")),
                "total": len(results),
                "tokens": sum(result.get("total_tokens") or 0 for result in results),
                "cost_usd": sum(result.get("cost_usd") or 0 for result in results),
                "mtime": mtime,
            }
        )
    return records


def result_to_dict(result: ParticipantResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": result.name,
        "ok": result.ok,
        "model": result.model,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "wall_elapsed_seconds": (
            round(result.wall_elapsed_seconds, 3)
            if result.wall_elapsed_seconds is not None
            else None
        ),
        "command": result.command,
        "output": result.output,
        "error": result.error,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
    }
    if result.recovered_after_launch_retry:
        payload["recovered_after_launch_retry"] = True
    if result.repair_retry_recovered:
        payload["repair_retry_recovered"] = True
    if result.recovered_after_timeout:
        payload["recovered_after_timeout"] = True
    if result.terse_retry_attempted:
        payload["terse_retry_attempted"] = True
    if getattr(result, "model_fallback_used", None):
        payload["model_fallback_used"] = result.model_fallback_used
    if getattr(result, "recovered_after_quota", False):
        payload["recovered_after_quota"] = True
    if result.section_repair_attempted:
        payload["section_repair_attempted"] = True
    if getattr(result, "empty_retry_attempted", False):
        payload["empty_retry_attempted"] = True
    if getattr(result, "recovered_after_empty_retry", False):
        payload["recovered_after_empty_retry"] = True
    # Failure diagnostics (exit status + stderr tail) — omitted on success
    # and for hosted peers so the common transcript shape is unchanged.
    if getattr(result, "exit_code", None) is not None and not result.ok:
        payload["exit_code"] = result.exit_code
    if getattr(result, "stderr_tail", None):
        payload["stderr_tail"] = result.stderr_tail
    if getattr(result, "tool_call_status", None) is not None:
        payload["tool_call_status"] = result.tool_call_status
    if result.prompt_chars is not None:
        payload["prompt_chars"] = result.prompt_chars
    if result.from_cache:
        payload["from_cache"] = True
        if result.cache_hit_seconds is not None:
            payload["cache_hit_seconds"] = result.cache_hit_seconds
    if result.stance is not None:
        payload["stance"] = result.stance
    # Envelope fields are emitted only when present so transcripts from
    # peers that never supply them stay readable. List fields are emitted
    # when non-empty; scalar fields when not None.
    envelope_lists = {
        "blockers": list(result.blockers or ()),
        "evidence": list(result.evidence or ()),
        "tests_to_run": list(result.tests_to_run or ()),
        "assumptions": list(result.assumptions or ()),
    }
    for field_name in ("effort", "confidence", "risk"):
        value = getattr(result, field_name, None)
        if value is not None:
            payload[field_name] = value
    for field_name, items in envelope_lists.items():
        if items:
            payload[field_name] = items
    # continue_debate (round-2 vote) and evidence_verification_failures
    # (failed [VERIFIED:...] cites) are part of the documented envelope /
    # citations surface — emit them so the transcript JSON matches the
    # docstrings and the MCP structured_results shape.
    if getattr(result, "continue_debate", None) is not None:
        payload["continue_debate"] = result.continue_debate
    if getattr(result, "evidence_verification_failures", None):
        payload["evidence_verification_failures"] = list(
            result.evidence_verification_failures
        )
    from llm_council.adapters import classify_error

    error_kind = classify_error(result.error)
    if error_kind is not None:
        payload["error_kind"] = error_kind
    return payload


def convergence_summary_lines(metadata: dict[str, Any]) -> list[str]:
    """Render per-round convergence tallies as bullet lines for the markdown header.

    Returns an empty list when no convergence data is recorded (i.e. fewer than
    two rounds ran or the orchestrator did not stamp metadata).
    """
    convergence = metadata.get("convergence")
    if not isinstance(convergence, dict) or not convergence:
        return []
    lines: list[str] = []
    for round_key in sorted(convergence.keys(), key=lambda k: int(k)):
        records = convergence.get(round_key) or []
        if not isinstance(records, list) or not records:
            continue
        states = [r.get("state") for r in records if isinstance(r, dict)]
        counts = tally_states(states)
        insufficient = sum(1 for s in states if s == "insufficient")
        classified_total = counts["converged"] + counts["refining"] + counts["diverging"]
        parts = []
        for state in ("converged", "refining", "diverging"):
            if counts[state]:
                parts.append(f"{counts[state]} {state}")
        if insufficient:
            parts.append(f"{insufficient} insufficient")
        summary = ", ".join(parts) if parts else "no signal"
        prefix = ""
        if classified_total > 0 and counts["converged"] == classified_total:
            prefix = "**ALL CONVERGED** — "
        lines.append(f"- Convergence (round {round_key}): {prefix}{summary}")
    return lines


def deliberation_summary(metadata: dict[str, Any]) -> str:
    status = metadata.get("deliberation_status")
    if status == "ran_no_labeled_disagreement":
        return "ran; no labeled disagreement remained"
    if status == "ran_max_rounds_unresolved":
        return "ran; max rounds reached with labeled disagreement"
    if status == "skipped_no_labeled_disagreement":
        return "skipped, no labeled disagreement detected"
    if status == "skipped_max_rounds":
        return "skipped, max rounds is 1"
    if status == "pending":
        return "pending"
    if metadata.get("deliberated"):
        return "ran"
    if metadata.get("deliberation_requested"):
        return "skipped"
    return "not requested"


def result_round(name: str) -> int:
    match = ROUND_SUFFIX_RE.search(name)
    return int(match.group(1)) if match else 1


def final_round_results(results: list[ParticipantResult]) -> list[ParticipantResult]:
    if not results:
        return []
    final_round = max(result_round(result.name) for result in results)
    return [result for result in results if result_round(result.name) == final_round]


def final_decision_label(results: list[ParticipantResult]) -> str:
    """Return the unique leading final-round peer label, or ``unknown``.

    The dashboard headline is peer-vote telemetry, not mutable caller metadata
    or the optional synthesis chair's memo. A tie has no council decision and
    therefore must not be resolved by label ordering.
    """

    return summarize_recommendations(final_round_results(results)).recommendation


_RECOMMENDATION_PREFIX_RE = re.compile(
    r"^RECOMMENDATION:\s*(?:yes|no|tradeoff)\s*[-–—:]?\s*",
    re.IGNORECASE,
)


def _strip_recommendation_prefix(line: str) -> str:
    return _RECOMMENDATION_PREFIX_RE.sub("", line, count=1).strip()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _participant_disagreement_entry(result: ParticipantResult) -> dict[str, Any]:
    if result.ok:
        label = recommendation_label(result.output)
        summary = _strip_recommendation_prefix(recommendation_line(result.output))
    else:
        label = None
        summary = _first_nonempty_line(result.error or "")
    return {"name": result.name, "ok": result.ok, "label": label, "summary": summary}


def remaining_disagreement_payload(
    final_results: list[ParticipantResult], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    if not metadata.get("final_disagreement_detected"):
        return None
    if not final_results:
        return None
    counts = recommendation_counts(final_results)
    return {
        "status": metadata.get("deliberation_status"),
        "ran_max_rounds_unresolved": metadata.get("deliberation_status")
        == "ran_max_rounds_unresolved",
        "counts": counts,
        "participants": [_participant_disagreement_entry(r) for r in final_results],
    }


def _minority_callout(remaining: dict[str, Any]) -> str | None:
    """Scannable minority note for the remaining-disagreement count line.

    Returns a string like ``minority: codex, gemini held no`` when there is a
    single CLEAR majority trinary label and a non-empty minority of OTHER
    trinary labels. Returns ``None`` (skip the callout) when:

    - there is no clear majority (two or more trinary labels tie for the top),
    - the council is unanimous (no minority), or
    - no trinary label was emitted at all.

    ``unknown`` / ``None`` labels are intentionally excluded from both the
    majority computation and the minority callout — they're already shown in
    the per-peer label list, and surfacing them here would be noise.
    """
    counts = remaining["counts"]
    trinary = {label: counts[label] for label in ("yes", "no", "tradeoff")}
    top = max(trinary.values())
    if top == 0:
        return None
    leaders = [label for label, n in trinary.items() if n == top]
    if len(leaders) != 1:
        # Ambiguous tie among top labels → no single majority.
        return None
    majority = leaders[0]
    minority: list[tuple[str, str]] = []
    for entry in remaining["participants"]:
        label = entry.get("label")
        if label in ("yes", "no", "tradeoff") and label != majority:
            minority.append((entry["name"], label))
    if not minority:
        return None
    # Group minority peers by the label they held so the callout reads
    # naturally even when the minority itself is split across labels.
    by_label: dict[str, list[str]] = {}
    for name, label in minority:
        by_label.setdefault(label, []).append(name)
    parts = [
        f"{', '.join(names)} held {label}" for label, names in by_label.items()
    ]
    return "minority: " + "; ".join(parts)


def _missing_label_reason(result: ParticipantResult) -> str:
    if result.ok:
        if recommendation_label(result.output) == "unknown":
            return "missing label"
        return "labeled"
    if is_timeout_error(result.error):
        return "timeout"
    if is_context_overflow_error(result.error):
        return "context overflow"
    return "failed"


def context_overflow_excluded_names(
    results: list[ParticipantResult],
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.ok or not is_context_overflow_error(result.error):
            continue
        base = ROUND_SUFFIX_RE.sub("", result.name)
        if base in seen:
            continue
        seen.add(base)
        names.append(base)
    return names


def context_overflow_records(
    results: list[ParticipantResult],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        if result.ok or not is_context_overflow_error(result.error):
            continue
        records.append(
            {
                "name": result.name,
                "estimated_tokens": result.prompt_tokens,
                "error": result.error,
            }
        )
    return records


def _participant_quorum_entry(result: ParticipantResult) -> dict[str, Any]:
    if result.ok:
        label = recommendation_label(result.output)
        return {
            "name": result.name,
            "ok": True,
            "label": None if label == "unknown" else label,
            "reason": _missing_label_reason(result),
        }
    return {
        "name": result.name,
        "ok": False,
        "label": None,
        "reason": _missing_label_reason(result),
        "error": _first_nonempty_line(result.error or ""),
    }


def quorum_summary(
    final_results: list[ParticipantResult], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Pure helper: derive labeled_quorum / min_quorum / degraded from results.

    Prefers values stamped onto metadata by the orchestrator; falls back to
    recomputing from final_results so transcripts written from older runs (or
    raw test fixtures) remain coherent.
    """
    labeled = metadata.get("labeled_quorum")
    if labeled is None:
        labeled = labeled_quorum_count(final_results)
    threshold = metadata.get("min_quorum")
    if threshold is None:
        threshold = default_min_quorum(len(final_results))
    degraded = metadata.get("degraded")
    if degraded is None:
        degraded = labeled < threshold
    return {
        "labeled_quorum": int(labeled),
        "min_quorum": int(threshold),
        "degraded": bool(degraded),
    }


def degraded_consensus_payload(
    final_results: list[ParticipantResult], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    summary = quorum_summary(final_results, metadata)
    if not summary["degraded"]:
        return None
    missing = [
        _participant_quorum_entry(result)
        for result in final_results
        if _missing_label_reason(result) != "labeled"
    ]
    return {
        "labeled_quorum": summary["labeled_quorum"],
        "min_quorum": summary["min_quorum"],
        "missing": missing,
    }


def write_transcript(
    markdown_path: Path,
    json_path: Path,
    *,
    question: str,
    mode: str,
    current: str | None,
    participants: list[str],
    prompt: str,
    results: list[ParticipantResult],
    transparent: bool = False,
    metadata: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
) -> None:
    ensure_private_transcript_dir(markdown_path.parent)

    metadata = metadata or {}
    ok_count = sum(1 for result in results if result.ok)
    final_results = final_round_results(results)
    final_ok_count = sum(1 for result in final_results if result.ok)
    elapsed_total = sum(result.elapsed_seconds for result in results)
    # None (rendered "n/a") when NO participant reported the figure —
    # text-mode CLI peers have no metering hook, and summing their Nones
    # to 0 reads as "this run was free" rather than "telemetry
    # unavailable". A partial sum (some peers reported) is still shown:
    # the header says "reported", not "total".
    reported_tokens = [r.total_tokens for r in results if r.total_tokens is not None]
    reported_costs = [r.cost_usd for r in results if r.cost_usd is not None]
    token_total = sum(reported_tokens) if reported_tokens else None
    cost_total = sum(reported_costs) if reported_costs else None
    recommendations = recommendation_counts(final_results)
    quorum = quorum_summary(final_results, metadata)
    quorum_bullet = (
        f"- Quorum: {quorum['labeled_quorum']} of {len(final_results)} peers "
        f"labeled (min: {quorum['min_quorum']})"
    )
    if quorum["degraded"]:
        quorum_bullet += " — **DEGRADED**"
    overflow_names = context_overflow_excluded_names(final_results)
    overflow_bullet = (
        [f"- Excluded for context overflow: {', '.join(overflow_names)}"]
        if overflow_names
        else []
    )
    # Context files dropped by chunking never reached any peer — a silent
    # drop reads as full coverage, so the transcript states it loudly.
    # MCP runs store every event in `chunk_events`; CLI runs store only the
    # latest under `diff_chunking`.
    chunk_records = metadata.get("chunk_events") or [
        metadata.get("diff_chunking") or {}
    ]
    dropped_context = sorted(
        {
            str(path)
            for record in chunk_records
            if isinstance(record, dict)
            for path in (record.get("dropped_files") or [])
        }
    )
    dropped_bullet = (
        [
            "- ⚠️ Context files DROPPED by chunking (no peer saw their "
            "contents): " + ", ".join(f"`{p}`" for p in dropped_context)
        ]
        if dropped_context
        else []
    )
    default_model_peers = metadata.get("cli_default_model_peers") or []
    default_model_bullet = (
        [
            "- Peers on account-default models (may differ from the host "
            "session's model): "
            + ", ".join(f"`{p}`" for p in default_model_peers)
        ]
        if default_model_peers
        else []
    )
    refused_peers = metadata.get("content_refused_peers") or []
    refused_bullet = (
        [
            "- ⚠️ Content-policy refusals (peer dropped from quorum; "
            "rephrase the request as verification rather than attack): "
            + ", ".join(
                f"`{entry.get('peer')}` — {entry.get('message') or 'refused'}"
                for entry in refused_peers
                if isinstance(entry, dict)
            )
        ]
        if refused_peers
        else []
    )
    lines = [
        "# LLM Council Transcript",
        "",
        f"- Mode: `{mode}`",
        f"- Current agent: `{current or 'unknown'}`",
        f"- Participants: {', '.join(f'`{name}`' for name in participants)}",
        f"- Successful responses: {ok_count}/{len(results)} total",
        f"- Final-round successful responses: {final_ok_count}/{len(final_results)}",
        f"- Participant elapsed total: `{elapsed_total:.1f}s`",
        f"- Tokens reported: `{'n/a' if token_total is None else token_total}`",
        "- Cost reported: "
        f"`{'n/a' if cost_total is None else f'${cost_total:.6f}'}`",
        f"- Rounds: `{metadata.get('rounds', 1)}`",
        f"- Deliberation: {deliberation_summary(metadata)}",
        *convergence_summary_lines(metadata),
        *(
            [f"- Parent run: `{parent_run_id}`"]
            if parent_run_id
            else []
        ),
        "- Recommendations (final round): "
        f"`{recommendations['yes']} yes / {recommendations['no']} no / "
        f"{recommendations['tradeoff']} tradeoff / {recommendations['unknown']} unknown`",
        quorum_bullet,
        *overflow_bullet,
        *dropped_bullet,
        *default_model_bullet,
        *refused_bullet,
        "",
        "## Question",
        "",
        question.strip(),
        "",
    ]

    images = metadata.get("images") or []
    if images:
        lines.extend(["## Images", ""])
        for entry in images:
            label = entry.get("path") or "?"
            mime = entry.get("mime") or "?"
            size = entry.get("size")
            sha = (entry.get("sha256") or "")[:12]
            size_str = f"{size} bytes" if size is not None else "?"
            lines.append(f"- `{label}` ({mime}, {size_str}, sha256:{sha})")
        lines.append("")

    if transparent:
        lines.extend(["## Model Comparison", ""])
        lines.extend(model_comparison(results))
        lines.append("")

    lines.extend(["## Participant Responses", ""])

    for result in results:
        if result.ok:
            status = "ok"
        elif is_timeout_error(result.error):
            status = "timeout"
        elif is_context_overflow_error(result.error):
            status = "excluded"
        else:
            status = "error"
        cache_tag = " [cached]" if result.from_cache else ""
        lines.extend(
            [
                f"### {result.name} ({status}){cache_tag}",
                "",
                f"- Model: `{result.model or 'cli default (unreported)'}`",
                f"- Attempt elapsed: `{result.elapsed_seconds:.1f}s`",
            ]
        )
        if result.wall_elapsed_seconds is not None:
            lines.append(f"- Wall elapsed: `{result.wall_elapsed_seconds:.1f}s`")
        if result.total_tokens is not None:
            lines.append(f"- Tokens: `{result.total_tokens}`")
        if result.cost_usd is not None:
            lines.append(f"- Cost: `${result.cost_usd:.6f}`")
        if result.command:
            lines.append(f"- Command: `{command_for_display(result.command)}`")
        lines.append("")
        if result.ok:
            lines.extend([result.output.strip() or "[empty response]", ""])
        else:
            lines.extend(["```", result.error.strip() or "[unknown error]", "```", ""])
            if getattr(result, "exit_code", None) is not None:
                lines.append(f"- Exit status: `{result.exit_code}`")
            if getattr(result, "empty_retry_attempted", False):
                lines.append("- Empty-response re-run: attempted, also failed")
            if getattr(result, "exit_code", None) is not None or getattr(
                result, "empty_retry_attempted", False
            ):
                lines.append("")
            stderr_tail = getattr(result, "stderr_tail", None)
            if stderr_tail:
                lines.extend(
                    [
                        "Stderr tail (last "
                        f"{len(stderr_tail)} chars the CLI wrote before it "
                        "exited or was killed):",
                        "",
                        "```",
                        stderr_tail.strip(),
                        "```",
                        "",
                    ]
                )
            if result.output.strip():
                lines.extend(["Captured output:", "", result.output.strip(), ""])

    remaining = remaining_disagreement_payload(final_results, metadata)
    if remaining is not None:
        counts = remaining["counts"]
        lines.extend(["## Remaining disagreement", ""])
        count_line = (
            "Recommendations (final round): "
            f"{counts['yes']} yes / {counts['no']} no / "
            f"{counts['tradeoff']} tradeoff / {counts['unknown']} unknown"
        )
        minority = _minority_callout(remaining)
        if minority:
            count_line += f" — {minority}"
        lines.append(count_line)
        lines.append("")
        for entry in remaining["participants"]:
            label = entry["label"] or "—"
            summary = entry["summary"] or "—"
            lines.append(f"- {entry['name']}: {label} — {summary}")
        if remaining["ran_max_rounds_unresolved"]:
            rounds_run = metadata.get("rounds")
            rounds_phrase = (
                f" ({rounds_run})" if isinstance(rounds_run, int) else ""
            )
            lines.extend(
                [
                    "",
                    f"Deliberation reached the maximum configured rounds{rounds_phrase} "
                    "without the council converging on a single recommendation.",
                ]
            )
        lines.append("")

    degraded = degraded_consensus_payload(final_results, metadata)
    if degraded is not None:
        lines.extend(["## Degraded consensus", ""])
        if degraded["missing"]:
            lines.append(
                f"**{degraded['labeled_quorum']} of {len(final_results)} peers produced a "
                f"label, below the configured minimum of {degraded['min_quorum']}.** "
                "Treat the recommendation above with caution: the surviving "
                "peer(s) may not be representative of the council."
            )
            lines.append("")
            lines.append("Peers that did not label:")
            lines.append("")
            for entry in degraded["missing"]:
                reason = entry.get("reason") or "—"
                detail = entry.get("error")
                if detail:
                    lines.append(f"- {entry['name']}: {reason} — {detail}")
                else:
                    lines.append(f"- {entry['name']}: {reason}")
            lines.append("")
        else:
            lines.append(
                f"**The configured `min_quorum` of {degraded['min_quorum']} exceeds "
                f"the {degraded['labeled_quorum']} peer(s) that produced a label, "
                "even though every selected peer responded.** This is a configuration "
                "issue, not a participant failure: lower `min_quorum` or add more "
                "peers if you want a non-degraded result."
            )
            lines.append("")

    # H2 independence warning (advisory-only). Rendered near the quorum /
    # degraded summary; only present when the orchestrator fired it. Does
    # NOT affect quorum/degraded — purely informational.
    independence_warning = metadata.get("independence_warning")
    if isinstance(independence_warning, dict):
        distinct = independence_warning.get("distinct_vendors")
        required = independence_warning.get("required")
        families = independence_warning.get("families") or []
        labeled = independence_warning.get("labeled_quorum")
        lines.append(
            f"- ⚠️ Independence warning: all {labeled} labeled vote(s) came "
            f"from {distinct} vendor family/families "
            f"(families: {', '.join(families) if families else '—'}); "
            f"required ≥ {required} distinct. Same-vendor agreement may "
            "overstate independent corroboration."
        )

    finding_matrix_md = metadata.get("finding_matrix")
    if isinstance(finding_matrix_md, dict) and (
        finding_matrix_md.get("consensus_blockers")
        or finding_matrix_md.get("single_peer_concerns")
    ):
        lines.extend(["## Finding Matrix", ""])
        consensus = finding_matrix_md.get("consensus_blockers") or []
        if consensus:
            lines.append("**Consensus blockers** (>=2 peers, overlapping verified ranges):")
            lines.append("")
            for entry in consensus:
                peers = ", ".join(entry.get("peers") or [])
                location = ""
                path = entry.get("path")
                if path:
                    lo = entry.get("start_line")
                    hi = entry.get("end_line")
                    location = f" at `{path}:{lo}-{hi}`"
                lines.append(
                    f"- {entry.get('id')} [{entry.get('severity')}]{location} — {peers}"
                )
                claim = (entry.get("claim") or "").strip()
                if claim:
                    lines.append(f"  - {claim}")
            lines.append("")
        singles = finding_matrix_md.get("single_peer_concerns") or []
        if singles:
            lines.append("**Single-peer concerns:**")
            lines.append("")
            for entry in singles:
                peer = entry.get("peer") or "?"
                location = ""
                path = entry.get("path")
                if path:
                    lo = entry.get("start_line")
                    hi = entry.get("end_line")
                    location = f" at `{path}:{lo}-{hi}`"
                    if entry.get("unverified"):
                        location += " (unverified)"
                elif entry.get("unverified"):
                    location = " (unverified)"
                lines.append(
                    f"- {peer} [{entry.get('severity')}]{location}"
                )
                claim = (entry.get("claim") or "").strip()
                if claim:
                    lines.append(f"  - {claim}")
            lines.append("")

    synthesis_md = metadata.get("synthesis")
    if (
        isinstance(synthesis_md, dict)
        and synthesis_md.get("ok")
        and (synthesis_md.get("output") or "").strip()
    ):
        # The synthesis chair is an opt-in, PAID extra call; its decision memo
        # was previously preserved only in the JSON transcript, invisible on
        # the human-facing markdown surface. The chair's `output` already
        # carries the structured ## Decision / ## Consensus blockers / ## Dissent
        # sections, so render it verbatim under a header plus the parsed label.
        chair = synthesis_md.get("chair") or "?"
        decision = synthesis_md.get("decision_label") or "unknown"
        lines.extend([f"## Synthesis (chair: {chair})", ""])
        lines.append(f"**Decision:** {decision}")
        lines.append("")
        lines.append(synthesis_md["output"].strip())
        lines.append("")

    fence = markdown_fence(prompt)
    lines.extend(["## Prompt Sent", "", f"{fence}text", prompt, fence, ""])

    deliberation_prompts = metadata.get("deliberation_prompts")
    if isinstance(deliberation_prompts, dict):
        for round_key in sorted(deliberation_prompts.keys()):
            text = deliberation_prompts[round_key]
            if not isinstance(text, str) or not text:
                continue
            round_fence = markdown_fence(text)
            lines.extend(
                [
                    f"## Round {round_key} Prompt",
                    "",
                    f"{round_fence}text",
                    text,
                    round_fence,
                    "",
                ]
            )
    _atomic_write_private(markdown_path, "\n".join(lines))

    # These keys live at the TOP level of the JSON payload for downstream
    # consumers (dashboards, external tooling). We extract them from `metadata`
    # and remove them there to avoid double-serialization (the same dict
    # appearing under both `metadata.<key>` and `json_payload.<key>`).
    # Omitted entirely when the producing pass (findings) did not run.
    # Shallow copy so the in-memory `metadata` mutation does not surprise
    # the caller (orchestrator continues to use its own reference after
    # `write_transcript` returns).
    metadata = dict(metadata)
    LIFTED_KEYS = ("finding_matrix",)
    lifted = {k: metadata.pop(k) for k in LIFTED_KEYS if k in metadata}

    json_payload: dict[str, Any] = {
        "question": question,
        "mode": mode,
        "current": current,
        "participants": participants,
        "prompt": prompt,
        "metadata": metadata,
        "results": [result_to_dict(result) for result in results],
    }
    if parent_run_id:
        json_payload["parent_run_id"] = parent_run_id
    if remaining is not None:
        json_payload["remaining_disagreement"] = remaining
    if degraded is not None:
        json_payload["degraded_consensus"] = degraded
    overflow_records = context_overflow_records(results)
    if overflow_records:
        json_payload["context_overflow_excluded"] = overflow_records
    finding_matrix_payload = lifted.get("finding_matrix")
    if isinstance(finding_matrix_payload, dict) and (
        finding_matrix_payload.get("consensus_blockers")
        or finding_matrix_payload.get("single_peer_concerns")
    ):
        # Mirrors the shape used in MCP `structured_results`.
        json_payload["finding_matrix"] = finding_matrix_payload
    _atomic_write_private(json_path, json.dumps(json_payload, indent=2) + "\n")

    # Generate and write HTML transcript
    html_path = markdown_path.with_suffix(".html")
    html_content = _generate_html_dashboard(
        question=question,
        mode=mode,
        current=current,
        participants=participants,
        results=results,
        metadata=metadata,
        parent_run_id=parent_run_id,
        elapsed_total=elapsed_total,
        token_total=token_total,
        cost_total=cost_total,
        recommendations=recommendations,
        quorum=quorum,
    )
    try:
        _atomic_write_private(html_path, html_content)
    except OSError:
        pass


def _generate_html_dashboard(
    question: str,
    mode: str,
    current: str | None,
    participants: list[str],
    results: list[ParticipantResult],
    metadata: dict[str, Any],
    parent_run_id: str | None,
    elapsed_total: float,
    token_total: int | None,
    cost_total: float | None,
    recommendations: dict[str, int],
    quorum: dict[str, Any],
) -> str:
    import html
    def esc(text: str) -> str:
        return html.escape(text)

    # None means no participant reported the figure (text-mode CLI peers
    # have no metering hook) — render "n/a", never a misleading zero.
    token_display = "n/a" if token_total is None else str(token_total)
    cost_display = "n/a" if cost_total is None else f"${cost_total:.5f}"

    synthesis = metadata.get("synthesis") or {}
    decision = final_decision_label(results)
    decision_badge_class = f"badge-{decision.lower()}" if decision.lower() in ("yes", "no", "tradeoff") else "badge-unknown"

    peers_html = []
    for r in results:
        status = "ok" if r.ok else "error"
        cache_tag = " [cached]" if r.from_cache else ""
        from llm_council.deliberation import recommendation_label
        rec = recommendation_label(r.output) if r.ok else "unknown"
        rec_badge = f'<span class="badge badge-{rec.lower()}">{rec.upper()}</span>' if r.ok else ""
        
        stance = getattr(r, "stance", None)
        stance_class = f"stance-{stance}" if stance in ("for", "against", "neutral") else ""
        stance_label = f"Stance: {stance.upper()}" if stance else "Stance: GENERAL"

        peers_html.append(f"""
        <div class="card response-card {stance_class}">
            <div class="card-title">
                <div>
                    <strong>{esc(r.name)}</strong> 
                    <span style="font-size: 13px; color: var(--text-muted); margin-left: 8px;">
                        ({status}){cache_tag} &bull; {r.elapsed_seconds:.1f}s &bull; {r.total_tokens or 0} tokens &bull; ${r.cost_usd or 0:.6f}
                    </span>
                </div>
                <div>
                    {rec_badge}
                    <span class="badge" style="background-color: rgba(171, 125, 246, 0.15); color: #d3bcf6; border: 1px solid rgba(171, 125, 246, 0.4); margin-left: 8px;">
                        {esc(stance_label)}
                    </span>
                </div>
            </div>
            <pre>{esc(r.output) if r.ok else esc(r.error)}</pre>
        </div>
        """)

    synthesis_html = ""
    if synthesis.get("ok") and (synthesis.get("output") or "").strip():
        synthesis_html = f"""
        <div class="card" style="border-left: 4px solid var(--accent-color);">
            <div class="card-title">
                <strong>Synthesis Chair Report (Chair: {esc(synthesis.get("chair") or "?")})</strong>
                <span class="badge badge-{decision.lower()}">Decision: {esc(decision.upper())}</span>
            </div>
            <pre>{esc(synthesis["output"].strip())}</pre>
        </div>
        """

    quorum_msg = f"{quorum['labeled_quorum']} of {len(results)} peers labeled (min: {quorum['min_quorum']})"
    if quorum.get("degraded"):
        quorum_msg += " — DEGRADED"

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Council Transcript Dashboard</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --text-muted: #8b949e;
            --primary-color: #58a6ff;
            --success-color: #2ea44f;
            --danger-color: #f85149;
            --warning-color: #db6d28;
            --accent-color: #ab7df6;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}
        h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 600;
            background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 2em;
            text-transform: uppercase;
        }}
        .badge-yes {{ background-color: rgba(46, 164, 79, 0.15); color: #56d364; border: 1px solid rgba(46, 164, 79, 0.4); }}
        .badge-no {{ background-color: rgba(248, 81, 73, 0.15); color: #ff7b72; border: 1px solid rgba(248, 81, 73, 0.4); }}
        .badge-tradeoff {{ background-color: rgba(219, 109, 40, 0.15); color: #f0883e; border: 1px solid rgba(219, 109, 40, 0.4); }}
        .badge-unknown {{ background-color: rgba(139, 148, 158, 0.15); color: #c9d1d9; border: 1px solid rgba(139, 148, 158, 0.4); }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .stat-val {{
            font-size: 24px;
            font-weight: 700;
            margin-top: 8px;
            color: var(--primary-color);
        }}
        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}
        .tab:hover {{
            color: var(--text-color);
        }}
        .tab.active {{
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
        }}
        
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        .card-title {{
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 16px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        pre {{
            background-color: #0d1117;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            margin: 0;
        }}
        
        .response-card {{
            border-left: 4px solid var(--border-color);
        }}
        .response-card.stance-for {{ border-left-color: var(--success-color); }}
        .response-card.stance-against {{ border-left-color: var(--danger-color); }}
        .response-card.stance-neutral {{ border-left-color: var(--accent-color); }}
        
        .search-box {{
            width: 100%;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }}
        .search-box:focus {{
            border-color: var(--primary-color);
            outline: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>LLM Council Dashboard</h1>
                <div style="font-size: 14px; color: var(--text-muted); margin-top: 4px;">
                    Mode: <code>{esc(mode)}</code> &bull; Current Agent: <code>{esc(current or 'unknown')}</code>
                </div>
            </div>
            <div>
                <span class="badge {decision_badge_class}" style="font-size: 16px; padding: 6px 16px;">
                    Decision: {esc(decision.upper())}
                </span>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Elapsed Total</div>
                <div class="stat-val">{elapsed_total:.1f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tokens Reported</div>
                <div class="stat-val">{token_display}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cost (USD)</div>
                <div class="stat-val">{cost_display}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Quorum</div>
                <div class="stat-val" style="font-size: 16px; margin-top: 16px;">{esc(quorum_msg)}</div>
            </div>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('debate')">Debate Timeline</div>
            <div class="tab" onclick="switchTab('summary')">Executive Report</div>
            <div class="tab" onclick="switchTab('prompt')">Prompt & Context</div>
        </div>

        <div id="debate-content" class="tab-content active">
            <input type="text" class="search-box" id="search-input" placeholder="Search responses..." onkeyup="filterResponses()">
            <div id="responses-container">
                {"".join(peers_html)}
            </div>
        </div>

        <div id="summary-content" class="tab-content">
            {synthesis_html}
            <div class="card">
                <h3 style="margin-top: 0;">Vote Summary</h3>
                <p>Yes: <strong>{recommendations.get('yes', 0)}</strong></p>
                <p>No: <strong>{recommendations.get('no', 0)}</strong></p>
                <p>Tradeoff: <strong>{recommendations.get('tradeoff', 0)}</strong></p>
                <p>Unknown: <strong>{recommendations.get('unknown', 0)}</strong></p>
            </div>
        </div>

        <div id="prompt-content" class="tab-content">
            <div class="card">
                <div class="card-title"><strong>Original Prompt</strong></div>
                <pre>{esc(question)}</pre>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            const tabEl = Array.from(document.querySelectorAll('.tab')).find(t => t.textContent.toLowerCase().includes(tabId === 'prompt' ? 'prompt' : tabId === 'summary' ? 'executive' : 'debate'));
            if (tabEl) tabEl.classList.add('active');
            
            document.getElementById(tabId + '-content').classList.add('active');
        }}

        function filterResponses() {{
            const query = document.getElementById('search-input').value.toLowerCase();
            document.querySelectorAll('.response-card').forEach(card => {{
                const text = card.textContent.toLowerCase();
                if (text.includes(query)) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    return html_str



def markdown_fence(text: str) -> str:
    longest = 0
    for match in re.finditer(r"`+", text):
        longest = max(longest, len(match.group(0)))
    return "`" * max(3, longest + 1)
