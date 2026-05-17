"""Verified file:line citation primitives for evidence tags.

A peer can cite code with `[VERIFIED:path:start-end]` (start-end optional,
single line if absent). The orchestrator runs `verify_ref` against the repo
after participant results return; failures are recorded on the result for
operator visibility, but the entry is NOT dropped (coverage > filtering).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

_VERIFIED_TAG_RE = re.compile(
    r"\[\s*VERIFIED\s*:\s*(?P<path>[^\s:\]]+)\s*:\s*(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?\s*\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerifiedRef:
    path: str
    start_line: int
    end_line: int  # inclusive; equals start_line for single-line refs


def parse_verified_tag(text: str) -> VerifiedRef | None:
    """Return the first `[VERIFIED:path:start-end]` reference, or None."""
    m = _VERIFIED_TAG_RE.search(text or "")
    if not m:
        return None
    start = int(m.group("start"))
    end_raw = m.group("end")
    end = int(end_raw) if end_raw else start
    if end < start:
        start, end = end, start
    return VerifiedRef(path=m.group("path"), start_line=start, end_line=end)


def strip_verified_tag(text: str) -> str:
    """Remove the `[VERIFIED:...]` token from a string."""
    return _VERIFIED_TAG_RE.sub("", text or "").strip(" -—–:")


def verify_ref(ref: VerifiedRef, cwd: Path) -> bool:
    """True iff the path exists under cwd AND the line range is in bounds.

    Rejects path-traversal escapes (resolved path must be under cwd).
    Read errors (binary files, encoding issues) count as bound-check failure.
    """
    try:
        cwd_resolved = cwd.resolve()
        candidate = (cwd / ref.path).resolve()
        if not candidate.is_file():
            return False
        try:
            candidate.relative_to(cwd_resolved)
        except ValueError:
            return False
        with candidate.open("r", encoding="utf-8", errors="replace") as fh:
            line_count = sum(1 for _ in fh)
        return 1 <= ref.start_line <= line_count and ref.end_line <= line_count
    except (OSError, ValueError):
        return False


def verify_evidence_citations(results: list[Any], cwd: Path) -> None:
    """Mutate each result's evidence list to set `verified` on VERIFIED entries.

    For every entry with `tag == "verified"`, run `verify_ref` and set the
    `verified` key to True/False. Failed refs are appended to the result's
    `evidence_verification_failures` as `path:start-end` strings.

    Pure mutation (no return value) — matches the in-place style already used
    by `_with_envelope`. Safe to call multiple times; only flips `verified` if
    it is currently None.
    """
    for result in results:
        evidence = getattr(result, "evidence", None) or []
        failures: list[str] = []
        for entry in evidence:
            if not isinstance(entry, dict):
                continue
            if entry.get("tag") != "verified":
                continue
            if entry.get("verified") is not None:
                continue
            path = entry.get("path")
            start = entry.get("start_line")
            end = entry.get("end_line", start)
            if path is None or start is None:
                entry["verified"] = False
                failures.append(f"{path}:{start}-{end} (malformed)")
                continue
            ref = VerifiedRef(path=str(path), start_line=int(start), end_line=int(end))
            ok = verify_ref(ref, cwd)
            entry["verified"] = ok
            if not ok:
                failures.append(f"{ref.path}:{ref.start_line}-{ref.end_line}")
        if failures:
            existing = getattr(result, "evidence_verification_failures", None) or []
            result.evidence_verification_failures = [*existing, *failures]
