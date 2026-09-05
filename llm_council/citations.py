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
from llm_council.blocking import check_cancelled

MAX_CITATION_FILE_CHARS = 8 * 1024 * 1024
MAX_CITATION_RUN_CHARS = 32 * 1024 * 1024

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


class CitationVerifier:
    """Share line counts within a run, invalidating when a file changes.

    This verifies only path/range existence, never the truth of a claim.
    """

    def __init__(self, cwd: Path):
        self.cwd = cwd.resolve()
        self._counts: dict[Path, tuple[tuple[int, ...], int]] = {}
        self._remaining = MAX_CITATION_RUN_CHARS

    def verify(self, ref: VerifiedRef) -> bool:
        check_cancelled()
        try:
            if not 1 <= ref.start_line <= ref.end_line:
                return False
            candidate = (self.cwd / ref.path).resolve()
            if not candidate.is_relative_to(self.cwd) or not candidate.is_file():
                return False
            info = candidate.stat()
            identity = (info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino)
            cached = self._counts.get(candidate)
            if cached is None or cached[0] != identity:
                with candidate.open("r", encoding="utf-8", errors="replace") as fh:
                    count = 0
                    last = ""
                    remaining = min(MAX_CITATION_FILE_CHARS, self._remaining)
                    while remaining > 0:
                        check_cancelled()
                        chunk = fh.read(min(64 * 1024, remaining))
                        if not chunk:
                            count += bool(last and last != "\n")
                            break
                        remaining -= len(chunk)
                        self._remaining -= len(chunk)
                        count += chunk.count("\n")
                        last = chunk[-1:]
                    else:
                        # At a cap, only newline-terminated lines are proven.
                        # An incomplete final line must never count as verified.
                        if last and not fh.read(1):
                            count += last != "\n"
                cached = (identity, count)
                self._counts[candidate] = cached
            return ref.end_line <= cached[1]
        except (OSError, ValueError, RuntimeError, TypeError):
            return False


def verify_ref(ref: VerifiedRef, cwd: Path) -> bool:
    """True iff the path is within cwd and the line range is in bounds."""
    return CitationVerifier(cwd).verify(ref)


def verify_evidence_citations(
    results: list[Any], cwd: Path, *, verifier: CitationVerifier | None = None
) -> None:
    """Mutate each result's evidence list to set `verified` on VERIFIED entries.

    For every entry with `tag == "verified"`, run `verify_ref` and set the
    `verified` key to True/False. Failed refs are appended to the result's
    `evidence_verification_failures` as `path:start-end` strings.

    Pure mutation (no return value) — matches the in-place style already used
    by `_with_envelope`. Safe to call multiple times; only flips `verified` if
    it is currently None.
    """
    verifier = verifier or CitationVerifier(cwd)
    for result in results:
        check_cancelled()
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
            try:
                if path is None or start is None:
                    raise ValueError("missing path/range")
                ref = VerifiedRef(path=str(path), start_line=int(start), end_line=int(end))
            except (TypeError, ValueError, OverflowError):
                entry["verified"] = False
                failures.append(f"{path}:{start}-{end} (malformed)")
                continue
            ok = verifier.verify(ref)
            entry["verified"] = ok
            if not ok:
                failures.append(f"{ref.path}:{ref.start_line}-{ref.end_line}")
        if failures:
            existing = getattr(result, "evidence_verification_failures", None) or []
            result.evidence_verification_failures = [*existing, *failures]
