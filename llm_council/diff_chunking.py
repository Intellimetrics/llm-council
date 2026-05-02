"""Opt-in diff chunking strategies for oversized review prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VALID_STRATEGIES = ("fail", "head", "tail", "hash-aware")

_DIFF_GIT_RE = re.compile(r"^diff --git ", re.MULTILINE)
_DIFF_PATH_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_FILENAME_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+")
_PATHY_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+")
_BAREWORD_PATHS = frozenset(
    {"Makefile", "Dockerfile", "Rakefile", "Procfile", "Gemfile", "Justfile"}
)


@dataclass
class ChunkResult:
    """Outcome of applying a chunking strategy to a raw unified diff."""

    text: str
    strategy: str
    original_chars: int
    chunked_chars: int
    dropped_chars: int
    dropped_files: list[str] = field(default_factory=list)
    marker: str = ""
    triggered: bool = False


def chunk_diff(
    raw_diff: str,
    *,
    strategy: str,
    budget: int,
    question: str,
) -> ChunkResult:
    """Apply ``strategy`` to ``raw_diff`` so it fits within ``budget`` chars.

    For ``head`` / ``tail`` we keep the first / last ``budget`` characters
    minus the marker length and emit a sentinel announcing the dropped count.
    """

    original = raw_diff or ""
    original_chars = len(original)
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown chunk strategy '{strategy}'. "
            f"Expected one of: {', '.join(VALID_STRATEGIES)}"
        )
    if strategy == "fail":
        return ChunkResult(
            text=original,
            strategy=strategy,
            original_chars=original_chars,
            chunked_chars=original_chars,
            dropped_chars=0,
        )
    if budget <= 0:
        marker = (
            f"[diff dropped: chunking budget exhausted; original {original_chars} chars]"
        )
        return ChunkResult(
            text=marker,
            strategy=strategy,
            original_chars=original_chars,
            chunked_chars=len(marker),
            dropped_chars=original_chars,
            dropped_files=_all_paths(original),
            marker=marker,
            triggered=original_chars > 0,
        )
    if original_chars <= budget:
        return ChunkResult(
            text=original,
            strategy=strategy,
            original_chars=original_chars,
            chunked_chars=original_chars,
            dropped_chars=0,
        )

    if strategy == "head":
        return _chunk_head(original, budget=budget)
    if strategy == "tail":
        return _chunk_tail(original, budget=budget)
    return _chunk_hash_aware(original, budget=budget, question=question)


def _chunk_head(diff: str, *, budget: int) -> ChunkResult:
    original_chars = len(diff)
    # Reserve room for the marker. Use a placeholder length and then refine.
    placeholder = f"[diff truncated after head: dropped {original_chars} chars]"
    keep = max(0, budget - len(placeholder) - 2)
    body = diff[:keep]
    dropped = original_chars - len(body)
    marker = f"[diff truncated after head: dropped {dropped} chars]"
    text = body.rstrip() + "\n" + marker
    return ChunkResult(
        text=text,
        strategy="head",
        original_chars=original_chars,
        chunked_chars=len(text),
        dropped_chars=dropped,
        marker=marker,
        triggered=True,
    )


def _chunk_tail(diff: str, *, budget: int) -> ChunkResult:
    original_chars = len(diff)
    placeholder = f"[diff truncated before tail: dropped {original_chars} chars]"
    keep = max(0, budget - len(placeholder) - 2)
    body = diff[-keep:] if keep else ""
    dropped = original_chars - len(body)
    marker = f"[diff truncated before tail: dropped {dropped} chars]"
    text = marker + "\n" + body.lstrip()
    return ChunkResult(
        text=text,
        strategy="tail",
        original_chars=original_chars,
        chunked_chars=len(text),
        dropped_chars=dropped,
        marker=marker,
        triggered=True,
    )


def _chunk_hash_aware(diff: str, *, budget: int, question: str) -> ChunkResult:
    """Split on per-file ``diff --git`` boundaries and keep highest-relevance hunks.

    Each hunk is scored by (1) explicit filename mentions in the question,
    (2) extension affinity inferred from the question text, then (3) smaller
    hunks first as a tie-breaker to maximise file coverage. We greedily admit
    hunks in priority order until the next one would exceed the budget,
    emitting a marker that lists every file we had to drop.
    """

    original_chars = len(diff)
    hunks = _split_hunks(diff)
    if not hunks:
        return _chunk_head(diff, budget=budget)

    mentioned_paths = _filename_tokens(question)
    extension_hits = {token.rsplit(".", 1)[-1].lower() for token in mentioned_paths}
    keyword_extensions = _question_extension_hints(question)

    scored: list[tuple[int, int, int, int, _Hunk]] = []
    for index, hunk in enumerate(hunks):
        path = hunk.path or ""
        path_lower = path.lower()
        priority = 0
        if path and any(token.lower() in path_lower for token in mentioned_paths):
            priority -= 100
        if path:
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if ext and ext in extension_hits:
                priority -= 25
            if ext and ext in keyword_extensions:
                priority -= 10
        scored.append((priority, len(hunk.text), index, index, hunk))

    scored.sort(key=lambda row: (row[0], row[1], row[2]))

    sep = "\n"
    accepted: list[tuple[int, _Hunk]] = []
    used = 0
    dropped_files: list[str] = []
    for _priority, length, _idx, original_index, hunk in scored:
        addition = length + (len(sep) if accepted else 0)
        if used + addition > budget:
            dropped_files.append(hunk.path or "<unparsed>")
            continue
        accepted.append((original_index, hunk))
        used += addition

    if not accepted:
        # Budget too small for even the smallest whole-file block: head-truncate
        # the highest-priority hunk so the most relevant file still reaches the
        # council, and record the result as a hash-aware fallback so transcripts
        # don't claim a clean run.
        top_hunk = scored[0][4]
        head_only = _chunk_head(top_hunk.text, budget=budget)
        other_paths = [hunk.path or "<unparsed>" for _p, _l, _i, _o, hunk in scored[1:]]
        return ChunkResult(
            text=head_only.text,
            strategy="hash-aware",
            original_chars=original_chars,
            chunked_chars=head_only.chunked_chars,
            dropped_chars=original_chars - head_only.chunked_chars,
            dropped_files=other_paths,
            marker=head_only.marker,
            triggered=True,
        )

    accepted.sort(key=lambda row: row[0])
    body = sep.join(hunk.text for _idx, hunk in accepted)

    if dropped_files:
        marker_paths = ", ".join(dropped_files)
        marker = (
            f"[diff truncated by hash-aware chunker: dropped {len(dropped_files)} "
            f"file(s): {marker_paths}]"
        )
        # Try to fit the marker; if it overflows, drop more files.
        while accepted and len(body) + len(sep) + len(marker) > budget:
            _idx, hunk = accepted.pop()
            dropped_files.append(hunk.path or "<unparsed>")
            body = sep.join(item.text for _idx2, item in accepted)
            marker_paths = ", ".join(dropped_files)
            marker = (
                f"[diff truncated by hash-aware chunker: dropped {len(dropped_files)} "
                f"file(s): {marker_paths}]"
            )
        text = (body + "\n" + marker) if body else marker
    else:
        marker = ""
        text = body

    triggered = bool(dropped_files)
    dropped_chars = original_chars - len(body)
    return ChunkResult(
        text=text,
        strategy="hash-aware",
        original_chars=original_chars,
        chunked_chars=len(text),
        dropped_chars=max(0, dropped_chars),
        dropped_files=dropped_files,
        marker=marker,
        triggered=triggered,
    )


@dataclass
class _Hunk:
    path: str
    text: str


def _split_hunks(diff: str) -> list[_Hunk]:
    if not diff or "diff --git " not in diff:
        return []
    matches = list(_DIFF_GIT_RE.finditer(diff))
    if not matches:
        return []
    hunks: list[_Hunk] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(diff)
        block = diff[start:end].rstrip("\n")
        path_match = _DIFF_PATH_RE.match(block)
        path = path_match.group(2) if path_match else ""
        hunks.append(_Hunk(path=path, text=block))
    return hunks


def _all_paths(diff: str) -> list[str]:
    return [hunk.path or "<unparsed>" for hunk in _split_hunks(diff)]


def _filename_tokens(question: str) -> list[str]:
    if not question:
        return []
    seen: set[str] = set()
    tokens: list[str] = []

    def _add(value: str) -> None:
        normalized = value.strip(".,;:'\"()[]")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tokens.append(normalized)

    for match in _FILENAME_TOKEN_RE.findall(question):
        _add(match)
    for match in _PATHY_TOKEN_RE.findall(question):
        _add(match)
    for word in re.findall(r"[A-Za-z][A-Za-z0-9]+", question):
        if word in _BAREWORD_PATHS:
            _add(word)
    return tokens


_KEYWORD_EXTENSIONS = {
    "python": {"py"},
    "py": {"py"},
    "javascript": {"js", "mjs", "cjs"},
    "typescript": {"ts", "tsx"},
    "tsx": {"tsx"},
    "jsx": {"jsx"},
    "rust": {"rs"},
    "go": {"go"},
    "ruby": {"rb"},
    "java": {"java"},
    "kotlin": {"kt"},
    "swift": {"swift"},
    "cpp": {"cpp", "cc", "cxx", "hpp", "h"},
    "c++": {"cpp", "cc", "cxx", "hpp", "h"},
    "shell": {"sh", "bash"},
    "bash": {"sh", "bash"},
    "yaml": {"yaml", "yml"},
    "markdown": {"md"},
    "docs": {"md", "rst"},
    "documentation": {"md", "rst"},
    "config": {"yaml", "yml", "toml", "json", "ini"},
    "tests": {"py", "js", "ts"},
}


def _question_extension_hints(question: str) -> set[str]:
    if not question:
        return set()
    lowered = question.lower()
    hits: set[str] = set()
    for keyword, exts in _KEYWORD_EXTENSIONS.items():
        if keyword in lowered:
            hits.update(exts)
    return hits
