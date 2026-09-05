"""Semantic search over existing council transcripts (v0.9.0).

`search_similar` returns the top-k most-similar prior council runs given a
free-text query. Uses Jaccard tokenization (reuse `convergence.tokenize`)
over the prior question text. NO new dependencies; sentence-transformers
deferred until Jaccard proves insufficient.

Scope-cut: `find_contradictions` and `trace_evolution` are deferred to
v0.9.x. Search-only for v0.9.0.

Inspired by ai-counsel's `query_decisions` MCP tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from datetime import datetime
import heapq
import json
import os
from pathlib import Path
import threading
from typing import Any

from llm_council.convergence import jaccard_similarity, tokenize
from llm_council.adapters import ParticipantResult
from llm_council.blocking import check_cancelled
from llm_council.transcript import final_decision_label

_QUESTION_EXCERPT_MAX_CHARS = 150
# Keep only compact search records, never prompts or raw peer responses.
# Revalidate file identities on every query so rewrites and pruning are visible.
_INDEX_LOCK = threading.Lock()
_INDEX: OrderedDict[Path, dict[Path, tuple[tuple[int, ...], "_SearchRecord | None"]]] = OrderedDict()
_MAX_INDEX_DIRECTORIES = 8
_MAX_INDEX_RECORDS = 10_000
_MAX_DIRECTORY_ENTRIES = 100_000
_MAX_TRANSCRIPT_BYTES = 8 * 1024 * 1024
_MAX_QUESTION_CHARS = 4096
_MAX_QUERY_READ_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class _SearchRecord:
    tokens: frozenset[str]
    run_id: str
    question_excerpt: str
    recommendation_label: str | None
    timestamp: str


@dataclass(frozen=True)
class SimilarMatch:
    """A single prior council run that matches the query."""

    run_id: str
    similarity: float  # 0.0 to 1.0 (Jaccard ratio)
    question_excerpt: str  # first ~150 chars of the prior question
    recommendation_label: str | None  # includes directional "leaning-*" ties
    timestamp: str  # ISO-format derived from the run-id prefix when available


def _truncate_question(question: str) -> str:
    cleaned = (question or "").strip()
    if len(cleaned) <= _QUESTION_EXCERPT_MAX_CHARS:
        return cleaned
    # Use a Unicode ellipsis to flag truncation. Match the same byte-budget
    # downstream consumers (transcripts CLI summary, ai-counsel parity)
    # expect — characters not bytes.
    return cleaned[:_QUESTION_EXCERPT_MAX_CHARS].rstrip() + "…"


def _timestamp_from_run_id(run_id: str) -> str:
    """Parse the leading ``YYYYMMDD_HHMMSS`` prefix into an ISO-8601 string.

    Returns the empty string when the prefix is missing/unparseable — a
    transcript without a recognizable run-id prefix is uncommon (only
    hand-edited fixtures), so we tolerate it rather than crash the search.
    """
    if len(run_id) < 15 or run_id[8] != "_":
        return ""
    prefix = run_id[:15]
    try:
        dt = datetime.strptime(prefix, "%Y%m%d_%H%M%S")
    except ValueError:
        return ""
    return dt.isoformat()


def _final_round_label_from_results(results: list[dict[str, Any]]) -> str | None:
    """Use the dashboard's final-round decision, including ties and leanings."""
    peers = [
        ParticipantResult(
            name=str(r.get("name") or ""), ok=bool(r.get("ok")),
            output=str(r.get("output") or ""), error=str(r.get("error") or ""),
            elapsed_seconds=0,
        )
        for r in results if isinstance(r, dict) and not r.get("is_ranking_round")
    ]
    label = final_decision_label(peers)
    return None if label == "unknown" else label


def _read_search_record(path: Path) -> _SearchRecord | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_TRANSCRIPT_BYTES + 1)
        if len(raw) > _MAX_TRANSCRIPT_BYTES:
            return None
        data = json.loads(raw)
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    tokens = frozenset(tokenize(question[:_MAX_QUESTION_CHARS]))
    if not tokens:
        return None
    results = data.get("results")
    return _SearchRecord(
        tokens, path.stem, _truncate_question(question),
        _final_round_label_from_results(results if isinstance(results, list) else []),
        _timestamp_from_run_id(path.stem),
    )


def _search_records(base_dir: Path, diagnostics: dict[str, Any]) -> list[_SearchRecord]:
    base_dir = base_dir.resolve()
    # Retain only the newest bounded window before opening any transcripts.
    # scandir streams entries; glob can materialize a directory first.
    candidates: list[tuple[int, str, Path, tuple[int, ...]]] = []
    scanned = eligible = oversized = 0
    entry_limit_reached = False
    with os.scandir(base_dir) as entries:
        for entry in entries:
            check_cancelled()
            scanned += 1
            if scanned > _MAX_DIRECTORY_ENTRIES:
                entry_limit_reached = True
                break
            try:
                if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                    continue
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if info.st_size > _MAX_TRANSCRIPT_BYTES:
                oversized += 1
                continue
            eligible += 1
            identity = (info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino)
            item = (info.st_mtime_ns, entry.name, Path(entry.path), identity)
            if len(candidates) < _MAX_INDEX_RECORDS:
                heapq.heappush(candidates, item)
            elif item > candidates[0]:
                heapq.heapreplace(candidates, item)
    diagnostics.update(
        limited=entry_limit_reached or oversized > 0 or eligible > len(candidates),
        entry_limit_reached=entry_limit_reached,
        skipped_oversize_files=oversized,
        eligible_files=eligible, searched_files=len(candidates),
        max_records=_MAX_INDEX_RECORDS, max_transcript_bytes=_MAX_TRANSCRIPT_BYTES,
        question_prefix_chars=_MAX_QUESTION_CHARS,
    )
    while not _INDEX_LOCK.acquire(timeout=.1):
        check_cancelled()
    try:
        previous = _INDEX.pop(base_dir, {})
        refreshed = {}
        records = []
        remaining_bytes = _MAX_QUERY_READ_BYTES
        deferred = 0
        for _, _, path, identity in sorted(candidates, reverse=True):
            check_cancelled()
            cached = previous.get(path)
            if cached and cached[0] == identity:
                record = cached[1]
            else:
                if identity[2] > remaining_bytes:
                    deferred += 1
                    continue
                remaining_bytes -= identity[2]
                record = _read_search_record(path)
            if len(refreshed) < _MAX_INDEX_RECORDS:
                refreshed[path] = (identity, record)
            if record is not None:
                records.append(record)
        _INDEX[base_dir] = refreshed
        diagnostics["deferred_files"] = deferred
        diagnostics["limited"] |= deferred > 0
        diagnostics["max_query_read_bytes"] = _MAX_QUERY_READ_BYTES
        while len(_INDEX) > _MAX_INDEX_DIRECTORIES:
            _INDEX.popitem(last=False)
        return records
    finally:
        _INDEX_LOCK.release()


def search_similar(
    query: str,
    top_k: int = 5,
    runs_dir: Path | None = None,
    *, diagnostics: dict[str, Any] | None = None,
) -> list[SimilarMatch]:
    """Top-k most-similar past councils by Jaccard token overlap on the
    original question text.

    Returns a list ranked similarity-descending. Empty list when the
    runs directory is missing, empty, or no transcript carries a
    ``question`` field. Malformed transcript JSON is silently skipped
    (mirrors `stats.load_transcript_files` and `transcript_records`).

    Search covers a bounded window of newest transcripts and the first 4096
    question characters. Optional diagnostics report limits and deferred reads.

    A degenerate query (empty after tokenization) returns ``[]`` rather
    than ranking everything at similarity ``0.0`` — the caller almost
    certainly wants no matches over an arbitrary ordering.
    """
    if top_k <= 0:
        return []
    base_dir = Path(runs_dir) if runs_dir is not None else Path(".llm-council/runs")
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    query_tokens = tokenize((query or "")[:_MAX_QUESTION_CHARS])
    if not query_tokens:
        return []

    records = _search_records(base_dir, diagnostics if diagnostics is not None else {})
    scored: list[tuple[float, SimilarMatch]] = []
    for record in records:
        similarity = jaccard_similarity(query_tokens, record.tokens)
        if similarity <= 0.0:
            continue
        match = SimilarMatch(
            run_id=record.run_id,
            similarity=similarity,
            question_excerpt=record.question_excerpt,
            recommendation_label=record.recommendation_label,
            timestamp=record.timestamp,
        )
        scored.append((similarity, match))

    # Sort similarity-desc; stable run-id-asc tiebreak so callers see
    # deterministic ordering across runs with identical Jaccard scores.
    return [match for _, match in heapq.nsmallest(
        top_k, scored, key=lambda item: (-item[0], item[1].run_id)
    )]
