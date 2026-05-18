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
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_council.convergence import jaccard_similarity, tokenize
from llm_council.deliberation import recommendation_label
from llm_council.stats import load_transcript_files
from llm_council.transcript import result_round

_QUESTION_EXCERPT_MAX_CHARS = 150
_VALID_LABELS = frozenset({"yes", "no", "tradeoff"})


@dataclass(frozen=True)
class SimilarMatch:
    """A single prior council run that matches the query."""

    run_id: str
    similarity: float  # 0.0 to 1.0 (Jaccard ratio)
    question_excerpt: str  # first ~150 chars of the prior question
    recommendation_label: str | None  # "yes" | "no" | "tradeoff" | None
    timestamp: str  # ISO-format derived from the run-id prefix when available


def _truncate_question(question: str) -> str:
    cleaned = (question or "").strip()
    if len(cleaned) <= _QUESTION_EXCERPT_MAX_CHARS:
        return cleaned
    # Use a Unicode ellipsis to flag truncation. Match the same byte-budget
    # downstream consumers (transcripts CLI summary, ai-counsel parity)
    # expect — characters not bytes.
    return cleaned[:_QUESTION_EXCERPT_MAX_CHARS].rstrip() + "…"


def _run_id_from_path(path_str: str) -> str:
    return Path(path_str).stem


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
    """Return the trinary recommendation label for the final round, if any.

    Walks the raw transcript ``results`` list (dicts, not ParticipantResult),
    keeps only the final round (mirroring `transcript._select_final_round_records`
    without depending on the private helper), runs the fence-aware label
    parser from `deliberation.recommendation_label`, and returns the first
    usable trinary vote (``yes`` / ``no`` / ``tradeoff``). Returns ``None``
    when no peer emitted a usable label.
    """
    if not results:
        return None
    rounds = [result_round(str(r.get("name") or "")) for r in results]
    final_round = max(rounds) if rounds else 1
    final_records = [
        result for result, rnd in zip(results, rounds) if rnd == final_round
    ]
    for record in final_records:
        if not record.get("ok"):
            continue
        output = record.get("output") or ""
        label = recommendation_label(output)
        if label in _VALID_LABELS:
            return label
    return None


def search_similar(
    query: str,
    top_k: int = 5,
    runs_dir: Path | None = None,
) -> list[SimilarMatch]:
    """Top-k most-similar past councils by Jaccard token overlap on the
    original question text.

    Returns a list ranked similarity-descending. Empty list when the
    runs directory is missing, empty, or no transcript carries a
    ``question`` field. Malformed transcript JSON is silently skipped
    (mirrors `stats.load_transcript_files` and `transcript_records`).

    A degenerate query (empty after tokenization) returns ``[]`` rather
    than ranking everything at similarity ``0.0`` — the caller almost
    certainly wants no matches over an arbitrary ordering.
    """
    if top_k <= 0:
        return []
    base_dir = Path(runs_dir) if runs_dir is not None else Path(".llm-council/runs")
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    query_tokens = tokenize(query or "")
    if not query_tokens:
        return []

    records = load_transcript_files(base_dir)
    scored: list[tuple[float, SimilarMatch]] = []
    for record in records:
        data = record.get("data") or {}
        if not isinstance(data, dict):
            continue
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        prior_tokens = tokenize(question)
        if not prior_tokens:
            # Tokenizer drained the question entirely (e.g. only stopwords).
            # Jaccard would be 0 by definition — drop rather than rank it
            # against everything else with the same score.
            continue
        similarity = jaccard_similarity(query_tokens, prior_tokens)
        if similarity <= 0.0:
            continue
        run_id = _run_id_from_path(str(record.get("path") or ""))
        results = data.get("results") or []
        label = _final_round_label_from_results(
            results if isinstance(results, list) else []
        )
        match = SimilarMatch(
            run_id=run_id,
            similarity=similarity,
            question_excerpt=_truncate_question(question),
            recommendation_label=label,
            timestamp=_timestamp_from_run_id(run_id),
        )
        scored.append((similarity, match))

    # Sort similarity-desc; stable run-id-asc tiebreak so callers see
    # deterministic ordering across runs with identical Jaccard scores.
    scored.sort(key=lambda item: (-item[0], item[1].run_id))
    return [match for _, match in scored[:top_k]]
