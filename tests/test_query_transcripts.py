"""Tests for the Jaccard-based transcript search (Phase 4 — v0.9.0 plan).

Mirrors the style of `tests/test_stats_reliability.py`: fixtures are synthesized
transcript JSON files written to `tmp_path`. No adapter or orchestrator calls.

Covers `llm_council.query.search_similar` and the MCP `query_transcripts`
handler in `llm_council.mcp_server`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_council.query import SimilarMatch, search_similar


def _write_transcript(
    runs_dir: Path,
    run_id: str,
    *,
    question: str,
    results: list[dict] | None = None,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": question,
        "mode": "review",
        "current": "claude",
        "participants": ["claude", "codex"],
        "prompt": "...",
        "results": results if results is not None else [],
    }
    path = runs_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _result(name: str, *, ok: bool = True, recommendation: str | None = "yes") -> dict:
    if recommendation is None:
        output = "I have no opinion."
    else:
        output = f"RECOMMENDATION: {recommendation}\nrationale"
    return {
        "name": name,
        "ok": ok,
        "model": "test-model",
        "elapsed_seconds": 1.0,
        "command": None,
        "output": output if ok else "",
        "error": "" if ok else "boom",
    }


# --- empty / degenerate cases --------------------------------------------


def test_empty_runs_dir_returns_empty(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    assert search_similar("anything", runs_dir=runs) == []


def test_missing_runs_dir_returns_empty(tmp_path):
    # Directory doesn't exist at all.
    runs = tmp_path / "nope" / "nada"
    assert search_similar("anything", runs_dir=runs) == []


def test_empty_query_returns_empty(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="Should we adopt fence-aware label parsing?",
    )
    # Stopword-only query tokenizes to the empty set → no matches.
    assert search_similar("", runs_dir=runs) == []
    assert search_similar("the and of", runs_dir=runs) == []


def test_top_k_zero_returns_empty(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="adopt fence-aware label parsing",
    )
    assert search_similar("fence aware label parsing", top_k=0, runs_dir=runs) == []


# --- happy path ----------------------------------------------------------


def test_single_matching_transcript_ranks_first(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="Should we adopt fence-aware label parsing for RECOMMENDATION?",
        results=[_result("claude", recommendation="yes")],
    )
    matches = search_similar(
        "fence-aware label parsing for recommendation",
        runs_dir=runs,
    )
    assert len(matches) == 1
    assert matches[0].similarity > 0.0
    assert matches[0].run_id == "20260517_120000_run-a"
    assert matches[0].recommendation_label == "yes"


def test_top_k_caps_result_count(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    for i, q in enumerate(
        [
            "fence-aware label parsing for recommendation",
            "label parsing tweaks for the recommendation envelope",
            "the parsing of fence labels and recommendations",
        ]
    ):
        _write_transcript(
            runs,
            f"20260517_12000{i}_run-{i}",
            question=q,
            results=[_result("claude", recommendation="yes")],
        )
    matches = search_similar("fence label parsing recommendation", top_k=2, runs_dir=runs)
    assert len(matches) == 2


def test_sorted_by_similarity_descending(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    # High-overlap question
    _write_transcript(
        runs,
        "20260517_120001_run-high",
        question="fence aware label parsing recommendation",
    )
    # Low-overlap question — shares only "parsing"
    _write_transcript(
        runs,
        "20260517_120002_run-low",
        question="parsing markdown for synthesis",
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert [m.run_id for m in matches] == [
        "20260517_120001_run-high",
        "20260517_120002_run-low",
    ]
    assert matches[0].similarity > matches[1].similarity


def test_similarity_in_unit_interval(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="fence aware label parsing recommendation",
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert 0.0 <= matches[0].similarity <= 1.0


# --- robustness ----------------------------------------------------------


def test_malformed_json_is_skipped(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    runs.mkdir(parents=True)
    (runs / "20260517_120000_broken.json").write_text("{not valid json", encoding="utf-8")
    _write_transcript(
        runs,
        "20260517_120001_run-ok",
        question="fence aware label parsing recommendation",
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].run_id == "20260517_120001_run-ok"


def test_missing_question_field_skipped(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    runs.mkdir(parents=True)
    (runs / "20260517_120000_no-question.json").write_text(
        json.dumps({"mode": "review", "results": []}), encoding="utf-8"
    )
    _write_transcript(
        runs,
        "20260517_120001_run-ok",
        question="fence aware label parsing recommendation",
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].run_id == "20260517_120001_run-ok"


def test_blank_question_skipped(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(runs, "20260517_120000_blank", question="   ")
    _write_transcript(
        runs,
        "20260517_120001_run-ok",
        question="fence aware label parsing recommendation",
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].run_id == "20260517_120001_run-ok"


# --- excerpt truncation --------------------------------------------------


def test_question_excerpt_truncates_with_ellipsis(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    long_question = (
        "Should we adopt fence-aware label parsing for the RECOMMENDATION envelope "
        "across every CLI participant family including Claude Code, Codex CLI, "
        "Gemini CLI, OpenRouter hosted, and local Ollama peers given the existing "
        "test corpus and the v0.8 release cadence?"
    )
    _write_transcript(runs, "20260517_120000_run-a", question=long_question)
    matches = search_similar("fence aware label parsing", runs_dir=runs)
    assert len(matches) == 1
    excerpt = matches[0].question_excerpt
    # The literal cap is 150 chars + the single ellipsis character.
    assert len(excerpt) <= 151
    assert excerpt.endswith("…")
    # The first chunk of the original question should still be there.
    assert excerpt.startswith("Should we adopt fence-aware")


def test_question_excerpt_no_ellipsis_when_short(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(runs, "20260517_120000_run-a", question="fence label parsing")
    matches = search_similar("fence label parsing", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].question_excerpt == "fence label parsing"
    assert "…" not in matches[0].question_excerpt


# --- recommendation label extraction -------------------------------------


@pytest.mark.parametrize("label", ["yes", "no", "tradeoff"])
def test_recommendation_label_extracted(tmp_path, label):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        f"20260517_120000_run-{label}",
        question="fence aware label parsing recommendation",
        results=[_result("claude", recommendation=label)],
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].recommendation_label == label


def test_missing_recommendation_returns_none_label(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="fence aware label parsing recommendation",
        results=[_result("claude", recommendation=None)],
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].recommendation_label is None


def test_failed_result_does_not_emit_label(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="fence aware label parsing recommendation",
        results=[_result("claude", ok=False, recommendation="yes")],
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].recommendation_label is None


def test_final_round_label_wins_over_round_one(tmp_path):
    """`recommendation_label` should be drawn from the final-round results.

    Round-1 vote was `no`; round-2 vote (after deliberation) flipped to
    `yes`. The search result should report the final-round outcome.
    """
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="fence aware label parsing recommendation",
        results=[
            _result("claude", recommendation="no"),
            {
                "name": "claude:round2",
                "ok": True,
                "model": "test-model",
                "elapsed_seconds": 1.0,
                "command": None,
                "output": "RECOMMENDATION: yes\nupdated rationale",
                "error": "",
            },
        ],
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].recommendation_label == "yes"


# --- timestamp / run-id parsing ------------------------------------------


def test_timestamp_iso_from_run_id(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="fence aware label parsing recommendation",
    )
    matches = search_similar("fence aware label parsing recommendation", runs_dir=runs)
    assert len(matches) == 1
    assert matches[0].timestamp == "2026-05-17T12:00:00"


# --- dataclass shape -----------------------------------------------------


def test_similar_match_is_frozen():
    match = SimilarMatch(
        run_id="x",
        similarity=0.5,
        question_excerpt="hi",
        recommendation_label=None,
        timestamp="",
    )
    with pytest.raises(Exception):
        match.similarity = 0.9  # type: ignore[misc]


# --- MCP handler wiring --------------------------------------------------


def test_mcp_handler_returns_matches_shape(tmp_path, monkeypatch):
    """End-to-end through `mcp_server.query_transcripts` with `working_directory`."""
    from llm_council import mcp_server

    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        question="fence aware label parsing recommendation",
        results=[_result("claude", recommendation="yes")],
    )

    # The MCP root guard pins working_directory inside LLM_COUNCIL_MCP_ROOT.
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    result = mcp_server.query_transcripts(
        {
            "query": "fence aware label parsing recommendation",
            "top_k": 5,
            "working_directory": str(tmp_path),
        }
    )
    assert "matches" in result
    assert isinstance(result["matches"], list)
    assert len(result["matches"]) == 1
    entry = result["matches"][0]
    assert entry["run_id"] == "20260517_120000_run-a"
    assert entry["recommendation_label"] == "yes"
    assert 0.0 < entry["similarity"] <= 1.0
    assert entry["timestamp"] == "2026-05-17T12:00:00"
    assert "fence" in entry["question_excerpt"]


def test_mcp_handler_rejects_empty_query(tmp_path, monkeypatch):
    from llm_council import mcp_server

    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="query"):
        mcp_server.query_transcripts(
            {"query": "   ", "working_directory": str(tmp_path)}
        )


def test_mcp_handler_rejects_invalid_top_k(tmp_path, monkeypatch):
    from llm_council import mcp_server

    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="top_k"):
        mcp_server.query_transcripts(
            {
                "query": "anything",
                "top_k": 0,
                "working_directory": str(tmp_path),
            }
        )
    with pytest.raises(ValueError, match="top_k"):
        mcp_server.query_transcripts(
            {
                "query": "anything",
                "top_k": 999,
                "working_directory": str(tmp_path),
            }
        )


def test_mcp_tool_registered_in_list_tools():
    """Smoke: confirm the schema appears alongside the other tool schemas."""
    from llm_council.mcp_server import query_transcripts_schema

    schema = query_transcripts_schema()
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False
