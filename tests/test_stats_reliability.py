"""Per-peer reliability counters (Phase C — v0.8 plan).

Covers the new `aggregate_reliability` view in `llm_council.stats`. All
fixtures are written directly to the transcripts dir; no adapter or
orchestrator calls are made.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_council.outcomes import OutcomeRecord, write_outcome
from llm_council.stats import aggregate_reliability, format_reliability_text


def _write_transcript(
    runs_dir: Path,
    run_id: str,
    *,
    participants: list[str],
    results: list[dict],
    mode: str = "review",
) -> Path:
    """Write a minimal transcript JSON the reliability aggregator can consume.

    `results` items follow the same shape as `result_to_dict` output —
    `name`, `ok`, `output`, optional `evidence` list-of-dicts.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": "test",
        "mode": mode,
        "current": participants[0] if participants else None,
        "participants": participants,
        "prompt": "test prompt",
        "results": results,
    }
    path = runs_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _result(
    name: str,
    *,
    ok: bool = True,
    recommendation: str = "yes",
    evidence: list[dict] | None = None,
) -> dict:
    output = f"RECOMMENDATION: {recommendation}\nrationale goes here"
    return {
        "name": name,
        "ok": ok,
        "model": "test-model",
        "elapsed_seconds": 1.0,
        "command": None,
        "output": output if ok else "",
        "error": "" if ok else "boom",
        "evidence": evidence or [],
    }


# --- useful_count + false_blocker_count ----------------------------------


def test_useful_and_false_blocker_counts(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    # Run A: claude says yes, codex says no — shipped clean.
    _write_transcript(
        runs,
        "20260517_120000_run-a",
        participants=["claude", "codex"],
        results=[
            _result("claude", recommendation="yes"),
            _result("codex", recommendation="no"),
        ],
    )
    write_outcome(
        tmp_path,
        OutcomeRecord(
            run_id="20260517_120000_run-a",
            decision="shipped",
            bug_found=False,
            winning_peer="claude",
        ),
    )

    out = aggregate_reliability(tmp_path)
    by_name = {row["name"]: row for row in out["peers"]}
    assert by_name["claude"]["outcomes_marked"] == 1
    assert by_name["claude"]["useful_count"] == 1
    # claude voted yes → not a false blocker.
    assert by_name["claude"]["false_blocker_count"] == 0
    # codex voted no but the change shipped clean → false blocker
    # (and NOT useful — useful/false_blocker are mutually exclusive).
    assert by_name["codex"]["outcomes_marked"] == 1
    assert by_name["codex"]["useful_count"] == 0
    assert by_name["codex"]["false_blocker_count"] == 1


def test_useful_count_credits_tradeoff_vote(tmp_path):
    """A `tradeoff` vote on shipped+no-bug is aligned with the outcome
    (the peer flagged risk but did not try to block); counts as useful."""
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-tradeoff",
        participants=["claude"],
        results=[_result("claude", recommendation="tradeoff")],
    )
    write_outcome(
        tmp_path,
        OutcomeRecord(
            run_id="20260517_120000_run-tradeoff",
            decision="shipped",
            bug_found=False,
        ),
    )
    out = aggregate_reliability(tmp_path)
    by_name = {row["name"]: row for row in out["peers"]}
    assert by_name["claude"]["useful_count"] == 1
    assert by_name["claude"]["false_blocker_count"] == 0


def test_useful_and_false_blocker_skip_unlabeled_results(tmp_path):
    """A peer with no usable RECOMMENDATION label (e.g. abdicated or
    ok=False) didn't actually vote — neither counter increments even
    though outcomes_marked goes up."""
    runs = tmp_path / ".llm-council" / "runs"
    # Build a result with ok=False (no output) — recommendation_label
    # returns None.
    _write_transcript(
        runs,
        "20260517_120000_run-abdicated",
        participants=["claude"],
        results=[_result("claude", ok=False)],
    )
    write_outcome(
        tmp_path,
        OutcomeRecord(
            run_id="20260517_120000_run-abdicated",
            decision="shipped",
            bug_found=False,
        ),
    )
    out = aggregate_reliability(tmp_path)
    by_name = {row["name"]: row for row in out["peers"]}
    assert by_name["claude"]["outcomes_marked"] == 1
    assert by_name["claude"]["useful_count"] == 0
    assert by_name["claude"]["false_blocker_count"] == 0


# --- unique_blocker_catch_count -----------------------------------------


def test_unique_blocker_catch_when_winning_peer_credits_bug(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-b",
        participants=["claude", "codex", "gemini"],
        results=[
            _result("claude", recommendation="no"),
            _result("codex", recommendation="yes"),
            _result("gemini", recommendation="yes"),
        ],
    )
    write_outcome(
        tmp_path,
        OutcomeRecord(
            run_id="20260517_120000_run-b",
            decision="reverted",
            bug_found=True,
            winning_peer="claude",
        ),
    )

    out = aggregate_reliability(tmp_path)
    by_name = {row["name"]: row for row in out["peers"]}
    # claude was credited as the catcher of a real bug.
    assert by_name["claude"]["unique_blocker_catch_count"] == 1
    # Reverted+bug counts as outcome_marked for everyone but useful for no-one.
    assert by_name["claude"]["useful_count"] == 0
    assert by_name["codex"]["unique_blocker_catch_count"] == 0
    assert by_name["codex"]["useful_count"] == 0


# --- verified_citation_rate ---------------------------------------------


def test_verified_citation_rate_mechanical_signal(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    # claude has 3 VERIFIED entries; 2 verified, 1 failed.
    _write_transcript(
        runs,
        "20260517_120000_run-c",
        participants=["claude", "gemini"],
        results=[
            _result(
                "claude",
                evidence=[
                    {"text": "x", "tag": "verified", "verified": True},
                    {"text": "y", "tag": "verified", "verified": True},
                    {"text": "z", "tag": "verified", "verified": False},
                    {"text": "background", "tag": "published"},  # ignored
                ],
            ),
            # gemini has no VERIFIED entries at all.
            _result(
                "gemini",
                evidence=[{"text": "bg", "tag": "published"}],
            ),
        ],
    )
    # No outcome marked — verified_citation_rate is a mechanical signal
    # that works without any operator labeling.

    out = aggregate_reliability(tmp_path)
    by_name = {row["name"]: row for row in out["peers"]}
    assert "claude" in by_name  # has VERIFIED entries → included
    assert by_name["claude"]["verified_citation_rate"] == pytest.approx(2 / 3)
    assert by_name["claude"]["verified_total"] == 3
    # gemini: zero VERIFIED entries → not in the table at all (no
    # signal vs outcome counters).
    assert "gemini" not in by_name


def test_verified_citation_rate_none_when_no_verified_entries(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-d",
        participants=["claude"],
        results=[
            _result(
                "claude",
                evidence=[{"text": "x", "tag": "published"}],
            ),
        ],
    )
    write_outcome(
        tmp_path,
        OutcomeRecord(
            run_id="20260517_120000_run-d",
            decision="shipped",
            bug_found=False,
            winning_peer="claude",
        ),
    )
    out = aggregate_reliability(tmp_path)
    by_name = {row["name"]: row for row in out["peers"]}
    # outcomes_marked > 0 keeps the row visible; verified_rate is None.
    assert by_name["claude"]["verified_citation_rate"] is None
    assert by_name["claude"]["verified_total"] == 0


# --- peer filter ---------------------------------------------------------


def test_aggregate_reliability_peer_filter(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-e",
        participants=["claude", "codex"],
        results=[
            _result("claude", recommendation="yes"),
            _result("codex", recommendation="yes"),
        ],
    )
    write_outcome(
        tmp_path,
        OutcomeRecord(
            run_id="20260517_120000_run-e",
            decision="shipped",
            bug_found=False,
        ),
    )
    out = aggregate_reliability(tmp_path, peer="claude")
    assert [row["name"] for row in out["peers"]] == ["claude"]


# --- multiple outcomes accumulate ---------------------------------------


def test_counts_accumulate_across_multiple_outcomes(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    for i, decision in enumerate(("shipped", "shipped", "reverted")):
        rid = f"2026051{i}_120000_run-{i}"
        bug = i == 2  # third one had a bug
        _write_transcript(
            runs,
            rid,
            participants=["claude"],
            results=[_result("claude", recommendation="yes")],
        )
        write_outcome(
            tmp_path,
            OutcomeRecord(
                run_id=rid,
                decision=decision,  # type: ignore[arg-type]
                bug_found=bug,
                winning_peer="claude" if bug else None,
            ),
        )
    out = aggregate_reliability(tmp_path)
    claude = next(row for row in out["peers"] if row["name"] == "claude")
    assert claude["outcomes_marked"] == 3
    assert claude["useful_count"] == 2  # two shipped+no-bug
    assert claude["unique_blocker_catch_count"] == 1  # one reverted+bug, winning=claude


# --- no outcomes -> stable empty shape ----------------------------------


def test_empty_state_returns_stable_shape(tmp_path):
    out = aggregate_reliability(tmp_path)
    assert out == {
        "total_outcomes": 0,
        "transcripts_considered": 0,
        "filters": {"peer": None},
        "peers": [],
    }


# --- format_reliability_text -------------------------------------------


def test_format_reliability_text_empty_peers(tmp_path):
    out = aggregate_reliability(tmp_path)
    text = format_reliability_text(out)
    assert "outcomes: 0" in text
    assert "no peer reliability signal" in text


def test_format_reliability_text_with_rows(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _write_transcript(
        runs,
        "20260517_120000_run-f",
        participants=["claude"],
        results=[
            _result(
                "claude",
                evidence=[
                    {"text": "a", "tag": "verified", "verified": True},
                    {"text": "b", "tag": "verified", "verified": False},
                ],
            )
        ],
    )
    write_outcome(
        tmp_path,
        OutcomeRecord(
            run_id="20260517_120000_run-f",
            decision="shipped",
            bug_found=False,
        ),
    )
    out = aggregate_reliability(tmp_path)
    text = format_reliability_text(out)
    assert "claude" in text
    assert "50%" in text  # 1 verified / 2 total
