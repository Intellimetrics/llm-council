"""Outcome record persistence (Phase C — v0.8 plan).

Covers `OutcomeRecord` serialization, the `.llm-council/outcomes/` sidecar
layout, and the partial-prefix `resolve_run_id` lookup. CLI behavior is
exercised separately via the smoke test in the cmd_run path; this file
focuses on the module's API surface so future refactors have a fast
regression net.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


from llm_council.outcomes import (
    OutcomeRecord,
    iter_outcomes,
    outcomes_dir,
    read_outcome,
    resolve_run_id,
    write_outcome,
)


def _make_transcript(runs_dir: Path, run_id: str) -> Path:
    """Write a minimal transcript JSON the resolver can find."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": "test",
        "mode": "review",
        "current": "claude",
        "participants": ["claude", "codex"],
        "prompt": "test prompt",
        "results": [],
    }
    path = runs_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- Round-trip persistence ----------------------------------------------


def test_outcome_record_round_trip(tmp_path):
    record = OutcomeRecord(
        run_id="20260517_120000_test-prompt",
        decision="shipped",
        bug_found=False,
        winning_peer="claude",
        note="merged as proposed",
    )
    path = write_outcome(tmp_path, record)
    assert path.exists()
    loaded = read_outcome(tmp_path, record.run_id)
    assert loaded is not None
    assert loaded.run_id == record.run_id
    assert loaded.decision == "shipped"
    assert loaded.bug_found is False
    assert loaded.winning_peer == "claude"
    assert loaded.note == "merged as proposed"
    # marked_at survives the round trip (down to second precision is fine
    # — ISO-8601 preserves microseconds).
    assert loaded.marked_at == record.marked_at


def test_write_outcome_creates_sidecar_path(tmp_path):
    record = OutcomeRecord(
        run_id="20260517_120000_test", decision="rejected"
    )
    path = write_outcome(tmp_path, record)
    expected = tmp_path / ".llm-council" / "outcomes" / "20260517_120000_test.json"
    assert path == expected
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "20260517_120000_test"
    assert payload["decision"] == "rejected"
    assert payload["schema_version"] == 1


def test_write_outcome_overwrites_existing(tmp_path):
    record = OutcomeRecord(
        run_id="20260517_120000_test",
        decision="shipped",
        bug_found=False,
        note="first try",
    )
    write_outcome(tmp_path, record)

    overwrite = OutcomeRecord(
        run_id="20260517_120000_test",
        decision="reverted",
        bug_found=True,
        note="actually we had to roll it back",
    )
    write_outcome(tmp_path, overwrite)

    loaded = read_outcome(tmp_path, "20260517_120000_test")
    assert loaded is not None
    assert loaded.decision == "reverted"
    assert loaded.bug_found is True
    assert loaded.note == "actually we had to roll it back"


def test_read_outcome_missing_returns_none(tmp_path):
    assert read_outcome(tmp_path, "20260101_000000_nothing") is None


def test_read_outcome_malformed_json_returns_none(tmp_path):
    target = outcomes_dir(tmp_path) / "20260517_120000_bad.json"
    target.write_text("{not valid json", encoding="utf-8")
    assert read_outcome(tmp_path, "20260517_120000_bad") is None


def test_read_outcome_invalid_decision_returns_none(tmp_path):
    target = outcomes_dir(tmp_path) / "20260517_120000_bad.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "20260517_120000_bad",
                "decision": "nope",
                "marked_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    assert read_outcome(tmp_path, "20260517_120000_bad") is None


def test_read_outcome_missing_required_fields_returns_none(tmp_path):
    target = outcomes_dir(tmp_path) / "20260517_120000_bad.json"
    target.write_text(
        json.dumps({"schema_version": 1, "decision": "shipped"}),
        encoding="utf-8",
    )
    assert read_outcome(tmp_path, "20260517_120000_bad") is None


# --- iter_outcomes ordering -----------------------------------------------


def test_iter_outcomes_returns_marked_at_desc(tmp_path):
    base = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    older = OutcomeRecord(
        run_id="20260517_120000_older",
        decision="shipped",
        marked_at=base,
    )
    newer = OutcomeRecord(
        run_id="20260517_130000_newer",
        decision="reverted",
        marked_at=base + timedelta(hours=1),
    )
    middle = OutcomeRecord(
        run_id="20260517_123000_middle",
        decision="unknown",
        marked_at=base + timedelta(minutes=30),
    )
    write_outcome(tmp_path, older)
    write_outcome(tmp_path, newer)
    write_outcome(tmp_path, middle)

    ordered = list(iter_outcomes(tmp_path))
    assert [r.run_id for r in ordered] == [
        "20260517_130000_newer",
        "20260517_123000_middle",
        "20260517_120000_older",
    ]


def test_iter_outcomes_empty_dir_yields_nothing(tmp_path):
    assert list(iter_outcomes(tmp_path)) == []


def test_iter_outcomes_skips_malformed_entries(tmp_path):
    good = OutcomeRecord(run_id="20260517_120000_ok", decision="shipped")
    write_outcome(tmp_path, good)
    bad = outcomes_dir(tmp_path) / "20260517_120000_bad.json"
    bad.write_text("not json at all", encoding="utf-8")
    ordered = list(iter_outcomes(tmp_path))
    assert [r.run_id for r in ordered] == ["20260517_120000_ok"]


# --- resolve_run_id ------------------------------------------------------


def test_resolve_run_id_unique_prefix(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _make_transcript(runs, "20260517_120000_unique-question")
    assert (
        resolve_run_id(tmp_path, "20260517_120000")
        == "20260517_120000_unique-question"
    )


def test_resolve_run_id_full_filename(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _make_transcript(runs, "20260517_120000_unique-question")
    assert (
        resolve_run_id(tmp_path, "20260517_120000_unique-question.json")
        == "20260517_120000_unique-question"
    )


def test_resolve_run_id_ambiguous_prefix_returns_none(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _make_transcript(runs, "20260517_120000_first-question")
    _make_transcript(runs, "20260517_120000_second-question")
    # Both share the timestamp prefix; resolver must refuse rather than
    # silently pick one.
    assert resolve_run_id(tmp_path, "20260517_120000") is None


def test_resolve_run_id_no_match_returns_none(tmp_path):
    runs = tmp_path / ".llm-council" / "runs"
    _make_transcript(runs, "20260517_120000_q")
    assert resolve_run_id(tmp_path, "19990101_000000") is None


def test_resolve_run_id_bad_input_returns_none(tmp_path):
    assert resolve_run_id(tmp_path, "") is None
    assert resolve_run_id(tmp_path, "not-a-timestamp") is None


def test_resolve_run_id_no_transcripts_dir_returns_none(tmp_path):
    # No `.llm-council/runs/` exists yet — common before any council
    # run has happened in a fresh checkout.
    assert resolve_run_id(tmp_path, "20260517_120000") is None


def test_resolve_run_id_exact_stem_when_ambiguous(tmp_path):
    """If a prefix matches multiple files BUT one is an exact stem, use it.

    Mirrors find_transcript_by_id's behavior so a partial prefix doesn't
    accidentally beat a complete one.
    """
    runs = tmp_path / ".llm-council" / "runs"
    # The exact stem (no suffix slug) is the one we want.
    _make_transcript(runs, "20260517_120000")
    _make_transcript(runs, "20260517_120000_with-suffix")
    assert resolve_run_id(tmp_path, "20260517_120000") == "20260517_120000"


def test_resolve_run_id_honors_custom_transcripts_dir(tmp_path):
    """With a relocated transcripts_dir the default cwd/.llm-council/runs is
    empty, so prefix resolution must use the passed-in dir or it silently
    fails (the outcome-marking bug)."""
    custom = tmp_path / "somewhere" / "else" / "runs"
    _make_transcript(custom, "20260517_120000_relocated")
    # Without the dir, the default location has nothing -> None.
    assert resolve_run_id(tmp_path, "20260517_120000") is None
    # With it, the prefix resolves.
    assert (
        resolve_run_id(tmp_path, "20260517_120000", transcripts_dir=custom)
        == "20260517_120000_relocated"
    )


# --- outcomes_dir creates the directory ----------------------------------


def test_outcomes_dir_creates_if_missing(tmp_path):
    target = outcomes_dir(tmp_path)
    assert target == tmp_path / ".llm-council" / "outcomes"
    assert target.is_dir()
