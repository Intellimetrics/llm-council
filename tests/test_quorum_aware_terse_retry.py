"""Quorum-aware terse-retry skip (field issue #1, 2026-08).

A chronically slow peer that times out AFTER the round already has
``min_quorum`` labeled votes used to burn another 30-120s on a terse
retry that (in every observed field case) re-timed-out — doubling the
wasted wall time on a run that was already viable without it. The
``QuorumTracker`` is written by ``run_participants`` as peers finish and
consulted by the CLI pipeline's terse-retry gate.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from llm_council.adapters import (
    ParticipantResult,
    QuorumTracker,
    TERSE_RETRY_SKIPPED_QUORUM_SUFFIX,
    run_cli_participant,
    run_participants,
)

TIMEOUT_ERROR = (
    "Timeout: `codex` did not respond within 623s (prompt was 8720 chars)."
)


def _labeled(name: str) -> ParticipantResult:
    return ParticipantResult(
        name=name,
        ok=True,
        output="RECOMMENDATION: yes - fine",
        error="",
        elapsed_seconds=1.0,
    )


def test_quorum_tracker_counts_only_labeled_ok_results():
    tracker = QuorumTracker(2)
    assert tracker.met() is False

    tracker.record(_labeled("a"))
    assert tracker.labeled == 1
    assert tracker.met() is False

    # Failure: never counts.
    tracker.record(
        ParticipantResult(
            name="b", ok=False, output="", error="boom", elapsed_seconds=0.0
        )
    )
    # ok but unlabeled: never counts.
    tracker.record(
        ParticipantResult(
            name="c", ok=True, output="no label here", error="", elapsed_seconds=0.0
        )
    )
    assert tracker.labeled == 1

    tracker.record(_labeled("d"))
    assert tracker.met() is True


def _run_pipeline(tracker: QuorumTracker | None) -> tuple[ParticipantResult, int]:
    call_count = {"n": 0}

    async def fake_run_cli_once(
        name, cfg, prompt, cwd, *, start, mode_multiplier=None, mode=None
    ):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                ParticipantResult(
                    name=name,
                    ok=False,
                    output="",
                    error=TIMEOUT_ERROR,
                    elapsed_seconds=623.0,
                    prompt_chars=len(prompt),
                ),
                {"nonzero_exit": False, "stderr": "", "exited": False},
            )
        return (
            ParticipantResult(
                name=name,
                ok=True,
                output="RECOMMENDATION: tradeoff - terse recovered",
                error="",
                elapsed_seconds=15.0,
            ),
            {"nonzero_exit": False, "stderr": "", "exited": True},
        )

    with patch("llm_council.adapters._run_cli_once", side_effect=fake_run_cli_once):
        with patch("llm_council.adapters._cache_lookup", return_value=(None, None)):
            with patch(
                "llm_council.adapters._maybe_persist_cache", return_value=None
            ):
                result = asyncio.run(
                    run_cli_participant(
                        "codex",
                        {"type": "cli", "timeout": 600, "command": "codex"},
                        "question",
                        "/tmp",  # type: ignore[arg-type]
                        quorum_tracker=tracker,
                    )
                )
    return result, call_count["n"]


def test_terse_retry_skipped_when_quorum_already_met():
    tracker = QuorumTracker(2)
    tracker.record(_labeled("claude"))
    tracker.record(_labeled("antigravity"))
    assert tracker.met() is True

    result, calls = _run_pipeline(tracker)
    assert calls == 1, "terse retry must NOT fire once quorum is met"
    assert result.ok is False
    assert result.error.startswith("Timeout:")  # classify_error unchanged
    assert result.error.endswith(TERSE_RETRY_SKIPPED_QUORUM_SUFFIX)
    assert result.terse_retry_attempted is False


def test_terse_retry_still_fires_below_quorum():
    tracker = QuorumTracker(2)
    tracker.record(_labeled("claude"))
    assert tracker.met() is False

    result, calls = _run_pipeline(tracker)
    assert calls == 2, "below quorum the retry must fire as before"
    assert result.ok is True
    assert result.recovered_after_timeout is True


def test_terse_retry_unchanged_without_tracker():
    result, calls = _run_pipeline(None)
    assert calls == 2
    assert result.ok is True


def test_run_participants_records_into_tracker_and_emits_skip_event():
    tracker = QuorumTracker(1)
    events: list[dict] = []

    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        if name == "slow":
            return ParticipantResult(
                name=name,
                ok=False,
                output="",
                error=TIMEOUT_ERROR + TERSE_RETRY_SKIPPED_QUORUM_SUFFIX,
                elapsed_seconds=623.0,
            )
        return _labeled(name)

    with patch(
        "llm_council.adapters.run_participant", side_effect=fake_run_participant
    ):
        results = asyncio.run(
            run_participants(
                ["fast", "slow"],
                {
                    "fast": {"type": "cli", "timeout": 10, "command": "true"},
                    "slow": {"type": "cli", "timeout": 10, "command": "true"},
                },
                "question",
                "/tmp",  # type: ignore[arg-type]
                progress=events.append,
                quorum_tracker=tracker,
            )
        )

    assert len(results) == 2
    assert tracker.labeled == 1  # only the labeled ok result counted
    skip_events = [e for e in events if e.get("event") == "terse_retry_skipped"]
    assert len(skip_events) == 1
    assert skip_events[0]["participant"] == "slow"
    assert skip_events[0]["reason"] == "quorum_met"


def test_skip_flag_validates_as_boolean():
    import pytest

    from llm_council.config import validate_config
    from llm_council.defaults import DEFAULT_CONFIG
    import copy

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["defaults"]["skip_terse_retry_when_quorum_met"] = "false"
    with pytest.raises(ValueError, match="skip_terse_retry_when_quorum_met"):
        validate_config(config)
