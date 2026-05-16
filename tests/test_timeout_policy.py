"""Timeout policy tests (Changes 1a, 1b, 1c).

Covers the timeout resolver, terse-retry on timeout, and the
prompt-size telemetry that lets users see whether the timeout wall is
the real bottleneck. For the pass-7 anchor that ties these to the
actual failure mode the council surfaced, see
`tests/test_pass7_regression.py`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.adapters import (
    ParticipantResult,
    TERSE_RETRY_TIMEOUT_SECONDS,
    _build_terse_retry_prompt,
    _resolve_effective_timeout,
    _terse_retry_enabled,
    is_timeout_error,
)
from llm_council.stats import _timeout_prompt_size_bucket


# --- Change 1a: _resolve_effective_timeout -------------------------------

def test_resolve_effective_timeout_no_multiplier_returns_base():
    """`mode_multiplier=None` returns int(cfg['timeout'])."""
    assert _resolve_effective_timeout({"timeout": 240}, None) == 240


def test_resolve_effective_timeout_falls_back_to_default():
    """Missing `timeout` key uses the base default."""
    assert _resolve_effective_timeout({}, None) == 240


def test_resolve_effective_timeout_consensus_doubles():
    """consensus mode's 2.0x raises 240s to 480s."""
    assert _resolve_effective_timeout({"timeout": 240}, 2.0) == 480


def test_resolve_effective_timeout_deliberate_one_point_five():
    """deliberate mode's 1.5x rounds 240s to 360s exactly."""
    assert _resolve_effective_timeout({"timeout": 240}, 1.5) == 360


def test_resolve_effective_timeout_respects_user_raised_base():
    """User-raised base composes with the multiplier (the whole point)."""
    assert _resolve_effective_timeout({"timeout": 600}, 2.0) == 1200


def test_resolve_effective_timeout_rounds_up():
    """Non-integer products ceil up so 1.5x of 99s lands at 149, not 148."""
    assert _resolve_effective_timeout({"timeout": 99}, 1.5) == 149
    assert _resolve_effective_timeout({"timeout": 100}, 1.5) == 150


def test_resolve_effective_timeout_treats_one_x_as_base():
    """A 1.0x multiplier returns the base unchanged (no rounding noise)."""
    assert _resolve_effective_timeout({"timeout": 240}, 1.0) == 240


def test_resolve_effective_timeout_ignores_invalid_multiplier():
    """Negative or zero multiplier falls back to base; never reduces."""
    assert _resolve_effective_timeout({"timeout": 240}, 0.0) == 240
    assert _resolve_effective_timeout({"timeout": 240}, -1.0) == 240


def test_resolve_effective_timeout_custom_base_default():
    """openai_compatible / ollama paths pass `base_default=180`."""
    assert _resolve_effective_timeout({}, 2.0, base_default=180) == 360


# --- Change 1b: terse-retry plumbing -------------------------------------

def test_terse_retry_enabled_by_default():
    """A bare config opts in."""
    assert _terse_retry_enabled({}) is True


def test_terse_retry_disabled_explicitly():
    """Per-participant `terse_retry_on_timeout: false` opts out."""
    assert _terse_retry_enabled({"terse_retry_on_timeout": False}) is False


def test_terse_retry_disabled_when_retries_zero():
    """`retries: 0` means "no extra calls" — that includes terse-retry."""
    assert _terse_retry_enabled({"retries": 0}) is False


def test_build_terse_retry_prompt_appends_directive():
    """The prompt is wrapped with a recovery directive marker."""
    out = _build_terse_retry_prompt("original question")
    assert out.startswith("original question")
    assert "Timeout recovery directive" in out
    # The directive must explicitly tell peers to cover REQUIRED sections —
    # otherwise the section-coverage validator would flag the terse retry
    # for skipping sections it told the peer to skip.
    assert "REQUIRED" in out


def test_terse_retry_timeout_constant_is_60():
    """A retry that needs more than 60s isn't a timeout-recovery story."""
    assert TERSE_RETRY_TIMEOUT_SECONDS == 60


def test_is_timeout_error_recognizes_cli_prefixes():
    assert is_timeout_error("Timeout: claude did not respond")
    assert is_timeout_error("TimeoutError: timed out")


def test_is_timeout_error_recognizes_httpx_prefixes():
    """openai_compatible / ollama serialize httpx timeouts by class name.
    Terse-retry must fire on all of them, not just CLI subprocess timeouts."""
    assert is_timeout_error("ReadTimeout: 30s")
    assert is_timeout_error("ConnectTimeout: connect failed")
    assert is_timeout_error("WriteTimeout: send failed")
    assert is_timeout_error("PoolTimeout: pool exhausted")
    assert is_timeout_error("TimeoutException: general timeout")


def test_is_timeout_error_rejects_unrelated_errors():
    assert not is_timeout_error("CliExitNonZero: process exited")
    assert not is_timeout_error("AbdicatedResponse: blocked")
    assert not is_timeout_error("")


# --- Change 1c: timeout-by-prompt-size bucketing ------------------------

def test_timeout_prompt_size_bucket_boundaries():
    """The pass-7 prompt was 14_263 chars — exactly in the medium band."""
    assert _timeout_prompt_size_bucket(0) == "small"
    assert _timeout_prompt_size_bucket(3_999) == "small"
    assert _timeout_prompt_size_bucket(4_000) == "small"
    assert _timeout_prompt_size_bucket(4_001) == "medium"
    assert _timeout_prompt_size_bucket(14_263) == "medium"  # pass-7 anchor
    assert _timeout_prompt_size_bucket(20_000) == "medium"
    assert _timeout_prompt_size_bucket(20_001) == "large"
    assert _timeout_prompt_size_bucket(60_000) == "large"
    assert _timeout_prompt_size_bucket(60_001) == "xlarge"
    assert _timeout_prompt_size_bucket(200_000) == "xlarge"


def test_timeout_prompt_size_bucket_none_is_small():
    """A timeout without a prompt_chars field (legacy transcript) maps to
    `small` so legacy entries don't poison the larger buckets."""
    assert _timeout_prompt_size_bucket(None) == "small"
    assert _timeout_prompt_size_bucket(0) == "small"
    assert _timeout_prompt_size_bucket(-5) == "small"


def test_new_peer_bucket_initializes_timeout_buckets():
    """All four size buckets exist at 0; timeout_recoveries at 0."""
    from llm_council.stats import _new_peer_bucket

    bucket = _new_peer_bucket()
    assert bucket["timeout_by_prompt_size"] == {
        "small": 0, "medium": 0, "large": 0, "xlarge": 0,
    }
    assert bucket["timeout_recoveries"] == 0


def test_stats_buckets_timeout_by_size():
    """A timed-out result with prompt_chars=14_263 lands in the medium bucket."""
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "consensus",
                "results": [
                    {
                        "name": "claude",
                        "ok": False,
                        "error_kind": "timeout",
                        "prompt_chars": 14_263,
                        "output": "",
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    assert by_peer["claude"]["timeout_by_prompt_size"]["medium"] == 1
    assert by_peer["claude"]["timeout_by_prompt_size"]["small"] == 0


def test_stats_counts_timeout_recoveries():
    """A successful run with recovered_after_timeout=True bumps the counter."""
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "consensus",
                "results": [
                    {
                        "name": "claude",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        "recovered_after_timeout": True,
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    assert by_peer["claude"]["timeout_recoveries"] == 1


def test_participant_result_carries_prompt_chars_field():
    """Smoke check that the dataclass field exists and defaults to None."""
    pr = ParticipantResult(name="x", ok=True, output="", error="", elapsed_seconds=0)
    assert pr.prompt_chars is None
    assert pr.recovered_after_timeout is False


# --- prompt_chars population on success and recovery paths (pass-8 fix #7) -

def test_cli_success_populates_prompt_chars():
    """Successful CLI runs must record prompt_chars so the new recoveries
    bucket can correlate with prompt size. Before the pass-8 fix only the
    timeout-failure branch set this field, which left `timeout_recoveries`
    uncrossable with `timeout_by_prompt_size`."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    medium_prompt = "x" * 14_263  # pass-7 anchor; lands in `medium` bucket

    class _FakeProc:
        returncode = 0

        async def communicate(self, _data):
            return (
                b"PART 6 - RECOMMENDATION\nRECOMMENDATION: yes - ok\n",
                b"",
            )

        async def wait(self):
            return 0

    async def _go() -> ParticipantResult:
        with patch(
            "llm_council.adapters.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_FakeProc()),
        ):
            from llm_council.adapters import run_cli_participant

            return await run_cli_participant(
                "claude",
                {
                    "type": "cli",
                    "command": "echo",
                    "args": [],
                    "timeout": 60,
                    # Disable the require_sections gate so the bare label
                    # output above is treated as a valid success.
                    "require_sections": False,
                },
                medium_prompt,
                Path("."),
            )

    result = asyncio.run(_go())
    assert result.ok is True, result.error
    assert result.prompt_chars == len(medium_prompt)


def test_recovered_after_timeout_populates_prompt_chars():
    """The terse-retry success path must carry `prompt_chars=len(original
    prompt)` — not the terse retry's length — so recoveries land in the
    same size bucket as the timeout that triggered them."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    medium_prompt = "x" * 14_263

    class _TimeoutProc:
        returncode = None

        async def communicate(self, _data):
            await asyncio.sleep(10)
            return (b"", b"")

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode or 0

    class _SuccessProc:
        returncode = 0

        async def communicate(self, _data):
            return (
                b"PART 6 - RECOMMENDATION\nRECOMMENDATION: tradeoff - ok\n",
                b"",
            )

        async def wait(self):
            return 0

    calls = {"n": 0}

    async def _factory(*_args, **_kwargs):
        calls["n"] += 1
        return _TimeoutProc() if calls["n"] == 1 else _SuccessProc()

    async def _go() -> ParticipantResult:
        with patch(
            "llm_council.adapters.asyncio.create_subprocess_exec",
            new=_factory,
        ):
            from llm_council.adapters import run_cli_participant

            return await run_cli_participant(
                "claude",
                {
                    "type": "cli",
                    "command": "echo",
                    "args": [],
                    # First call uses 1s and times out; terse-retry uses
                    # TERSE_RETRY_TIMEOUT_SECONDS (60s, fixed) which the
                    # mocked success proc returns immediately.
                    "timeout": 1,
                    "require_sections": False,
                },
                medium_prompt,
                Path("."),
            )

    result = asyncio.run(_go())
    assert result.ok is True, result.error
    assert result.recovered_after_timeout is True
    # Most important: prompt_chars reflects the ORIGINAL prompt's length,
    # not the terse retry's (longer, includes the recovery directive).
    assert result.prompt_chars == len(medium_prompt)


def test_stats_buckets_recoveries_by_prompt_size():
    """A successful run with recovered_after_timeout=True and
    prompt_chars=14_263 lands in the `medium` recoveries bucket and not the
    others."""
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "consensus",
                "results": [
                    {
                        "name": "claude",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        "recovered_after_timeout": True,
                        "prompt_chars": 14_263,
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    assert by_peer["claude"]["timeout_recoveries"] == 1
    assert by_peer["claude"]["timeout_recoveries_by_prompt_size"] == {
        "small": 0, "medium": 1, "large": 0, "xlarge": 0,
    }


def test_new_peer_bucket_initializes_recoveries_by_prompt_size():
    """All four size buckets exist at 0 in the new aggregator field."""
    from llm_council.stats import _new_peer_bucket

    bucket = _new_peer_bucket()
    assert bucket["timeout_recoveries_by_prompt_size"] == {
        "small": 0, "medium": 0, "large": 0, "xlarge": 0,
    }


# --- pass-8 regression: silent-failure mode in terse retry --------------

def test_pass8_terse_retry_failure_is_visible_in_result():
    """End-to-end: when both the original CLI call AND the terse retry
    time out, the returned ParticipantResult must carry
    ``terse_retry_attempted=True`` and an annotated error suffix.

    This is the pass-8 dogfood regression: the v0.7.0 transcript at
    ``20260516_131440_pass-8-*`` showed ``elapsed_seconds=240.341`` and
    ``error_kind=timeout`` for claude, with NO field indicating whether
    the terse retry fired. The original code returned the unmodified
    original result whenever the retry failed, making the failure
    indistinguishable from "retry never fired". This test simulates
    the exact double-timeout subprocess shape and asserts the new
    annotation is in place so transcripts can tell the two failure
    modes apart.
    """
    import asyncio
    from unittest.mock import patch

    pass7_anchor_prompt = "x" * 14_394  # exact pass-8 transcript size

    class _AlwaysTimingOutProc:
        """Subprocess mock that never returns in time."""

        def __init__(self) -> None:
            self.returncode: int | None = None

        async def communicate(self, _data):
            await asyncio.sleep(10)
            return (b"", b"")

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode if self.returncode is not None else 0

    calls = {"n": 0}

    async def _factory(*_args, **_kwargs):
        calls["n"] += 1
        return _AlwaysTimingOutProc()

    async def _go() -> ParticipantResult:
        with patch(
            "llm_council.adapters.asyncio.create_subprocess_exec",
            new=_factory,
        ):
            from llm_council.adapters import run_cli_participant

            return await run_cli_participant(
                "claude",
                {
                    "type": "cli",
                    "command": "echo",
                    "args": [],
                    # Tiny timeouts so the test runs fast; the retry path
                    # itself is what we're verifying, not real wall-clock.
                    # `max_prompt_chars` high so the retry's longer prompt
                    # (original + directive) isn't skipped.
                    "timeout": 1,
                    "max_prompt_chars": 100_000,
                    "require_sections": False,
                },
                pass7_anchor_prompt,
                Path("."),
            )

    result = asyncio.run(_go())

    # The retry must have fired (two subprocess launches).
    assert calls["n"] == 2, (
        f"Expected 2 _run_cli_once launches (original + terse retry), "
        f"got {calls['n']}. The terse-retry path did not fire — this is "
        f"the pass-8 silent-failure mode."
    )
    # Result must still be a failure (retry also timed out).
    assert result.ok is False
    # The error must still classify as a timeout (so quorum math is
    # unchanged and downstream consumers branching on `error_kind` keep
    # working) — but it must now carry the terse-retry-failed suffix.
    from llm_council.adapters import classify_error, TERSE_RETRY_TIMEOUT_SECONDS

    assert classify_error(result.error) == "timeout"
    assert result.error.startswith("Timeout:")
    assert "Terse-retry-on-timeout was attempted" in result.error, (
        f"Expected the failed-retry annotation in the error, got: "
        f"{result.error[:200]}"
    )
    assert f"{TERSE_RETRY_TIMEOUT_SECONDS}s budget" in result.error
    # The NEW field is the load-bearing signal — without it, transcripts
    # cannot distinguish "retry fired and failed" from "retry never fired".
    assert result.terse_retry_attempted is True
    # `recovered_after_timeout` stays False because the retry did not
    # succeed; the two flags are NOT redundant.
    assert result.recovered_after_timeout is False
    # `prompt_chars` stays the original (the multiplier-scaled budget's
    # call), so timeout_by_prompt_size buckets the right call.
    assert result.prompt_chars == len(pass7_anchor_prompt)


def test_pass8_terse_retry_failure_lands_in_stats_attempts_bucket():
    """A failed terse-retry must increment the new `terse_retry_attempts`
    stats counter, distinct from `timeout_recoveries`."""
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "review",
                "results": [
                    {
                        "name": "claude",
                        "ok": False,
                        "error_kind": "timeout",
                        "prompt_chars": 14_394,
                        "terse_retry_attempted": True,
                        "output": "",
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    # Attempt counted.
    assert by_peer["claude"]["terse_retry_attempts"] == 1
    # No recovery counted (the retry failed).
    assert by_peer["claude"]["timeout_recoveries"] == 0
    # Original timeout still buckets correctly.
    assert by_peer["claude"]["timeout_by_prompt_size"]["medium"] == 1


def test_pass8_terse_retry_recovery_also_counts_as_attempt():
    """A successful terse-retry counts as BOTH an attempt and a recovery.
    The two stats fields together let the operator compute
    attempts - recoveries = "retries that fired but failed".
    """
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "review",
                "results": [
                    {
                        "name": "claude",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        "recovered_after_timeout": True,
                        "terse_retry_attempted": True,
                        "prompt_chars": 14_394,
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    assert by_peer["claude"]["terse_retry_attempts"] == 1
    assert by_peer["claude"]["timeout_recoveries"] == 1


def test_stats_recovery_without_prompt_chars_falls_back_to_small():
    """Legacy transcripts (recovered_after_timeout=True but no prompt_chars)
    must still aggregate without crashing; they map to the `small` bucket
    so they don't poison the larger ones."""
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "consensus",
                "results": [
                    {
                        "name": "claude",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        "recovered_after_timeout": True,
                        # no prompt_chars field
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    assert by_peer["claude"]["timeout_recoveries"] == 1
    assert by_peer["claude"]["timeout_recoveries_by_prompt_size"]["small"] == 1
