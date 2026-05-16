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
