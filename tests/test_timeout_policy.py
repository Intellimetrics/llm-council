"""Timeout policy tests (Changes 1a, 1b, 1c).

Covers the timeout resolver, terse-retry on timeout, and the
prompt-size telemetry that lets users see whether the timeout wall is
the real bottleneck. For the pass-7 anchor that ties these to the
actual failure mode the council surfaced, see
`tests/test_pass7_regression.py`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from llm_council import adapters as adapters_module
from llm_council.adapters import (
    ERROR_KIND_TIMEOUT,
    ParticipantResult,
    TERSE_RETRY_TIMEOUT_SECONDS,
    _build_terse_retry_prompt,
    _resolve_effective_timeout,
    _terse_retry_enabled,
    classify_error,
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


# --- v0.12.0 size-scaled timeout + proportional terse-retry -----------


def test_resolve_effective_timeout_adds_size_bonus_above_threshold():
    """Prompts above 4KB get +5s per KB by default."""
    # 4KB exactly → no bonus.
    assert _resolve_effective_timeout({"timeout": 240}, None, prompt_chars=4096) == 240
    # 5KB → bonus of (1024/1024)*5 = 5s.
    assert _resolve_effective_timeout({"timeout": 240}, None, prompt_chars=5120) == 245
    # 26KB → bonus of ((26*1024 - 4096)/1024)*5 = 110s → 350s total.
    assert _resolve_effective_timeout({"timeout": 240}, None, prompt_chars=26 * 1024) == 350


def test_resolve_effective_timeout_size_bonus_capped():
    """Runaway prompts can't inflate the timeout past +600s."""
    # Massive prompt: bonus would be (10000-4)KB * 5 = ~50000s; capped at 600s.
    # 10MB prompt → effective = 240 + 600 = 840.
    huge = 10 * 1024 * 1024
    assert _resolve_effective_timeout({"timeout": 240}, None, prompt_chars=huge) == 840


def test_resolve_effective_timeout_size_bonus_disabled_by_per_kb_zero():
    """`timeout_per_kb_chars: 0` opts out of size scaling per peer."""
    assert (
        _resolve_effective_timeout(
            {"timeout": 240, "timeout_per_kb_chars": 0},
            None,
            prompt_chars=100_000,
        )
        == 240
    )


def test_resolve_effective_timeout_size_bonus_composes_with_multiplier():
    """consensus 2.0x multiplies the (base + size_bonus) total."""
    # base=240, prompt=26KB → bonus=110 → (240+110)*2 = 700.
    assert (
        _resolve_effective_timeout({"timeout": 240}, 2.0, prompt_chars=26 * 1024)
        == 700
    )


def test_resolve_effective_timeout_per_peer_override():
    """A user pinning a different slope per peer is honored."""
    # 10s/KB instead of 5s/KB → 5KB prompt adds 10s, not 5s.
    cfg = {"timeout": 240, "timeout_per_kb_chars": 10.0}
    assert _resolve_effective_timeout(cfg, None, prompt_chars=5120) == 250


def test_terse_retry_budget_proportional_with_floor_and_ceiling():
    """40% of original, floor 30s, ceiling 120s."""
    from llm_council.adapters import _terse_retry_budget

    # Floor at 30s for tiny originals (1s → 0.4s, raised to 30s).
    assert _terse_retry_budget(1) == 30
    # Ceiling at 120s for huge originals (600s → 240s, capped at 120s).
    assert _terse_retry_budget(600) == 120
    # Mid: 240s * 0.4 = 96s.
    assert _terse_retry_budget(240) == 96
    # Boundary cases.
    assert _terse_retry_budget(75) == 30  # 30s = max(30, round(75*0.4=30))
    assert _terse_retry_budget(80) == 32   # 80*0.4=32, between floor and ceiling
    assert _terse_retry_budget(300) == 120  # 300*0.4=120, hits ceiling exactly


def test_drop_missing_key_participants_drops_openrouter_without_env():
    """Missing api_key_env env var → peer dropped from active list."""
    from llm_council.orchestrator import _drop_missing_key_participants
    import os as _os

    # Ensure the env var is NOT set (defensive — pytest isolates env
    # but be explicit).
    _os.environ.pop("MISSING_FOR_DROP_TEST", None)
    cfg = {
        "openrouter_a": {"type": "openrouter", "api_key_env": "MISSING_FOR_DROP_TEST"},
        "cli_b": {"type": "cli", "command": "true"},
        "ollama_c": {"type": "ollama"},
    }
    active, dropped = _drop_missing_key_participants(
        ["openrouter_a", "cli_b", "ollama_c"], cfg
    )
    assert active == ["cli_b", "ollama_c"]
    assert len(dropped) == 1
    assert dropped[0]["peer"] == "openrouter_a"
    assert dropped[0]["api_key_env"] == "MISSING_FOR_DROP_TEST"


def test_drop_missing_key_participants_keeps_peer_when_env_set(monkeypatch):
    """Env var present → peer stays in the active list."""
    from llm_council.orchestrator import _drop_missing_key_participants

    monkeypatch.setenv("PRESENT_KEY", "x")
    cfg = {
        "openrouter_a": {"type": "openrouter", "api_key_env": "PRESENT_KEY"},
    }
    active, dropped = _drop_missing_key_participants(["openrouter_a"], cfg)
    assert active == ["openrouter_a"]
    assert dropped == []


def test_drop_missing_key_participants_uses_default_env_name():
    """`api_key_env` absent defaults to OPENROUTER_API_KEY for openrouter peers."""
    from llm_council.orchestrator import _drop_missing_key_participants
    import os as _os

    _os.environ.pop("OPENROUTER_API_KEY", None)
    cfg = {"openrouter_a": {"type": "openrouter"}}
    active, dropped = _drop_missing_key_participants(["openrouter_a"], cfg)
    assert active == []
    assert dropped[0]["api_key_env"] == "OPENROUTER_API_KEY"


def test_drop_missing_key_participants_drops_openai_compatible_with_explicit_env(
    monkeypatch,
):
    """v0.12.2 fix: openai_compatible peers WITH an explicit api_key_env
    are pre-dropped when that env var is unset — same as openrouter peers.
    The user told us which env var to check; honor that intent."""
    import os as _os
    from llm_council.orchestrator import _drop_missing_key_participants

    _os.environ.pop("EXPLICIT_OPENAI_COMPAT_KEY", None)
    cfg = {
        "vllm_remote": {
            "type": "openai_compatible",
            "base_url": "https://my-vllm.example.com/v1",
            "api_key_env": "EXPLICIT_OPENAI_COMPAT_KEY",
        }
    }
    active, dropped = _drop_missing_key_participants(["vllm_remote"], cfg)
    assert active == []
    assert len(dropped) == 1
    assert dropped[0]["peer"] == "vllm_remote"
    assert dropped[0]["api_key_env"] == "EXPLICIT_OPENAI_COMPAT_KEY"


def test_drop_missing_key_participants_uses_default_for_openai_compatible_without_explicit_env(
    monkeypatch,
):
    """The OpenAI-compatible adapter requires a key even for loopback and
    defaults an omitted api_key_env to OPENROUTER_API_KEY. Pre-drop must use
    that same contract instead of letting an unrunnable peer reach quorum."""
    import os as _os
    from llm_council.orchestrator import _drop_missing_key_participants

    _os.environ.pop("OPENROUTER_API_KEY", None)
    cfg = {
        "local_vllm": {
            "type": "openai_compatible",
            "base_url": "http://127.0.0.1:8000/v1",
            # No api_key_env: adapter uses OPENROUTER_API_KEY.
        },
    }
    active, dropped = _drop_missing_key_participants(["local_vllm"], cfg)
    assert active == []
    assert dropped == [
        {
            "peer": "local_vllm",
            "family": "local_vllm",
            "api_key_env": "OPENROUTER_API_KEY",
        }
    ]


def test_drop_missing_key_participants_keeps_openai_compatible_with_env_set(
    monkeypatch,
):
    """openai_compatible peer with explicit api_key_env that IS set stays
    active — same as the openrouter happy path, but exercising the new
    branch."""
    from llm_council.orchestrator import _drop_missing_key_participants

    monkeypatch.setenv("PRESENT_OPENAI_COMPAT_KEY", "dummy")
    cfg = {
        "remote_openai": {
            "type": "openai_compatible",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "PRESENT_OPENAI_COMPAT_KEY",
        }
    }
    active, dropped = _drop_missing_key_participants(["remote_openai"], cfg)
    assert active == ["remote_openai"]
    assert dropped == []


def test_execute_council_missing_key_peer_does_not_degrade_run(
    monkeypatch, tmp_path: Path
):
    """A peer with a missing api_key_env should drop out + emit a
    `peer_missing_api_key` event + appear in metadata.missing_key_peers
    BUT NOT count toward the quorum denominator (so the run isn't
    flagged as degraded just because one peer was unconfigured)."""
    import os as _os
    from llm_council import orchestrator as _orchestrator_module
    from llm_council.orchestrator import execute_council

    _os.environ.pop("ABSENT_FOR_DEGRADE_TEST", None)

    async def fake_run_participants(participants, *args, **kwargs):
        # Only the active peers (after key-drop) reach this point.
        return [
            ParticipantResult(name, True, f"RECOMMENDATION: yes - {name}", "", 1.0)
            for name in participants
        ]

    monkeypatch.setattr(
        _orchestrator_module, "run_participants", fake_run_participants
    )

    participant_cfg = {
        "claude": {"type": "cli", "command": "claude", "args": []},
        "codex": {"type": "cli", "command": "codex", "args": []},
        "absent_remote": {
            "type": "openrouter",
            "model": "x/y",
            "api_key_env": "ABSENT_FOR_DEGRADE_TEST",
        },
    }
    _, metadata = asyncio.run(
        execute_council(
            ["claude", "codex", "absent_remote"],
            participant_cfg,
            "q",
            tmp_path,
            {},
        )
    )
    # The missing-key peer was dropped from the run.
    assert "missing_key_peers" in metadata
    assert metadata["missing_key_peers"][0]["peer"] == "absent_remote"
    assert metadata["missing_key_peers"][0]["api_key_env"] == "ABSENT_FOR_DEGRADE_TEST"
    # The run is NOT degraded — the remaining two peers met quorum.
    assert metadata["degraded"] is False
    # Event was emitted before council_start.
    events = metadata["progress_events"]
    missing_events = [e for e in events if e.get("event") == "peer_missing_api_key"]
    assert len(missing_events) == 1
    assert missing_events[0]["peer"] == "absent_remote"


def test_idle_read_helper_raises_on_silence():
    """`_read_stream_with_idle_deadline` must raise TimeoutError when no
    data arrives within the idle window. Wall-clock cap is enforced
    separately by the outer `asyncio.wait_for` in `_run_cli_once`."""
    import asyncio as _asyncio
    from llm_council.adapters import _read_stream_with_idle_deadline

    class _SilentStream:
        async def read(self, n):
            await _asyncio.sleep(10)  # never returns within idle window
            return b""

    with pytest.raises(TimeoutError):
        _asyncio.run(_read_stream_with_idle_deadline(_SilentStream(), idle_timeout=0.05))


def test_idle_read_helper_returns_all_bytes_on_eof():
    """Idle reader returns the accumulated bytes once the stream EOFs."""
    import asyncio as _asyncio
    from llm_council.adapters import _read_stream_with_idle_deadline

    chunks = [b"hello ", b"world", b""]

    class _ChunkedStream:
        def __init__(self):
            self.i = 0

        async def read(self, n):
            chunk = chunks[self.i]
            self.i += 1
            return chunk

    out = _asyncio.run(_read_stream_with_idle_deadline(_ChunkedStream(), idle_timeout=1.0))
    assert out == b"hello world"


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


# --- prompt_chars population on success and recovery paths (pass-8 fix #7) -

def test_cli_success_populates_prompt_chars():
    """Successful CLI runs must record prompt_chars so the new recoveries
    bucket can correlate with prompt size. Before the pass-8 fix only the
    timeout-failure branch set this field, which left `timeout_recoveries`
    uncrossable with `timeout_by_prompt_size`."""
    import asyncio
    from unittest.mock import AsyncMock

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
                    # First call uses 1s and times out; terse-retry budget
                    # is proportional (v0.12.0) but the mocked success
                    # proc returns immediately so the exact budget is
                    # irrelevant. Disable size scaling so the 14K-char
                    # prompt doesn't inflate the 1s base into ~50s,
                    # which would let the first call succeed.
                    "timeout": 1,
                    "timeout_per_kb_chars": 0,
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
                    # (original + directive) isn't skipped. Disable size
                    # scaling (v0.12.0) so the 14K-char prompt doesn't
                    # inflate the 1s base into ~50s.
                    "timeout": 1,
                    "timeout_per_kb_chars": 0,
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
    from llm_council.adapters import classify_error, _terse_retry_budget

    assert classify_error(result.error) == "timeout"
    assert result.error.startswith("Timeout:")
    assert "Terse-retry-on-timeout was attempted" in result.error, (
        f"Expected the failed-retry annotation in the error, got: "
        f"{result.error[:200]}"
    )
    # v0.12.0: the budget is proportional to the ORIGINAL timeout
    # (1s here → floor of 30s), so the suffix names the real budget
    # rather than the legacy 60s constant. The test cfg disabled size
    # scaling so original_timeout stays at 1s for budget math.
    expected_budget = _terse_retry_budget(1)
    assert f"{expected_budget}s budget" in result.error
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
# --- pass-8 finding #6: classify_error/is_timeout_error symmetry ---------
# v0.7.0 extended is_timeout_error to recognize httpx timeout class names
# so terse-retry fires on hosted (openai_compatible / ollama) timeouts.
# classify_error was NOT updated, so unrecovered hosted timeouts landed in
# `downstream_error` (or `unknown`) instead of `timeout`, breaking
# `stats.timeout_by_prompt_size` and `timeout_recoveries`. These tests
# lock in the symmetry: every prefix is_timeout_error accepts MUST map to
# error_kind="timeout".

def test_classify_error_recognizes_cli_timeout_prefixes():
    """CLI subprocess prefixes — the original two."""
    assert classify_error("Timeout: claude did not respond") == ERROR_KIND_TIMEOUT
    assert classify_error("TimeoutError: timed out") == ERROR_KIND_TIMEOUT


def test_classify_error_recognizes_httpx_timeout_prefixes():
    """httpx-backed peers serialize timeouts as f"{class_name}: ...".
    All five must classify as `timeout` so hosted timeouts show up in
    stats.timeout_by_prompt_size instead of being lost to downstream_error."""
    assert classify_error("ReadTimeout: 30s elapsed") == ERROR_KIND_TIMEOUT
    assert classify_error("ConnectTimeout: connect failed") == ERROR_KIND_TIMEOUT
    assert classify_error("WriteTimeout: send failed") == ERROR_KIND_TIMEOUT
    assert classify_error("PoolTimeout: pool exhausted") == ERROR_KIND_TIMEOUT
    assert classify_error("TimeoutException: general timeout") == ERROR_KIND_TIMEOUT


def test_classify_error_and_is_timeout_error_share_prefix_set():
    """Belt-and-braces: every prefix is_timeout_error accepts MUST classify
    as timeout. If someone adds a new httpx exception to _TIMEOUT_PREFIXES
    but forgets to think about classify_error, this fails fast."""
    for prefix in adapters_module._TIMEOUT_PREFIXES:
        sample = f"{prefix} some details"
        assert is_timeout_error(sample), f"is_timeout_error rejected {prefix!r}"
        assert classify_error(sample) == ERROR_KIND_TIMEOUT, (
            f"classify_error did not return 'timeout' for {prefix!r}"
        )


def test_classify_error_keeps_non_timeout_downstream_errors_downstream():
    """The fix must not turn HTTPStatusError / ConnectError etc into timeouts.
    These remain `downstream_error` so the "service blew up vs. timed out"
    distinction stays intact in stats."""
    assert classify_error("HTTPStatusError: 503") == "downstream_error"
    assert classify_error("ConnectError: connection refused") == "downstream_error"
    assert classify_error("RemoteProtocolError: bad framing") == "downstream_error"


# --- End-to-end: hosted httpx ReadTimeout → error_kind="timeout" ---------
# Mocks _request_with_retries to always raise httpx.ReadTimeout. Both the
# initial call AND the terse-retry will fail, so the final result must
# carry an error string whose classify_error() is "timeout". Pre-fix this
# would be "downstream_error" because "ReadTimeout" substring-matched the
# downstream_markers list.

def _always_read_timeout():
    async def _raise(client, method, url, **kwargs):
        raise httpx.ReadTimeout("simulated read timeout", request=None)
    return _raise


def test_openai_compatible_read_timeout_classifies_as_timeout(monkeypatch):
    """A ReadTimeout that survives the terse-retry must produce
    error_kind="timeout" for stats / transcripts / MCP --json."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(
        adapters_module, "_request_with_retries", _always_read_timeout()
    )

    result = asyncio.run(
        adapters_module.run_openai_compatible_participant(
            "router",
            {
                "type": "openai_compatible",
                "model": "z-ai/glm-test",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "timeout": 1,
                # Skip cache hits in this test environment.
                "cache_ttl_seconds": 0,
            },
            "what is 2+2?",
        )
    )
    assert result.ok is False
    assert result.error.startswith("ReadTimeout:"), result.error
    assert classify_error(result.error) == ERROR_KIND_TIMEOUT
    # Sanity: terse-retry was actually attempted (not recovered).
    assert result.recovered_after_timeout is False


def test_ollama_read_timeout_classifies_as_timeout(monkeypatch):
    """Same contract for the ollama adapter path."""
    monkeypatch.setattr(
        adapters_module, "_request_with_retries", _always_read_timeout()
    )

    result = asyncio.run(
        adapters_module.run_ollama_participant(
            "local",
            {
                "type": "ollama",
                "model": "qwen3:q4",
                "base_url": "http://localhost:11434",
                "timeout": 1,
                "cache_ttl_seconds": 0,
            },
            "what is 2+2?",
        )
    )
    assert result.ok is False
    assert result.error.startswith("ReadTimeout:"), result.error
    assert classify_error(result.error) == ERROR_KIND_TIMEOUT
    assert result.recovered_after_timeout is False
