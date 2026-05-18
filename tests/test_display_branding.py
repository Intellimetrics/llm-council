"""Tests for the brand-token/peer-accent helpers and the MCP progress
callback shape (display.format_progress_message + mcp_server progress
wiring). See the v0.10 plan §7 test list.
"""
from __future__ import annotations

import asyncio
import io
import os
from unittest.mock import MagicMock

import pytest

from llm_council import display
from llm_council.cli import _make_progress_printer


# ----- display.peer_accent -----------------------------------------------


def test_peer_accent_deterministic_by_roster_index():
    roster = ["claude", "codex", "gemini"]
    assert display.peer_accent("claude", roster) == display.PEER_ACCENT_PALETTE[0]
    assert display.peer_accent("codex", roster) == display.PEER_ACCENT_PALETTE[1]
    assert display.peer_accent("gemini", roster) == display.PEER_ACCENT_PALETTE[2]


def test_peer_accent_returns_none_when_not_in_roster():
    assert display.peer_accent("openrouter-gpt5", ["claude", "codex"]) is None


def test_peer_accent_wraps_when_roster_exceeds_palette():
    palette_size = len(display.PEER_ACCENT_PALETTE)
    long_roster = [f"peer{i}" for i in range(palette_size + 2)]
    # Wrap-around: palette[0] is reused at position palette_size.
    assert display.peer_accent(long_roster[palette_size], long_roster) == display.PEER_ACCENT_PALETTE[0]


# ----- display.wants_quiet ----------------------------------------------


def test_wants_quiet_off_by_default(monkeypatch):
    monkeypatch.delenv("LLM_COUNCIL_QUIET", raising=False)
    assert display.wants_quiet() is False


def test_wants_quiet_truthy_values_enable(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("LLM_COUNCIL_QUIET", value)
        assert display.wants_quiet() is True, f"failed for value={value!r}"


def test_wants_quiet_falsy_values_keep_off(monkeypatch):
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("LLM_COUNCIL_QUIET", value)
        assert display.wants_quiet() is False, f"failed for value={value!r}"


# ----- display.format_progress_message ----------------------------------


def test_progress_message_always_prefixed_with_brand_token():
    cases = [
        {"event": "council_start", "participants": ["a", "b"]},
        {"event": "participant_finish", "participant": "claude", "status": "ok", "elapsed_seconds": 12.3},
        {"event": "participant_slow", "participant": "codex", "elapsed_seconds": 90, "timeout_seconds": 180},
        {"event": "deliberation_round_start", "round": 2},
        {"event": "cross_rank_start", "peer_count": 4},
        {"event": "synthesis_start", "chair": "claude"},
        {"event": "council_finish", "ok": 4, "total": 4},
    ]
    for event in cases:
        msg = display.format_progress_message(event)
        assert msg is not None, f"unexpected None for {event}"
        assert msg.startswith(f"{display.BRAND_TOKEN}{display.BRAND_SEP}"), (
            f"missing prefix in {msg!r}"
        )


def test_progress_message_suppresses_participant_start():
    # Plan §3: participant_start is suppressed (noise multiplier when
    # N peers fire concurrently; participant_finish is the visible signal).
    assert display.format_progress_message(
        {"event": "participant_start", "participant": "claude"}
    ) is None


def test_progress_message_suppresses_noise_events():
    for kind in (
        "images_skipped",
        "truncated_for_deliberation",
        "deliberation_skip_participants",
        "convergence",
        "context_files_chunked",
    ):
        assert display.format_progress_message({"event": kind}) is None, kind


def test_progress_advancing_events_set_matches_doc():
    # `preflight_failed` is the v0.10.1 addition — without it, the
    # progress bar stalled when a local peer's preflight ping failed
    # because those peers never reach `participant_finish`.
    assert display.PROGRESS_ADVANCING_EVENTS == frozenset(
        {
            "participant_finish",
            "preflight_failed",
            "cross_rank_complete",
            "synthesis_finish",
        }
    )


# ----- mcp_server._build_mcp_progress_callback --------------------------


def test_progress_callback_no_token_returns_none():
    from llm_council.mcp_server import _build_mcp_progress_callback

    session = MagicMock()
    cb = _build_mcp_progress_callback(session, None, planned_total=5.0)
    assert cb is None


def test_progress_callback_no_session_returns_none():
    from llm_council.mcp_server import _build_mcp_progress_callback

    cb = _build_mcp_progress_callback(None, "token-1", planned_total=5.0)
    assert cb is None


def test_progress_callback_quiet_env_returns_none(monkeypatch):
    from llm_council.mcp_server import _build_mcp_progress_callback

    monkeypatch.setenv("LLM_COUNCIL_QUIET", "1")
    cb = _build_mcp_progress_callback(MagicMock(), "token-1", planned_total=5.0)
    assert cb is None


def _drain_loop_tasks():
    """Spin the event loop briefly so create_task'd notifications fire."""
    async def _yield():
        # Two awaits to flush scheduled tasks even when wrapped in gather.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    asyncio.run(_yield())


def test_progress_callback_emits_for_interesting_events(monkeypatch):
    from llm_council.mcp_server import _build_mcp_progress_callback

    sent: list[dict] = []

    class FakeSession:
        async def send_progress_notification(
            self, progress_token, progress, total=None, message=None, related_request_id=None
        ):
            sent.append(
                {"token": progress_token, "progress": progress, "total": total, "message": message}
            )

    monkeypatch.delenv("LLM_COUNCIL_QUIET", raising=False)

    async def _exercise():
        cb = _build_mcp_progress_callback(FakeSession(), "tok", planned_total=5.0)
        assert cb is not None
        # 2 peers x 1 round + 1 council_finish.
        cb({"event": "council_start", "participants": ["claude", "codex"]})
        cb({"event": "participant_start", "participant": "claude"})  # suppressed
        cb({"event": "participant_finish", "participant": "claude", "status": "ok", "elapsed_seconds": 10.0})
        cb({"event": "participant_finish", "participant": "codex", "status": "ok", "elapsed_seconds": 12.0})
        cb({"event": "council_finish", "ok": 2, "total": 2})
        # Yield twice to flush the create_task scheduled notifications.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_exercise())

    # council_start + 2 x participant_finish + council_finish = 4. The
    # participant_start in between is suppressed.
    assert len(sent) == 4
    messages = [entry["message"] for entry in sent]
    assert all(m.startswith(f"{display.BRAND_TOKEN}{display.BRAND_SEP}") for m in messages)
    # Progress is non-decreasing.
    progresses = [entry["progress"] for entry in sent]
    assert progresses == sorted(progresses)
    # Final progress clamps to planned_total.
    assert progresses[-1] == 5.0
    # Token forwarded unchanged.
    assert all(entry["token"] == "tok" for entry in sent)
    assert all(entry["total"] == 5.0 for entry in sent)


def test_progress_callback_swallows_transport_errors():
    """A failing send_progress_notification must not break the council."""
    from llm_council.mcp_server import _build_mcp_progress_callback

    class FailingSession:
        async def send_progress_notification(self, *a, **k):
            raise RuntimeError("simulated transport failure")

    async def _exercise():
        cb = _build_mcp_progress_callback(FailingSession(), "tok", planned_total=3.0)
        # Must not raise even though every send fails.
        cb({"event": "council_start", "participants": ["a"]})
        cb({"event": "participant_finish", "participant": "a", "status": "ok", "elapsed_seconds": 1.0})
        cb({"event": "council_finish", "ok": 1, "total": 1})
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    # If the except: pass in _send leaks, this raises.
    asyncio.run(_exercise())


# ----- CLI _make_progress_printer with LLM_COUNCIL_QUIET ----------------


def test_cli_quiet_strips_color_but_preserves_gutter_layout(monkeypatch, capsys):
    """Plan §7 invariant: layout still prints under QUIET; ANSI gone."""
    monkeypatch.setenv("LLM_COUNCIL_QUIET", "1")
    # Force wants_color True so the only thing turning color off is QUIET.
    monkeypatch.setattr(display, "wants_color", lambda *a, **k: True)

    printer = _make_progress_printer(["claude", "codex"])
    printer(
        {
            "event": "participant_finish",
            "participant": "claude",
            "status": "ok",
            "elapsed_seconds": 12.3,
            "round": 1,
        }
    )
    out = capsys.readouterr().out
    # ANSI bytes stripped.
    assert "\x1b[" not in out
    # Right-aligned gutter still present (6 spaces + "claude" = 12).
    assert "      claude " in out


def test_cli_per_peer_accent_uses_palette_when_color_enabled(monkeypatch, capsys):
    monkeypatch.delenv("LLM_COUNCIL_QUIET", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(display, "wants_color", lambda *a, **k: True)

    printer = _make_progress_printer(["claude", "codex", "gemini"])
    printer(
        {
            "event": "participant_finish",
            "participant": "codex",
            "status": "ok",
            "elapsed_seconds": 5.0,
            "round": 1,
        }
    )
    out = capsys.readouterr().out
    # codex is roster index 1 → magenta.
    assert display.PEER_ACCENT_PALETTE[1] in out
    # And NOT the default bold-cyan gutter color for this peer's gutter
    # token. (Default gutter could legitimately appear in other tokens
    # later, but for a single participant_finish line we should only see
    # the peer-accent color wrapping the right-aligned token.)
    assert display.ANSI_GUTTER not in out


def test_cli_default_peer_uses_gutter_color_when_not_in_roster(monkeypatch, capsys):
    monkeypatch.delenv("LLM_COUNCIL_QUIET", raising=False)
    monkeypatch.setattr(display, "wants_color", lambda *a, **k: True)

    printer = _make_progress_printer(["claude", "codex"])
    printer(
        {
            "event": "participant_finish",
            "participant": "stranger",
            "status": "ok",
            "elapsed_seconds": 1.0,
            "round": 1,
        }
    )
    out = capsys.readouterr().out
    # No accent color matches → falls back to default ANSI_GUTTER.
    assert display.ANSI_GUTTER in out


# ----- v0.10.1 fixes (council-surfaced) ---------------------------------


def test_preflight_failed_advances_progress_counter(monkeypatch):
    """v0.10.1 fix: preflight_failed peers are stripped from run_targets
    and never emit participant_finish. Without listing them in
    PROGRESS_ADVANCING_EVENTS the bar stalls until council_finish clamps.
    """
    from llm_council.mcp_server import _build_mcp_progress_callback

    sent: list[dict] = []

    class FakeSession:
        async def send_progress_notification(self, progress_token, progress, total=None, message=None, related_request_id=None):
            sent.append({"progress": progress, "message": message})

    monkeypatch.delenv("LLM_COUNCIL_QUIET", raising=False)

    async def _exercise():
        # 3 peers; one preflight-fails. planned_total = 3*1 + 1 = 4.
        cb = _build_mcp_progress_callback(FakeSession(), "tok", planned_total=4.0)
        assert cb is not None
        cb({"event": "council_start", "participants": ["claude", "codex", "ollama"]})
        cb({"event": "preflight_failed", "participant": "ollama", "round": 1, "error": "unreachable"})
        cb({"event": "participant_finish", "participant": "claude", "status": "ok", "elapsed_seconds": 5.0})
        cb({"event": "participant_finish", "participant": "codex", "status": "ok", "elapsed_seconds": 6.0})
        cb({"event": "council_finish", "ok": 2, "total": 3})
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_exercise())

    progresses = [entry["progress"] for entry in sent]
    # council_start (0) → preflight (1) → claude (2) → codex (3) → finish (4 clamp).
    # Without the fix, preflight wouldn't advance → 0, 0, 1, 2, 4 — visible stall.
    assert progresses == [0.0, 1.0, 2.0, 3.0, 4.0], progresses
    # Preflight-failed message is also visible.
    assert any("preflight failed" in m for m in (entry["message"] for entry in sent))


def test_progress_callback_keeps_strong_refs_under_burst(monkeypatch):
    """v0.10.1 fix: asyncio.create_task uses weak refs in the event loop.
    Without strong refs, an in-flight notification can be GC'd mid-await
    and silently disappear. Fire 50 events, force a GC pass mid-burst,
    assert every notification was delivered.
    """
    import gc
    from llm_council.mcp_server import _build_mcp_progress_callback

    delivered: list[float] = []

    class SlowSession:
        async def send_progress_notification(self, progress_token, progress, total=None, message=None, related_request_id=None):
            # Yield once so the task is mid-await when GC runs. Without
            # the strong-ref fix, the task can be collected at this point.
            await asyncio.sleep(0)
            delivered.append(progress)

    monkeypatch.delenv("LLM_COUNCIL_QUIET", raising=False)

    async def _exercise():
        cb = _build_mcp_progress_callback(SlowSession(), "tok", planned_total=100.0)
        assert cb is not None
        # 50 advancing events; each schedules an asyncio task.
        for _ in range(50):
            cb({
                "event": "participant_finish",
                "participant": "claude",
                "status": "ok",
                "elapsed_seconds": 1.0,
            })
        # Force a GC pass — without strong refs in the closure set, any
        # task suspended at the `await asyncio.sleep(0)` above is fair
        # game and would be collected before delivery.
        gc.collect()
        gc.collect()
        # Drain.
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(_exercise())

    # All 50 notifications delivered.
    assert len(delivered) == 50, f"expected 50 delivered, got {len(delivered)}"
