"""Tests for the ``CONTINUE_DEBATE: yes|no`` envelope tag (v0.8.1).

Peers may emit ``CONTINUE_DEBATE: yes|no`` alongside the other envelope
fields. The orchestrator skips the optional round-2 deliberation when
every label-producing peer voted ``no`` (unanimity, not 66%).
Abdicated / unlabeled peers are excluded from BOTH numerator and
denominator. A single-peer council cannot trigger the skip
(``len(denominator) >= 2`` floor).

Mirrors the envelope-parsing patterns in ``test_effort_contract.py``
and the orchestrator fixture style in ``test_synthesis_gating.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from llm_council.adapters import (
    ABDICATED_ERROR_PREFIX,
    ParticipantResult,
    _extract_response_envelope,
    _result_from_cache_payload,
    _with_envelope,
)
from llm_council.cache import build_payload as cache_build_payload


# --- Parsing tests -------------------------------------------------------

def test_continue_debate_yes_parses():
    env = _extract_response_envelope(
        "RECOMMENDATION: yes - ship\nCONTINUE_DEBATE: yes\n"
    )
    assert env["continue_debate"] == "yes"


def test_continue_debate_no_parses():
    env = _extract_response_envelope(
        "RECOMMENDATION: no - hold\nCONTINUE_DEBATE: no\n"
    )
    assert env["continue_debate"] == "no"


def test_continue_debate_case_insensitive():
    env = _extract_response_envelope(
        "RECOMMENDATION: tradeoff - mixed\ncontinue_debate: NO\n"
    )
    assert env["continue_debate"] == "no"


def test_continue_debate_ignored_inside_fenced_block():
    """Fence-aware, same rule as other envelope fields. A CONTINUE_DEBATE
    line inside a code fence is example syntax, not a real vote."""
    text = (
        "RECOMMENDATION: yes - ship\n"
        "```\n"
        "CONTINUE_DEBATE: no\n"
        "```\n"
    )
    env = _extract_response_envelope(text)
    assert env["continue_debate"] is None


def test_continue_debate_missing_stays_none():
    env = _extract_response_envelope(
        "RECOMMENDATION: yes - ship it\nEFFORT: full\n"
    )
    assert env["continue_debate"] is None


def test_continue_debate_bullet_form_parses():
    env = _extract_response_envelope(
        "RECOMMENDATION: no - blocked\n- CONTINUE_DEBATE: yes\n"
    )
    assert env["continue_debate"] == "yes"


# --- ParticipantResult round-trip ---------------------------------------

def test_participant_result_default_is_none():
    r = ParticipantResult("a", True, "RECOMMENDATION: yes - ok", "", 0.1)
    assert r.continue_debate is None


def test_with_envelope_populates_continue_debate():
    r = ParticipantResult(
        "a",
        True,
        "RECOMMENDATION: tradeoff - ok\nCONTINUE_DEBATE: no\n",
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.continue_debate == "no"


def test_cache_round_trip_preserves_continue_debate():
    """A result with ``continue_debate="no"`` must survive write+read."""
    r = ParticipantResult(
        "a",
        True,
        "RECOMMENDATION: no - hold\nCONTINUE_DEBATE: no\n",
        "",
        1.5,
        continue_debate="no",
    )
    payload = cache_build_payload(
        participant_name=r.name,
        prompt="q",
        key="k",
        output=r.output,
        recommendation_label="no",
        elapsed_seconds=r.elapsed_seconds,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model=None,
        command=None,
        continue_debate=r.continue_debate,
    )
    assert payload.get("continue_debate") == "no"
    rehydrated = _result_from_cache_payload("a", payload)
    assert rehydrated.continue_debate == "no"


def test_cache_round_trip_omits_continue_debate_when_none():
    """Absence semantics: cached payloads without the field rehydrate to None."""
    payload = cache_build_payload(
        participant_name="a",
        prompt="q",
        key="k",
        output="RECOMMENDATION: yes - ok",
        recommendation_label="yes",
        elapsed_seconds=1.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model=None,
        command=None,
        continue_debate=None,
    )
    # Payload should not carry the key for absent votes — keeps payloads tight.
    assert "continue_debate" not in payload
    rehydrated = _result_from_cache_payload("a", payload)
    assert rehydrated.continue_debate is None


# --- Orchestrator gating ------------------------------------------------

def _result(name, *, label, continue_debate=None, ok=True, error=""):
    """Build a fixture ParticipantResult with the envelope field populated.

    The orchestrator gate consults ``result.continue_debate`` directly, so
    we synthesize it here rather than relying on parsing.
    """
    output = f"RECOMMENDATION: {label} - reason"
    if continue_debate is not None:
        output += f"\nCONTINUE_DEBATE: {continue_debate}"
    return ParticipantResult(
        name=name,
        ok=ok,
        output=output,
        error=error,
        elapsed_seconds=1.0,
        continue_debate=continue_debate,
    )


def _run_orchestrator(round1_factory):
    """Run ``execute_council`` with a single-round fake. Returns (results, metadata)."""
    import llm_council.orchestrator as orch_module

    selected_names: list[str] = []

    async def fake_run_participants(selected, *args, **kwargs):
        # Capture names so the test fixture can return one per participant.
        selected_names[:] = list(selected)
        return [round1_factory(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    participants = ["a", "b", "c"]
    with patch.object(orch_module, "run_participants", side_effect=fake_run_participants):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            return asyncio.run(
                orch_module.execute_council(
                    participants=participants,
                    participant_cfg={n: {"type": "cli"} for n in participants},
                    prompt="q",
                    cwd=Path("."),
                    config={"defaults": {}},
                    deliberate=True,
                    max_rounds=2,
                )
            )


def test_orchestrator_skips_round2_on_unanimous_no():
    """3 peers, all vote ``CONTINUE_DEBATE: no`` with disagreement — round 2 SKIPPED."""
    labels = {"a": "yes", "b": "no", "c": "tradeoff"}

    def factory(name):
        return _result(name, label=labels[name], continue_debate="no")

    _, metadata = _run_orchestrator(factory)

    assert metadata["disagreement_detected"] is True
    assert metadata["deliberation_status"] == "skipped_continue_debate_unanimous"
    assert metadata["rounds"] == 1
    assert metadata["deliberated"] is False
    # Progress event surfaced for operator-visible telemetry.
    events = [
        e for e in metadata["progress_events"]
        if e.get("event") == "deliberation_skipped"
    ]
    assert len(events) == 1
    assert events[0]["reason"] == "continue_debate_unanimous"
    assert events[0]["no_votes"] == 3
    assert events[0]["denominator"] == 3


def test_orchestrator_runs_round2_when_one_peer_votes_yes():
    """2 vote no + 1 votes yes — not unanimous, round 2 RUNS."""
    labels = {"a": "yes", "b": "no", "c": "tradeoff"}
    cd = {"a": "no", "b": "no", "c": "yes"}

    def factory(name):
        return _result(name, label=labels[name], continue_debate=cd[name])

    _, metadata = _run_orchestrator(factory)

    # The fixture returns the SAME round-1 results on round 2 (the fake
    # doesn't differentiate), so disagreement persists and the status
    # reflects deliberation completing the max-rounds budget. The key
    # assertion is that round 2 actually ran.
    assert metadata["deliberated"] is True
    assert metadata["rounds"] == 2
    assert metadata["deliberation_status"] != "skipped_continue_debate_unanimous"


def test_orchestrator_runs_round2_when_one_peer_missing_vote():
    """2 vote no + 1 has no CONTINUE_DEBATE tag — silent peer != skip vote, round 2 RUNS."""
    labels = {"a": "yes", "b": "no", "c": "tradeoff"}
    cd = {"a": "no", "b": "no", "c": None}

    def factory(name):
        return _result(name, label=labels[name], continue_debate=cd[name])

    _, metadata = _run_orchestrator(factory)

    assert metadata["deliberated"] is True
    assert metadata["rounds"] == 2
    assert metadata["deliberation_status"] != "skipped_continue_debate_unanimous"


def test_orchestrator_abdicated_peer_excluded_from_denominator():
    """3 peers, 2 vote ``no`` + 1 abdicated. Denominator is 2, both votes
    are ``no``, so the unanimity check fires and round 2 is skipped."""
    import llm_council.orchestrator as orch_module

    def factory(name):
        if name == "c":
            # Abdicated: ok=False, no usable RECOMMENDATION label
            # downstream — fully excluded from the denominator. The
            # peer's `continue_debate` is irrelevant here.
            return ParticipantResult(
                name=name,
                ok=False,
                output="RECOMMENDATION: no - too complex\nEFFORT: blocked",
                error=f"{ABDICATED_ERROR_PREFIX} fixture",
                elapsed_seconds=1.0,
                effort="blocked",
            )
        labels = {"a": "yes", "b": "no"}
        return _result(name, label=labels[name], continue_debate="no")

    async def fake_run_participants(selected, *args, **kwargs):
        return [factory(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(orch_module, "run_participants", side_effect=fake_run_participants):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            _, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=["a", "b", "c"],
                    participant_cfg={n: {"type": "cli"} for n in ["a", "b", "c"]},
                    prompt="q",
                    cwd=Path("."),
                    config={"defaults": {}},
                    deliberate=True,
                    max_rounds=2,
                )
            )

    assert metadata["deliberation_status"] == "skipped_continue_debate_unanimous"
    events = [
        e for e in metadata["progress_events"]
        if e.get("event") == "deliberation_skipped"
    ]
    assert len(events) == 1
    assert events[0]["no_votes"] == 2
    assert events[0]["denominator"] == 2


def test_orchestrator_single_peer_unanimous_no_still_runs_round2():
    """Degenerate single-peer council: ``len(denominator) >= 2`` floor
    prevents a lone peer from voting itself out of deliberation. But
    with only one peer, ``has_disagreement`` is also False, so round 2
    never runs anyway. The key assertion is that the skip status is
    NOT stamped — we want the no-disagreement reason to win, not a
    spurious unanimity stamp from a degenerate council."""
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return [
            _result(name, label="no", continue_debate="no")
            for name in selected
        ]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(orch_module, "run_participants", side_effect=fake_run_participants):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            _, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=["a"],
                    participant_cfg={"a": {"type": "cli"}},
                    prompt="q",
                    cwd=Path("."),
                    config={"defaults": {}},
                    deliberate=True,
                    max_rounds=2,
                )
            )

    # With only one peer, no disagreement exists; deliberation skip
    # reason should be "no labeled disagreement", NOT the unanimity
    # gate (which is guarded by `len(denominator) >= 2`).
    assert metadata["deliberation_status"] == "skipped_no_labeled_disagreement"
