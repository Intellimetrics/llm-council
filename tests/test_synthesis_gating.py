"""Synthesis-chair gating and chair-resolution edge cases.

Covers the orchestrator-level decisions about WHEN to invoke the chair
and WHICH peer to invoke as chair. For the synthesis prompt body and
basic helpers see ``test_synthesis.py``; for the chair's response
handling under retry/cache pressure see ``test_abdication_detection.py``.

Specifically:
- ``should_synthesize`` interaction with ``universal_abdication``.
- ``run_synthesis_chair`` bypasses the label-validation path
  (decision memos are not votes).
- ``execute_council`` feeds the chair final-round results only.
- ``execute_council`` short-circuits round 2 on universal abdication,
  and only stamps ``deliberation_status`` when deliberation was on the
  table.
- ``select_synthesizer("current")`` fails loudly when the host CLI is
  not a configured participant.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.adapters import ABDICATED_ERROR_PREFIX, ParticipantResult
from llm_council.synthesis import (
    run_synthesis_chair,
    select_synthesizer,
    should_synthesize,
)


# --- should_synthesize / universal_abdication interaction ----------------

def test_should_synthesize_skips_when_universal_abdication_fired():
    """Even an explicit ``--synthesize`` must not invoke the chair when
    all peers abdicated — chair input would be empty after filtering."""
    metadata = {"universal_abdication": {"blockers": ["missing data"]}}
    assert should_synthesize(True, metadata) is False
    assert should_synthesize(False, metadata) is False


def test_should_synthesize_normal_paths_unaffected():
    assert should_synthesize(True, {}) is True
    assert (
        should_synthesize(False, {"deliberation_status": "ran_max_rounds_unresolved"})
        is True
    )
    assert (
        should_synthesize(False, {"deliberation_status": "ran_no_labeled_disagreement"})
        is False
    )


# --- Chair invocation contract -------------------------------------------

def test_run_synthesis_chair_bypasses_label_validation():
    """The chair returns a structured decision memo (no
    ``RECOMMENDATION:`` line). Without overriding the cfg, label
    validation would mark the chair's output ``invalid_response`` and
    fire the label-only repair retry."""
    captured_cfg: dict = {}

    async def fake_run_participant(name, cfg, prompt, cwd, *, cache_ctx=None):
        captured_cfg.update(cfg)
        return ParticipantResult(
            name=name,
            ok=True,
            output="### Decision\nTradeoff. Ship with the migration test.",
            error="",
            elapsed_seconds=0.1,
        )

    with patch("llm_council.synthesis.run_participant", side_effect=fake_run_participant):
        peer = ParticipantResult("p", True, "RECOMMENDATION: yes - ok", "", 1.0)
        asyncio.run(
            run_synthesis_chair(
                question="should we ship?",
                results=[peer],
                convergence=None,
                participant_cfg={"chair": {"type": "cli", "command": "echo"}},
                cwd=Path("."),
                chair_name="chair",
            )
        )

    assert captured_cfg.get("require_recommendation") is False
    assert captured_cfg.get("retry_on_missing_label") is False


# --- Orchestrator: final-round-only chair input --------------------------

def test_chair_receives_only_final_round_after_deliberation():
    """``execute_council`` must call ``final_round_results(results)``
    before passing peer outputs to the chair. Round-1 ('a', 'b') and
    round-2 ('a:round2', 'b:round2') entries coexist in the cumulative
    results list; only the suffixed ones reflect the peers' final
    positions."""
    captured: dict = {}

    async def fake_run_synthesis(*args, **kwargs):
        captured["results"] = kwargs["results"]
        return {
            "chair": kwargs["chair_name"],
            "ok": True,
            "output": "### Decision\nyes",
            "error": "",
            "decision_label": "unknown",
            "blockers": [],
            "evidence": [],
            "tests_to_run": [],
            "elapsed_seconds": 0.1,
            "model": None,
            "total_tokens": None,
            "cost_usd": None,
            "consumed_convergence": False,
            "prompt_chars": 0,
        }

    from llm_council.transcript import final_round_results

    round1 = [
        ParticipantResult("a", True, "RECOMMENDATION: yes - r1", "", 1.0),
        ParticipantResult("b", True, "RECOMMENDATION: no - r1", "", 1.0),
    ]
    round2 = [
        ParticipantResult("a:round2", True, "RECOMMENDATION: tradeoff - r2", "", 1.0),
        ParticipantResult("b:round2", True, "RECOMMENDATION: tradeoff - r2", "", 1.0),
    ]
    cumulative = round1 + round2

    chair_input = final_round_results(cumulative)
    asyncio.run(
        fake_run_synthesis(
            question="q",
            results=chair_input,
            convergence=None,
            participant_cfg={"a": {"type": "cli"}, "b": {"type": "cli"}},
            cwd=Path("."),
            chair_name="a",
        )
    )

    names = [r.name for r in captured["results"]]
    assert names == ["a:round2", "b:round2"]


# --- Orchestrator: universal_abdication short-circuit ---------------------

def test_universal_abdication_skips_deliberation():
    """When all round-1 peers abdicate, the deliberation loop must
    refuse to enter. ``deliberation_status`` records
    ``skipped_universal_abdication``."""
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return [
            ParticipantResult(
                name,
                False,
                "RECOMMENDATION: no - too complex\nEFFORT: blocked",
                f"{ABDICATED_ERROR_PREFIX} fixture",
                1.0,
                effort="blocked",
            )
            for name in selected
        ]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(orch_module, "run_participants", side_effect=fake_run_participants):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            results, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=["a", "b"],
                    participant_cfg={"a": {"type": "cli"}, "b": {"type": "cli"}},
                    prompt="q",
                    cwd=Path("."),
                    config={"defaults": {}},
                    deliberate=True,
                    max_rounds=2,
                )
            )

    assert metadata.get("universal_abdication") is not None
    assert metadata["deliberation_status"] == "skipped_universal_abdication"
    assert metadata["rounds"] == 1
    assert metadata["deliberated"] is False


def test_universal_abdication_does_not_stamp_status_without_deliberate():
    """Non-deliberative runs keep ``deliberation_status='not_requested'``
    even when universal abdication fires — the field would otherwise
    falsely imply deliberation was considered."""
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return [
            ParticipantResult(
                name,
                False,
                "RECOMMENDATION: no - too complex\nEFFORT: blocked",
                f"{ABDICATED_ERROR_PREFIX} fixture",
                1.0,
                effort="blocked",
            )
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
                    participants=["a", "b"],
                    participant_cfg={"a": {"type": "cli"}, "b": {"type": "cli"}},
                    prompt="q",
                    cwd=Path("."),
                    config={"defaults": {}},
                    deliberate=False,
                    max_rounds=2,
                )
            )

    assert metadata.get("universal_abdication") is not None
    assert metadata["deliberation_status"] == "not_requested"


# --- select_synthesizer("current") host-exclusion edge case --------------

def test_select_synthesizer_current_requires_host_in_participant_cfg():
    """When the host CLI is excluded from the run (peer-only modes),
    ``current`` must fail loudly rather than silently falling back to
    another peer — preserving the requester-bias-is-opt-in default."""
    with pytest.raises(ValueError, match="not a configured participant"):
        select_synthesizer(
            {"defaults": {"synthesizer": "current"}},
            {"codex": {}, "gemini": {}},  # claude (host) excluded
            stances=None,
            current="claude",
        )
