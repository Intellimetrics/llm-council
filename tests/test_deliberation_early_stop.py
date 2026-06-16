"""No-new-movement early-stop for deliberation (L5, opt-in, default OFF).

When ``deliberation_early_stop`` is enabled (per-mode or via
``defaults``), the orchestrator stops the deliberation loop early once an
extra round produces NO new movement: a non-diverging convergence signal
AND an unchanged vote tally. Both conditions are required — a "converged"
similarity can co-exist with a still-split vote, so the vote-tally
comparison corroborates the Jaccard signal.

Only meaningful for modes with ``max_rounds >= 3``; with the default
``max_rounds=2`` a single deliberation round runs and the early-stop check
never gets a second round to compare against.

Uses the orchestrator fixture harness shared with
``test_synthesis_gating.py`` / ``test_continue_debate.py``: ``run_participants``
is stubbed to return the SAME (stable, split) results on every round, so
votes persist identically and consecutive rounds are byte-identical —
exactly the no-new-movement condition.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from llm_council.adapters import ParticipantResult


# Long, stable per-peer rationales: enough content tokens (>=10) that the
# convergence detector classifies (identical text across rounds -> Jaccard
# 1.0 -> "converged", which is non-diverging) rather than returning
# "insufficient" on a too-short body.
_STABLE_OUTPUTS = {
    "a": (
        "RECOMMENDATION: yes - the migration path is safe and reversible.\n"
        "The schema change preserves backward compatibility because the new "
        "column is nullable and the existing readers ignore unknown fields "
        "entirely during the rollout window.\n"
    ),
    "b": (
        "RECOMMENDATION: no - the rollback procedure is undocumented and risky.\n"
        "Without a tested down-migration the deployment cannot recover from a "
        "partial failure, so the change should hold until that gap is closed "
        "and verified against staging.\n"
    ),
}


def _stable_result(name: str) -> ParticipantResult:
    return ParticipantResult(
        name=name,
        ok=True,
        output=_STABLE_OUTPUTS[name],
        error="",
        elapsed_seconds=1.0,
    )


def _run_deep_audit(*, config: dict) -> tuple[list, dict]:
    """Run a deep-audit-like council (deliberate, max_rounds=3) where every
    round reproduces the same split, stable votes. Returns (results, metadata)."""
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return [_stable_result(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    participants = ["a", "b"]
    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            return asyncio.run(
                orch_module.execute_council(
                    participants=participants,
                    participant_cfg={n: {"type": "cli"} for n in participants},
                    prompt="should we ship the migration?",
                    cwd=Path("."),
                    config=config,
                    deliberate=True,
                    max_rounds=3,
                )
            )


def test_early_stop_fires_on_no_new_movement():
    """Flag ON + a round that reproduces the prior split with no divergence
    -> loop stops before round 3, status is ``stopped_no_new_movement``, and
    a ``deliberation_early_stop`` event is emitted."""
    _, metadata = _run_deep_audit(
        config={"defaults": {"deliberation_early_stop": True}}
    )

    # max_rounds=3 would otherwise run rounds 2 AND 3; early-stop breaks
    # after the first deliberation round (round 2).
    assert metadata["rounds"] == 2
    assert metadata["deliberated"] is True
    assert metadata["deliberation_status"] == "stopped_no_new_movement"
    # final_disagreement is still computed (the split persists).
    assert metadata["final_disagreement_detected"] is True

    stop_events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "deliberation_early_stop"
    ]
    assert len(stop_events) == 1
    assert stop_events[0]["reason"] == "no_new_movement"
    assert stop_events[0]["round"] == 2
    # The current vote tally is carried on the event.
    assert stop_events[0]["counts"]["yes"] == 1
    assert stop_events[0]["counts"]["no"] == 1


def test_no_early_stop_when_flag_off_runs_full_rounds():
    """Same no-new-movement scenario with the flag OFF (absent) -> the loop
    runs the full deliberation budget and the status is the legacy
    ``ran_max_rounds_unresolved`` (split persists). No early-stop event."""
    _, metadata = _run_deep_audit(config={"defaults": {}})

    # All three rounds run (round 1 + two deliberation rounds).
    assert metadata["rounds"] == 3
    assert metadata["deliberated"] is True
    assert metadata["deliberation_status"] == "ran_max_rounds_unresolved"

    stop_events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "deliberation_early_stop"
    ]
    assert stop_events == []


def test_no_early_stop_on_max_rounds_2_even_when_enabled():
    """Guard (codex WU9 review): with the flag ON but ``max_rounds=2`` the
    single deliberation round is the last one anyway — early-stop must NOT
    fire (nothing to skip) and must NOT relabel the run as
    ``stopped_no_new_movement``. The early-stop check is gated on
    ``round_number < max_rounds`` so only deep-audit (>=3) is affected."""
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return [_stable_result(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    participants = ["a", "b"]
    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            _, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=participants,
                    participant_cfg={n: {"type": "cli"} for n in participants},
                    prompt="q",
                    cwd=Path("."),
                    config={"defaults": {"deliberation_early_stop": True}},
                    deliberate=True,
                    max_rounds=2,
                )
            )

    # The single deliberation round ran; status is the legacy one, NOT
    # stopped_no_new_movement, and no early-stop event was emitted.
    assert metadata["rounds"] == 2
    assert metadata["deliberation_status"] == "ran_max_rounds_unresolved"
    stop_events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "deliberation_early_stop"
    ]
    assert stop_events == []


def test_mode_explicit_false_overrides_true_default():
    """None-aware precedence: a mode-explicit ``deliberation_early_stop:
    false`` must override a ``true`` default (NOT an ``or`` chain)."""
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return [_stable_result(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    participants = ["a", "b"]
    config = {
        "defaults": {"deliberation_early_stop": True},
        "modes": {"deep-audit": {"deliberation_early_stop": False}},
    }
    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            _, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=participants,
                    participant_cfg={n: {"type": "cli"} for n in participants},
                    prompt="q",
                    cwd=Path("."),
                    config=config,
                    mode="deep-audit",
                    deliberate=True,
                    max_rounds=3,
                )
            )

    # Mode-explicit false wins -> no early stop, full budget runs.
    assert metadata["rounds"] == 3
    assert metadata["deliberation_status"] == "ran_max_rounds_unresolved"
