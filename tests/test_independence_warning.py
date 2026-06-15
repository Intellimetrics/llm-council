"""Tests for the optional independence warning (H2).

Advisory-only signal: when every labeled vote in the final round comes from
the same vendor *family*, correlated same-vendor agreement can masquerade as
independent corroboration. The orchestrator surfaces a NEW
``metadata['independence_warning']`` dict and a ``single_vendor_quorum``
progress event when the resolved distinct-vendor floor is unmet. It NEVER
drops a peer and NEVER overloads ``metadata['degraded']`` / ``min_quorum`` /
``labeled_quorum``. Default OFF (threshold unset).

Mirrors the orchestrator fixture style in ``test_continue_debate.py`` and
``test_synthesis_gating.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from llm_council.adapters import ParticipantResult


def _result(name, *, label="yes"):
    """Build a labeled-vote fixture ParticipantResult."""
    return ParticipantResult(
        name=name,
        ok=True,
        output=f"RECOMMENDATION: {label} - reason",
        error="",
        elapsed_seconds=1.0,
    )


def _run(participant_cfg, config, *, participants=None):
    """Run a single-round ``execute_council`` over the given cfg/config.

    The fake ``run_participants`` returns one labeled vote per selected
    participant; deliberation is off so only round 1 runs.
    """
    import llm_council.orchestrator as orch_module

    names = participants or list(participant_cfg.keys())

    async def fake_run_participants(selected, *args, **kwargs):
        return [_result(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    mode = config.get("_mode_for_test")
    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            return asyncio.run(
                orch_module.execute_council(
                    participants=names,
                    participant_cfg=participant_cfg,
                    prompt="q",
                    cwd=Path("."),
                    config=config,
                    deliberate=False,
                    max_rounds=1,
                    mode=mode,
                )
            )


def test_warning_fires_when_all_votes_share_one_family():
    """Two peers, both `acme` family, threshold 2 → warning present."""
    cfg = {
        "a": {"type": "cli", "family": "acme"},
        "b": {"type": "cli", "family": "acme"},
    }
    config = {"defaults": {"min_distinct_vendors": 2}}

    _, metadata = _run(cfg, config)

    warning = metadata.get("independence_warning")
    assert warning is not None
    assert warning["distinct_vendors"] == 1
    assert warning["required"] == 2
    assert warning["families"] == ["acme"]
    assert warning["labeled_quorum"] == metadata["labeled_quorum"]

    events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "single_vendor_quorum"
    ]
    assert len(events) == 1
    assert events[0]["distinct_vendors"] == 1
    assert events[0]["required"] == 2
    assert events[0]["families"] == ["acme"]

    # CRITICAL: degraded must be untouched. Both peers labeled, so this is
    # NOT a below-quorum-count run.
    assert metadata["degraded"] is False


def test_warning_absent_when_threshold_met():
    """Two peers across distinct families, threshold 2 → key ABSENT."""
    cfg = {
        "a": {"type": "cli", "family": "acme"},
        "b": {"type": "cli", "family": "globex"},
    }
    config = {"defaults": {"min_distinct_vendors": 2}}

    _, metadata = _run(cfg, config)

    assert "independence_warning" not in metadata
    events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "single_vendor_quorum"
    ]
    assert events == []
    assert metadata["degraded"] is False


def test_feature_off_when_threshold_unset():
    """No threshold configured → key ABSENT, no event, even single-family."""
    cfg = {
        "a": {"type": "cli", "family": "acme"},
        "b": {"type": "cli", "family": "acme"},
    }
    config = {"defaults": {}}

    _, metadata = _run(cfg, config)

    assert "independence_warning" not in metadata
    events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "single_vendor_quorum"
    ]
    assert events == []
    assert metadata["degraded"] is False


def test_mode_override_takes_precedence_over_global():
    """Per-mode `require_distinct_vendors` resolves before the global default.

    Global default is 1 (would not fire for a single family); the mode
    override of 2 makes the single-family run trip the warning.
    """
    cfg = {
        "a": {"type": "cli", "family": "acme"},
        "b": {"type": "cli", "family": "acme"},
    }
    config = {
        "defaults": {"min_distinct_vendors": 1},
        "modes": {"review": {"require_distinct_vendors": 2}},
        "_mode_for_test": "review",
    }

    _, metadata = _run(cfg, config)

    warning = metadata.get("independence_warning")
    assert warning is not None
    assert warning["required"] == 2
    assert metadata["degraded"] is False


def test_family_falls_back_to_base_name_when_unset():
    """A participant cfg without `family` uses its base name as the family.

    Two peers, no explicit family → two distinct 'families' (their names),
    so the count is 2 and a threshold of 2 is MET (warning absent).
    """
    cfg = {
        "a": {"type": "cli"},
        "b": {"type": "cli"},
    }
    config = {"defaults": {"min_distinct_vendors": 2}}

    _, metadata = _run(cfg, config)

    assert "independence_warning" not in metadata
    assert metadata["degraded"] is False


def test_no_warning_when_zero_labeled_votes():
    """Threshold enabled but NO peer produced a usable label → no warning.

    With zero labeled votes there is no consensus to mistake for independent
    corroboration (the run is already `degraded` on quorum count), so the
    single-vendor warning must NOT fire. Regression guard for the codex
    review finding in WU2.
    """
    import llm_council.orchestrator as orch_module

    cfg = {
        "a": {"type": "cli", "family": "acme"},
        "b": {"type": "cli", "family": "acme"},
    }
    config = {"defaults": {"min_distinct_vendors": 2}}

    async def fake_run_participants(selected, *args, **kwargs):
        # ok=True but NO RECOMMENDATION label → not a labeled vote.
        return [
            ParticipantResult(
                name=name,
                ok=True,
                output="some prose with no recommendation label",
                error="",
                elapsed_seconds=1.0,
            )
            for name in selected
        ]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            _, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=list(cfg.keys()),
                    participant_cfg=cfg,
                    prompt="q",
                    cwd=Path("."),
                    config=config,
                    deliberate=False,
                    max_rounds=1,
                )
            )

    assert "independence_warning" not in metadata
    events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "single_vendor_quorum"
    ]
    assert events == []
