"""Regression tests for pass-5 council review (v0.5.1 → v0.5.2).

Pass-5 voted RECOMMENDATION: tradeoff with 7 follow-ups across the 4
peers; this file locks in v0.5.2's response to each.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.adapters import (
    ABDICATED_ERROR_PREFIX,
    CacheContext,
    ParticipantResult,
    _maybe_persist_cache,
    _with_envelope,
    classify_error,
)
from llm_council.synthesis import select_synthesizer, should_synthesize


# --- Fix A: overbroad repair_retry_recovered guard dropped ---------------

def test_with_envelope_abdicates_when_repaired_response_itself_abdicates():
    """Belt-and-suspenders bug from v0.5.1: skipping abdication when
    repair_retry_recovered=True let a *legitimately* abdicating repaired
    response slip through. The parse-source strip is the correctness
    mechanism — abdication detection must still run on whatever the
    repaired attempt actually contained."""
    output = (
        "[recovered after retry] First attempt was missing the required "
        "RECOMMENDATION label; second attempt is shown below.\n"
        "\n"
        "--- Repaired response ---\n"
        "RECOMMENDATION: no - I genuinely cannot evaluate this\n"
        "EFFORT: blocked\n"
        "\n"
        "--- Original response (first attempt) ---\n"
        "Some original prose without a label.\n"
    )
    r = ParticipantResult(
        "peer", True, output, "", 1.0, repair_retry_recovered=True
    )
    out = _with_envelope(r)
    assert out.ok is False, (
        "repaired response that itself abdicates must be flagged — "
        "repair_retry_recovered is not a free pass"
    )
    assert out.error.startswith(ABDICATED_ERROR_PREFIX)
    assert classify_error(out.error) == "abdicated"


# --- Revert #10: cache writes abdications for offline re-derivation -----

def test_abdication_cached_then_rederived_on_hit(tmp_path: Path):
    """Confirms the v0.5.1 fix-revert: abdication output IS written to the
    cache, and the read-side _with_envelope re-derivation flips ok=False.

    This preserves the user-visible "failed runs are not counted" invariant
    without paying the peer twice for the same abdication.
    """
    abdication_output = (
        "RECOMMENDATION: no - too complex to evaluate\n"
        "EFFORT: blocked\n"
    )
    # Adapter writes ok=True to cache (the flip to ok=False happens later in
    # run_participant via _with_envelope).
    r = ParticipantResult("peer", True, abdication_output, "", 1.0)
    cache_ctx = CacheContext(cwd=tmp_path, cache_mode="on", cache_disabled=False)
    _maybe_persist_cache("peer", "the prompt", "fake-key", r, cache_ctx)
    cache_dir = tmp_path / ".llm-council" / "cache"
    cached_files = list(cache_dir.glob("*.json"))
    assert len(cached_files) == 1, "abdication output must be cached"

    # Re-derivation: simulating run_participant pulling from cache. We
    # build a fresh ParticipantResult with from_cache=True (mimicking
    # _result_from_cache_payload) and call _with_envelope as the runtime
    # entry point would.
    rehydrated = ParticipantResult(
        "peer", True, abdication_output, "", 1.0, from_cache=True
    )
    out = _with_envelope(rehydrated)
    assert out.ok is False, (
        "cache-hit re-derivation must still flag abdication — "
        "this is the correctness mechanism that allows caching the output"
    )
    assert classify_error(out.error) == "abdicated"


# --- Fix D: synthesis skipped after universal_abdication ---------------

def test_should_synthesize_false_when_universal_abdication_fired():
    """Even an explicit --synthesize must not invoke the chair when all
    peers abdicated — chair input would be empty after final-round +
    ok-only filtering."""
    metadata = {"universal_abdication": {"blockers": ["missing data"]}}
    assert should_synthesize(True, metadata) is False
    assert should_synthesize(False, metadata) is False


def test_should_synthesize_unaffected_when_universal_abdication_absent():
    """Sanity: removing the universal_abdication key restores normal trigger."""
    assert should_synthesize(True, {}) is True
    assert (
        should_synthesize(False, {"deliberation_status": "ran_max_rounds_unresolved"})
        is True
    )


# --- Fix F: deliberation_status only stamped when deliberate=True --------

def test_universal_abdication_does_not_stamp_status_without_deliberate():
    """If the run was non-deliberative, `deliberation_status=skipped_*` is
    misleading metadata — deliberation was never under consideration."""
    import asyncio
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return [
            ParticipantResult(
                name,
                False,
                "RECOMMENDATION: no - too complex\nEFFORT: blocked",
                f"{ABDICATED_ERROR_PREFIX} test fixture",
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
                    deliberate=False,  # the key difference vs pass-4 test
                    max_rounds=2,
                )
            )

    assert metadata.get("universal_abdication") is not None
    # Pass-5 fix F: do NOT overwrite the not_requested status when
    # deliberation wasn't even on the table.
    assert metadata["deliberation_status"] == "not_requested", (
        "non-deliberative runs must not be relabeled 'skipped_universal_abdication'"
    )


# --- UX tests for previously uncovered v0.5.1 patches ------------------

def test_select_synthesizer_current_requires_host_in_participant_cfg():
    """Pass-5 codex flagged: the 'current' synthesizer must fail loudly
    when the host CLI is excluded from the run (peer-only modes, etc.)
    rather than silently fall back to a peer."""
    with pytest.raises(ValueError, match="not a configured participant"):
        select_synthesizer(
            {"defaults": {"synthesizer": "current"}},
            {"codex": {}, "gemini": {}},  # claude (host) deliberately excluded
            stances=None,
            current="claude",
        )
