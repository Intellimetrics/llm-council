"""Tests for `check_promotion_gate` (Phase E — v0.8 plan).

The promotion gate compares two `SuiteScorecard` aggregates and returns
`PromotionResult(promoted=True)` only when:
    1. B.blocker_recall - A.blocker_recall >= recall_lift (default 0.05)
    2. B.signal_to_noise_ratio >= A.signal_to_noise_ratio * snr_floor_ratio
       (default 0.85)

Either gate failing blocks promotion. The mode stays `experimental: true`
in `defaults.py` until both gates pass on the canonical fixture set.
"""

from __future__ import annotations

import json

import pytest

from llm_council.eval.runner import (
    SuiteScorecard,
    check_promotion_gate,
)


def _suite(
    mode: str,
    *,
    recall: float,
    snr: float,
    extra: dict | None = None,
) -> SuiteScorecard:
    """Build a minimal SuiteScorecard with just the aggregate metrics
    the gate cares about. Other fields stay at defaults.
    """
    aggregate = {
        "blocker_recall": recall,
        "signal_to_noise_ratio": snr,
    }
    if extra:
        aggregate.update(extra)
    return SuiteScorecard(
        mode=mode,
        council_version="test",
        timestamp="2026-05-17T00:00:00+00:00",
        aggregate_metrics=aggregate,
    )


# --- happy path ---------------------------------------------------------


def test_both_conditions_met_promotes():
    baseline = _suite("review", recall=0.50, snr=2.0)
    candidate = _suite("review-with-tools", recall=0.60, snr=1.9)
    result = check_promotion_gate(baseline, candidate)
    assert result.promoted is True
    assert pytest.approx(result.recall_delta, abs=1e-6) == 0.10
    assert pytest.approx(result.snr_ratio, abs=1e-6) == 1.9 / 2.0
    # At least one PASS reason for each gate.
    joined = " ".join(result.reasons)
    assert "recall_lift_met" in joined
    assert "snr_floor_met" in joined


def test_exactly_at_threshold_promotes():
    """Boundary: recall delta is exactly 0.05, SNR ratio is exactly 0.85."""
    baseline = _suite("review", recall=0.50, snr=2.0)
    # candidate SNR = 1.7 == 2.0 * 0.85 exactly
    candidate = _suite("review-with-tools", recall=0.55, snr=1.7)
    result = check_promotion_gate(baseline, candidate)
    assert result.promoted is True


# --- failure: SNR collapsed --------------------------------------------


def test_recall_lift_met_but_snr_collapsed_blocks():
    baseline = _suite("review", recall=0.50, snr=2.0)
    # SNR drops to 1.5 (75% of baseline) — below the 85% floor.
    candidate = _suite("review-with-tools", recall=0.60, snr=1.5)
    result = check_promotion_gate(baseline, candidate)
    assert result.promoted is False
    joined = " ".join(result.reasons)
    assert "snr_collapse" in joined
    # Recall gate should be reported as MET even though promotion fails.
    assert "recall_lift_met" in joined


# --- failure: recall lift missing --------------------------------------


def test_recall_lift_not_met_blocks():
    baseline = _suite("review", recall=0.50, snr=2.0)
    # Only +3pp lift — below the 5pp floor.
    candidate = _suite("review-with-tools", recall=0.53, snr=2.0)
    result = check_promotion_gate(baseline, candidate)
    assert result.promoted is False
    joined = " ".join(result.reasons)
    assert "recall_lift_missing" in joined
    assert "snr_floor_met" in joined


def test_recall_regression_blocks():
    """Candidate scores LOWER recall than baseline — obviously blocked."""
    baseline = _suite("review", recall=0.60, snr=2.0)
    candidate = _suite("review-with-tools", recall=0.50, snr=2.0)
    result = check_promotion_gate(baseline, candidate)
    assert result.promoted is False
    assert result.recall_delta == pytest.approx(-0.10, abs=1e-6)
    joined = " ".join(result.reasons)
    assert "recall_lift_missing" in joined


# --- failure: both gates fail ------------------------------------------


def test_both_conditions_fail_reports_both_reasons():
    baseline = _suite("review", recall=0.50, snr=2.0)
    # Neither lift nor SNR floor met.
    candidate = _suite("review-with-tools", recall=0.51, snr=1.4)
    result = check_promotion_gate(baseline, candidate)
    assert result.promoted is False
    joined = " ".join(result.reasons)
    assert "recall_lift_missing" in joined
    assert "snr_collapse" in joined


# --- threshold customization ------------------------------------------


def test_default_thresholds_are_0_05_and_0_85():
    """A 4pp lift fails the 5pp default; a 6pp lift passes."""
    baseline = _suite("review", recall=0.50, snr=2.0)
    candidate_4pp = _suite("rwt", recall=0.54, snr=2.0)
    candidate_6pp = _suite("rwt", recall=0.56, snr=2.0)
    assert check_promotion_gate(baseline, candidate_4pp).promoted is False
    assert check_promotion_gate(baseline, candidate_6pp).promoted is True

    # SNR: 86% of baseline passes default, 84% fails.
    candidate_snr_high = _suite("rwt", recall=0.56, snr=2.0 * 0.86)
    candidate_snr_low = _suite("rwt", recall=0.56, snr=2.0 * 0.84)
    assert check_promotion_gate(baseline, candidate_snr_high).promoted is True
    assert check_promotion_gate(baseline, candidate_snr_low).promoted is False


def test_custom_thresholds_honored():
    """Looser thresholds let a marginal lift through; tighter ones reject it."""
    baseline = _suite("review", recall=0.50, snr=2.0)
    candidate = _suite("rwt", recall=0.52, snr=1.8)
    # Loose: 2pp lift OK, 90% SNR (>=0.5*floor) — both pass.
    loose = check_promotion_gate(
        baseline, candidate, recall_lift=0.01, snr_floor_ratio=0.5
    )
    assert loose.promoted is True
    # Tight: require 10pp lift — fails.
    tight = check_promotion_gate(
        baseline, candidate, recall_lift=0.10, snr_floor_ratio=0.5
    )
    assert tight.promoted is False


# --- edge cases --------------------------------------------------------


def test_zero_baseline_snr_passes_floor_trivially():
    """When baseline SNR is 0, any non-negative candidate SNR clears the floor."""
    baseline = _suite("review", recall=0.50, snr=0.0)
    candidate = _suite("rwt", recall=0.60, snr=0.0)
    result = check_promotion_gate(baseline, candidate)
    # Recall delta is +0.10, SNR is trivially OK.
    assert result.promoted is True
    # `snr_ratio` is None to signal "ratio undefined" (not 0.0, which
    # would conflate with a candidate that has zero signal).
    assert result.snr_ratio is None
    joined = " ".join(result.reasons)
    assert "snr_floor_met" in joined


def test_zero_baseline_snr_with_positive_candidate_serializes_as_none():
    """Baseline SNR=0 + candidate SNR>0 → ratio is `None` (not 0.0).

    Regression guard against the prior conflation where the gate
    collapsed `float('inf')` to `0.0` for JSON, making "candidate has
    positive SNR over a zero baseline" indistinguishable from "candidate
    SNR ratio is zero (candidate has no signal)".
    """
    baseline = _suite("review", recall=0.50, snr=0.0)
    candidate = _suite("rwt", recall=0.60, snr=2.5)
    result = check_promotion_gate(baseline, candidate)
    assert result.promoted is True
    assert result.snr_ratio is None
    # Round-trip through JSON: None becomes null and back.
    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)
    assert decoded["snr_ratio"] is None


def test_empty_aggregates_treated_as_zero():
    """No aggregate metrics → treat both as 0; only recall lift needs to satisfy."""
    baseline = SuiteScorecard(
        mode="review", council_version="t", timestamp="t", aggregate_metrics={}
    )
    candidate = SuiteScorecard(
        mode="rwt", council_version="t", timestamp="t", aggregate_metrics={}
    )
    result = check_promotion_gate(baseline, candidate)
    # delta = 0, lift not met.
    assert result.promoted is False


# --- JSON serialization ------------------------------------------------


def test_promotion_result_to_dict_roundtrips_json():
    """`PromotionResult.to_dict()` must round-trip through json.dumps/loads."""
    baseline = _suite("review", recall=0.50, snr=2.0)
    candidate = _suite("rwt", recall=0.60, snr=1.9)
    result = check_promotion_gate(baseline, candidate)
    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)
    assert decoded["promoted"] is True
    assert decoded["recall_delta"] == pytest.approx(0.10, abs=1e-6)
    assert decoded["snr_ratio"] == pytest.approx(1.9 / 2.0, abs=1e-6)
    assert isinstance(decoded["reasons"], list)
    assert len(decoded["reasons"]) == 2


# --- cross_rank_correlation_floor gate (was untested) -------------------


def test_cross_rank_floor_missing_correlation_blocks():
    """Floor set but no correlation supplied -> blocked with a 'missing' reason,
    even when recall + SNR both pass."""
    baseline = _suite("review", recall=0.50, snr=2.0)
    candidate = _suite("review-with-tools", recall=0.60, snr=1.9)
    result = check_promotion_gate(
        baseline, candidate, cross_rank_correlation_floor=0.5
    )
    assert result.promoted is False
    assert any("cross_rank_correlation_missing" in r for r in result.reasons)


def test_cross_rank_floor_met_allows_promotion():
    """Correlation >= floor -> gate is satisfied; promotion rides on recall+SNR."""
    baseline = _suite("review", recall=0.50, snr=2.0)
    candidate = _suite("review-with-tools", recall=0.60, snr=1.9)
    result = check_promotion_gate(
        baseline,
        candidate,
        cross_rank_correlation_floor=0.5,
        cross_rank_correlation=0.6,
    )
    assert result.promoted is True
    assert any("cross_rank_correlation_met" in r for r in result.reasons)


def test_cross_rank_floor_low_blocks_despite_recall_and_snr():
    """Correlation below floor blocks promotion even when recall+SNR pass."""
    baseline = _suite("review", recall=0.50, snr=2.0)
    candidate = _suite("review-with-tools", recall=0.60, snr=1.9)
    result = check_promotion_gate(
        baseline,
        candidate,
        cross_rank_correlation_floor=0.5,
        cross_rank_correlation=0.3,
    )
    assert result.promoted is False
    assert any("cross_rank_correlation_low" in r for r in result.reasons)


