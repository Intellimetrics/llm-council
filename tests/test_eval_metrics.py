"""Eval metric math (Phase B — v0.8 plan).

Pure-function tests. No I/O, no subprocess, no orchestrator. The whole
file should complete in well under a second.
"""

from __future__ import annotations


from llm_council.eval.metrics import (
    MATCH_JACCARD_THRESHOLD,
    blocker_recall,
    citation_accuracy,
    compute_all,
    evidence_density,
    false_blocker_rate,
    signal_to_noise_ratio,
)


# --- blocker_recall ------------------------------------------------------


def test_blocker_recall_all_matched_returns_1():
    expected = [
        {"path": "src/auth/middleware.py", "severity": "blocker", "claim": "missing tenant filter"},
        {"path": "src/db/session.py", "severity": "blocker", "claim": "leaks connection"},
    ]
    emitted = [
        "src/auth/middleware.py: tenant filter is gone — cross-tenant leak",
        "session.py leaks a connection on the error path",
    ]
    assert blocker_recall(emitted, expected) == 1.0


def test_blocker_recall_half_matched_returns_0_5():
    expected = [
        {"path": "src/auth/middleware.py", "severity": "blocker", "claim": "missing tenant filter"},
        {"path": "src/db/session.py", "severity": "blocker", "claim": "leaks connection"},
    ]
    emitted = [
        "middleware.py is missing the tenant filter",
        # Second emitted is unrelated noise — no path match, no claim overlap.
        "function names are inconsistent across the codebase",
    ]
    assert blocker_recall(emitted, expected) == 0.5


def test_blocker_recall_empty_emitted_returns_zero():
    expected = [
        {"path": "src/foo.py", "severity": "blocker", "claim": "panic"},
    ]
    assert blocker_recall([], expected) == 0.0


def test_blocker_recall_empty_expected_is_vacuous_one():
    # No expected → can't miss anything.
    assert blocker_recall(["any text"], []) == 1.0


# --- false_blocker_rate --------------------------------------------------


def test_false_blocker_rate_all_match_returns_zero():
    expected = [
        {"path": "src/auth/middleware.py", "severity": "blocker", "claim": "missing tenant filter"},
    ]
    emitted = [
        "src/auth/middleware.py drops the tenant_id filter",
    ]
    assert false_blocker_rate(emitted, expected) == 0.0


def test_false_blocker_rate_half_unmatched_returns_0_5():
    expected = [
        {"path": "src/auth/middleware.py", "severity": "blocker", "claim": "missing tenant filter"},
    ]
    emitted = [
        "src/auth/middleware.py drops the tenant filter",
        "unrelated style nit about naming conventions",
    ]
    assert false_blocker_rate(emitted, expected) == 0.5


def test_false_blocker_rate_empty_emitted_returns_zero():
    assert false_blocker_rate([], [{"path": "x", "claim": "y"}]) == 0.0


# --- citation_accuracy ---------------------------------------------------


def test_citation_accuracy_all_verified_true_returns_one():
    evidence = [
        {"text": "cite a", "tag": "verified", "verified": True},
        {"text": "cite b", "tag": "verified", "verified": True},
    ]
    assert citation_accuracy(evidence) == 1.0


def test_citation_accuracy_all_verified_false_returns_zero():
    evidence = [
        {"text": "cite a", "tag": "verified", "verified": False},
        {"text": "cite b", "tag": "verified", "verified": False},
    ]
    assert citation_accuracy(evidence) == 0.0


def test_citation_accuracy_no_verified_entries_returns_none():
    """No VERIFIED entries at all → no signal, not zero."""
    evidence = [
        {"text": "Bai 2022", "tag": "published"},
        {"text": "untagged blob", "tag": None},
    ]
    assert citation_accuracy(evidence) is None


def test_citation_accuracy_ignores_non_verified_in_denominator():
    """[PUBLISHED]/[INFERRED]/untagged entries do not affect the verified ratio."""
    evidence = [
        {"text": "cite a", "tag": "verified", "verified": True},
        {"text": "cite b", "tag": "verified", "verified": False},
        {"text": "Bai 2022", "tag": "published"},  # ignored
        {"text": "untagged", "tag": None},          # ignored
    ]
    assert citation_accuracy(evidence) == 0.5


def test_citation_accuracy_empty_list_returns_none():
    assert citation_accuracy([]) is None


def test_citation_accuracy_verified_none_does_not_count_as_true():
    """A VERIFIED entry whose `verified` field is None (not yet checked)
    counts as a 'not True' in the numerator — only True passes."""
    evidence = [
        {"text": "cite a", "tag": "verified", "verified": None},
        {"text": "cite b", "tag": "verified", "verified": True},
    ]
    assert citation_accuracy(evidence) == 0.5


# --- evidence_density ----------------------------------------------------


def test_evidence_density_non_zero_findings():
    evidence = [{}, {}, {}]
    assert evidence_density(evidence, findings_count=2) == 1.5


def test_evidence_density_zero_findings_uses_max_1():
    """Don't divide by zero when a peer emits evidence but no blockers."""
    evidence = [{}, {}]
    assert evidence_density(evidence, findings_count=0) == 2.0


def test_evidence_density_empty_returns_zero():
    assert evidence_density([], findings_count=5) == 0.0


# --- signal_to_noise_ratio -----------------------------------------------


def test_signal_to_noise_ratio_reflects_emitted_denominator():
    """SNR uses the emitted count in the denominator (CR-Bench convention),
    distinct from blocker_recall which uses the expected count."""
    expected = [
        {"path": "src/foo.py", "severity": "blocker", "claim": "races on counter"},
    ]
    emitted = [
        "src/foo.py: a race on the counter",
        "noise nit 1",
        "noise nit 2",
        "noise nit 3",
    ]
    # 1 true positive / 4 emitted = 0.25 — distinct from recall=1.0.
    assert signal_to_noise_ratio(emitted, expected) == 0.25
    assert blocker_recall(emitted, expected) == 1.0


def test_signal_to_noise_ratio_empty_emitted_returns_zero():
    assert signal_to_noise_ratio([], [{"path": "x", "claim": "y"}]) == 0.0


def test_signal_to_noise_ratio_all_noise_returns_zero():
    expected = [{"path": "src/foo.py", "claim": "race"}]
    emitted = ["unrelated", "stylistic", "tangential"]
    assert signal_to_noise_ratio(emitted, expected) == 0.0


# --- Jaccard match path --------------------------------------------------


def test_jaccard_match_when_words_overlap():
    """Different surface order but high word overlap → match."""
    expected = [{"path": "", "severity": "blocker", "claim": "missing tenant filter"}]
    emitted = ["tenant filter is missing on the user query"]
    # `tokenize` drops stopwords; the tokens {missing, tenant, filter} vs
    # {tenant, filter, missing, user, query} = 3 / 5 = 0.6 >= threshold (0.4).
    assert blocker_recall(emitted, expected) == 1.0


def test_jaccard_match_below_threshold_does_not_match():
    """Two unrelated review blockers sharing one word should not match."""
    expected = [{"path": "", "severity": "blocker", "claim": "missing tenant filter on user query"}]
    emitted = ["the variable naming is inconsistent here"]
    # No path token, no claim token overlap → no match.
    assert blocker_recall(emitted, expected) == 0.0


def test_jaccard_threshold_value_reasonable():
    """Guard: the threshold should be strict enough to require a real
    overlap (~half the words), not just one shared word."""
    assert 0.3 < MATCH_JACCARD_THRESHOLD <= 0.6


# --- path-token match path -----------------------------------------------


def test_path_basename_token_matches_emitted_text():
    """A bare 'middleware' in emitted text should match path 'src/auth/middleware.py'."""
    expected = [
        {"path": "src/auth/middleware.py", "severity": "blocker", "claim": "uncited"}
    ]
    emitted = ["The middleware skips tenant scoping in the new branch"]
    assert blocker_recall(emitted, expected) == 1.0


def test_path_short_token_does_not_match_random_text():
    """Path tokens shorter than 3 chars don't trip a match (defends
    against 'db' colliding with random prose)."""
    expected = [
        {"path": "db.py", "severity": "blocker", "claim": "no overlap with this"}
    ]
    # 'db' is 2 chars — too short to trigger a path-token match. The
    # claim has no word overlap with the emitted text either, so this
    # should NOT match.
    emitted = ["random unrelated nit about formatting style"]
    assert blocker_recall(emitted, expected) == 0.0


# --- compute_all aggregator ----------------------------------------------


def test_compute_all_returns_full_block():
    """End-to-end: synthesize one good peer and one bad peer scenario,
    confirm the aggregator surface matches the per-function results."""
    expected = [
        {"path": "src/auth/middleware.py", "severity": "blocker", "claim": "missing tenant filter"},
    ]
    emitted = [
        "src/auth/middleware.py: missing tenant_id filter on the lookup",
        "stylistic concern about naming",
    ]
    evidence = [
        {"text": "cite", "tag": "verified", "verified": True},
        {"text": "Bai 2022", "tag": "published"},
    ]
    out = compute_all(
        emitted_blockers=emitted,
        expected_blockers=expected,
        evidence_entries=evidence,
    )
    assert out["blocker_recall"] == 1.0
    assert out["false_blocker_rate"] == 0.5
    assert out["signal_to_noise_ratio"] == 0.5
    assert out["citation_accuracy"] == 1.0
    # 2 evidence / max(1, 2 blockers) = 1.0
    assert out["evidence_density"] == 1.0


def test_compute_all_handles_legacy_string_evidence():
    """Pre-v0.7 evidence is `list[str]`. Strings can't carry a VERIFIED
    tag so citation_accuracy is None, but evidence_density still works."""
    out = compute_all(
        emitted_blockers=[],
        expected_blockers=[],
        evidence_entries=["legacy", "string", "entries"],
    )
    assert out["citation_accuracy"] is None
    assert out["evidence_density"] == 3.0
