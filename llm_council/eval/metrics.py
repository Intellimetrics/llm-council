"""Eval metrics for council scorecards (Phase B — v0.8 plan).

Pure functions that operate on synthesized inputs — DO NOT import the
runner or any fixture loader. Each function takes only the data it
needs explicitly so tests can exercise the math without spinning up
the orchestrator or hitting disk.

Input shapes:

- emitted_blockers: list[str]
    The `blockers` envelope list from a `ParticipantResult`. Plain
    strings (no tag parsing) per the v0.7 envelope contract.

- expected_blockers: list[dict]
    Truth labels from a fixture's `expected_blockers.json`:
        [{"path": "src/foo.py",
          "severity": "blocker",
          "claim": "missing tenant filter"}, ...]

- evidence_entries: list[dict]
    The `evidence` envelope list from a `ParticipantResult`. Each entry
    is `{text, tag, ...}` where tag is one of None | "published" |
    "observable" | "inferred" | "speculative" | "verified". VERIFIED
    entries carry `verified: True|False|None` after Phase A's
    orchestrator pass.

Definitions (CR-Bench convention):

- blocker_recall    = matched_expected / total_expected
- false_blocker_rate = unmatched_emitted / total_emitted
- citation_accuracy = verified_true / total_verified  (None if no VERIFIED)
- evidence_density  = len(evidence) / max(1, findings_count)
- signal_to_noise_ratio = true_positive_blockers / max(1, total_emitted)

Match an emitted blocker text to an expected blocker by EITHER:
  (a) the expected file-path token appearing inside the emitted text, OR
  (b) Jaccard token overlap >= MATCH_JACCARD_THRESHOLD on the lowercase
      stopword-filtered token sets of the emitted text and the expected
      `claim`.

Reuses `llm_council.convergence.tokenize` so we share one stopword
list / tokenization regime with the deliberation convergence detector.
"""

from __future__ import annotations

from typing import Any, Iterable

from llm_council.convergence import tokenize

# Threshold for claim-text overlap. Tuned for the kind of short claim
# strings fixtures carry (e.g. "missing tenant filter" vs "tenant filter
# is missing on the user query" — Jaccard ~ 0.5). Anything below this
# is too loose: two unrelated review blockers can share 1 word out of 4.
MATCH_JACCARD_THRESHOLD = 0.4


def _path_tokens(path: str) -> set[str]:
    """Salient path components ignoring extensions / common separators."""
    if not path:
        return set()
    # Strip extension off the leaf so "middleware.py" → "middleware".
    parts: list[str] = []
    for chunk in path.replace("\\", "/").split("/"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "." in chunk:
            chunk = chunk.rsplit(".", 1)[0]
        if chunk:
            parts.append(chunk.lower())
    return {p for p in parts if p}


def _emitted_matches_expected(emitted: str, expected: dict[str, Any]) -> bool:
    """True iff `emitted` plausibly refers to the same finding as `expected`.

    Match by file-path token OR Jaccard claim-text overlap.
    """
    emitted_text = (emitted or "").strip()
    if not emitted_text:
        return False

    # (a) file-path token match
    path = str(expected.get("path") or "")
    if path:
        # Cheap substring check first — exact path appears in the prose.
        if path in emitted_text:
            return True
        # Otherwise check salient path tokens (basename without ext).
        emitted_lower = emitted_text.lower()
        for tok in _path_tokens(path):
            if len(tok) >= 3 and tok in emitted_lower:
                return True

    # (b) Jaccard token overlap on claim text
    claim = str(expected.get("claim") or "")
    if claim:
        emitted_tokens = tokenize(emitted_text)
        claim_tokens = tokenize(claim)
        if emitted_tokens and claim_tokens:
            inter = len(emitted_tokens & claim_tokens)
            union = len(emitted_tokens | claim_tokens)
            if union > 0 and (inter / union) >= MATCH_JACCARD_THRESHOLD:
                return True

    return False


def _matched_indices(
    emitted_blockers: list[str],
    expected_blockers: list[dict[str, Any]],
) -> tuple[set[int], set[int]]:
    """Greedy bipartite-ish match.

    Returns (matched_emitted_indices, matched_expected_indices). Each
    emitted entry can match at most one expected (and vice versa) so
    duplicate-emit doesn't artificially inflate recall.
    """
    matched_emitted: set[int] = set()
    matched_expected: set[int] = set()
    for ei, emitted in enumerate(emitted_blockers or []):
        for xi, expected in enumerate(expected_blockers or []):
            if xi in matched_expected:
                continue
            if _emitted_matches_expected(emitted, expected):
                matched_emitted.add(ei)
                matched_expected.add(xi)
                break
    return matched_emitted, matched_expected


def blocker_recall(
    emitted_blockers: list[str],
    expected_blockers: list[dict[str, Any]],
) -> float:
    """Fraction of expected blockers that were caught by an emitted entry.

    Empty `expected_blockers` returns 1.0 (vacuous — nothing to miss).
    """
    if not expected_blockers:
        return 1.0
    _, matched_expected = _matched_indices(emitted_blockers, expected_blockers)
    return len(matched_expected) / len(expected_blockers)


def false_blocker_rate(
    emitted_blockers: list[str],
    expected_blockers: list[dict[str, Any]],
) -> float:
    """Fraction of emitted blockers that did NOT match any expected entry.

    Empty `emitted_blockers` returns 0.0 (no false alarms emitted).
    """
    if not emitted_blockers:
        return 0.0
    matched_emitted, _ = _matched_indices(emitted_blockers, expected_blockers)
    unmatched = len(emitted_blockers) - len(matched_emitted)
    return unmatched / len(emitted_blockers)


def citation_accuracy(evidence_entries: list[dict[str, Any]]) -> float | None:
    """Verified-citation accuracy among the VERIFIED-tagged entries.

    Returns None when there are no VERIFIED entries (no signal, not zero).
    Non-VERIFIED tags ([PUBLISHED], [INFERRED], untagged, …) do not enter
    the denominator — they are a different evidence shape entirely and
    are not mechanically verifiable.
    """
    verified_entries = [
        e for e in (evidence_entries or [])
        if isinstance(e, dict) and e.get("tag") == "verified"
    ]
    if not verified_entries:
        return None
    hits = sum(1 for e in verified_entries if e.get("verified") is True)
    return hits / len(verified_entries)


def evidence_density(
    evidence_entries: list[dict[str, Any]] | list[Any],
    findings_count: int,
) -> float:
    """Evidence entries per finding. Findings_count uses `max(1, n)` to
    avoid div-by-zero when a peer emits evidence but no blockers."""
    n_evidence = len(evidence_entries or [])
    denom = max(1, int(findings_count))
    return n_evidence / denom


def signal_to_noise_ratio(
    emitted_blockers: list[str],
    expected_blockers: list[dict[str, Any]],
) -> float:
    """CR-Bench SNR: true-positive blockers / total emitted blockers.

    Distinct from `blocker_recall` in the denominator: recall measures
    "how many expected did we catch", SNR measures "of what we said,
    how much was right". Empty emitted returns 0.0 (no signal at all).
    """
    if not emitted_blockers:
        return 0.0
    matched_emitted, _ = _matched_indices(emitted_blockers, expected_blockers)
    return len(matched_emitted) / max(1, len(emitted_blockers))


# Convenience aggregator used by the runner. Not part of the brief's
# minimum surface but the runner needs SOMETHING to call uniformly per
# participant, and forcing the runner to re-derive `matched_indices`
# duplicates work.
def compute_all(
    *,
    emitted_blockers: list[str],
    expected_blockers: list[dict[str, Any]],
    evidence_entries: list[Any],
) -> dict[str, Any]:
    """One pass that returns the full metric block for a peer."""
    # Normalize evidence to a list of dicts for citation_accuracy. Plain
    # strings (legacy/untagged) cannot have a VERIFIED tag so they are
    # filtered out — they would be counted as non-verified and skipped.
    evidence_dicts = [e for e in (evidence_entries or []) if isinstance(e, dict)]
    return {
        "blocker_recall": blocker_recall(emitted_blockers, expected_blockers),
        "false_blocker_rate": false_blocker_rate(emitted_blockers, expected_blockers),
        "citation_accuracy": citation_accuracy(evidence_dicts),
        "evidence_density": evidence_density(
            evidence_entries or [], len(emitted_blockers or [])
        ),
        "signal_to_noise_ratio": signal_to_noise_ratio(
            emitted_blockers, expected_blockers
        ),
    }


__all__ = [
    "MATCH_JACCARD_THRESHOLD",
    "blocker_recall",
    "citation_accuracy",
    "compute_all",
    "evidence_density",
    "false_blocker_rate",
    "signal_to_noise_ratio",
]
