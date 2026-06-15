"""Per-finding agreement matrix tests (Phase F — v0.8 plan).

Covers extraction of the FINDINGS envelope block, mechanical clustering
across peers by overlapping verified line ranges, and the
serialization shape used in transcripts / MCP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_council.citations import VerifiedRef
from llm_council.findings import (
    Finding,
    FindingCluster,
    FindingMatrix,
    cluster_findings,
    extract_findings,
    matrix_to_dict,
)


# --- extract_findings ---------------------------------------------------


def test_extract_findings_well_formed():
    text = """
RECOMMENDATION: no - several issues.

FINDINGS:
- id: F1
  severity: blocker
  claim: tenant filter missing on user query
  evidence: [VERIFIED:src/auth/middleware.py:42-58]
- id: F2
  severity: medium
  claim: missing rollback for new migration
  evidence: [VERIFIED:migrations/20260515_roles.sql:1-20]
"""
    out = extract_findings(text, "claude")
    assert len(out) == 2
    assert out[0].id == "F1"
    assert out[0].peer == "claude"
    assert out[0].severity == "blocker"
    assert out[0].claim == "tenant filter missing on user query"
    assert out[0].verified_ref == VerifiedRef(
        path="src/auth/middleware.py", start_line=42, end_line=58
    )
    assert out[1].id == "F2"
    assert out[1].severity == "medium"
    assert out[1].verified_ref.path == "migrations/20260515_roles.sql"


def test_extract_findings_empty_when_no_section():
    assert extract_findings("RECOMMENDATION: yes - lgtm", "codex") == []


def test_extract_findings_empty_input():
    assert extract_findings("", "codex") == []
    assert extract_findings(None, "codex") == []  # type: ignore[arg-type]


def test_extract_findings_tolerates_indentation_variants():
    text = """
FINDINGS:
  - id: F1
    severity: blocker
    claim: nested indent works
    evidence: [VERIFIED:a.py:1-5]
- id: F2
  severity: nit
  claim: also fine at zero indent
"""
    out = extract_findings(text, "gemini")
    assert {f.id for f in out} == {"F1", "F2"}
    f1 = next(f for f in out if f.id == "F1")
    assert f1.verified_ref is not None
    assert f1.verified_ref.path == "a.py"


def test_extract_findings_missing_optional_fields():
    """A finding with only id+claim still materializes; severity defaults."""
    text = """
FINDINGS:
- id: F1
  claim: only the bare minimum
"""
    out = extract_findings(text, "claude")
    assert len(out) == 1
    assert out[0].claim == "only the bare minimum"
    assert out[0].severity == "nit"
    assert out[0].verified_ref is None


def test_extract_findings_skips_entries_without_claim():
    text = """
FINDINGS:
- id: F1
  severity: blocker
- id: F2
  severity: medium
  claim: this one has a claim
"""
    out = extract_findings(text, "codex")
    assert len(out) == 1
    assert out[0].id == "F2"


def test_extract_findings_normalises_unknown_severity():
    text = """
FINDINGS:
- id: F1
  severity: SCREAMING_LOUD
  claim: severity gets normalized
"""
    out = extract_findings(text, "claude")
    assert out[0].severity == "nit"


def test_extract_findings_ignores_fenced_block():
    """A FINDINGS block inside a fence is example code, not a real entry."""
    text = """
```
FINDINGS:
- id: F1
  severity: blocker
  claim: should not be parsed
```

RECOMMENDATION: yes - lgtm
"""
    out = extract_findings(text, "claude")
    assert out == []


def test_extract_findings_terminator_envelope_header_ends_block():
    text = """
FINDINGS:
- id: F1
  severity: blocker
  claim: first finding
TESTS_TO_RUN:
- pytest -q
"""
    out = extract_findings(text, "claude")
    assert len(out) == 1
    assert out[0].claim == "first finding"


def test_extract_findings_assigns_default_id():
    """A claim without an explicit id is skipped (id is required to start a bullet)."""
    text = """
FINDINGS:
  claim: orphan
"""
    out = extract_findings(text, "claude")
    assert out == []


# --- cluster_findings ----------------------------------------------------


def _make_finding(
    peer: str,
    fid: str,
    severity: str,
    path: str,
    lo: int,
    hi: int,
    *,
    verified: bool | None = True,
) -> Finding:
    return Finding(
        id=fid,
        peer=peer,
        severity=severity,
        claim=f"{peer} flagged {fid}",
        verified_ref=VerifiedRef(path=path, start_line=lo, end_line=hi),
        verified=verified,
    )


def test_cluster_findings_consensus_two_peers_overlapping():
    findings_by_peer = {
        "claude": [_make_finding("claude", "F1", "blocker", "auth.py", 40, 60)],
        "codex": [_make_finding("codex", "F1", "blocker", "auth.py", 50, 70)],
    }
    matrix = cluster_findings(findings_by_peer)
    # Consensus is implicit: any item in `matrix.clusters` came from
    # >=2 distinct peers. The old `cluster.consensus` boolean was
    # always True for that reason and has been removed.
    assert len(matrix.clusters) == 1
    cluster = matrix.clusters[0]
    assert set(cluster.peers) == {"claude", "codex"}
    assert cluster.severity == "blocker"
    assert cluster.verified_path == "auth.py"
    assert cluster.verified_range == (40, 70)
    assert matrix.single_peer_concerns == []


def test_cluster_findings_single_peer_verified_goes_to_concerns():
    findings_by_peer = {
        "gemini": [_make_finding("gemini", "F1", "medium", "x.py", 10, 20)],
    }
    matrix = cluster_findings(findings_by_peer)
    assert matrix.clusters == []
    assert len(matrix.single_peer_concerns) == 1
    assert matrix.single_peer_concerns[0].peer == "gemini"


def test_cluster_findings_unverified_findings_never_cluster():
    """No verified_ref -> single-peer concern even when claims overlap."""
    f1 = Finding(
        id="F1",
        peer="claude",
        severity="blocker",
        claim="prose-only finding",
        verified_ref=None,
    )
    f2 = Finding(
        id="F2",
        peer="codex",
        severity="blocker",
        claim="another prose-only finding",
        verified_ref=None,
    )
    matrix = cluster_findings({"claude": [f1], "codex": [f2]})
    assert matrix.clusters == []
    assert len(matrix.single_peer_concerns) == 2


def test_cluster_findings_does_not_cluster_when_verified_false():
    """A peer's VERIFIED tag that failed verification cannot anchor a cluster."""
    findings_by_peer = {
        "claude": [
            _make_finding("claude", "F1", "blocker", "ghost.py", 10, 20, verified=False)
        ],
        "codex": [
            _make_finding("codex", "F1", "blocker", "ghost.py", 12, 25, verified=False)
        ],
    }
    matrix = cluster_findings(findings_by_peer)
    assert matrix.clusters == []
    assert len(matrix.single_peer_concerns) == 2


def test_cluster_findings_cluster_severity_picks_most_severe():
    findings_by_peer = {
        "claude": [_make_finding("claude", "F1", "medium", "x.py", 1, 10)],
        "codex": [_make_finding("codex", "F1", "blocker", "x.py", 5, 15)],
        "gemini": [_make_finding("gemini", "F1", "nit", "x.py", 8, 12)],
    }
    matrix = cluster_findings(findings_by_peer)
    assert len(matrix.clusters) == 1
    cluster = matrix.clusters[0]
    assert cluster.severity == "blocker"
    assert set(cluster.peers) == {"claude", "codex", "gemini"}


def test_cluster_findings_three_overlapping_peers_one_cluster():
    findings_by_peer = {
        "claude": [_make_finding("claude", "F1", "blocker", "x.py", 10, 30)],
        "codex": [_make_finding("codex", "F1", "blocker", "x.py", 20, 40)],
        "gemini": [_make_finding("gemini", "F1", "blocker", "x.py", 35, 50)],
    }
    matrix = cluster_findings(findings_by_peer)
    # claude<->codex overlap, codex<->gemini overlap; merge pass should
    # fold all three into one cluster.
    assert len(matrix.clusters) == 1
    assert set(matrix.clusters[0].peers) == {"claude", "codex", "gemini"}
    assert matrix.clusters[0].verified_range == (10, 50)


def test_cluster_findings_non_overlapping_same_file_separate_clusters():
    findings_by_peer = {
        "claude": [
            _make_finding("claude", "F1", "blocker", "x.py", 1, 10),
            _make_finding("claude", "F2", "medium", "x.py", 100, 110),
        ],
        "codex": [
            _make_finding("codex", "F1", "blocker", "x.py", 5, 12),
            _make_finding("codex", "F2", "medium", "x.py", 105, 115),
        ],
    }
    matrix = cluster_findings(findings_by_peer)
    assert len(matrix.clusters) == 2
    ranges = sorted(c.verified_range for c in matrix.clusters)
    assert ranges == [(1, 12), (100, 115)]


def test_cluster_findings_same_peer_two_findings_no_self_cluster():
    """One peer flagging two overlapping ranges does not produce consensus."""
    findings_by_peer = {
        "claude": [
            _make_finding("claude", "F1", "blocker", "x.py", 1, 10),
            _make_finding("claude", "F2", "blocker", "x.py", 5, 15),
        ],
    }
    matrix = cluster_findings(findings_by_peer)
    # Both findings cluster mechanically (same path, overlapping ranges)
    # but only one peer contributed -> consensus=False -> single_peer_concerns.
    assert matrix.clusters == []
    assert len(matrix.single_peer_concerns) == 2


def test_cluster_findings_empty_input():
    matrix = cluster_findings({})
    assert matrix.clusters == []
    assert matrix.single_peer_concerns == []
    assert matrix.is_empty()


# --- matrix_to_dict round-trip ------------------------------------------


def test_matrix_to_dict_consensus_shape_stable():
    findings_by_peer = {
        "claude": [_make_finding("claude", "F1", "blocker", "auth.py", 40, 60)],
        "codex": [_make_finding("codex", "F1", "blocker", "auth.py", 50, 70)],
    }
    matrix = cluster_findings(findings_by_peer)
    payload = matrix_to_dict(matrix)
    assert payload["consensus_blockers"]
    entry = payload["consensus_blockers"][0]
    assert entry["severity"] == "blocker"
    assert set(entry["peers"]) == {"claude", "codex"}
    assert entry["path"] == "auth.py"
    assert entry["start_line"] == 40
    assert entry["end_line"] == 70
    assert "claim" in entry
    assert payload["single_peer_concerns"] == []


def test_matrix_to_dict_single_peer_shape_stable():
    f = Finding(
        id="F1",
        peer="gemini",
        severity="medium",
        claim="prose-only finding",
        verified_ref=None,
    )
    matrix = cluster_findings({"gemini": [f]})
    payload = matrix_to_dict(matrix)
    assert payload["consensus_blockers"] == []
    assert len(payload["single_peer_concerns"]) == 1
    entry = payload["single_peer_concerns"][0]
    assert entry["peer"] == "gemini"
    assert entry["claim"] == "prose-only finding"
    assert entry.get("unverified") is True
    assert "path" not in entry


def test_matrix_to_dict_unverified_flag_for_failed_verification():
    f = _make_finding("claude", "F1", "blocker", "ghost.py", 1, 5, verified=False)
    matrix = cluster_findings({"claude": [f]})
    payload = matrix_to_dict(matrix)
    assert payload["consensus_blockers"] == []
    entry = payload["single_peer_concerns"][0]
    assert entry["path"] == "ghost.py"
    assert entry.get("unverified") is True


# --- matrix_to_dict three-tier gating partition (L4) --------------------


def test_matrix_to_dict_gating_partitions_by_severity():
    """Consensus clusters partition by severity: blocker -> blocking,
    medium -> non_blocking, nit -> suggestion. All single-peer concerns
    are suggestion-tier, alongside the nit consensus cluster."""
    findings_by_peer = {
        # blocker consensus cluster (auth.py)
        "claude": [
            _make_finding("claude", "B1", "blocker", "auth.py", 40, 60),
            _make_finding("claude", "M1", "medium", "db.py", 5, 15),
            _make_finding("claude", "N1", "nit", "fmt.py", 1, 3),
        ],
        "codex": [
            _make_finding("codex", "B1", "blocker", "auth.py", 50, 70),
            _make_finding("codex", "M1", "medium", "db.py", 8, 20),
            _make_finding("codex", "N1", "nit", "fmt.py", 2, 4),
        ],
        # single-peer concern (no corroboration)
        "gemini": [_make_finding("gemini", "S1", "blocker", "lonely.py", 1, 9)],
    }
    matrix = cluster_findings(findings_by_peer)
    payload = matrix_to_dict(matrix)

    gating = payload["gating"]
    # Severity of each cluster, indexed by the partition it lands in.
    assert [e["severity"] for e in gating["blocking"]] == ["blocker"]
    assert [e["severity"] for e in gating["non_blocking"]] == ["medium"]

    # suggestion holds the nit consensus cluster PLUS the single-peer concern.
    suggestion_severities = [e["severity"] for e in gating["suggestion"]]
    assert "nit" in suggestion_severities
    assert len(gating["suggestion"]) == 2  # nit cluster + single-peer concern
    # The single-peer concern (by peer key) is present in suggestion.
    assert any(e.get("peer") == "gemini" for e in gating["suggestion"])

    # Partition is exhaustive: every consensus blocker landed in exactly
    # one of blocking/non_blocking/suggestion, and the single-peer concern
    # landed in suggestion only.
    total = (
        len(gating["blocking"])
        + len(gating["non_blocking"])
        + len(gating["suggestion"])
    )
    assert total == len(payload["consensus_blockers"]) + len(
        payload["single_peer_concerns"]
    )

    # Entries are referenced, not copied — same object identity.
    assert gating["blocking"][0] is payload["consensus_blockers"][0]


def test_matrix_to_dict_gating_always_present_and_empty():
    """An empty matrix still yields a gating block with three empty lists,
    mirroring the always-present-keys design of the consensus fields."""
    payload = matrix_to_dict(FindingMatrix())
    assert payload["gating"] == {
        "blocking": [],
        "non_blocking": [],
        "suggestion": [],
    }
    assert payload["consensus_blockers"] == []
    assert payload["single_peer_concerns"] == []


# --- build_matrix_from_results integration ------------------------------


@dataclass
class _StubResult:
    name: str
    output: str = ""
    evidence_verification_failures: list[str] = field(default_factory=list)


def test_build_matrix_from_results_strips_round_suffix(tmp_path):
    from llm_council.findings import build_matrix_from_results

    text = """
FINDINGS:
- id: F1
  severity: blocker
  claim: same bug from claude
  evidence: [VERIFIED:auth.py:10-30]
"""
    text2 = """
FINDINGS:
- id: F1
  severity: blocker
  claim: same bug from codex
  evidence: [VERIFIED:auth.py:20-40]
"""
    results = [
        _StubResult(name="claude:round2", output=text),
        _StubResult(name="codex:round2", output=text2),
    ]
    matrix = build_matrix_from_results(results)
    assert len(matrix.clusters) == 1
    assert set(matrix.clusters[0].peers) == {"claude", "codex"}


def test_build_matrix_from_results_honors_evidence_failures():
    """When the result's evidence_verification_failures list contains the
    verified key, the finding cannot anchor a cluster."""
    from llm_council.findings import build_matrix_from_results

    text_claude = """
FINDINGS:
- id: F1
  severity: blocker
  claim: ghost cite
  evidence: [VERIFIED:ghost.py:10-30]
"""
    text_codex = """
FINDINGS:
- id: F1
  severity: blocker
  claim: same ghost
  evidence: [VERIFIED:ghost.py:20-40]
"""
    results = [
        _StubResult(
            name="claude",
            output=text_claude,
            evidence_verification_failures=["ghost.py:10-30"],
        ),
        _StubResult(
            name="codex",
            output=text_codex,
            evidence_verification_failures=["ghost.py:20-40"],
        ),
    ]
    matrix = build_matrix_from_results(results)
    assert matrix.clusters == []
    assert len(matrix.single_peer_concerns) == 2
