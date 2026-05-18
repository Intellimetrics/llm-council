"""Synthesis chair integration with the FindingMatrix (Phase F).

Covers the optional ``finding_matrix=`` parameter on
``build_synthesis_prompt`` and ``run_synthesis_chair``: backward-compat
when omitted/empty, and the appended block when non-empty.
"""

from __future__ import annotations

from llm_council.adapters import ParticipantResult
from llm_council.citations import VerifiedRef
from llm_council.findings import (
    Finding,
    FindingMatrix,
    cluster_findings,
)
from llm_council.synthesis import build_synthesis_prompt


def _ok_result(name: str, label: str = "yes") -> ParticipantResult:
    return ParticipantResult(
        name=name,
        ok=True,
        model="test",
        elapsed_seconds=1.0,
        command=None,
        output=f"RECOMMENDATION: {label} - looks fine\n\nrationale here",
        error="",
    )


# --- build_synthesis_prompt backward-compat -----------------------------


def test_build_synthesis_prompt_matrix_none_unchanged():
    """When finding_matrix is None, the prompt shape matches v0.7.x."""
    results = [_ok_result("claude"), _ok_result("codex")]
    prompt_legacy = build_synthesis_prompt("Q?", results, convergence=None)
    prompt_none = build_synthesis_prompt(
        "Q?", results, convergence=None, finding_matrix=None
    )
    assert prompt_legacy == prompt_none
    assert "Finding matrix" not in prompt_none
    assert "CONSENSUS BLOCKERS" not in prompt_none
    assert "SINGLE-PEER CONCERNS" not in prompt_none


def test_build_synthesis_prompt_matrix_empty_unchanged():
    """An empty FindingMatrix is treated as None — no appended block."""
    results = [_ok_result("claude"), _ok_result("codex")]
    empty = FindingMatrix()
    prompt_legacy = build_synthesis_prompt("Q?", results, convergence=None)
    prompt_empty = build_synthesis_prompt(
        "Q?", results, convergence=None, finding_matrix=empty
    )
    assert prompt_legacy == prompt_empty
    assert "Finding matrix" not in prompt_empty


# --- build_synthesis_prompt with non-empty matrix -----------------------


def _consensus_matrix() -> FindingMatrix:
    findings_by_peer = {
        "claude": [
            Finding(
                id="F1",
                peer="claude",
                severity="blocker",
                claim="tenant filter missing on user query",
                verified_ref=VerifiedRef(
                    path="src/auth/middleware.py", start_line=42, end_line=58
                ),
                verified=True,
            ),
        ],
        "codex": [
            Finding(
                id="F1",
                peer="codex",
                severity="blocker",
                claim="auth bypass via tenant id",
                verified_ref=VerifiedRef(
                    path="src/auth/middleware.py", start_line=50, end_line=65
                ),
                verified=True,
            ),
        ],
        "gemini": [
            Finding(
                id="G1",
                peer="gemini",
                severity="medium",
                claim="missing rollback for new migration",
                verified_ref=VerifiedRef(
                    path="migrations/20260515_roles.sql", start_line=1, end_line=20
                ),
                verified=True,
            ),
        ],
    }
    return cluster_findings(findings_by_peer)


def test_build_synthesis_prompt_includes_consensus_block():
    matrix = _consensus_matrix()
    results = [_ok_result("claude"), _ok_result("codex"), _ok_result("gemini")]
    prompt = build_synthesis_prompt(
        "Q?", results, convergence=None, finding_matrix=matrix
    )
    assert "CONSENSUS BLOCKERS" in prompt
    assert "SINGLE-PEER CONCERNS" in prompt
    # Consensus cluster names both contributing peers and the verified location.
    assert "claude" in prompt and "codex" in prompt
    assert "src/auth/middleware.py:42-65" in prompt
    # Single-peer concern shows the peer name + claim.
    assert "gemini" in prompt
    assert "missing rollback for new migration" in prompt


def test_build_synthesis_prompt_consensus_cluster_lists_all_peers():
    """A 3-peer cluster names all of them, in order."""
    findings_by_peer = {
        "claude": [
            Finding(
                id="F1", peer="claude", severity="blocker", claim="bug",
                verified_ref=VerifiedRef(path="x.py", start_line=10, end_line=30),
                verified=True,
            )
        ],
        "codex": [
            Finding(
                id="F1", peer="codex", severity="blocker", claim="bug",
                verified_ref=VerifiedRef(path="x.py", start_line=20, end_line=40),
                verified=True,
            )
        ],
        "gemini": [
            Finding(
                id="F1", peer="gemini", severity="medium", claim="bug",
                verified_ref=VerifiedRef(path="x.py", start_line=35, end_line=50),
                verified=True,
            )
        ],
    }
    matrix = cluster_findings(findings_by_peer)
    results = [_ok_result("claude"), _ok_result("codex"), _ok_result("gemini")]
    prompt = build_synthesis_prompt(
        "Q?", results, convergence=None, finding_matrix=matrix
    )
    assert "claude" in prompt
    assert "codex" in prompt
    assert "gemini" in prompt
    # Most-severe-wins: cluster severity is blocker even though gemini said medium.
    assert "[blocker]" in prompt


def test_build_synthesis_prompt_single_peer_unverified_marker():
    findings_by_peer = {
        "codex": [
            Finding(
                id="N1",
                peer="codex",
                severity="nit",
                claim="consider extracting helper for repeated regex",
                verified_ref=None,
            ),
        ],
    }
    matrix = cluster_findings(findings_by_peer)
    results = [_ok_result("codex")]
    prompt = build_synthesis_prompt(
        "Q?", results, convergence=None, finding_matrix=matrix
    )
    assert "SINGLE-PEER CONCERNS" in prompt
    assert "codex" in prompt
    assert "unverified" in prompt
    assert "consider extracting helper" in prompt
