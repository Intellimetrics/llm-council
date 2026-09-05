"""Per-finding agreement matrix (Phase F — v0.8 plan).

Peers may emit a structured ``FINDINGS:`` envelope section listing specific
issues with severity and (preferably) a ``[VERIFIED:path:start-end]`` evidence
ref. After all peer results return, the orchestrator clusters findings
across peers by overlapping verified line ranges + severity class. The
synthesis chair uses the cluster to distinguish consensus blockers
(>= 2 peers, overlapping verified ranges) from single-peer concerns.

Mechanical clustering only — no fuzzy prose matching. Unverified findings
(no VERIFIED tag, or VERIFIED with verified=False) are reported as
single-peer entries and do not contribute to consensus clusters.

The matrix is NOT fed back to peers during deliberation rounds. Pre-MAD
work shows that forcing convergence in-round depresses signal-to-noise.
This module is consumed by synthesis only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from llm_council.citations import (
    CitationVerifier,
    VerifiedRef,
    parse_verified_tag,
    strip_verified_tag,
)


# Severity ordering: blocker (most severe) < medium < nit. Lower numeric
# rank means more severe; the cluster's representative severity is the
# lowest rank across members.
_SEVERITY_RANK = {"blocker": 0, "medium": 1, "nit": 2}
_SEVERITY_DEFAULT = "nit"


@dataclass(frozen=True)
class Finding:
    """One peer-emitted finding parsed from the FINDINGS envelope block.

    ``verified_ref`` is the optional ``[VERIFIED:path:start-end]`` reference.
    ``verified`` is filled in by the orchestrator's existing citation
    verification pass (None when no VERIFIED tag, True/False afterward).
    """

    id: str
    peer: str
    severity: str
    claim: str
    verified_ref: VerifiedRef | None = None
    verified: bool | None = None


@dataclass
class FindingCluster:
    """A cluster of verified findings from >=2 peers (consensus is implicit).

    Single-peer findings route to ``FindingMatrix.single_peer_concerns``
    instead, so every `FindingCluster` instance is by definition a
    consensus cluster. The previous explicit ``consensus`` boolean
    was always ``True`` for any cluster that ever made it into
    ``FindingMatrix.clusters``.
    """

    id: str
    severity: str
    peers: list[str]
    findings: list[Finding]
    verified_path: str | None
    verified_range: tuple[int, int] | None


@dataclass
class FindingMatrix:
    clusters: list[FindingCluster] = field(default_factory=list)
    single_peer_concerns: list[Finding] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.clusters and not self.single_peer_concerns


# --- parsing -------------------------------------------------------------

# Header that opens the FINDINGS block. Mirrors the existing envelope
# header style (`_ENVELOPE_LIST_HEADER_RE` in adapters.py) but lives here
# so this module stays self-contained for the synthesis path.
_FINDINGS_HEADER_RE = re.compile(
    r"""
    ^\s*(?:>\s*)?(?:\*\*)?
    FINDINGS
    (?:\*\*)?\s*:\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A new bullet entry starts with `- id:` (or `* id:` / `+ id:`). The id
# value itself is captured to support arbitrary identifiers (F1, blocker-1,
# etc.).
_FINDING_ENTRY_START_RE = re.compile(
    r"""
    ^\s*[-*+]\s*
    id\s*:\s*(?P<id>\S.*?)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Continuation lines: nested `key: value` under a finding entry. Indentation
# is tolerated rather than required since peers vary.
_FINDING_FIELD_RE = re.compile(
    r"""
    ^\s*[-*+]?\s*
    (?P<key>severity|claim|evidence)
    \s*:\s*(?P<value>.*?)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A fresh top-level header anywhere outside the FINDINGS block terminates
# parsing. Re-use the same set the envelope parser handles plus PART/##
# headings used by the section validator.
_ENVELOPE_TERMINATOR_RE = re.compile(
    r"""
    ^\s*(?:>\s*)?(?:\*\*)?
    (?:EFFORT|CONFIDENCE|RISK|BLOCKERS|EVIDENCE|TESTS_TO_RUN|ASSUMPTIONS|RECOMMENDATION|FINDINGS)
    (?:\*\*)?\s*:
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_severity(raw: str | None) -> str:
    if not raw:
        return _SEVERITY_DEFAULT
    lowered = raw.strip().lower()
    if lowered in _SEVERITY_RANK:
        return lowered
    return _SEVERITY_DEFAULT


def _flush_finding(
    pending: dict[str, Any] | None,
    peer_name: str,
    out: list[Finding],
) -> None:
    """Materialize a pending bullet into a Finding when it has a claim."""
    if not pending:
        return
    claim_raw = (pending.get("claim") or "").strip()
    if not claim_raw:
        # Missing required field — skip silently per the brief.
        return
    evidence_raw = pending.get("evidence") or ""
    verified = parse_verified_tag(evidence_raw)
    # The claim line shouldn't include the VERIFIED token, but strip
    # defensively in case the peer emitted it there instead.
    claim_text = strip_verified_tag(claim_raw) or claim_raw
    out.append(
        Finding(
            id=str(pending.get("id") or "").strip() or f"F{len(out) + 1}",
            peer=peer_name,
            severity=_normalize_severity(pending.get("severity")),
            claim=claim_text,
            verified_ref=verified,
            verified=None if verified is None else False,
        )
    )


def extract_findings(participant_output: str, peer_name: str) -> list[Finding]:
    """Parse the optional FINDINGS block from a participant's output.

    Recognised shape (indentation is flexible)::

        FINDINGS:
        - id: F1
          severity: blocker
          claim: tenant filter missing on user query
          evidence: [VERIFIED:src/auth/middleware.py:42-58]

    Tolerant of:
    - missing optional fields (only ``claim`` is required; without it, the
      entry is silently skipped)
    - varying indentation on continuation lines
    - unknown severity values (normalized to ``nit``)
    - multiple FINDINGS blocks (later blocks append to the list)
    - fenced code blocks (ignored — matches the envelope parser's rule)

    Returns ``[]`` when no FINDINGS section is present.
    """
    if not participant_output:
        return []

    out: list[Finding] = []
    in_fence = False
    in_findings_block = False
    pending: dict[str, Any] | None = None

    for raw_line in participant_output.splitlines():
        stripped = raw_line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            # Fence boundary closes any in-progress entry.
            _flush_finding(pending, peer_name, out)
            pending = None
            in_findings_block = False
            continue
        if in_fence:
            continue

        if _FINDINGS_HEADER_RE.match(raw_line):
            # Flush any prior pending and enter findings mode.
            _flush_finding(pending, peer_name, out)
            pending = None
            in_findings_block = True
            continue

        if not in_findings_block:
            continue

        # Blank line: close current entry; remain in block (an additional
        # blank could indicate end, but tolerate spacing between entries).
        if not stripped:
            _flush_finding(pending, peer_name, out)
            pending = None
            continue

        # An entry-start (`- id: ...`) always re-anchors a finding,
        # even when its `severity:`/`claim:`/`evidence:` continuation
        # lines look like envelope headers (EVIDENCE!). Try entry-start
        # and field-continuation matches BEFORE the envelope terminator
        # check, so an indented `  evidence: [VERIFIED:...]` does not get
        # mis-classified as the start of the EVIDENCE envelope block.
        start = _FINDING_ENTRY_START_RE.match(raw_line)
        if start:
            _flush_finding(pending, peer_name, out)
            pending = {"id": start.group("id").strip()}
            continue

        field_match = _FINDING_FIELD_RE.match(raw_line)
        if field_match and pending is not None:
            key = field_match.group("key").lower()
            value = field_match.group("value").strip()
            # Don't overwrite already-set fields (first wins) to avoid
            # multi-line bullet noise blowing away the parsed value.
            if key not in pending or not pending[key]:
                pending[key] = value
            continue

        # Another envelope header (or RECOMMENDATION) at this point ends
        # the FINDINGS block. Checked AFTER the entry-start / field-
        # continuation matches above so an indented continuation line
        # never closes the block prematurely.
        if _ENVELOPE_TERMINATOR_RE.match(raw_line) and not _FINDINGS_HEADER_RE.match(
            raw_line
        ):
            _flush_finding(pending, peer_name, out)
            pending = None
            in_findings_block = False
            continue

        # Unrecognized line inside FINDINGS: ignore. Don't terminate the
        # block — peers sometimes interleave commentary.

    # End of output: flush any in-flight entry.
    _flush_finding(pending, peer_name, out)
    return out


# --- clustering ----------------------------------------------------------


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True iff two inclusive line ranges intersect."""
    return a[0] <= b[1] and b[0] <= a[1]


def _representative_severity(findings: list[Finding]) -> str:
    """Lowest-rank (most severe) severity across the cluster's members."""
    if not findings:
        return _SEVERITY_DEFAULT
    return min(findings, key=lambda f: _SEVERITY_RANK.get(f.severity, 99)).severity


def _union_range(findings: list[Finding]) -> tuple[int, int] | None:
    """Union the verified line ranges across a cluster's members."""
    ranges = [
        (f.verified_ref.start_line, f.verified_ref.end_line)
        for f in findings
        if f.verified_ref is not None
    ]
    if not ranges:
        return None
    return (min(r[0] for r in ranges), max(r[1] for r in ranges))


def cluster_findings(by_peer: dict[str, list[Finding]]) -> FindingMatrix:
    """Cluster verified findings across peers by overlapping line ranges.

    Algorithm:
    1. Collect all verified findings (``verified_ref`` is not None AND
       ``verified`` is not ``False``).
    2. Build clusters greedily: two findings join the same cluster iff
       they share a path AND their line ranges overlap. Transitive merges
       are handled by the iterative pass.
    3. A cluster is ``consensus=True`` iff it contains contributions from
       >= 2 distinct peers.
    4. Single-peer clusters (and all unverified findings, and verified
       findings whose ``verified`` flag is ``False``) flow to
       ``single_peer_concerns``.
    """
    by_peer = {peer: list(items) for peer, items in (by_peer or {}).items()}

    verifiable: list[Finding] = []
    leftover: list[Finding] = []
    for peer, items in by_peer.items():
        for finding in items:
            if finding.verified_ref is None:
                leftover.append(finding)
                continue
            if finding.verified is False:
                leftover.append(finding)
                continue
            verifiable.append(finding)

    # Greedy clustering. For each finding, find an existing cluster whose
    # path matches and whose union-range overlaps; otherwise start a new
    # cluster. Repeat until stable (one pass usually suffices given the
    # union widens the range monotonically).
    clusters_raw: list[list[Finding]] = []
    for finding in verifiable:
        ref = finding.verified_ref
        assert ref is not None
        joined = False
        for cluster in clusters_raw:
            head_ref = cluster[0].verified_ref
            assert head_ref is not None
            if head_ref.path != ref.path:
                continue
            current_range = _union_range(cluster)
            if current_range is None:
                continue
            if _ranges_overlap(
                (ref.start_line, ref.end_line), current_range
            ):
                cluster.append(finding)
                joined = True
                break
        if not joined:
            clusters_raw.append([finding])

    # Merge pass: clusters that now share path + overlapping union may
    # need to be coalesced (e.g. two ranges A,B don't overlap each other
    # but a later C overlaps both).
    merged_changed = True
    while merged_changed:
        merged_changed = False
        for i in range(len(clusters_raw)):
            for j in range(i + 1, len(clusters_raw)):
                ci, cj = clusters_raw[i], clusters_raw[j]
                if not ci or not cj:
                    continue
                p_i = ci[0].verified_ref.path  # type: ignore[union-attr]
                p_j = cj[0].verified_ref.path  # type: ignore[union-attr]
                if p_i != p_j:
                    continue
                r_i = _union_range(ci)
                r_j = _union_range(cj)
                if r_i is None or r_j is None:
                    continue
                if _ranges_overlap(r_i, r_j):
                    ci.extend(cj)
                    clusters_raw[j] = []
                    merged_changed = True
        clusters_raw = [c for c in clusters_raw if c]

    clusters: list[FindingCluster] = []
    singles: list[Finding] = list(leftover)
    for idx, members in enumerate(clusters_raw, start=1):
        peers_seen: list[str] = []
        for f in members:
            if f.peer not in peers_seen:
                peers_seen.append(f.peer)
        path = members[0].verified_ref.path  # type: ignore[union-attr]
        rng = _union_range(members)
        is_consensus = len(peers_seen) >= 2
        if is_consensus:
            clusters.append(
                FindingCluster(
                    id=f"C{idx}",
                    severity=_representative_severity(members),
                    peers=peers_seen,
                    findings=members,
                    verified_path=path,
                    verified_range=rng,
                )
            )
        else:
            # Single-peer "cluster": a peer's verified finding that no
            # one else corroborated. These flow to single_peer_concerns
            # rather than constructing a non-consensus FindingCluster
            # (which the dataclass no longer represents).
            singles.extend(members)

    return FindingMatrix(
        clusters=clusters,
        single_peer_concerns=singles,
    )


# --- serialization for transcripts / MCP ---------------------------------


def matrix_to_dict(matrix: FindingMatrix) -> dict[str, Any]:
    """JSON-serializable shape used in transcript JSON and MCP results.

    Returns an empty dict-like with both keys present even when the matrix
    is empty so downstream callers can read it without branching. Callers
    that want to omit the field entirely should check ``is_empty()`` first.
    """

    consensus_blockers: list[dict[str, Any]] = []
    for cluster in matrix.clusters:
        # Pick the first contributing finding as the representative claim
        # — the cluster severity already encodes the "most severe wins"
        # rule, and peers' wording differs; we don't try to merge prose.
        rep_claim = cluster.findings[0].claim if cluster.findings else ""
        entry: dict[str, Any] = {
            "id": cluster.id,
            "severity": cluster.severity,
            "peers": list(cluster.peers),
            "claim": rep_claim,
        }
        if cluster.verified_path is not None:
            entry["path"] = cluster.verified_path
        if cluster.verified_range is not None:
            entry["start_line"] = cluster.verified_range[0]
            entry["end_line"] = cluster.verified_range[1]
        consensus_blockers.append(entry)

    single_peer_concerns: list[dict[str, Any]] = []
    for finding in matrix.single_peer_concerns:
        entry = {
            "peer": finding.peer,
            "id": finding.id,
            "severity": finding.severity,
            "claim": finding.claim,
        }
        if finding.verified_ref is not None:
            entry["path"] = finding.verified_ref.path
            entry["start_line"] = finding.verified_ref.start_line
            entry["end_line"] = finding.verified_ref.end_line
            if finding.verified is False:
                entry["unverified"] = True
        else:
            entry["unverified"] = True
        single_peer_concerns.append(entry)

    # Derived three-tier gating partition (advisory; no new parsing/severities).
    # The severity ladder is exactly {"blocker", "medium", "nit"} (default
    # "nit"); blocker -> blocking, medium -> non_blocking, nit -> suggestion.
    # Any unexpected severity falls through to the safe `suggestion` default.
    # All single-peer concerns are suggestion-tier by construction (no
    # corroboration). Entry dicts are referenced, not copied.
    blocking: list[dict[str, Any]] = []
    non_blocking: list[dict[str, Any]] = []
    suggestion: list[dict[str, Any]] = []
    for entry in consensus_blockers:
        severity = entry.get("severity")
        if severity == "blocker":
            blocking.append(entry)
        elif severity == "medium":
            non_blocking.append(entry)
        else:
            suggestion.append(entry)
    suggestion.extend(single_peer_concerns)

    return {
        "consensus_blockers": consensus_blockers,
        "single_peer_concerns": single_peer_concerns,
        "gating": {
            "blocking": blocking,
            "non_blocking": non_blocking,
            "suggestion": suggestion,
        },
    }


def build_matrix_from_results(
    results: list[Any],
    *, verifier: CitationVerifier | None = None,
) -> FindingMatrix:
    """Build a FindingMatrix from final-round results.

    A tag is not a receipt. Verify FINDINGS references directly, including
    references never repeated in EVIDENCE. Without a verifier, only explicit
    positive EVIDENCE receipts can contribute to consensus clusters.
    """

    by_peer: dict[str, list[Finding]] = {}
    for result in results:
        if getattr(result, "ok", True) is False:
            continue
        output = getattr(result, "output", None) or ""
        peer_name = getattr(result, "name", "?")
        # Strip any `:roundN` suffix so multi-round entries share a peer
        # identity — matters when the orchestrator passes only final-round
        # results but their names still carry the suffix.
        base_name = peer_name.split(":round", 1)[0]
        findings = extract_findings(output, base_name)
        if not findings:
            continue
        # Use the result's already-computed verification failures (a list
        # of ``"path:start-end"`` strings) to mark Finding.verified=False
        # where the verified ref turned out to be unreal.
        failures = getattr(result, "evidence_verification_failures", None) or []
        failed_keys = set()
        for entry in failures:
            failed_keys.add(str(entry).split(" ", 1)[0])
        positive_refs = {
            (e.get("path"), e.get("start_line"), e.get("end_line", e.get("start_line")))
            for e in (getattr(result, "evidence", None) or [])
            if isinstance(e, dict) and e.get("tag") == "verified" and e.get("verified") is True
        }
        materialized: list[Finding] = []
        for f in findings:
            verified = f.verified
            if f.verified_ref is not None:
                key = f"{f.verified_ref.path}:{f.verified_ref.start_line}-{f.verified_ref.end_line}"
                ref = f.verified_ref
                verified = (
                    verifier.verify(ref) if verifier is not None
                    else (ref.path, ref.start_line, ref.end_line) in positive_refs
                )
                if key in failed_keys:
                    verified = False
                if not verified and key not in failed_keys:
                    result.evidence_verification_failures = [
                        *(getattr(result, "evidence_verification_failures", None) or []), key
                    ]
                    failed_keys.add(key)
            materialized.append(
                Finding(
                    id=f.id,
                    peer=f.peer,
                    severity=f.severity,
                    claim=f.claim,
                    verified_ref=f.verified_ref,
                    verified=verified,
                )
            )
        by_peer[base_name] = materialized
    return cluster_findings(by_peer)


__all__ = [
    "Finding",
    "FindingCluster",
    "FindingMatrix",
    "build_matrix_from_results",
    "cluster_findings",
    "extract_findings",
    "matrix_to_dict",
]
