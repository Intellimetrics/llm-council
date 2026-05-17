"""Verified file:line citation primitives (Phase A — v0.8 plan).

Covers the `[VERIFIED:path:start-end]` evidence tag: regex parsing,
shape produced by `_parse_tagged_entry`, on-disk verification by
`verify_ref`, and the in-place mutation done by
`verify_evidence_citations` after participant results return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llm_council.adapters import _parse_tagged_entry
from llm_council.citations import (
    VerifiedRef,
    parse_verified_tag,
    strip_verified_tag,
    verify_evidence_citations,
    verify_ref,
)


# --- parse_verified_tag --------------------------------------------------

def test_parse_verified_tag_single_line():
    ref = parse_verified_tag("[VERIFIED:file.py:42]")
    assert ref == VerifiedRef(path="file.py", start_line=42, end_line=42)


def test_parse_verified_tag_range():
    ref = parse_verified_tag("[VERIFIED:src/foo.py:10-20]")
    assert ref == VerifiedRef(path="src/foo.py", start_line=10, end_line=20)


def test_parse_verified_tag_malformed_no_line():
    assert parse_verified_tag("[VERIFIED:no_lines]") is None


def test_parse_verified_tag_malformed_empty():
    assert parse_verified_tag("[VERIFIED:]") is None


def test_parse_verified_tag_malformed_path_only():
    assert parse_verified_tag("[VERIFIED:file.py]") is None


def test_parse_verified_tag_case_insensitive():
    ref = parse_verified_tag("[verified:file.py:7]")
    assert ref == VerifiedRef(path="file.py", start_line=7, end_line=7)


def test_parse_verified_tag_swaps_reversed_range():
    """Reversed range `20-10` normalises to start=10, end=20."""
    ref = parse_verified_tag("[VERIFIED:f.py:20-10]")
    assert ref == VerifiedRef(path="f.py", start_line=10, end_line=20)


def test_parse_verified_tag_returns_none_on_empty_input():
    assert parse_verified_tag("") is None


# --- strip_verified_tag --------------------------------------------------

def test_strip_verified_tag_removes_token():
    text = "see [VERIFIED:foo.py:1-3] for the bug"
    assert strip_verified_tag(text) == "see  for the bug"


# --- _parse_tagged_entry round-trip ------------------------------------

def test_parse_tagged_entry_verified_shape():
    out = _parse_tagged_entry("[VERIFIED:src/foo.py:10-20] race condition here")
    assert out["tag"] == "verified"
    assert out["path"] == "src/foo.py"
    assert out["start_line"] == 10
    assert out["end_line"] == 20
    assert out["verified"] is None
    assert "race condition here" in out["text"]
    assert "VERIFIED" not in out["text"]


def test_parse_tagged_entry_strips_verified_token():
    out = _parse_tagged_entry("[VERIFIED:foo.py:7] kaboom")
    assert "[VERIFIED" not in out["text"]
    assert "kaboom" in out["text"]


def test_parse_tagged_entry_published_regression():
    """Pre-Phase-A shape must still work for non-VERIFIED tags."""
    out = _parse_tagged_entry("[PUBLISHED] Bai et al. 2022")
    assert out == {"text": "Bai et al. 2022", "tag": "published"}


def test_parse_tagged_entry_untagged_regression():
    out = _parse_tagged_entry("a claim with no tag")
    assert out == {"text": "a claim with no tag", "tag": None}


# --- verify_ref filesystem tests ---------------------------------------

def _write_lines(path: Path, n_lines: int) -> None:
    path.write_text("\n".join(f"line {i}" for i in range(1, n_lines + 1)) + "\n")


def test_verify_ref_existing_file_in_bounds(tmp_path):
    target = tmp_path / "foo.py"
    _write_lines(target, 50)
    ref = VerifiedRef(path="foo.py", start_line=10, end_line=20)
    assert verify_ref(ref, tmp_path) is True


def test_verify_ref_single_line_in_bounds(tmp_path):
    target = tmp_path / "foo.py"
    _write_lines(target, 50)
    ref = VerifiedRef(path="foo.py", start_line=50, end_line=50)
    assert verify_ref(ref, tmp_path) is True


def test_verify_ref_nonexistent_file(tmp_path):
    ref = VerifiedRef(path="nope.py", start_line=1, end_line=1)
    assert verify_ref(ref, tmp_path) is False


def test_verify_ref_out_of_bounds(tmp_path):
    target = tmp_path / "foo.py"
    _write_lines(target, 10)
    ref = VerifiedRef(path="foo.py", start_line=5, end_line=99)
    assert verify_ref(ref, tmp_path) is False


def test_verify_ref_start_below_one(tmp_path):
    target = tmp_path / "foo.py"
    _write_lines(target, 10)
    ref = VerifiedRef(path="foo.py", start_line=0, end_line=5)
    assert verify_ref(ref, tmp_path) is False


def test_verify_ref_path_traversal_rejected(tmp_path):
    """A `../sibling` escape resolves outside cwd and must fail."""
    sibling_dir = tmp_path.parent / "escape_target"
    sibling_dir.mkdir(exist_ok=True)
    outside = sibling_dir / "leak.py"
    _write_lines(outside, 5)
    try:
        ref = VerifiedRef(path="../escape_target/leak.py", start_line=1, end_line=2)
        assert verify_ref(ref, tmp_path) is False
    finally:
        outside.unlink()
        sibling_dir.rmdir()


def test_verify_ref_directory_not_file(tmp_path):
    (tmp_path / "sub").mkdir()
    ref = VerifiedRef(path="sub", start_line=1, end_line=1)
    assert verify_ref(ref, tmp_path) is False


# --- verify_evidence_citations mutation --------------------------------

@dataclass
class _StubResult:
    """Minimal stand-in for ParticipantResult — only fields we touch."""
    evidence: list = field(default_factory=list)
    evidence_verification_failures: list[str] = field(default_factory=list)


def test_verify_evidence_citations_sets_verified_true(tmp_path):
    target = tmp_path / "foo.py"
    _write_lines(target, 50)
    result = _StubResult(
        evidence=[
            {
                "text": "bug at line 10",
                "tag": "verified",
                "path": "foo.py",
                "start_line": 10,
                "end_line": 20,
                "verified": None,
            }
        ]
    )
    verify_evidence_citations([result], tmp_path)
    assert result.evidence[0]["verified"] is True
    assert result.evidence_verification_failures == []


def test_verify_evidence_citations_sets_verified_false_and_records_failure(tmp_path):
    result = _StubResult(
        evidence=[
            {
                "text": "ghost cite",
                "tag": "verified",
                "path": "missing.py",
                "start_line": 1,
                "end_line": 5,
                "verified": None,
            }
        ]
    )
    verify_evidence_citations([result], tmp_path)
    assert result.evidence[0]["verified"] is False
    assert result.evidence_verification_failures == ["missing.py:1-5"]


def test_verify_evidence_citations_skips_non_verified_tags(tmp_path):
    result = _StubResult(
        evidence=[
            {"text": "Bai 2022", "tag": "published"},
            {"text": "just an obs", "tag": "observable"},
            "raw string entry",
        ]
    )
    verify_evidence_citations([result], tmp_path)
    # Non-verified entries are untouched.
    assert result.evidence[0] == {"text": "Bai 2022", "tag": "published"}
    assert result.evidence_verification_failures == []


def test_verify_evidence_citations_idempotent(tmp_path):
    """Running twice does not re-verify or double-append failures."""
    result = _StubResult(
        evidence=[
            {
                "text": "ghost",
                "tag": "verified",
                "path": "missing.py",
                "start_line": 1,
                "end_line": 5,
                "verified": None,
            }
        ]
    )
    verify_evidence_citations([result], tmp_path)
    verify_evidence_citations([result], tmp_path)
    assert result.evidence[0]["verified"] is False
    assert result.evidence_verification_failures == ["missing.py:1-5"]


def test_verify_evidence_citations_handles_empty_evidence(tmp_path):
    result = _StubResult(evidence=[])
    verify_evidence_citations([result], tmp_path)
    assert result.evidence_verification_failures == []


def test_verify_evidence_citations_malformed_entry(tmp_path):
    """A VERIFIED entry without `path` or `start_line` is marked failed."""
    result = _StubResult(
        evidence=[
            {"text": "broken", "tag": "verified", "verified": None}
        ]
    )
    verify_evidence_citations([result], tmp_path)
    assert result.evidence[0]["verified"] is False
    assert len(result.evidence_verification_failures) == 1
    assert "malformed" in result.evidence_verification_failures[0]
