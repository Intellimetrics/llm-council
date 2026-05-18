"""Eval runner (Phase B — v0.8 plan).

Exercises the fixture loader, the runner against a stubbed
`execute_council_fn`, and the JSON round-trip. NO real subprocess
or API calls — every test uses an injected stub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from llm_council.eval.runner import (
    Fixture,
    FixtureScorecard,
    SuiteScorecard,
    iter_fixture_dirs,
    load_fixture,
    run_fixture,
    run_suite,
    to_json,
)


# --- Stub ParticipantResult -----------------------------------------------


@dataclass
class _StubResult:
    """Mirrors the fields the runner reads off `ParticipantResult`."""

    name: str
    ok: bool = True
    error: str = ""
    from_cache: bool = False
    blockers: list[str] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)


# --- Fixture helpers ------------------------------------------------------


def _write_fixture(
    root: Path,
    fid: str,
    prompt: str,
    blockers: list[dict[str, Any]],
) -> Path:
    fdir = root / fid
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "prompt.md").write_text(prompt, encoding="utf-8")
    (fdir / "expected_blockers.json").write_text(
        json.dumps({"blockers": blockers}), encoding="utf-8"
    )
    return fdir


# --- load_fixture --------------------------------------------------------


def test_load_fixture_reads_prompt_and_expected(tmp_path):
    blockers = [
        {"path": "src/foo.py", "severity": "blocker", "claim": "race condition"}
    ]
    fdir = _write_fixture(
        tmp_path, "case_a", "Review this PR for issues.", blockers
    )
    fixture = load_fixture(fdir)
    assert isinstance(fixture, Fixture)
    assert fixture.id == "case_a"
    assert fixture.prompt == "Review this PR for issues."
    assert fixture.expected_blockers == blockers


def test_load_fixture_missing_prompt_raises(tmp_path):
    fdir = tmp_path / "broken"
    fdir.mkdir()
    (fdir / "expected_blockers.json").write_text('{"blockers": []}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_fixture(fdir)


def test_load_fixture_missing_expected_raises(tmp_path):
    fdir = tmp_path / "broken"
    fdir.mkdir()
    (fdir / "prompt.md").write_text("hi", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_fixture(fdir)


def test_load_fixture_malformed_json_raises(tmp_path):
    fdir = tmp_path / "broken"
    fdir.mkdir()
    (fdir / "prompt.md").write_text("hi", encoding="utf-8")
    (fdir / "expected_blockers.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_fixture(fdir)


def test_load_fixture_missing_blockers_key_raises(tmp_path):
    fdir = tmp_path / "broken"
    fdir.mkdir()
    (fdir / "prompt.md").write_text("hi", encoding="utf-8")
    (fdir / "expected_blockers.json").write_text('{"not_blockers": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_fixture(fdir)


def test_iter_fixture_dirs_skips_dotfiles_and_pycache(tmp_path):
    _write_fixture(tmp_path, "case_a", "p", [])
    _write_fixture(tmp_path, "case_b", "p", [])
    # Noise that should be skipped.
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "prompt.md").write_text("noise", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "prompt.md").write_text("noise", encoding="utf-8")
    # A child dir without prompt.md should also be skipped.
    (tmp_path / "case_c_incomplete").mkdir()

    dirs = sorted(p.name for p in iter_fixture_dirs(tmp_path))
    assert dirs == ["case_a", "case_b"]


def test_iter_fixture_dirs_nonexistent_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert list(iter_fixture_dirs(missing)) == []


# --- run_fixture ---------------------------------------------------------


def test_run_fixture_calls_stub_and_returns_scorecard(tmp_path):
    blockers = [
        {"path": "src/auth/middleware.py", "severity": "blocker",
         "claim": "missing tenant filter"}
    ]
    fdir = _write_fixture(tmp_path, "case_tenant", "Review middleware.", blockers)
    fixture = load_fixture(fdir)

    calls: list[tuple[str, str]] = []

    def stub(prompt: str, mode: str):
        calls.append((prompt, mode))
        return [
            _StubResult(
                name="claude",
                ok=True,
                blockers=[
                    "src/auth/middleware.py drops the tenant filter on lookup",
                ],
                evidence=[
                    {
                        "text": "see middleware.py",
                        "tag": "verified",
                        "verified": True,
                    },
                ],
            ),
            _StubResult(
                name="codex",
                ok=True,
                blockers=["unrelated style nit"],
                evidence=[],
            ),
        ]

    sc = run_fixture(fixture, mode="review", execute_council_fn=stub)
    assert isinstance(sc, FixtureScorecard)
    assert sc.fixture_id == "case_tenant"
    assert len(sc.peers) == 2
    assert calls == [("Review middleware.", "review")]
    # claude caught the tenant blocker → recall=1.0; codex's bullet
    # doesn't match → recall=0.0. Fixture-level max should be 1.0.
    assert sc.aggregate_metrics["blocker_recall_max"] == 1.0
    assert sc.aggregate_metrics["peer_count"] == 2
    # claude metrics surface the verified citation hit.
    claude = next(p for p in sc.peers if p.name == "claude")
    assert claude.metrics["citation_accuracy"] == 1.0
    assert claude.metrics["blocker_recall"] == 1.0
    codex = next(p for p in sc.peers if p.name == "codex")
    assert codex.metrics["blocker_recall"] == 0.0
    assert codex.metrics["citation_accuracy"] is None


def test_run_fixture_marks_cache_miss_when_cache_only(tmp_path):
    fdir = _write_fixture(tmp_path, "case_c", "p", [])
    fixture = load_fixture(fdir)

    def stub(prompt: str, mode: str):
        return [_StubResult(name="claude", ok=True, from_cache=False)]

    sc = run_fixture(
        fixture, mode="review", execute_council_fn=stub, cache_only=True
    )
    assert sc.cache_miss is True


def test_run_fixture_no_cache_miss_when_all_from_cache(tmp_path):
    fdir = _write_fixture(tmp_path, "case_d", "p", [])
    fixture = load_fixture(fdir)

    def stub(prompt: str, mode: str):
        return [_StubResult(name="claude", ok=True, from_cache=True)]

    sc = run_fixture(
        fixture, mode="review", execute_council_fn=stub, cache_only=True
    )
    assert sc.cache_miss is False


# --- run_suite -----------------------------------------------------------


def test_run_suite_aggregates_across_fixtures(tmp_path):
    _write_fixture(
        tmp_path,
        "case_one",
        "p1",
        [{"path": "src/a.py", "severity": "blocker", "claim": "bug a"}],
    )
    _write_fixture(
        tmp_path,
        "case_two",
        "p2",
        [{"path": "src/b.py", "severity": "blocker", "claim": "bug b"}],
    )

    def stub(prompt: str, mode: str):
        # Always catches the right blocker.
        target = "src/a.py" if prompt == "p1" else "src/b.py"
        return [
            _StubResult(
                name="claude",
                ok=True,
                blockers=[f"{target} has the bug"],
                evidence=[],
            )
        ]

    suite = run_suite(tmp_path, mode="review", execute_council_fn=stub)
    assert isinstance(suite, SuiteScorecard)
    assert len(suite.fixtures) == 2
    assert {f.fixture_id for f in suite.fixtures} == {"case_one", "case_two"}
    assert suite.aggregate_metrics["fixture_count"] == 2
    assert suite.aggregate_metrics["blocker_recall"] == 1.0


def test_run_suite_handles_empty_fixtures_dir(tmp_path):
    """No fixtures → empty suite, no crash."""
    suite = run_suite(
        tmp_path, mode="review", execute_council_fn=lambda p, m: []
    )
    assert suite.fixtures == []
    assert suite.aggregate_metrics == {}


# --- JSON round-trip -----------------------------------------------------


def test_to_json_round_trips(tmp_path):
    _write_fixture(
        tmp_path,
        "case_r",
        "prompt",
        [{"path": "src/x.py", "severity": "blocker", "claim": "boom"}],
    )

    def stub(prompt: str, mode: str):
        return [
            _StubResult(
                name="claude",
                ok=True,
                blockers=["src/x.py: boom"],
                evidence=[{"text": "c", "tag": "verified", "verified": True}],
            )
        ]

    suite = run_suite(tmp_path, mode="review", execute_council_fn=stub)
    payload = to_json(suite)
    parsed = json.loads(payload)
    assert parsed["mode"] == "review"
    assert "council_version" in parsed
    assert "timestamp" in parsed
    assert parsed["fixtures"][0]["fixture_id"] == "case_r"
    assert parsed["fixtures"][0]["peers"][0]["metrics"]["blocker_recall"] == 1.0


def test_suite_scorecard_includes_metadata(tmp_path):
    suite = run_suite(
        tmp_path, mode="custom-mode", execute_council_fn=lambda p, m: []
    )
    assert suite.mode == "custom-mode"
    assert suite.council_version  # populated from llm_council.__version__
    assert suite.timestamp  # ISO timestamp populated
    # Round-trip the metadata via .to_dict()
    as_dict = suite.to_dict()
    assert as_dict["mode"] == "custom-mode"
    assert as_dict["council_version"]
    assert as_dict["timestamp"]


# --- End-to-end stub assertion -------------------------------------------


def test_runner_does_not_import_orchestrator_at_module_load():
    """Critical constraint: the runner module must not import
    execute_council eagerly — it accepts the function as a parameter so
    test stubs work and so we don't pay orchestrator import cost in unit
    tests that just want to score synthetic outputs."""
    import llm_council.eval.runner as runner_mod

    # The runner module should not have execute_council as a top-level
    # symbol. (It's allowed to import it lazily inside a CLI helper.)
    assert not hasattr(runner_mod, "execute_council")
