"""Tests for user-authorable, composable review-focus bundles.

Covers ``llm_council.review_skills`` discovery + resolution + rendering,
the additive ``context.apply_per_peer_directives`` focus path, and the
orchestrator ``execute_council(..., focus=...)`` provenance + directive
threading.

Focus bundles are INERT PROMPT TEXT only — they shape WHAT peers scrutinize
and grant no tool/write capability. Backward compatibility is mandatory: the
no-focus path must behave exactly as before.

Orchestrator stubbing mirrors ``tests/test_independence_warning.py`` /
``tests/test_continue_debate.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.adapters import ParticipantResult
from llm_council.context import apply_per_peer_directives
from llm_council.review_skills import (
    FocusNotFound,
    ReviewSkill,
    discover_review_skills,
    render_focus_directive,
    resolve_focus,
)


# --- Fixture helpers ------------------------------------------------------

def _write_bundle(root: Path, dir_name: str, content: str) -> Path:
    bundle = root / ".llm-council" / "review-skills" / dir_name
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "SKILL.md").write_text(content, encoding="utf-8")
    return bundle


_VALID = (
    "---\n"
    "name: security-review\n"
    "description: Scrutinize for authz gaps and injection.\n"
    "---\n"
    "Flag authz/authn gaps and injection. Cite file:line. Do not edit.\n"
)


# --- Discovery ------------------------------------------------------------

def test_valid_bundle_parses(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    skills, skipped = discover_review_skills(tmp_path)
    assert skipped == []
    assert set(skills) == {"security-review"}
    skill = skills["security-review"]
    assert isinstance(skill, ReviewSkill)
    assert skill.name == "security-review"
    assert skill.description == "Scrutinize for authz gaps and injection."
    assert skill.body.startswith("Flag authz/authn gaps")
    assert skill.body.endswith("Do not edit.")  # body stripped
    expected = hashlib.sha256(skill.body.encode("utf-8")).hexdigest()
    assert skill.sha256 == expected
    assert skill.path.endswith("SKILL.md")


def test_name_mismatch_dir_is_skipped(tmp_path):
    # name in frontmatter != directory name
    _write_bundle(
        tmp_path,
        "wrong-dir",
        "---\nname: security-review\ndescription: x.\n---\nbody\n",
    )
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert len(skipped) == 1
    assert skipped[0]["name"] == "wrong-dir"
    assert "directory name" in skipped[0]["reason"]


def test_too_long_name_is_skipped(tmp_path):
    long_name = "a" * 65
    _write_bundle(
        tmp_path,
        long_name,
        f"---\nname: {long_name}\ndescription: x.\n---\nbody\n",
    )
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert len(skipped) == 1
    assert "64" in skipped[0]["reason"]


def test_invalid_name_chars_skipped(tmp_path):
    _write_bundle(
        tmp_path,
        "Bad_Name",
        "---\nname: Bad_Name\ndescription: x.\n---\nbody\n",
    )
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert "^[a-z0-9-]+$" in skipped[0]["reason"]


def test_missing_frontmatter_is_skipped(tmp_path):
    _write_bundle(tmp_path, "no-fm", "just a body with no frontmatter\n")
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert len(skipped) == 1
    assert "frontmatter" in skipped[0]["reason"]


def test_empty_body_is_skipped(tmp_path):
    _write_bundle(
        tmp_path,
        "empty-body",
        "---\nname: empty-body\ndescription: x.\n---\n   \n",
    )
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert "body is empty" in skipped[0]["reason"]


def test_missing_description_is_skipped(tmp_path):
    _write_bundle(
        tmp_path,
        "no-desc",
        "---\nname: no-desc\n---\nbody\n",
    )
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert "description" in skipped[0]["reason"]


def test_no_skill_md_in_dir_is_skipped(tmp_path):
    bundle = tmp_path / ".llm-council" / "review-skills" / "orphan"
    bundle.mkdir(parents=True)
    (bundle / "NOTES.md").write_text("not a skill", encoding="utf-8")
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert skipped[0]["name"] == "orphan"
    assert "SKILL.md" in skipped[0]["reason"]


def test_discovery_walks_up(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    skills, _ = discover_review_skills(nested)
    assert "security-review" in skills


def test_no_directory_returns_empty(tmp_path):
    skills, skipped = discover_review_skills(tmp_path)
    assert skills == {}
    assert skipped == []


def test_lenient_one_bad_one_good(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    _write_bundle(tmp_path, "broken", "no frontmatter here\n")
    skills, skipped = discover_review_skills(tmp_path)
    assert set(skills) == {"security-review"}
    assert len(skipped) == 1
    assert skipped[0]["name"] == "broken"


# --- resolve_focus --------------------------------------------------------

def test_resolve_focus_in_requested_order(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    _write_bundle(
        tmp_path,
        "test-gaps",
        "---\nname: test-gaps\ndescription: tests.\n---\nFlag missing tests.\n",
    )
    resolved, skipped = resolve_focus(["test-gaps", "security-review"], tmp_path)
    assert [s.name for s in resolved] == ["test-gaps", "security-review"]
    assert skipped == []


def test_resolve_focus_unknown_raises_with_available(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    with pytest.raises(FocusNotFound) as exc:
        resolve_focus(["nope"], tmp_path)
    assert exc.value.missing == ["nope"]
    assert exc.value.available == ["security-review"]
    assert "security-review" in str(exc.value)


def test_resolve_focus_surfaces_skipped(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    _write_bundle(tmp_path, "broken", "no frontmatter\n")
    resolved, skipped = resolve_focus(["security-review"], tmp_path)
    assert [s.name for s in resolved] == ["security-review"]
    assert len(skipped) == 1
    assert skipped[0]["name"] == "broken"


def test_sha256_stable_for_identical_body(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    s1, _ = resolve_focus(["security-review"], tmp_path)
    s2, _ = resolve_focus(["security-review"], tmp_path)
    assert s1[0].sha256 == s2[0].sha256


# --- render_focus_directive ----------------------------------------------

def test_render_empty_is_blank():
    assert render_focus_directive([]) == ""


def test_render_combines_with_delimiters(tmp_path):
    _write_bundle(tmp_path, "security-review", _VALID)
    _write_bundle(
        tmp_path,
        "test-gaps",
        "---\nname: test-gaps\ndescription: tests.\n---\nFlag missing tests.\n",
    )
    resolved, _ = resolve_focus(["security-review", "test-gaps"], tmp_path)
    directive = render_focus_directive(resolved)
    assert "=== REVIEW FOCUS: security-review ===" in directive
    assert "=== REVIEW FOCUS: test-gaps ===" in directive
    assert "Flag missing tests." in directive
    # security-review block precedes test-gaps block (requested order)
    assert directive.index("security-review") < directive.index("test-gaps")


# --- apply_per_peer_directives (additive composition) ---------------------

def test_focus_directive_appended():
    out = apply_per_peer_directives(
        "BASE PROMPT",
        mode="review",
        family="claude",
        focus_directive="=== REVIEW FOCUS: x ===\nbody",
    )
    assert out.startswith("BASE PROMPT")
    assert out.endswith("=== REVIEW FOCUS: x ===\nbody")
    assert "\n\n=== REVIEW FOCUS: x ===" in out


def test_focus_composes_with_stance():
    out = apply_per_peer_directives(
        "BASE PROMPT",
        mode="consensus",
        family="claude",
        stance="against",
        focus_directive="FOCUS BLOCK",
    )
    # both the stance block AND the focus block are present
    assert "INDIVIDUAL ASSIGNMENT" in out
    assert "FOCUS BLOCK" in out
    # focus is appended AFTER the stance block
    assert out.index("INDIVIDUAL ASSIGNMENT") < out.index("FOCUS BLOCK")


def test_no_focus_identical_to_before():
    base = "BASE PROMPT"
    with_none = apply_per_peer_directives(
        base, mode="review", family="claude", focus_directive=None
    )
    without_kw = apply_per_peer_directives(base, mode="review", family="claude")
    assert with_none == without_kw == base


# --- Orchestrator threading + provenance ----------------------------------

def _result(name, *, label="yes"):
    return ParticipantResult(
        name=name,
        ok=True,
        output=f"RECOMMENDATION: {label} - reason",
        error="",
        elapsed_seconds=1.0,
    )


def _run_with_focus(focus, *, participant_cfg=None, config=None):
    """Run execute_council capturing the focus_directive kwarg seen by
    the stubbed run_participants."""
    import llm_council.orchestrator as orch_module

    cfg = participant_cfg or {
        "a": {"type": "cli", "family": "acme"},
        "b": {"type": "cli", "family": "acme"},
    }
    cfg_obj = config or {"defaults": {}}
    captured: dict[str, object] = {}

    async def fake_run_participants(selected, *args, **kwargs):
        captured["focus_directive"] = kwargs.get("focus_directive")
        return [_result(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            results, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=list(cfg.keys()),
                    participant_cfg=cfg,
                    prompt="q",
                    cwd=Path("."),
                    config=cfg_obj,
                    deliberate=False,
                    max_rounds=1,
                    mode="review",
                    focus=focus,
                )
            )
    return results, metadata, captured


def test_orchestrator_stamps_applied_focus_and_threads_directive():
    skill = ReviewSkill(
        name="security-review",
        description="d",
        body="Flag authz gaps.",
        path="/tmp/SKILL.md",
        sha256="abc123",
    )
    _, metadata, captured = _run_with_focus([skill])
    assert metadata["applied_focus"] == [
        {"name": "security-review", "sha256": "abc123"}
    ]
    # the rendered directive reached run_participants
    directive = captured["focus_directive"]
    assert directive
    assert "=== REVIEW FOCUS: security-review ===" in directive
    assert "Flag authz gaps." in directive


def test_orchestrator_no_focus_absent_and_unchanged():
    _, metadata, captured = _run_with_focus(None)
    assert "applied_focus" not in metadata
    # backward-compat: no focus => empty directive passed through
    assert captured["focus_directive"] == ""


def test_shipped_example_bundles_do_not_trip_required_section_validator():
    """Shipped example bundles must not contain a `PART N — TITLE (REQUIRED)`
    header.

    A focus directive is appended to the peer prompt, and the section-
    coverage validator scans that combined prompt (adapters runs
    `required_sections_missing(peer_prompt, ...)`). So a bundle body carrying
    a REQUIRED-section header WOULD be enforced on every peer response —
    intended only when an author opts in, never incidentally. Guard the
    bundles we ship from tripping it. Regression guard for the codex review
    finding in WU4.
    """
    from llm_council.sections import REQUIRED_SECTION_HEADER_RE

    examples_dir = (
        Path(__file__).resolve().parent.parent / "examples" / "review-skills"
    )
    skill_files = sorted(examples_dir.glob("*/SKILL.md"))
    assert skill_files, "expected shipped example bundles under examples/review-skills/"
    for skill_file in skill_files:
        text = skill_file.read_text(encoding="utf-8")
        assert not REQUIRED_SECTION_HEADER_RE.search(text), (
            f"example bundle {skill_file.parent.name!r} contains a PART N "
            "(REQUIRED) header that would be enforced by the section-coverage "
            "validator"
        )


def test_focus_required_header_flows_into_section_validator():
    """Characterize the known interaction: a bundle body that DOES declare a
    `PART N — TITLE (REQUIRED)` header becomes an enforced section because the
    rendered directive is part of the peer prompt. Documented behavior, not a
    bug — authors opt in by writing the header."""
    from llm_council.context import apply_per_peer_directives
    from llm_council.sections import required_sections

    directive = (
        "=== REVIEW FOCUS: strict ===\n"
        "PART 1 — THREAT MODEL (REQUIRED)\nEnumerate attacker capabilities."
    )
    base = "Review this change."
    assert required_sections(base) == []
    peer_prompt = apply_per_peer_directives(
        base, mode=None, family=None, focus_directive=directive
    )
    reqs = required_sections(peer_prompt)
    assert any(
        isinstance(r, dict) and r.get("num") == "1" for r in reqs
    ), "a focus-declared REQUIRED section should be enforced via the peer prompt"
