"""Tests for the synthesis chair (Pick B) helpers."""

from __future__ import annotations

import pytest

from llm_council.adapters import ABDICATED_ERROR_PREFIX, ParticipantResult
from llm_council.synthesis import (
    build_synthesis_prompt,
    select_synthesizer,
    should_synthesize,
    universal_abdication,
)


def test_should_synthesize_explicit_always_runs():
    assert should_synthesize(True, {}) is True
    assert should_synthesize(True, {"deliberation_status": "ran_no_labeled_disagreement"}) is True


def test_should_synthesize_skips_on_agreement_auto():
    """Pass-3 trigger logic: agreement does NOT auto-trigger — wasted spend."""
    assert (
        should_synthesize(False, {"deliberation_status": "ran_no_labeled_disagreement"})
        is False
    )


def test_should_synthesize_auto_on_unresolved():
    assert (
        should_synthesize(False, {"deliberation_status": "ran_max_rounds_unresolved"})
        is True
    )


def test_select_synthesizer_loud_fail_when_unset():
    with pytest.raises(ValueError, match="defaults.synthesizer is not configured"):
        select_synthesizer({"defaults": {}}, {"a": {}}, stances=None, current=None)


def test_select_synthesizer_explicit_participant():
    chair = select_synthesizer(
        {"defaults": {"synthesizer": "codex"}},
        {"claude": {}, "codex": {}, "gemini": {}},
        stances=None,
        current=None,
    )
    assert chair == "codex"


def test_select_synthesizer_neutral_peer_resolution():
    chair = select_synthesizer(
        {"defaults": {"synthesizer": "neutral_peer"}},
        {"claude": {}, "codex": {}, "gemini": {}},
        stances={"claude": "for", "codex": "against", "gemini": "neutral"},
        current=None,
    )
    assert chair == "gemini"


def test_select_synthesizer_neutral_peer_no_stances_raises():
    with pytest.raises(ValueError, match="no stances were assigned"):
        select_synthesizer(
            {"defaults": {"synthesizer": "neutral_peer"}},
            {"claude": {}},
            stances=None,
            current=None,
        )


def test_select_synthesizer_neutral_peer_no_neutral_assigned_raises():
    with pytest.raises(ValueError, match="no participant in this run has stance=neutral"):
        select_synthesizer(
            {"defaults": {"synthesizer": "neutral_peer"}},
            {"a": {}, "b": {}},
            stances={"a": "for", "b": "against"},
            current=None,
        )


def test_select_synthesizer_current_routes_to_host():
    chair = select_synthesizer(
        {"defaults": {"synthesizer": "current"}},
        {"claude": {}, "codex": {}},
        stances=None,
        current="claude",
    )
    assert chair == "claude"


def test_select_synthesizer_unknown_name_raises():
    with pytest.raises(ValueError, match="not a configured participant"):
        select_synthesizer(
            {"defaults": {"synthesizer": "ghost"}},
            {"claude": {}},
            stances=None,
            current=None,
        )


def test_universal_abdication_all_blocked_returns_merged_blockers():
    results = [
        ParticipantResult(
            "a",
            False,
            "RECOMMENDATION: no - too complex\nEFFORT: blocked",
            f"{ABDICATED_ERROR_PREFIX} blocked",
            1.0,
            blockers=["missing migration"],
        ),
        ParticipantResult(
            "b",
            False,
            "RECOMMENDATION: no - cannot evaluate\nEFFORT: blocked",
            f"{ABDICATED_ERROR_PREFIX} blocked",
            1.0,
            blockers=["missing migration", "no env config"],
        ),
    ]
    payload = universal_abdication(results)
    assert payload is not None
    assert payload["recommendation"] == "unknown"
    assert payload["reason"] == "all_peers_abdicated"
    # Dedup across peers, preserving first-seen order.
    assert payload["blockers"] == ["missing migration", "no env config"]
    assert payload["abdicated_peers"] == ["a", "b"]


def test_universal_abdication_mixed_returns_none():
    """If even one peer voted, do not short-circuit."""
    results = [
        ParticipantResult(
            "a", False, "EFFORT: blocked", f"{ABDICATED_ERROR_PREFIX}", 1.0
        ),
        ParticipantResult("b", True, "RECOMMENDATION: yes - looks fine", "", 1.0),
    ]
    assert universal_abdication(results) is None


def test_universal_abdication_empty_returns_none():
    assert universal_abdication([]) is None


def test_build_synthesis_prompt_cites_peers_by_name():
    results = [
        ParticipantResult(
            "claude",
            True,
            "RECOMMENDATION: yes - ship it\nLooks good overall.",
            "",
            1.0,
            stance="for",
            blockers=["needs test"],
        ),
        ParticipantResult(
            "codex",
            True,
            "RECOMMENDATION: no - migration risk",
            "",
            1.0,
            stance="against",
        ),
    ]
    prompt = build_synthesis_prompt("Should we ship?", results, None)
    # Chair must see peer names and labels and a clear "do not vote" frame.
    assert "claude (label=yes" in prompt
    assert "codex (label=no" in prompt
    assert "stance=for" in prompt
    assert "stance=against" in prompt
    assert "Your output" in prompt
    assert "headline recommendation comes" in prompt
    assert "Consensus blockers" in prompt
    assert "Verification plan" in prompt


def test_build_synthesis_prompt_includes_convergence_when_provided():
    results = [
        ParticipantResult("a", True, "RECOMMENDATION: yes - ok", "", 1.0),
    ]
    convergence = {
        "2": [{"participant": "a", "state": "converged", "similarity": 0.92}],
    }
    prompt = build_synthesis_prompt("Q?", results, convergence)
    assert "Convergence (pre-computed; do not re-derive)" in prompt
    assert "state=converged" in prompt
    assert "0.92" in prompt


def test_build_synthesis_prompt_how_positions_moved_only_when_deliberated():
    """The chair-narrated '### How positions moved' section is conditional:
    request it only on a multi-round (convergence-bearing) run, never on a
    single-round run where there is no movement to narrate."""
    results = [
        ParticipantResult("a", True, "RECOMMENDATION: yes - ok", "", 1.0),
    ]
    # Single-round run: no convergence data → section absent.
    single = build_synthesis_prompt("Q?", results, None)
    assert "### How positions moved" not in single

    # Deliberated run: convergence present → section requested.
    convergence = {
        "2": [{"participant": "a", "state": "converged", "similarity": 0.92}],
    }
    multi = build_synthesis_prompt("Q?", results, convergence)
    assert "### How positions moved" in multi


def test_build_synthesis_prompt_decision_preserves_dissent_directive():
    """L1: the Decision directive must instruct the chair to name genuine
    remaining disagreement rather than paper over it."""
    results = [ParticipantResult("a", True, "RECOMMENDATION: yes - ok", "", 1.0)]
    prompt = build_synthesis_prompt("Q?", results, None)
    assert "did not converge" in prompt
    assert "papering over it" in prompt


def test_build_synthesis_prompt_consensus_blockers_request_attribution():
    """M2: each consensus blocker bullet should be attributed to the peers
    who raised it."""
    results = [ParticipantResult("a", True, "RECOMMENDATION: yes - ok", "", 1.0)]
    prompt = build_synthesis_prompt("Q?", results, None)
    assert "claude, gemini: <blocker>" in prompt


def test_build_synthesis_prompt_even_handed_moderator_frame():
    """L2: the chair preamble must frame the chair as an even-handed
    moderator, not a third debater."""
    results = [ParticipantResult("a", True, "RECOMMENDATION: yes - ok", "", 1.0)]
    prompt = build_synthesis_prompt("Q?", results, None)
    assert "moderator, not a third debater" in prompt


def test_build_synthesis_prompt_renders_structured_evidence_readably():
    """Evidence entries are ``list[{text, tag}]`` since v0.7.0
    (``adapters._parse_tagged_entry``). The chair prompt must format
    each entry as ``[TAG] text`` (or just ``text`` when untagged) rather
    than injecting raw stringified Python dict literals like
    ``{'text': '...', 'tag': 'PUBLISHED'}`` — those would be unreadable
    noise for the chair LLM."""
    results = [
        ParticipantResult(
            "gemini",
            True,
            "RECOMMENDATION: yes - ship\nLooks good.",
            "",
            1.0,
            evidence=[
                {"text": "tests pass in CI", "tag": "observable"},
                {"text": "docs at example.com confirm", "tag": "published"},
                {"text": "untagged claim with no source", "tag": None},
            ],
            blockers=["plain string blocker"],
        ),
    ]
    prompt = build_synthesis_prompt("Q?", results, None)
    # The bug we are guarding against: raw dict literals leaking into the
    # prompt. Any of these substrings indicate the regression returned.
    assert "{'text':" not in prompt
    assert "'tag':" not in prompt
    assert "{\"text\":" not in prompt
    assert "'PUBLISHED'" not in prompt
    # Tagged entries render as [TAG] text; untagged as bare text.
    assert "[OBSERVABLE] tests pass in CI" in prompt
    assert "[PUBLISHED] docs at example.com confirm" in prompt
    assert "untagged claim with no source" in prompt
    # Sibling list field (blockers) is still list[str]; render unchanged.
    assert "plain string blocker" in prompt


def test_build_synthesis_prompt_truncates_oversize():
    """When the assembled prompt exceeds max_chars, truncation marker appears.

    Note: per-peer rationale is already capped (MAX_RATIONALE_CHARS=320),
    so a single huge response will not overflow. Use multiple peers + a
    small budget to exercise the prompt-level truncation path.
    """
    results = [
        ParticipantResult(
            f"peer{i}", True, "RECOMMENDATION: yes - ok\n" + ("filler " * 50), "", 1.0
        )
        for i in range(8)
    ]
    prompt = build_synthesis_prompt("Q?", results, None, max_chars=500)
    assert len(prompt) <= 500
    assert "synthesis prompt truncated" in prompt
