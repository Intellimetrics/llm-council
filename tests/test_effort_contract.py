"""Tests for the optional Pick A response envelope and abdication detection."""

from __future__ import annotations

from llm_council.adapters import (
    ABDICATED_ERROR_PREFIX,
    ERROR_KIND_ABDICATED,
    KNOWN_ERROR_KINDS,
    ParticipantResult,
    _extract_response_envelope,
    _is_abdication,
    _with_envelope,
    classify_error,
)


def test_envelope_parses_full_shape():
    text = (
        "RECOMMENDATION: tradeoff - some risk\n"
        "EFFORT: full\n"
        "CONFIDENCE: high\n"
        "RISK: medium\n"
        "\n"
        "BLOCKERS:\n"
        "- missing migration\n"
        "- no tenant test\n"
        "\n"
        "EVIDENCE:\n"
        "- src/auth.py:42\n"
        "\n"
        "TESTS_TO_RUN:\n"
        "- pytest tests/auth/\n"
        "\n"
        "ASSUMPTIONS:\n"
        "- staging mirrors prod schema\n"
    )
    env = _extract_response_envelope(text)
    assert env["effort"] == "full"
    assert env["confidence"] == "high"
    assert env["risk"] == "medium"
    assert env["blockers"] == ["missing migration", "no tenant test"]
    # v0.7: evidence parsed into list[{text, tag}] for tag distribution
    # telemetry. Untagged entries return `tag=None`.
    assert env["evidence"] == [{"text": "src/auth.py:42", "tag": None}]
    assert env["tests_to_run"] == ["pytest tests/auth/"]
    assert env["assumptions"] == ["staging mirrors prod schema"]


def test_envelope_ignores_fenced_blocks():
    text = (
        "RECOMMENDATION: yes - ship it\n"
        "```\n"
        "EFFORT: blocked\n"
        "RISK: critical\n"
        "```\n"
        "EFFORT: full\n"
    )
    env = _extract_response_envelope(text)
    # Fenced lines must not contribute — only the EFFORT outside the fence.
    assert env["effort"] == "full"
    assert env["risk"] is None


def test_envelope_empty_input_safe():
    env = _extract_response_envelope("")
    assert env["effort"] is None
    assert env["blockers"] == []


def test_with_envelope_populates_dataclass():
    r = ParticipantResult(
        "a",
        True,
        "RECOMMENDATION: tradeoff - ok\nEFFORT: limited\nBLOCKERS:\n- need spec",
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True
    assert out.effort == "limited"
    assert out.blockers == ["need spec"]


def test_abdication_classified_and_terminal():
    """Peer says blocked but lists no concrete missing artifact — abdication."""
    r = ParticipantResult(
        "a",
        True,
        "RECOMMENDATION: no - too complex to evaluate\nEFFORT: blocked\n",
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is False, "abdication must flip ok=False so quorum drops"
    assert out.error.startswith(ABDICATED_ERROR_PREFIX)
    assert out.effort == "blocked"
    assert classify_error(out.error) == ERROR_KIND_ABDICATED
    assert ERROR_KIND_ABDICATED in KNOWN_ERROR_KINDS


def test_blocked_with_concrete_blockers_is_not_abdication():
    """Self-reported blocked + named missing artifact = useful, not abdication."""
    r = ParticipantResult(
        "b",
        True,
        (
            "RECOMMENDATION: no - need data\n"
            "EFFORT: blocked\n"
            "BLOCKERS:\n"
            "- missing migration SQL file\n"
            "- no env config\n"
        ),
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True, "blocked-with-blockers is honest, not abdication"
    assert out.blockers == ["missing migration SQL file", "no env config"]
    assert out.error == ""


def test_substantive_no_vote_not_misclassified():
    """A 'no' vote without EFFORT: blocked must remain a valid vote."""
    r = ParticipantResult(
        "c",
        True,
        "RECOMMENDATION: no - real issue here\nThis breaks tenant isolation.",
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True
    assert out.effort is None


def test_abdication_blocked_with_assumptions_not_abdication():
    """Naming assumptions explicitly counts as effort, not abdication."""
    r = ParticipantResult(
        "d",
        True,
        (
            "RECOMMENDATION: tradeoff - depends\n"
            "EFFORT: blocked\n"
            "ASSUMPTIONS:\n"
            "- the deploy script handles secrets\n"
        ),
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True


def test_is_abdication_requires_label():
    """Abdication only fires when there is a RECOMMENDATION label to invalidate."""
    env = {"effort": "blocked", "blockers": [], "assumptions": []}
    assert _is_abdication(env, "EFFORT: blocked\n") is False


def test_classify_error_routes_abdication():
    assert classify_error(f"{ABDICATED_ERROR_PREFIX} sample") == ERROR_KIND_ABDICATED


# --- v0.10.x self-review regressions -------------------------------------
# When the council reviewed itself in the v0.10.1 self-review run, three
# envelope-parser bugs surfaced in `structured_results` even though the
# raw markdown was intact:
#   1. RISK truncated to first word ("the" / "upstream")
#   2. comma-separated inline EVIDENCE / TESTS_TO_RUN collapsed to one
#      mangled entry (the commas left behind after tag-stripping)
#   3. BLOCKERS: none stored as ["none"] rather than []
# These tests pin the fixes.


def test_risk_preserves_full_sentence_and_case():
    """RISK is contractually a sentence (context.py prompt), not an enum.
    The v0.10.1 bug shared a single-word value regex with EFFORT /
    CONFIDENCE / CONTINUE_DEBATE, truncating sentences to the first word
    and lowercasing them."""
    text = (
        "RECOMMENDATION: yes - ship\n"
        "RISK: The single biggest risk is external-contract drift "
        "at the MCP schema and native-CLI permission flags.\n"
    )
    env = _extract_response_envelope(text)
    assert env["risk"] == (
        "The single biggest risk is external-contract drift "
        "at the MCP schema and native-CLI permission flags."
    )


def test_risk_short_word_value_still_works():
    """Backward-compat: peers that emit `RISK: medium` enum-style
    continue to be parsed (now case-preserving since RISK is free-form)."""
    env = _extract_response_envelope("RECOMMENDATION: tradeoff - ok\nRISK: medium\n")
    assert env["risk"] == "medium"


def test_risk_strips_trailing_markdown_emphasis():
    """Tolerate `**bold**` styling around the value."""
    env = _extract_response_envelope(
        "RECOMMENDATION: yes - ok\nRISK: **upstream CLI flag deprecations**\n"
    )
    assert env["risk"] == "upstream CLI flag deprecations"


def test_inline_evidence_splits_comma_separated_verified_cites():
    """Real peers emit multiple `[VERIFIED:...]` cites on one EVIDENCE
    line. Before the fix this collapsed to a single entry with `text=", , ,"`
    and only the first cite's metadata. After: one entry per cite."""
    text = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE: [VERIFIED:llm_council/defaults.py:123-223], "
        "[VERIFIED:llm_council/adapters.py:2871-2881], "
        "[VERIFIED:llm_council/orchestrator.py:448-604]\n"
    )
    env = _extract_response_envelope(text)
    assert len(env["evidence"]) == 3
    paths = [entry["path"] for entry in env["evidence"]]
    assert paths == [
        "llm_council/defaults.py",
        "llm_council/adapters.py",
        "llm_council/orchestrator.py",
    ]
    assert all(entry["tag"] == "verified" for entry in env["evidence"])


def test_inline_tests_to_run_splits_comma_separated_commands():
    """Gemini's actual v0.10.1 output: `TESTS_TO_RUN: pytest a, pytest b,
    pytest c`. Before the fix this was one item. After: three."""
    env = _extract_response_envelope(
        "RECOMMENDATION: yes - ok\n"
        "TESTS_TO_RUN: pytest tests/test_timeout_policy.py, "
        "pytest tests/test_evidence_tags.py, "
        "pytest tests/test_adapters_safety.py\n"
    )
    assert env["tests_to_run"] == [
        "pytest tests/test_timeout_policy.py",
        "pytest tests/test_evidence_tags.py",
        "pytest tests/test_adapters_safety.py",
    ]


def test_inline_blockers_none_normalizes_to_empty_list():
    """`BLOCKERS: none` is the documented sentinel for "no blockers".
    Storing `["none"]` is wrong on two fronts: it misrepresents the
    semantic in downstream consumers (MCP `structured_results`) and it
    defeats abdication detection because the truthiness check treats
    `["none"]` as a real blocker."""
    env = _extract_response_envelope(
        "RECOMMENDATION: yes - ok\nBLOCKERS: none\nASSUMPTIONS: n/a\n"
    )
    assert env["blockers"] == []
    assert env["assumptions"] == []


def test_abdication_with_none_sentinels_classifies_correctly():
    """Concrete fallout from the sentinel normalization: a peer that
    says EFFORT: blocked + BLOCKERS: none + ASSUMPTIONS: none is
    abdicating (won't name what's blocking it). Before the fix, both
    list fields were `["none"]` (truthy), so abdication didn't fire."""
    r = ParticipantResult(
        "e",
        True,
        (
            "RECOMMENDATION: no - cannot evaluate\n"
            "EFFORT: blocked\n"
            "BLOCKERS: none\n"
            "ASSUMPTIONS: none\n"
        ),
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is False, (
        "EFFORT: blocked with sentinel-only blockers and assumptions "
        "must classify as abdication"
    )
    assert out.error.startswith(ABDICATED_ERROR_PREFIX)


def test_inline_evidence_still_handles_single_entry():
    """Backward-compat with the pass-9 single-inline-entry shape
    (`test_envelope_inline_evidence_line_form_is_captured`): a single
    inline entry with no commas continues to produce one item."""
    env = _extract_response_envelope(
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE: [PUBLISHED] - tagged inline form\n"
    )
    assert env["evidence"] == [
        {"text": "tagged inline form", "tag": "published"},
    ]
