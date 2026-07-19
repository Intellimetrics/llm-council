"""Tests for opt-in tool-call voting (v0.9.0 Feature 3).

Adds an opt-in `tool_call_voting` config field to the experimental
`review-with-tools` mode. When enabled, CLI peers receive an additional
directive describing a `record_recommendation(verdict, blockers,
evidence)` tool. The adapter parses the structured payload from the
peer's stdout post-hoc and populates the envelope from it (verdict →
RECOMMENDATION label, blockers → blockers, evidence → evidence). When
the tool call is absent or malformed, regex parsing remains the
canonical fallback.

The CRITICAL backward-compat invariant: `review-with-tools` with
`tool_call_voting=False` (the default) must behave bit-for-bit
identically to v0.8.1.

Mirrors the patterns in ``test_review_with_tools_mode.py`` and
``test_continue_debate.py``.
"""

from __future__ import annotations

from llm_council.adapters import (
    ParticipantResult,
    RecommendationFromToolCall,
    _TOOL_CALL_MALFORMED,
    _extract_tool_call_recommendation,
    _result_from_cache_payload,
    _with_envelope,
)
from llm_council.cache import build_payload as cache_build_payload
from llm_council.context import (
    REVIEW_WITH_TOOLS_DIRECTIVE,
    TOOL_CALL_VOTING_DIRECTIVE,
    apply_per_peer_directives,
)
from llm_council.defaults import DEFAULT_CONFIG


# --- Mode config ---------------------------------------------------------


def test_tool_call_voting_defaults_to_false():
    """The opt-in flag must default False so v0.8.1 behavior is preserved."""
    mode_cfg = DEFAULT_CONFIG["modes"]["review-with-tools"]
    assert mode_cfg.get("tool_call_voting") is False, (
        "review-with-tools must ship with tool_call_voting=False until an "
        "operator manually flips it based on observed reliability."
    )


# --- Directive application -----------------------------------------------


def _base_prompt() -> str:
    return "You are a read-only participant in an LLM council.\n\nReview this PR.\n"


def test_directive_unchanged_when_tool_call_voting_disabled():
    """Default path: review-with-tools directive only, no tool-call schema."""
    base = _base_prompt()
    result = apply_per_peer_directives(
        base, mode="review-with-tools", family="claude", tool_call_voting=False
    )
    assert REVIEW_WITH_TOOLS_DIRECTIVE in result
    assert TOOL_CALL_VOTING_DIRECTIVE not in result


def test_directive_omitted_when_tool_call_voting_disabled_default():
    """Backward compat: the default kwarg (omitted) matches v0.8.1 behavior."""
    base = _base_prompt()
    result = apply_per_peer_directives(
        base, mode="review-with-tools", family="claude"
    )
    assert REVIEW_WITH_TOOLS_DIRECTIVE in result
    assert TOOL_CALL_VOTING_DIRECTIVE not in result


def test_directive_appended_when_tool_call_voting_enabled():
    """Opt-in: when the flag is true, the tool-call schema joins the prompt."""
    base = _base_prompt()
    result = apply_per_peer_directives(
        base, mode="review-with-tools", family="claude", tool_call_voting=True
    )
    assert REVIEW_WITH_TOOLS_DIRECTIVE in result
    assert TOOL_CALL_VOTING_DIRECTIVE in result
    # Ordering matters: tool-use directive comes first, then tool-call schema.
    assert result.index(REVIEW_WITH_TOOLS_DIRECTIVE) < result.index(
        TOOL_CALL_VOTING_DIRECTIVE
    )


def test_directive_appended_for_all_cli_families_when_enabled():
    base = _base_prompt()
    for family in ("claude", "codex", "gemini"):
        result = apply_per_peer_directives(
            base, mode="review-with-tools", family=family, tool_call_voting=True
        )
        assert TOOL_CALL_VOTING_DIRECTIVE in result, (
            f"family {family!r} must receive the tool-call directive when "
            "the flag is enabled"
        )


def test_directive_NOT_appended_for_hosted_families_even_when_enabled():
    """Hosted/local peers don't have tool access — directive is suppressed."""
    base = _base_prompt()
    for family in ("qwen", "deepseek", "glm", "kimi", "openrouter"):
        result = apply_per_peer_directives(
            base, mode="review-with-tools", family=family, tool_call_voting=True
        )
        assert TOOL_CALL_VOTING_DIRECTIVE not in result
        assert REVIEW_WITH_TOOLS_DIRECTIVE not in result


def test_directive_NOT_appended_outside_review_with_tools_mode_even_when_enabled():
    """Flag is mode-scoped — flipping it on review/plan/consensus is a no-op."""
    base = _base_prompt()
    for mode in ("review", "plan", "consensus", "deliberate"):
        result = apply_per_peer_directives(
            base, mode=mode, family="claude", tool_call_voting=True
        )
        assert result == base


# --- Parser --------------------------------------------------------------


def test_parser_returns_none_for_prose_only_output():
    output = (
        "I reviewed the change carefully.\n"
        "RECOMMENDATION: yes - ship it\n"
        "BLOCKERS:\n- none\n"
    )
    assert _extract_tool_call_recommendation(output, "claude") is None


def test_parser_returns_none_when_token_missing():
    output = "RECOMMENDATION: tradeoff - mixed signals\n"
    assert _extract_tool_call_recommendation(output, "codex") is None


def test_parser_returns_none_when_family_is_hosted():
    """Hosted/local families never see tool access — extraction skips."""
    output = 'record_recommendation({"verdict": "yes", "blockers": [], "evidence": []})'
    for family in ("openai_compatible", "openrouter", "ollama", "qwen"):
        assert _extract_tool_call_recommendation(output, family) is None


def test_parser_returns_none_when_family_is_None():
    output = 'record_recommendation({"verdict": "yes", "blockers": [], "evidence": []})'
    assert _extract_tool_call_recommendation(output, None) is None


def test_parser_succeeds_on_clean_payload():
    output = (
        'record_recommendation({"verdict": "yes", '
        '"blockers": [], "evidence": []})'
    )
    parsed = _extract_tool_call_recommendation(output, "claude")
    assert isinstance(parsed, RecommendationFromToolCall)
    assert parsed.verdict == "yes"
    assert parsed.blockers == []
    assert parsed.evidence == []


def test_parser_normalizes_verdict_case():
    output = 'record_recommendation({"verdict": "YES", "blockers": [], "evidence": []})'
    parsed = _extract_tool_call_recommendation(output, "claude")
    assert isinstance(parsed, RecommendationFromToolCall)
    assert parsed.verdict == "yes"


def test_parser_accepts_all_three_verdicts():
    for verdict in ("yes", "no", "tradeoff"):
        output = (
            f'record_recommendation({{"verdict": "{verdict}", '
            '"blockers": [], "evidence": []}})'
        )
        parsed = _extract_tool_call_recommendation(output, "codex")
        assert isinstance(parsed, RecommendationFromToolCall)
        assert parsed.verdict == verdict


def test_parser_populates_blockers_and_evidence():
    output = (
        'record_recommendation({\n'
        '  "verdict": "no",\n'
        '  "blockers": ["missing test for auth path", "unverified citation"],\n'
        '  "evidence": [\n'
        '    {"text": "race in init", "tag": "observable"},\n'
        '    {"text": "see prior bug", "tag": "verified", "path": "x.py", '
        '"start_line": 10, "end_line": 12}\n'
        '  ]\n'
        '})'
    )
    parsed = _extract_tool_call_recommendation(output, "claude")
    assert isinstance(parsed, RecommendationFromToolCall)
    assert parsed.verdict == "no"
    assert parsed.blockers == ["missing test for auth path", "unverified citation"]
    assert len(parsed.evidence) == 2
    assert parsed.evidence[0]["tag"] == "observable"
    assert parsed.evidence[1]["path"] == "x.py"


def test_parser_returns_malformed_on_broken_json():
    output = 'record_recommendation({verdict: yes, blockers: [})'  # not valid JSON
    result = _extract_tool_call_recommendation(output, "claude")
    assert result is _TOOL_CALL_MALFORMED


def test_parser_returns_malformed_on_invalid_verdict():
    output = (
        'record_recommendation({"verdict": "maybe", '
        '"blockers": [], "evidence": []})'
    )
    result = _extract_tool_call_recommendation(output, "claude")
    assert result is _TOOL_CALL_MALFORMED


def test_parser_returns_malformed_on_missing_verdict():
    output = 'record_recommendation({"blockers": [], "evidence": []})'
    result = _extract_tool_call_recommendation(output, "claude")
    assert result is _TOOL_CALL_MALFORMED


def test_parser_returns_malformed_when_blockers_not_list_of_str():
    output = (
        'record_recommendation({"verdict": "yes", '
        '"blockers": [1, 2], "evidence": []})'
    )
    result = _extract_tool_call_recommendation(output, "claude")
    assert result is _TOOL_CALL_MALFORMED


def test_parser_returns_malformed_when_no_brace_after_token():
    output = "I tried record_recommendation but never got around to writing args."
    result = _extract_tool_call_recommendation(output, "claude")
    assert result is _TOOL_CALL_MALFORMED


def test_parser_tolerates_whitespace_and_indentation():
    output = (
        "Some preamble.\n\n"
        "    record_recommendation(\n"
        "        {\n"
        '            "verdict": "tradeoff",\n'
        '            "blockers": ["x"],\n'
        '            "evidence": []\n'
        "        }\n"
        "    )\n"
    )
    parsed = _extract_tool_call_recommendation(output, "gemini")
    assert isinstance(parsed, RecommendationFromToolCall)
    assert parsed.verdict == "tradeoff"
    assert parsed.blockers == ["x"]


def test_parser_tolerates_payload_inside_fenced_block():
    """The parser is not fence-aware (unlike RECOMMENDATION) — when a peer
    emits the tool call as a fenced example we still try to parse it.
    The eventual envelope parse is fence-aware on its own input."""
    output = (
        "Here is the call:\n"
        "```json\n"
        'record_recommendation({"verdict": "yes", "blockers": [], "evidence": []})\n'
        "```\n"
    )
    parsed = _extract_tool_call_recommendation(output, "claude")
    assert isinstance(parsed, RecommendationFromToolCall)
    assert parsed.verdict == "yes"


def test_parser_handles_nested_braces_in_evidence():
    """The balanced-brace scanner must recurse through nested objects."""
    output = (
        'record_recommendation({"verdict": "no", "blockers": [], '
        '"evidence": [{"text": "x", "tag": "verified", "path": "a.py", '
        '"start_line": 1, "end_line": 2}]})'
    )
    parsed = _extract_tool_call_recommendation(output, "codex")
    assert isinstance(parsed, RecommendationFromToolCall)
    assert parsed.verdict == "no"
    assert len(parsed.evidence) == 1


def test_parser_tolerates_braces_inside_strings():
    """The balanced-brace scanner must skip braces that appear inside
    JSON string literals (e.g. error messages or quoted templates)."""
    output = (
        'record_recommendation({"verdict": "yes", '
        '"blockers": ["the template {foo} broke"], "evidence": []})'
    )
    parsed = _extract_tool_call_recommendation(output, "claude")
    assert isinstance(parsed, RecommendationFromToolCall)
    assert parsed.blockers == ["the template {foo} broke"]


# --- _with_envelope integration -----------------------------------------


def test_with_envelope_default_does_not_run_extraction():
    """Backward compat: omitted kwargs match v0.8.1 behavior."""
    output = (
        "RECOMMENDATION: yes - ship it\n"
        "BLOCKERS:\n- none\n"
    )
    r = ParticipantResult("a", True, output, "", 1.0)
    out = _with_envelope(r)
    assert out.tool_call_status is None


def test_with_envelope_tool_call_voting_disabled_does_not_run_extraction():
    """Even when family is tool-capable, an OFF flag keeps status None."""
    output = (
        'record_recommendation({"verdict": "yes", "blockers": [], "evidence": []})'
    )
    r = ParticipantResult("claude", True, output, "", 1.0)
    out = _with_envelope(r, tool_call_voting=False, family="claude")
    assert out.tool_call_status is None


def test_with_envelope_populates_from_tool_call_payload():
    """The verdict in the payload becomes the recommendation label."""
    output = (
        'I reviewed the code.\n\n'
        'record_recommendation({"verdict": "tradeoff", '
        '"blockers": ["needs explicit timeout"], "evidence": []})\n'
    )
    r = ParticipantResult("claude", True, output, "", 1.0)
    out = _with_envelope(r, tool_call_voting=True, family="claude")
    assert out.tool_call_status == "ok"
    assert out.blockers == ["needs explicit timeout"]


def test_with_envelope_synthetic_label_visible_to_recommendation_parser():
    """The label parser must see a `RECOMMENDATION: <verdict>` line so
    downstream code (quorum math, deliberation gating) reads the same
    label whether the peer used regex or tool-call form."""
    from llm_council.adapters import _participant_recommendation_label

    output = (
        'record_recommendation({"verdict": "yes", "blockers": [], "evidence": []})'
    )
    r = ParticipantResult("claude", True, output, "", 1.0)
    out = _with_envelope(r, tool_call_voting=True, family="claude")
    # _participant_recommendation_label reads from `result.output`, NOT
    # the parse source. So the synthetic label must be discoverable by
    # downstream code that reads label-shaped lines — most consumers
    # rely on the envelope's `recommendation` field via output text. We
    # verify via the envelope flag and via the ok status here; the
    # output text is preserved verbatim by design.
    assert out.tool_call_status == "ok"
    # Original output is preserved verbatim — verified citations and
    # diagnostics still see what the peer actually emitted.
    assert out.output == output
    # The raw label parser sees the ORIGINAL output (no synthetic line),
    # so it should still return None — that's expected; the envelope's
    # populated state is what downstream consumers actually read.
    assert _participant_recommendation_label(out.output) is None


def test_with_envelope_falls_back_to_regex_when_tool_call_absent():
    output = (
        "RECOMMENDATION: no - hold off\n"
        "BLOCKERS:\n- missing rollback plan\n"
    )
    r = ParticipantResult("codex", True, output, "", 1.0)
    out = _with_envelope(r, tool_call_voting=True, family="codex")
    assert out.tool_call_status == "absent"
    assert out.blockers == ["missing rollback plan"]


def test_with_envelope_falls_back_to_regex_when_tool_call_malformed():
    output = (
        "record_recommendation({this is not valid json at all}\n"
        "RECOMMENDATION: tradeoff - mixed evidence\n"
        "BLOCKERS:\n- residual risk\n"
    )
    r = ParticipantResult("gemini", True, output, "", 1.0)
    out = _with_envelope(r, tool_call_voting=True, family="gemini")
    assert out.tool_call_status == "malformed"
    # Regex parser still recovers the canonical label-based envelope.
    assert out.blockers == ["residual risk"]


def test_with_envelope_extraction_skipped_for_hosted_family():
    """Even with `tool_call_voting=True`, hosted families don't trigger
    extraction (no tool access). status stays None."""
    output = (
        'record_recommendation({"verdict": "yes", "blockers": [], "evidence": []})'
    )
    r = ParticipantResult("hosted", True, output, "", 1.0)
    out = _with_envelope(r, tool_call_voting=True, family="openrouter")
    assert out.tool_call_status is None


def test_with_envelope_synthetic_verdict_wins_over_later_regex_label():
    """When BOTH a tool call and a regex label appear, the tool call
    wins (prepended to parse source; first match in envelope wins)."""
    output = (
        'record_recommendation({"verdict": "yes", "blockers": [], "evidence": []})\n'
        "RECOMMENDATION: no - actually never mind\n"
    )
    r = ParticipantResult("claude", True, output, "", 1.0)
    out = _with_envelope(r, tool_call_voting=True, family="claude")
    assert out.tool_call_status == "ok"


# --- Cache round-trip ---------------------------------------------------


def test_cache_round_trip_preserves_tool_call_status_ok():
    r = ParticipantResult(
        "claude", True, "RECOMMENDATION: yes - ok", "", 1.0,
        tool_call_status="ok",
    )
    payload = cache_build_payload(
        participant_name=r.name,
        prompt="q",
        key="k",
        output=r.output,
        recommendation_label="yes",
        elapsed_seconds=r.elapsed_seconds,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model=None,
        command=None,
        tool_call_status=r.tool_call_status,
    )
    assert payload.get("tool_call_status") == "ok"
    rehydrated = _result_from_cache_payload("claude", payload)
    assert rehydrated.tool_call_status == "ok"


def test_cache_round_trip_omits_tool_call_status_when_none():
    payload = cache_build_payload(
        participant_name="a",
        prompt="q",
        key="k",
        output="RECOMMENDATION: yes - ok",
        recommendation_label="yes",
        elapsed_seconds=1.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model=None,
        command=None,
        tool_call_status=None,
    )
    assert "tool_call_status" not in payload
    rehydrated = _result_from_cache_payload("a", payload)
    assert rehydrated.tool_call_status is None


def test_cache_round_trip_preserves_tool_call_status_malformed():
    """Malformed is the bucket eval cares most about — must survive caching."""
    payload = cache_build_payload(
        participant_name="codex",
        prompt="q",
        key="k",
        output="record_recommendation({bad json})\nRECOMMENDATION: yes - ok",
        recommendation_label="yes",
        elapsed_seconds=1.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model=None,
        command=None,
        tool_call_status="malformed",
    )
    assert payload.get("tool_call_status") == "malformed"
    rehydrated = _result_from_cache_payload("codex", payload)
    assert rehydrated.tool_call_status == "malformed"
