"""Section-coverage validator (Change 2).

Anchors: pass-7 transcript at `.llm-council/runs/20260516_100758_*`. The
end-to-end "real pass-7 response" test lives in
`tests/test_pass7_regression.py`; this file covers the parser, matcher,
and validation-error wiring in isolation.
"""

from __future__ import annotations

from llm_council.adapters import (
    INCOMPLETE_RESPONSE_PREFIX,
    KNOWN_ERROR_KINDS,
    _response_validation_error,
    classify_error,
)
from llm_council.sections import (
    REQUIRED_SECTION_HEADER_RE,
    _extract_salient_tokens,
    _section_present,
    required_sections,
    required_sections_missing,
)


# --- Regex detection ------------------------------------------------------

def test_regex_matches_pass7_part2_marker():
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "2"


def test_regex_matches_pass7_part6_marker_with_qualifier():
    """`(REQUIRED BY COUNCIL INVARIANTS)` form must match too."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "PART 6 — RECOMMENDATION (REQUIRED BY COUNCIL INVARIANTS)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "6"


def test_regex_ignores_unmarked_part_headers():
    """`PART 3 — FORCED YES/NO` without `(REQUIRED)` is NOT detected.
    The validator only fires on explicitly-marked sections."""
    text = "PART 3 — FORCED YES/NO ON THE HIGH-STAKES CLAIMS"
    assert not list(REQUIRED_SECTION_HEADER_RE.finditer(text))


def test_regex_ignores_required_in_prose():
    """A `Required subsections:` line in prose isn't a section header."""
    text = "Required subsections:\nP1. ONE behavior...\n"
    assert not list(REQUIRED_SECTION_HEADER_RE.finditer(text))


def test_regex_tolerates_hyphen_dash_variants():
    """Em-dash, en-dash, and ASCII hyphen all work as the separator."""
    for sep in ("—", "–", "-"):
        line = f"PART 2 {sep} GRID (REQUIRED)"
        matches = list(REQUIRED_SECTION_HEADER_RE.finditer(line))
        assert len(matches) == 1, f"separator {sep!r} should match"


# --- Salient-token extraction --------------------------------------------

def test_salient_tokens_keep_4plus_caps_words():
    assert _extract_salient_tokens("CONCEPT-BY-CONCEPT GRID") == ["CONCEPT", "CONCEPT", "GRID"]


def test_salient_tokens_drop_stopwords():
    assert "AND" not in _extract_salient_tokens("FORCED YES AND NO")
    assert "OR" not in _extract_salient_tokens("FORCED YES OR NO")


def test_salient_tokens_drop_short_tokens():
    """Tokens shorter than 4 chars (UI, BY, IT) are too generic to match on."""
    tokens = _extract_salient_tokens("UI BY IT")
    assert tokens == []


# --- _section_present matcher --------------------------------------------

def test_section_present_via_literal_part_n():
    """Response says `## PART 2 — Notes`; matches the PART 2 requirement
    without needing salient tokens."""
    response_upper = "## PART 2 — NOTES HERE\nSome content\n"
    req = {"num": "2", "title_tokens": ["IRRELEVANT"]}
    assert _section_present(response_upper, req) is True


def test_section_present_via_paraphrased_title():
    """Response uses paraphrased header (codex used `**Concept Grid**`
    in pass-7); both salient tokens must appear within a 200-char window."""
    response_upper = "**CONCEPT GRID**\nC1. foo\n"
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert _section_present(response_upper, req) is True


def test_section_present_rejects_distant_tokens():
    """Tokens far apart aren't a section; they're prose accidentally
    mentioning both words. The 200-char window guards against this."""
    response_upper = "CONCEPT was used" + ("x" * 500) + "GRID was used"
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert _section_present(response_upper, req) is False


def test_section_present_no_tokens_defaults_to_present():
    """A title with no salient tokens defaults to present (don't false-pos)."""
    response_upper = "anything"
    req = {"num": "2", "title_tokens": []}
    assert _section_present(response_upper, req) is True


# --- required_sections_missing end-to-end --------------------------------

def test_returns_empty_when_prompt_has_no_required_markers():
    """No (REQUIRED) headers = no-op. The validator stays silent."""
    assert required_sections_missing("just a question", "any response") == []


def test_skips_part6_recommendation():
    """PART 6 (RECOMMENDATION) is the existing label check's job — the
    section validator must not double-fault."""
    prompt = (
        "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)\n"
        "PART 6 — RECOMMENDATION (REQUIRED BY COUNCIL INVARIANTS)\n"
    )
    # Response satisfies PART 2 but omits the RECOMMENDATION label
    # entirely. PART 6 must still be skipped by the section validator —
    # the absence of the label is the label validator's responsibility.
    response = "**Concept Grid**\nC1. foo\n"
    missing = required_sections_missing(prompt, response)
    assert missing == []


def test_skips_part6_recommendation_and_rationale():
    """`PART 6 — RECOMMENDATION AND RATIONALE` yields title_tokens
    `["RECOMMENDATION", "RATIONALE"]` (AND is a stopword). The exclusion
    must still fire — otherwise peers that emit a valid `RECOMMENDATION:`
    label get falsely failed for not paraphrasing "RATIONALE" in a
    section header. Pass-8 codex+gemini finding."""
    prompt = "PART 6 — RECOMMENDATION AND RATIONALE (REQUIRED)\n"
    response = "RECOMMENDATION: yes - rationale follows in prose\n"
    missing = required_sections_missing(prompt, response)
    assert missing == []


def test_skips_part6_recommendation_summary():
    """`PART 6 — RECOMMENDATION SUMMARY` -> title_tokens
    `["RECOMMENDATION", "SUMMARY"]`. Same shape as the AND RATIONALE
    case; exclusion must fire."""
    prompt = "PART 6 — RECOMMENDATION SUMMARY (REQUIRED)\n"
    response = "RECOMMENDATION: tradeoff - summary in prose\n"
    missing = required_sections_missing(prompt, response)
    assert missing == []


def test_skips_part_where_recommendation_is_leading_title_token():
    """Approach A: any title whose FIRST salient token is RECOMMENDATION
    is excluded — even if it isn't literally PART 6. This handles user
    renumbering (`PART 3 — RECOMMENDATION COMPONENTS (REQUIRED)`) without
    falling back to fragile num-based matching. Trade-off: a section
    titled `RECOMMENDATION COMPONENTS` is treated as the label's
    territory rather than gated independently. Reasonable because the
    label-bearing line is what the title is about."""
    prompt = "PART 3 — RECOMMENDATION COMPONENTS (REQUIRED)\n"
    response = "RECOMMENDATION: tradeoff - components inline\n"
    missing = required_sections_missing(prompt, response)
    assert missing == []


def test_does_not_skip_when_recommendation_is_secondary_modifier():
    """Conservative side of Approach A: if RECOMMENDATION is NOT the
    leading title token (e.g. `DETAILED RECOMMENDATION ANALYSIS`), the
    section gates normally. The real subject is `DETAILED ... ANALYSIS`,
    not the label."""
    prompt = "PART 3 — DETAILED RECOMMENDATION ANALYSIS (REQUIRED)\n"
    response = "RECOMMENDATION: yes - no analysis section\n"
    missing = required_sections_missing(prompt, response)
    assert missing == ["PART 3 — DETAILED RECOMMENDATION ANALYSIS"]


def test_returns_specific_label_when_section_missing():
    """Missing PART 2 is reported with its label so the repair-retry
    instruction can name it."""
    prompt = (
        "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)\n"
        "PART 3 — FORCED YES/NO (REQUIRED)\n"
    )
    response = "**Concept Grid**\nfoo"
    missing = required_sections_missing(prompt, response)
    assert missing == ["PART 3 — FORCED YES/NO"]


def test_required_sections_returns_parsed_dicts():
    """`required_sections` returns structured info, not just labels."""
    reqs = required_sections("PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)")
    assert len(reqs) == 1
    assert reqs[0]["num"] == "2"
    assert "CONCEPT" in reqs[0]["title_tokens"]


# --- Validation wiring + repair-retry contract ---------------------------

def test_validation_returns_incomplete_response_error():
    """When sections are missing, the validator produces an
    `IncompleteResponse:` error string with the missing label list."""
    prompt = "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)"
    response = "RECOMMENDATION: yes - ok\nNo grid here.\n"
    cfg = {"require_sections": True}
    error = _response_validation_error(response, cfg, prompt=prompt)
    assert error.startswith(INCOMPLETE_RESPONSE_PREFIX)
    assert "CONCEPT-BY-CONCEPT GRID" in error


def test_validation_passes_when_sections_present():
    prompt = "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)"
    response = "RECOMMENDATION: yes - ok\n**Concept Grid**\nC1. foo\n"
    cfg = {"require_sections": True}
    assert _response_validation_error(response, cfg, prompt=prompt) == ""


def test_validation_skips_section_check_when_disabled():
    """`require_sections: False` disables the check while keeping label
    validation. The pass-7 gemini three-bullet shape passes here."""
    prompt = "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)"
    response = "RECOMMENDATION: tradeoff - three bullets only"
    cfg = {"require_sections": False}
    assert _response_validation_error(response, cfg, prompt=prompt) == ""


def test_validation_skips_section_check_when_prompt_not_provided():
    """Older code paths that don't have `prompt` (e.g., synthesis-chair
    validation) must still work — the validator no-ops when prompt is None."""
    response = "RECOMMENDATION: yes - ok"
    cfg = {"require_sections": True}
    assert _response_validation_error(response, cfg) == ""


def test_validation_skips_when_label_already_missing():
    """Label-missing failures short-circuit before the section check —
    no double-faulting."""
    prompt = "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)"
    response = "no label here at all"
    cfg = {"require_sections": True}
    error = _response_validation_error(response, cfg, prompt=prompt)
    assert error.startswith("InvalidParticipantResponse:")
    # Should be the label error, NOT the section error.
    assert "missing required RECOMMENDATION label" in error


def test_classify_error_routes_incomplete_response():
    """The new error_kind must round-trip through classify_error."""
    assert (
        classify_error("IncompleteResponse: missing PART 2") == "incomplete_response"
    )


def test_known_error_kinds_includes_incomplete_response():
    """Guard against drift between adapters.py and mcp_server.py."""
    assert "incomplete_response" in KNOWN_ERROR_KINDS
