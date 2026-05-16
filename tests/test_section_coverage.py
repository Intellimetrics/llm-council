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


def test_section_present_no_tokens_requires_header_shaped_part_n():
    """A title with no salient tokens has no paraphrase route to fall back
    on, so the only acceptance signal is a header-shaped `PART N`
    mention. A bare response with neither is treated as missing — we
    cannot tell whether the peer addressed the section without some
    structural anchor."""
    req = {"num": "2", "title_tokens": []}
    # No anchor at all -> missing.
    assert _section_present("ANYTHING", req) is False
    # Header-shaped PART N at line start -> present.
    assert _section_present("## PART 2\nCONTENT", req) is True
    # Prose-only mention -> still missing.
    assert _section_present("I SKIPPED PART 2.", req) is False


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


# --- pass-8 finding #9 (codex): prose mentions don't satisfy sections ----

def test_skipped_part_n_does_not_count_as_present():
    """`I skipped PART 2 because of time` is a disclaimer, not a section.
    Pass-8 finding #9 (codex) anchor."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert _section_present("I SKIPPED PART 2 BECAUSE OF TIME.", req) is False


def test_not_addressed_part_n_does_not_count_as_present():
    """`PART 2 was not addressed` is a disclaimer, not a section."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert _section_present("PART 2 WAS NOT ADDRESSED.", req) is False


def test_part_n_referenced_in_prose_does_not_count():
    """`see PART 2 instructions` references the prompt, not the response.
    A peer pointing back at the user prompt has not delivered the section."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert (
        _section_present("SEE PART 2 INSTRUCTIONS IN THE USER PROMPT.", req)
        is False
    )


def test_unable_to_complete_part_n_does_not_count():
    """`I was unable to complete PART 2` is a disclaimer."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert (
        _section_present("I WAS UNABLE TO COMPLETE PART 2.", req) is False
    )


def test_part_n_missing_from_response_does_not_count():
    """`PART 2 missing` is a disclaimer."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert (
        _section_present("PART 2 MISSING FROM THIS ANALYSIS.", req) is False
    )


def test_part_n_omitted_does_not_count():
    """`PART 2 was omitted` is a disclaimer."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert _section_present("PART 2 WAS OMITTED.", req) is False


def test_markdown_header_part_n_counts_as_present():
    """`## PART 2: Concept Grid` is a header — peer is delivering."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert (
        _section_present("## PART 2: CONCEPT GRID\nC1. FOO", req) is True
    )


def test_bold_wrapped_part_n_counts_as_present():
    """`**PART 2 — CONCEPT GRID**` is a header — peer is delivering."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert (
        _section_present("**PART 2 — CONCEPT GRID**\nC1. FOO", req) is True
    )


def test_part_n_at_line_start_with_no_marker_counts():
    """Plain `PART 2 — CONCEPT GRID` at line start is a header. The
    structural route picks it up without `##`/`**` markers."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert (
        _section_present("PART 2 — CONCEPT GRID\nC1. FOO", req) is True
    )


def test_inline_part_n_with_title_nearby_counts():
    """`My concept grid analysis for PART 2: ...` puts the salient title
    token within the confirmation window of the literal PART N. The peer
    is at least claiming to address this section."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    assert (
        _section_present(
            "MY CONCEPT GRID ANALYSIS FOR PART 2: INCLUDES 14 ENTRIES.",
            req,
        )
        is True
    )


def test_title_tokens_route_unaffected_by_skip_prose_on_literal_route():
    """Documented edge: the literal-PART-N skip-prose guard only rejects
    the literal route. If the title tokens themselves happen to live
    near a skip-prose mention, the paraphrased-title fallback still
    fires. This stays close to the spirit of the original two-route
    matcher — narrowing only the literal-mention path, not the
    paraphrase path.

    Pass-8 finding #9 intentionally tightens the LITERAL route only.
    Tightening the paraphrase route would risk false-negatives on
    genuine paraphrased headers, which is the more dangerous failure
    mode."""
    req = {"num": "2", "title_tokens": ["CONCEPT", "GRID"]}
    # Literal route alone rejects (skip-prose in window).
    assert (
        _section_present("I SKIPPED PART 2 DUE TO TIME.", req) is False
    )
    # Paraphrase route still fires when title tokens are co-located,
    # even if the surrounding sentence skip-disclaims the section.
    # This is the documented escape valve.
    assert (
        _section_present(
            "I SKIPPED PART 2 (THE CONCEPT GRID) DUE TO TIME.", req
        )
        is True
    )


# --- pass-8 finding #9 (codex): prompt regex tolerates more header shapes ---

def test_regex_matches_markdown_header_prefix():
    """`## PART 2 — CONCEPT GRID (REQUIRED)` should match. v0.7.0
    required bare `PART N` at line start; pass-8 loosens this."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "## PART 2 — CONCEPT GRID (REQUIRED)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "2"


def test_regex_matches_triple_hash_header_prefix():
    """`### PART 3 — FORCED YES/NO (REQUIRED)` should match."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "### PART 3 — FORCED YES/NO (REQUIRED)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "3"


def test_regex_matches_colon_separator():
    """`PART 2: Concept Grid (REQUIRED)` should match. Colon is a
    common heading style alternative to em-dash."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "PART 2: Concept Grid (REQUIRED)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "2"


def test_regex_matches_markdown_bold_wrapper():
    """`**PART 2 — CONCEPT GRID (REQUIRED)**` should match. Bold-wrapped
    headings appear in markdown prompts."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "**PART 2 — CONCEPT GRID (REQUIRED)**"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "2"


def test_regex_matches_title_case():
    """`PART 2 — Concept Grid (REQUIRED)` should match. Some prompt
    authors write titles in title case rather than ALL CAPS."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "PART 2 — Concept Grid (REQUIRED)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "2"


def test_regex_matches_lowercase_part_and_required():
    """`Part 2 — concept grid (required)` should match. Case-insensitive
    matching covers paraphrased prompts that come back from automated
    tools or non-native-English authors."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "Part 2 — concept grid (required)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "2"


def test_regex_matches_combined_md_header_and_colon():
    """`## PART 2: Concept Grid (REQUIRED)` — markdown header AND colon
    separator together."""
    matches = list(REQUIRED_SECTION_HEADER_RE.finditer(
        "## PART 2: Concept Grid (REQUIRED)"
    ))
    assert len(matches) == 1
    assert matches[0].group("num") == "2"


def test_regex_extracts_salient_tokens_from_title_case():
    """`Concept Grid` title-case title still extracts uppercase tokens
    via title.upper() — paraphrase matching is unaffected."""
    reqs = required_sections("## PART 2: Concept Grid (REQUIRED)")
    assert len(reqs) == 1
    assert reqs[0]["title_tokens"] == ["CONCEPT", "GRID"]


def test_regex_still_rejects_inline_prose_required():
    """`Required subsections:` must still not match — the regex must
    only fire on actual section headers, not prose."""
    text = "Required subsections:\nP1. ONE behavior...\n"
    assert not list(REQUIRED_SECTION_HEADER_RE.finditer(text))


def test_regex_still_rejects_part_n_without_required_marker():
    """`PART 3 — FORCED YES/NO` without `(REQUIRED)` still doesn't
    match. The validator stays opt-in."""
    text = "## PART 3 — FORCED YES/NO ON THE HIGH-STAKES CLAIMS"
    assert not list(REQUIRED_SECTION_HEADER_RE.finditer(text))


def test_end_to_end_with_loosened_prompt_and_strict_response():
    """Full path: a prompt with markdown-bold + colon heading is
    detected, and a prose disclaimer in the response is flagged as
    missing."""
    prompt = "## PART 2: Concept Grid (REQUIRED)\n"
    response = "RECOMMENDATION: tradeoff\nI skipped PART 2 due to time."
    missing = required_sections_missing(prompt, response)
    assert missing == ["PART 2: Concept Grid"]


def test_end_to_end_with_loosened_prompt_and_paraphrased_response():
    """Full path: a markdown-bold prompt heading is detected, and a
    paraphrased response heading passes the salient-tokens route."""
    prompt = "**PART 2 — CONCEPT GRID (REQUIRED)**\n"
    response = "RECOMMENDATION: tradeoff\n## Concept Grid\nC1. foo"
    missing = required_sections_missing(prompt, response)
    assert missing == []
