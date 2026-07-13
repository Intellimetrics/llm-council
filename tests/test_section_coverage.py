"""Section-coverage validator (Change 2).

Anchors: pass-7 transcript at `.llm-council/runs/20260516_100758_*`. The
end-to-end "real pass-7 response" test lives in
`tests/test_pass7_regression.py`; this file covers the parser, matcher,
and validation-error wiring in isolation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path as _Path
from unittest.mock import patch as _patch

import llm_council.adapters as adapters_module
from llm_council.adapters import (
    INCOMPLETE_RESPONSE_PREFIX,
    KNOWN_ERROR_KINDS,
    UNTAGGED_EVIDENCE_PREFIX,
    ParticipantResult as _ParticipantResult,
    _merge_cli_section_retry,
    _merge_hosted_section_retry,
    _response_validation_error,
    classify_error,
    run_cli_participant,
    run_ollama_participant,
    run_openai_compatible_participant,
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


def test_regex_matches_separator_without_trailing_space():
    """Regression: a separator that abuts the title (`—OVERVIEW`, `:OVERVIEW`)
    must still be detected — otherwise the validator silently no-ops for that
    common typo and the section goes unchecked."""
    for line in (
        "PART 1 —OVERVIEW (REQUIRED)",
        "PART 1:OVERVIEW (REQUIRED)",
        "PART 1 -OVERVIEW (REQUIRED)",
    ):
        matches = list(REQUIRED_SECTION_HEADER_RE.finditer(line))
        assert len(matches) == 1, f"{line!r} should match"
        assert matches[0].group("num") == "1"
        assert "OVERVIEW" in matches[0].group("title").upper()


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


def test_shared_leading_token_does_not_falsely_satisfy_sibling():
    """Regression: two REQUIRED sections sharing a salient token (SECURITY).
    A response delivering only PART 1 but mentioning the sibling's distinctive
    word ("hardening") in nearby prose must NOT falsely satisfy PART 2."""
    prompt = (
        "PART 1 — SECURITY ANALYSIS (REQUIRED)\n"
        "PART 2 — SECURITY HARDENING (REQUIRED)\n"
    )
    only_part1 = "## SECURITY ANALYSIS\nWe should consider hardening later.\n"
    missing = required_sections_missing(prompt, only_part1)
    assert missing == ["PART 2 — SECURITY HARDENING"]


def test_shared_leading_token_accepts_both_real_sections():
    """The phrase route still accepts both sections when both are delivered
    (as paraphrased headers), so the collision fix doesn't over-reject."""
    prompt = (
        "PART 1 — SECURITY ANALYSIS (REQUIRED)\n"
        "PART 2 — SECURITY HARDENING (REQUIRED)\n"
    )
    both = "## Security Analysis\nfindings...\n## Security Hardening\nsteps...\n"
    assert required_sections_missing(prompt, both) == []


def test_shared_token_section_satisfied_by_literal_part_marker():
    """A collision-prone section is still satisfied by the explicit PART N
    header even if its title isn't paraphrased contiguously."""
    prompt = (
        "PART 1 — SECURITY ANALYSIS (REQUIRED)\n"
        "PART 2 — SECURITY HARDENING (REQUIRED)\n"
    )
    response = (
        "## SECURITY ANALYSIS\nfindings...\n"
        "## PART 2\nhere is how we harden the system\n"
    )
    assert required_sections_missing(prompt, response) == []


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
# --- Section-repair retry wired through hosted/local adapters ------------
#
# The CLI adapter has triggered the section-repair retry since v0.7.0.
# Pass-8 finding #3 (codex) was that the `openai_compatible` and
# `ollama` adapters only retried label-only failures and returned the
# `IncompleteResponse:` failure directly without a repair attempt.
# These tests pin the parity contract: hosted and local peers also get
# exactly one section-repair retry, with the same gating
# (`_should_section_repair`) and the same merge semantics as the CLI
# path.

_SECTION_PROMPT = (
    "Please answer with these sections:\n"
    "PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)\n"
    "PART 6 — RECOMMENDATION (REQUIRED BY COUNCIL INVARIANTS)\n"
)


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def json(self):
        return self._body


def _openai_body(content: str, finish_reason: str = "stop") -> dict:
    return {
        "model": "x-ai/test",
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        },
    }


def _ollama_body(content: str, done_reason: str = "stop") -> dict:
    return {
        "model": "qwen3:test",
        "message": {"content": content},
        "done_reason": done_reason,
    }


def test_openai_compatible_section_repair_recovers(monkeypatch):
    """openai_compatible peer returns response missing sections → triggers
    repair retry → success on retry → ok=True."""
    calls: list[dict] = []

    bodies = [
        # First response: label present, PART 2 missing.
        _openai_body("RECOMMENDATION: yes - ok\nNo grid here, sorry.\n"),
        # Section-repair retry: label + PART 2 grid present.
        _openai_body(
            "RECOMMENDATION: yes - ok\n## PART 2 — Concept Grid\nC1. foo\n"
        ),
    ]

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(bodies[len(calls) - 1])

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "router",
            {
                "type": "openai_compatible",
                "model": "x-ai/test",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
            },
            _SECTION_PROMPT,
        )
    )

    assert len(calls) == 2, "section-repair retry must fire a second HTTP call"
    # Second call's prompt contains the section-repair directive and the
    # missing-sections list.
    retry_user_content = calls[1]["messages"][-1]["content"]
    assert "PART 2 — CONCEPT-BY-CONCEPT GRID" in retry_user_content
    assert "REQUIRED sections" in retry_user_content
    # Merged result is OK with a section-themed recovery header.
    assert result.ok is True
    assert result.error == ""
    assert "[recovered after retry]" in result.output
    assert "REQUIRED sections" in result.output
    assert "PART 2 — Concept Grid" in result.output
    assert result.repair_retry_recovered is True


def test_openai_compatible_section_repair_exhausted(monkeypatch):
    """openai_compatible peer returns response missing sections → triggers
    repair retry → still missing → ok=False, error_kind=incomplete_response."""
    calls: list[dict] = []

    bodies = [
        # First response: label present, PART 2 missing.
        _openai_body("RECOMMENDATION: yes - ok\nNo grid.\n"),
        # Section-repair retry: label present, PART 2 still missing.
        _openai_body("RECOMMENDATION: yes - ok\nStill no grid, sorry.\n"),
    ]

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(bodies[len(calls) - 1])

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "router",
            {
                "type": "openai_compatible",
                "model": "x-ai/test",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
            },
            _SECTION_PROMPT,
        )
    )

    # Exactly two calls (original + one repair retry, no third).
    assert len(calls) == 2
    assert result.ok is False
    assert result.error.startswith(INCOMPLETE_RESPONSE_PREFIX)
    assert "after one repair retry" in result.error
    assert classify_error(result.error) == "incomplete_response"
    # Both attempts are visible in the merged output for the operator.
    assert "[retry exhausted]" in result.output
    assert "No grid." in result.output
    assert "Still no grid, sorry." in result.output


def test_openai_compatible_section_repair_disabled_by_require_sections_false(
    monkeypatch,
):
    """`require_sections: False` skips the section check entirely, so no
    section-repair retry can fire even on a response that would
    otherwise trigger it."""
    calls: list[dict] = []

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(
            _openai_body("RECOMMENDATION: tradeoff - terse three-bullet response")
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "router",
            {
                "type": "openai_compatible",
                "model": "x-ai/test",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "require_sections": False,
            },
            _SECTION_PROMPT,
        )
    )

    # Only the original call; the validator no-oped, so no retry.
    assert len(calls) == 1
    assert result.ok is True
    assert result.error == ""


def test_ollama_section_repair_recovers(monkeypatch):
    """ollama peer returns response missing sections → triggers repair
    retry → success on retry → ok=True."""
    calls: list[dict] = []

    bodies = [
        _ollama_body("RECOMMENDATION: yes - ok\nNo grid here, sorry.\n"),
        _ollama_body(
            "RECOMMENDATION: yes - ok\n## PART 2 — Concept Grid\nC1. foo\n"
        ),
    ]

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(bodies[len(calls) - 1])

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_ollama_participant(
            "local",
            {
                "type": "ollama",
                "model": "qwen3:test",
                "base_url": "http://localhost:11434",
            },
            _SECTION_PROMPT,
        )
    )

    assert len(calls) == 2
    retry_user_content = calls[1]["messages"][-1]["content"]
    assert "PART 2 — CONCEPT-BY-CONCEPT GRID" in retry_user_content
    assert "REQUIRED sections" in retry_user_content
    assert result.ok is True
    assert result.error == ""
    assert "[recovered after retry]" in result.output
    assert "REQUIRED sections" in result.output
    assert "PART 2 — Concept Grid" in result.output
    assert result.repair_retry_recovered is True


def test_ollama_section_repair_exhausted(monkeypatch):
    """ollama peer returns response missing sections → triggers repair
    retry → still missing → ok=False, error_kind=incomplete_response."""
    calls: list[dict] = []

    bodies = [
        _ollama_body("RECOMMENDATION: yes - ok\nNo grid.\n"),
        _ollama_body("RECOMMENDATION: yes - ok\nStill no grid, sorry.\n"),
    ]

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(bodies[len(calls) - 1])

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_ollama_participant(
            "local",
            {
                "type": "ollama",
                "model": "qwen3:test",
                "base_url": "http://localhost:11434",
            },
            _SECTION_PROMPT,
        )
    )

    assert len(calls) == 2
    assert result.ok is False
    assert result.error.startswith(INCOMPLETE_RESPONSE_PREFIX)
    assert "after one repair retry" in result.error
    assert classify_error(result.error) == "incomplete_response"
    assert "[retry exhausted]" in result.output
    assert "No grid." in result.output
    assert "Still no grid, sorry." in result.output


def test_openai_compatible_section_repair_respects_retries_zero(monkeypatch):
    """`retries: 0` is the user's "no extra calls" kill-switch. It must
    apply to section-repair just like label-repair — one HTTP call
    total, even when sections are missing."""
    calls: list[dict] = []

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(
            _openai_body("RECOMMENDATION: yes - ok\nNo grid here.\n")
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "router",
            {
                "type": "openai_compatible",
                "model": "x-ai/test",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "retries": 0,
            },
            _SECTION_PROMPT,
        )
    )

    assert len(calls) == 1
    assert result.ok is False
    assert result.error.startswith(INCOMPLETE_RESPONSE_PREFIX)
    # Original error wording (single attempt) — no "after one repair retry".
    assert "after one repair retry" not in result.error


# --- Pass-9 fix: section-repair recovers sections but evidence untagged ---
#
# Bug (pass-9 finding A, codex + claude-round-2 + gemini-round-2 +
# qwen-round-2): the section-repair merge functions
# (`_merge_cli_section_retry`, `_merge_hosted_section_retry`) only had
# branches for retry.ok==True and retry.error startswith
# INCOMPLETE_RESPONSE_PREFIX. When strict_evidence is enabled AND the
# section-repair retry succeeded at adding the missing sections BUT
# now-visible EVIDENCE bullets lacked epistemic tags, the retry
# produced `UntaggedEvidence:` (label → sections → evidence ordering
# in `_response_validation_error`). The fall-through `return original`
# silently discarded the retry, and the operator saw only the original
# `IncompleteResponse:` error. The fix preserves the retry's
# `UntaggedEvidence:` result (so `classify_error` returns
# `untagged_evidence`) with both attempts in the merged transcript,
# and sets `section_repair_attempted=True` to guard the
# strict-evidence wrapper from chaining a third call.


_UNTAGGED_AFTER_SECTIONS_OUTPUT = (
    "RECOMMENDATION: yes - sections now present\n"
    "## PART 2 — Concept Grid\n"
    "C1. foo\n"
    "EVIDENCE:\n"
    "- plain bullet with no tag\n"
)
_TAGGED_AFTER_SECTIONS_OUTPUT = (
    "RECOMMENDATION: yes - sections now present\n"
    "## PART 2 — Concept Grid\n"
    "C1. foo\n"
    "EVIDENCE:\n"
    "- plain bullet [PUBLISHED]\n"
)


def _untagged_after_sections_error() -> str:
    return (
        f"{UNTAGGED_EVIDENCE_PREFIX} 1 EVIDENCE entry/entries lack a "
        "[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE] tag while "
        "defaults.strict_evidence is true"
    )


def test_merge_cli_section_retry_preserves_untagged_evidence_retry():
    """Unit-test for the new third branch in `_merge_cli_section_retry`.

    A retry that fixes sections but has untagged EVIDENCE used to fall
    through `return original`. The fix preserves the retry's
    `UntaggedEvidence:` error, the merged transcript with both
    attempts, and sets `section_repair_attempted=True`."""
    original = _ParticipantResult(
        name="peer",
        ok=False,
        output="RECOMMENDATION: yes - ok\nNo grid here.\n",
        error=(
            f"{INCOMPLETE_RESPONSE_PREFIX} response had the RECOMMENDATION "
            "label but missed required sections: PART 2 — CONCEPT-BY-CONCEPT GRID"
        ),
        elapsed_seconds=1.0,
        model="peer-model",
    )
    retry = _ParticipantResult(
        name="peer",
        ok=False,
        output=_UNTAGGED_AFTER_SECTIONS_OUTPUT,
        error=_untagged_after_sections_error(),
        elapsed_seconds=1.5,
        model="peer-model",
    )

    merged = _merge_cli_section_retry(original, retry)

    assert merged.ok is False
    assert merged.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
    assert classify_error(merged.error) == "untagged_evidence"
    assert merged.section_repair_attempted is True
    # Both attempts must be visible in the merged transcript.
    assert "[retry exhausted]" in merged.output
    assert "Section-repair retry recovered" in merged.output
    assert "No grid here." in merged.output
    assert "PART 2 — Concept Grid" in merged.output


def test_merge_hosted_section_retry_preserves_untagged_evidence_retry():
    """Same as above but for the hosted/local merge function."""
    original = _ParticipantResult(
        name="endpoint",
        ok=False,
        output="RECOMMENDATION: yes - ok\nNo grid here.\n",
        error=(
            f"{INCOMPLETE_RESPONSE_PREFIX} response had the RECOMMENDATION "
            "label but missed required sections: PART 2 — CONCEPT-BY-CONCEPT GRID"
        ),
        elapsed_seconds=1.0,
        model="x-ai/test",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    retry = _ParticipantResult(
        name="endpoint",
        ok=False,
        output=_UNTAGGED_AFTER_SECTIONS_OUTPUT,
        error=_untagged_after_sections_error(),
        elapsed_seconds=1.5,
        model="x-ai/test",
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
    )

    merged = _merge_hosted_section_retry(original, retry)

    assert merged.ok is False
    assert merged.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
    assert classify_error(merged.error) == "untagged_evidence"
    assert merged.section_repair_attempted is True
    # The hosted variant carries token usage from the retry.
    assert merged.prompt_tokens == 12
    assert merged.completion_tokens == 8
    assert merged.total_tokens == 20
    # Both attempts visible.
    assert "[retry exhausted]" in merged.output
    assert "Section-repair retry recovered" in merged.output


def test_merge_cli_section_retry_success_path_sets_section_repair_attempted():
    """The success branch also flags `section_repair_attempted=True`
    so the wrapper guard fires uniformly across all merge branches.
    `repair_retry_recovered` and `section_repair_attempted` are both
    True in this case — they answer different questions."""
    original = _ParticipantResult(
        name="peer",
        ok=False,
        output="RECOMMENDATION: yes - ok\nNo grid here.\n",
        error=f"{INCOMPLETE_RESPONSE_PREFIX} missed: PART 2",
        elapsed_seconds=1.0,
    )
    retry = _ParticipantResult(
        name="peer",
        ok=True,
        output="RECOMMENDATION: yes\n## PART 2 — Concept Grid\nC1. foo\n",
        error="",
        elapsed_seconds=1.5,
    )
    merged = _merge_cli_section_retry(original, retry)
    assert merged.ok is True
    assert merged.repair_retry_recovered is True
    assert merged.section_repair_attempted is True


def test_openai_compatible_section_repair_surfaces_untagged_evidence(
    monkeypatch,
):
    """End-to-end: section-repair retry on openai_compatible produces
    sections + untagged evidence. Result must have
    `error_kind=untagged_evidence`, `section_repair_attempted=True`,
    and exactly TWO outer HTTP calls (NO third strict-evidence retry)."""
    calls: list[dict] = []

    bodies = [
        # First response: label present, PART 2 missing.
        _openai_body("RECOMMENDATION: yes - ok\nNo grid here.\n"),
        # Section-repair retry: PART 2 fixed BUT evidence has no tag.
        _openai_body(_UNTAGGED_AFTER_SECTIONS_OUTPUT),
    ]

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(bodies[len(calls) - 1])

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "router",
            {
                "type": "openai_compatible",
                "model": "x-ai/test",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "strict_evidence": True,
            },
            _SECTION_PROMPT,
        )
    )

    # Exactly two calls — no third strict-evidence repair retry.
    assert len(calls) == 2, (
        "expected one section-repair retry, no chained strict-evidence retry"
    )
    assert result.ok is False
    assert result.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
    assert classify_error(result.error) == "untagged_evidence"
    assert result.section_repair_attempted is True
    # Both attempts must be visible in the merged transcript.
    assert "Section-repair retry recovered" in result.output
    assert "No grid here." in result.output
    assert "PART 2 — Concept Grid" in result.output


def test_ollama_section_repair_surfaces_untagged_evidence(monkeypatch):
    """Same coverage on the ollama path."""
    calls: list[dict] = []

    bodies = [
        _ollama_body("RECOMMENDATION: yes - ok\nNo grid here.\n"),
        _ollama_body(_UNTAGGED_AFTER_SECTIONS_OUTPUT),
    ]

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(bodies[len(calls) - 1])

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_ollama_participant(
            "local",
            {
                "type": "ollama",
                "model": "qwen3:test",
                "base_url": "http://localhost:11434",
                "strict_evidence": True,
            },
            _SECTION_PROMPT,
        )
    )

    assert len(calls) == 2
    assert result.ok is False
    assert result.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
    assert classify_error(result.error) == "untagged_evidence"
    assert result.section_repair_attempted is True
    assert "Section-repair retry recovered" in result.output


def test_cli_section_repair_surfaces_untagged_evidence():
    """CLI path: section-repair retry produces sections + untagged
    evidence. Result must surface `untagged_evidence`, set
    `section_repair_attempted=True`, AND not chain a third retry."""
    call_count = {"n": 0}

    async def fake_run_cli_once(
        name, cfg, prompt, cwd, *, start, mode_multiplier=None, mode=None
    ):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                _ParticipantResult(
                    name=name,
                    ok=False,
                    output="RECOMMENDATION: yes - ok\nNo grid here.\n",
                    error=(
                        f"{INCOMPLETE_RESPONSE_PREFIX} response had the "
                        "RECOMMENDATION label but missed required sections: "
                        "PART 2 — CONCEPT-BY-CONCEPT GRID"
                    ),
                    elapsed_seconds=1.0,
                ),
                {"nonzero_exit": False, "stderr": "", "exited": True},
            )
        # Retry: sections fixed, evidence untagged.
        return (
            _ParticipantResult(
                name=name,
                ok=False,
                output=_UNTAGGED_AFTER_SECTIONS_OUTPUT,
                error=_untagged_after_sections_error(),
                elapsed_seconds=1.5,
            ),
            {"nonzero_exit": False, "stderr": "", "exited": True},
        )

    with _patch(
        "llm_council.adapters._run_cli_once", side_effect=fake_run_cli_once
    ), _patch(
        "llm_council.adapters._cache_lookup", return_value=(None, None)
    ), _patch("llm_council.adapters._maybe_persist_cache"):
        result = asyncio.run(
            run_cli_participant(
                "peer",
                {
                    "type": "cli",
                    "command": "peer",
                    "strict_evidence": True,
                    "timeout": 60,
                },
                _SECTION_PROMPT,
                _Path("/tmp"),
            )
        )

    # Two _run_cli_once calls: original + section-repair retry. NO third.
    assert call_count["n"] == 2, (
        "expected one section-repair retry, no chained strict-evidence retry"
    )
    assert result.ok is False
    assert result.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
    assert classify_error(result.error) == "untagged_evidence"
    assert result.section_repair_attempted is True
    assert "Section-repair retry recovered" in result.output


def test_strict_evidence_wrapper_guards_on_section_repair_attempted(
    monkeypatch,
):
    """Direct guard test: a result already carrying
    `section_repair_attempted=True` with `UntaggedEvidence:` must NOT
    trigger a strict-evidence repair retry. Constructed by having the
    section-repair retry surface untagged evidence and asserting only
    two HTTP calls total."""
    calls: list[dict] = []
    bodies = [
        _openai_body("RECOMMENDATION: yes\nNo grid here.\n"),
        _openai_body(_UNTAGGED_AFTER_SECTIONS_OUTPUT),
        # If the wrapper guard were missing, a third call would land
        # here with the strict-evidence retry directive. The assertion
        # below catches that — but we still seed a body just in case so
        # the test fails with a clear assert rather than IndexError.
        _openai_body(_TAGGED_AFTER_SECTIONS_OUTPUT),
    ]

    async def fake_request(client, method, url, **kwargs):
        calls.append(kwargs.get("json"))
        return _FakeResponse(bodies[len(calls) - 1])

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "router",
            {
                "type": "openai_compatible",
                "model": "x-ai/test",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "strict_evidence": True,
            },
            _SECTION_PROMPT,
        )
    )

    assert len(calls) == 2, (
        f"strict-evidence wrapper must not fire on a section-repair "
        f"result; got {len(calls)} calls"
    )
    assert result.section_repair_attempted is True


def test_section_repair_attempted_persists_through_cache(tmp_path):
    """Pass-8 fix #8 pattern: `section_repair_attempted` must round-trip
    through the cache so a cache hit on a sections-recovered result
    still carries the flag and the wrapper guard remains correct."""
    from llm_council.adapters import (
        CacheContext,
        _maybe_persist_cache,
        _result_from_cache_payload,
    )
    from llm_council.cache import read_cache

    output = (
        "[recovered after retry] First attempt was missing one or more "
        "REQUIRED sections; second attempt is shown below.\n\n"
        "--- Repaired response ---\n"
        "RECOMMENDATION: yes - ok\n## PART 2 — Concept Grid\nC1. foo\n\n"
        "--- Original response (first attempt) ---\n"
        "RECOMMENDATION: yes - ok\nNo grid here."
    )
    r = _ParticipantResult(
        name="peer",
        ok=True,
        output=output,
        error="",
        elapsed_seconds=2.0,
        repair_retry_recovered=True,
        section_repair_attempted=True,
    )
    cache_ctx = CacheContext(cwd=tmp_path, cache_mode="on", cache_disabled=False)
    _maybe_persist_cache("peer", "the prompt", "fake-key", r, cache_ctx)
    cached_files = list((tmp_path / ".llm-council" / "cache").glob("*.json"))
    assert len(cached_files) == 1
    payload = read_cache(cached_files[0], expected_key="fake-key")
    assert payload is not None
    assert payload.get("section_repair_attempted") is True
    rehydrated = _result_from_cache_payload("peer", payload)
    assert rehydrated.section_repair_attempted is True


def test_result_from_cache_payload_defaults_missing_section_repair_attempted():
    """Legacy payloads without the new key must rehydrate cleanly
    (default False), not crash on KeyError."""
    from llm_council.adapters import _result_from_cache_payload

    legacy_payload = {
        "output": "RECOMMENDATION: yes - fine",
        "elapsed_seconds": 1.0,
        "model": "test-model",
        # Intentionally missing: section_repair_attempted
    }
    r = _result_from_cache_payload("peer", legacy_payload)
    assert r.section_repair_attempted is False
    assert r.ok is True
