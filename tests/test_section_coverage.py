"""Section-coverage validator (Change 2).

Anchors: pass-7 transcript at `.llm-council/runs/20260516_100758_*`. The
end-to-end "real pass-7 response" test lives in
`tests/test_pass7_regression.py`; this file covers the parser, matcher,
and validation-error wiring in isolation.
"""

from __future__ import annotations

import asyncio

import llm_council.adapters as adapters_module
from llm_council.adapters import (
    INCOMPLETE_RESPONSE_PREFIX,
    KNOWN_ERROR_KINDS,
    _response_validation_error,
    classify_error,
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
