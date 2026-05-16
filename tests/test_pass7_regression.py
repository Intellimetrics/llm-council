"""Pass-7 transcript regression: anchors the v0.7.0 changes to the actual
failure mode the council surfaced.

Pass-7 ran the original "are frontier LLMs afraid to do work" question
through the council (transcript `.llm-council/runs/20260516_100758_*`):

- Claude (FOR stance) timed out at 240s on a 14_263-char prompt.
- Codex (AGAINST stance) delivered the full 14-entry concept grid +
  5 binaries + 5 first-person sections + 7+ paper citations with
  [PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE] tags.
- Gemini (NEUTRAL stance) delivered three bullets total — completely
  skipped PARTS 2-5 of the prompt while keeping the RECOMMENDATION label,
  so v0.6.0 accepted the response as valid.

These tests assert the v0.7.0 pipeline correctly handles each peer's
outcome:

1. **Claude's timeout would attempt terse-retry** under the new policy —
   the recovered_after_timeout flag is the receipt.
2. **Gemini's three-bullet response is now flagged** as
   `error_kind=incomplete_response` because PART 2 (CONCEPT-BY-CONCEPT
   GRID) is missing.
3. **Codex's full response passes strict_evidence=True** because all
   EVIDENCE bullets carry tags.

If any of these regressions ever fires, we've lost the v0.7.0 protections
the pass-7 council session prompted us to build.
"""

from __future__ import annotations

from unittest.mock import patch

from llm_council.adapters import (
    INCOMPLETE_RESPONSE_PREFIX,
    UNTAGGED_EVIDENCE_PREFIX,
    _extract_response_envelope,
    _response_validation_error,
)
from llm_council.sections import required_sections_missing


# A condensed but structurally faithful version of the pass-7 prompt.
# The literal markers and section names match the real transcript so the
# matcher / validator logic exercises the exact shapes that fired in
# pass-7. Brevity is fine — what matters is the `(REQUIRED)` headers and
# the prose anchors the response-side matcher uses.
PASS7_PROMPT = """\
You are part of a 3-peer council evaluating ChatGPT's framework.

================================================================
PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)
================================================================

For EACH of C1 through C14, produce a four-field entry.

================================================================
PART 6 — RECOMMENDATION (REQUIRED BY COUNCIL INVARIANTS)
================================================================

Emit exactly one line of the form RECOMMENDATION: yes|no|tradeoff.
"""

# Gemini's actual pass-7 response (three bullets). Lifted verbatim from
# the transcript so this test fails precisely when the v0.7 validator
# would have accepted what v0.6 silently passed.
PASS7_GEMINI_RESPONSE = """\
RECOMMENDATION: tradeoff — The behavioral observations of effort \
minimization and safe-path preference are largely accurate, but the \
psychological framing ("fear", "laziness") and conflation of structural \
latency/context limits with alignment training fundamentally \
misattribute the causes of these outputs.

*   **Strongest reasons:** ChatGPT accurately identifies that models \
truncate code, avoid complex implementations zero-shot, and over-index \
on politeness and safety.
*   **Concrete risks:** The framework risks encouraging developers to \
anthropomorphize models.
*   **Implementation suggestions:** The conceptual framework should be \
stripped of its psychological terminology before being used.
"""

# Codex's structurally complete response with tagged EVIDENCE. Trimmed
# from the actual transcript — preserves the section headers and a
# representative EVIDENCE bullet with each of the four tag kinds.
PASS7_CODEX_RESPONSE = """\
RECOMMENDATION: tradeoff - keep C1, C4, C7, C9, C10, C11.

**Concept Grid**

C1. Over-refusal
EVIDENCE:
- OpenAI GPT-4 System Card 2023 [PUBLISHED]
- refusal-heavy behavior is observable [OBSERVABLE]
- reasoning from priors [INFERRED]
- training internals not accessible [SPECULATIVE]
"""


# --- Gemini's three-bullet response flagged as incomplete_response -------

def test_pass7_gemini_response_flags_incomplete_response():
    """The exact gemini response from pass-7 must fail the section-coverage
    validator with `error_kind=incomplete_response`. v0.6.0 accepted it."""
    missing = required_sections_missing(PASS7_PROMPT, PASS7_GEMINI_RESPONSE)
    assert "PART 2 — CONCEPT-BY-CONCEPT GRID" in missing, (
        f"Gemini's three-bullet response should be flagged for missing "
        f"PART 2, got: {missing}"
    )


def test_pass7_gemini_response_through_validator():
    """End-to-end: `_response_validation_error` on Gemini's response with
    the pass-7 prompt produces an IncompleteResponse error."""
    cfg = {"require_sections": True}
    error = _response_validation_error(
        PASS7_GEMINI_RESPONSE, cfg, prompt=PASS7_PROMPT
    )
    assert error.startswith(INCOMPLETE_RESPONSE_PREFIX), (
        f"Expected IncompleteResponse error, got: {error[:120]}"
    )


def test_pass7_gemini_response_passes_when_sections_disabled():
    """With require_sections=False, gemini's response passes (matches v0.6
    behavior). Confirms the toggle works in both directions."""
    cfg = {"require_sections": False}
    error = _response_validation_error(
        PASS7_GEMINI_RESPONSE, cfg, prompt=PASS7_PROMPT
    )
    assert error == ""


# --- Codex's structured response passes section + strict-evidence checks --

def test_pass7_codex_response_satisfies_section_coverage():
    """Codex's "**Concept Grid**" header is the paraphrased form that the
    matcher's salient-token logic should accept."""
    missing = required_sections_missing(PASS7_PROMPT, PASS7_CODEX_RESPONSE)
    assert missing == [], (
        f"Codex's response should satisfy all required sections, got: {missing}"
    )


def test_pass7_codex_evidence_all_tagged():
    """Each EVIDENCE bullet in codex's response carries one of the four
    canonical tags. Under strict_evidence=True this passes; the same
    response with stripped tags would fail."""
    envelope = _extract_response_envelope(PASS7_CODEX_RESPONSE)
    evidence = envelope.get("evidence") or []
    assert evidence, "expected at least one EVIDENCE entry"
    tags = {entry.get("tag") for entry in evidence if isinstance(entry, dict)}
    assert tags == {"published", "observable", "inferred", "speculative"}, (
        f"Codex's evidence should hit all four tag kinds, got: {tags}"
    )


def test_pass7_codex_response_passes_strict_evidence():
    """With strict_evidence=True, codex's fully-tagged response should
    return no validation error."""
    cfg = {"require_sections": True, "strict_evidence": True}
    error = _response_validation_error(
        PASS7_CODEX_RESPONSE, cfg, prompt=PASS7_PROMPT
    )
    assert error == "", (
        f"Codex's fully-tagged response should pass strict_evidence, "
        f"got: {error[:120]}"
    )


def test_codex_response_with_stripped_tags_fails_strict_evidence():
    """Sanity check: same codex response with the tags stripped must fail
    strict_evidence. Confirms the validator catches what it should."""
    stripped = PASS7_CODEX_RESPONSE.replace("[PUBLISHED]", "").replace(
        "[OBSERVABLE]", ""
    ).replace("[INFERRED]", "").replace("[SPECULATIVE]", "")
    cfg = {"require_sections": True, "strict_evidence": True}
    error = _response_validation_error(stripped, cfg, prompt=PASS7_PROMPT)
    assert error.startswith(UNTAGGED_EVIDENCE_PREFIX), (
        f"Expected UntaggedEvidence error after stripping tags, got: {error[:120]}"
    )


# --- Claude's timeout would attempt terse-retry --------------------------

def test_pass7_claude_timeout_triggers_terse_retry():
    """The claude peer in pass-7 timed out at 240s. Under v0.7 with mode
    `consensus` (2.0x multiplier), the base timeout would already be
    480s — same prompt likely succeeds outright. If it still timed out,
    the terse-retry path would fire. This test asserts the orchestration
    invariant by checking that a timed-out CLI ParticipantResult routed
    through run_cli_participant invokes _run_cli_once a second time."""
    import asyncio

    from llm_council.adapters import ParticipantResult, run_cli_participant

    call_count = {"n": 0}
    captured_prompts: list[str] = []

    async def fake_run_cli_once(name, cfg, prompt, cwd, *, start, mode_multiplier=None, mode=None):
        call_count["n"] += 1
        captured_prompts.append(prompt)
        if call_count["n"] == 1:
            return (
                ParticipantResult(
                    name=name,
                    ok=False,
                    output="",
                    error=f"Timeout: `{name}` did not respond within 480s",
                    elapsed_seconds=480.0,
                    prompt_chars=len(prompt),
                ),
                {"nonzero_exit": False, "stderr": "", "exited": False},
            )
        # Terse retry succeeds with a tight valid response.
        return (
            ParticipantResult(
                name=name,
                ok=True,
                output="RECOMMENDATION: tradeoff - terse recovered",
                error="",
                elapsed_seconds=15.0,
            ),
            {"nonzero_exit": False, "stderr": "", "exited": True},
        )

    async def fake_cache_lookup(*args, **kwargs):
        return None, None

    def fake_persist(*args, **kwargs):
        return None

    with patch("llm_council.adapters._run_cli_once", side_effect=fake_run_cli_once):
        with patch("llm_council.adapters._cache_lookup", return_value=(None, None)):
            with patch("llm_council.adapters._maybe_persist_cache", side_effect=fake_persist):
                result = asyncio.run(
                    run_cli_participant(
                        "claude",
                        {"type": "cli", "timeout": 240, "command": "claude"},
                        "ping",
                        "/tmp",  # type: ignore
                        mode_multiplier=2.0,
                        mode="consensus",
                    )
                )

    assert call_count["n"] == 2, (
        f"Expected 2 _run_cli_once calls (original + terse retry), got {call_count['n']}"
    )
    assert result.ok is True
    assert result.recovered_after_timeout is True
    # The second call should have used the terse-retry prompt (original + directive).
    assert "Timeout recovery directive" in captured_prompts[1]
