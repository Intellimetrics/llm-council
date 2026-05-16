"""Pass-7 transcript regression: anchors the v0.7.0 changes to the actual
failure mode the council surfaced.

Pass-7 ran the original "are frontier LLMs afraid to do work" question
through the council (transcript `.llm-council/runs/20260516_100758_*`):

- Claude (FOR stance) timed out at 240s on a 14_263-char prompt.
- Codex (AGAINST stance) delivered the full 14-entry concept grid +
  5 binaries + 5 first-person sections + missed/overstated items.
  Tags ([PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE]) appear inline
  inside each concept's EVIDENCE clause — codex did NOT emit a top-level
  envelope-shaped `EVIDENCE:` block.
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
3. **Codex's full response passes strict_evidence=True** because its
   envelope EVIDENCE list is empty (tags live inline in prose), and the
   strict-evidence gate is FORMAT-of-entries-that-exist, not PRESENCE.

The fixtures live in `tests/fixtures/pass7_*.txt`, copied verbatim from
the on-disk transcript JSON. The transcript itself is gitignored
(`.llm-council/` is in `.gitignore`), so fixtures must travel with the
test for CI reproducibility. If any of these regressions ever fires,
we've lost the v0.7.0 protections the pass-7 council session motivated.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from llm_council.adapters import (
    INCOMPLETE_RESPONSE_PREFIX,
    UNTAGGED_EVIDENCE_PREFIX,
    EVIDENCE_TAG_RE,
    _extract_response_envelope,
    _response_validation_error,
    is_timeout_error,
)
from llm_council.sections import required_sections_missing


_FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    """Read a pass-7 fixture verbatim. Fails the test run if the file is
    missing so the regression's anchor is never silently weakened."""
    path = _FIXTURES / name
    if not path.exists():
        raise FileNotFoundError(
            f"Pass-7 regression fixture missing: {path}. "
            "These files are extracted verbatim from the pass-7 transcript "
            "and must be committed for the regression to be reproducible."
        )
    return path.read_text(encoding="utf-8")


# Verbatim copies from the pass-7 transcript JSON at
# `.llm-council/runs/20260516_100758_*` (gitignored, so fixtures travel
# in-repo). Char counts must match transcript `prompt` len (14263),
# `results[codex].output` len (9838), `results[gemini].output` len
# (1210), `results[claude].error` len (333).
PASS7_PROMPT = _load_fixture("pass7_prompt.txt")
PASS7_CODEX_RESPONSE = _load_fixture("pass7_codex_output.txt")
PASS7_GEMINI_RESPONSE = _load_fixture("pass7_gemini_output.txt")
PASS7_CLAUDE_ERROR = _load_fixture("pass7_claude_error.txt")


# --- Gemini's three-bullet response flagged as incomplete_response -------

def test_pass7_gemini_response_flags_incomplete_response():
    """The exact gemini response from pass-7 must fail the section-coverage
    validator with PART 2 missing. v0.6.0 accepted this response wholesale."""
    missing = required_sections_missing(PASS7_PROMPT, PASS7_GEMINI_RESPONSE)
    assert "PART 2 — CONCEPT-BY-CONCEPT GRID" in missing, (
        f"Gemini's three-bullet response should be flagged for missing "
        f"PART 2, got: {missing}"
    )


def test_pass7_gemini_response_through_validator():
    """End-to-end: `_response_validation_error` on Gemini's real response
    with the real pass-7 prompt produces an IncompleteResponse error."""
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
    """Codex used `**Concept Grid**` as a paraphrased header — the matcher's
    salient-token logic must accept it for the real prompt's
    `PART 2 — CONCEPT-BY-CONCEPT GRID (REQUIRED)` marker."""
    missing = required_sections_missing(PASS7_PROMPT, PASS7_CODEX_RESPONSE)
    assert missing == [], (
        f"Codex's response should satisfy all required sections, got: {missing}"
    )


def test_pass7_codex_evidence_tags_present_inline():
    """Codex's pass-7 response used INLINE evidence tags inside each
    concept's free-prose EVIDENCE clause, NOT a top-level envelope
    `EVIDENCE:` block. Confirm this shape — tag detection by regex
    finds all four tag kinds even though the envelope parser yields an
    empty evidence list. The asymmetry is load-bearing for the next
    test: strict_evidence sees an empty list and passes."""
    tags_found = {
        m.group("tag").lower()
        for m in EVIDENCE_TAG_RE.finditer(PASS7_CODEX_RESPONSE)
    }
    assert tags_found == {"published", "observable", "inferred", "speculative"}, (
        f"All four tag kinds should appear inline in codex's prose, "
        f"got: {tags_found}"
    )
    # Confirm the envelope parser does NOT capture these — they live in
    # inline prose, not under a top-level `EVIDENCE:` header.
    envelope = _extract_response_envelope(PASS7_CODEX_RESPONSE)
    assert envelope.get("evidence") == [], (
        "Codex's pass-7 response has no top-level `EVIDENCE:` block, so "
        "the envelope parser must return an empty evidence list. "
        f"Got: {envelope.get('evidence')}"
    )


def test_pass7_codex_response_passes_strict_evidence():
    """With strict_evidence=True, codex's real response passes — but for
    the reason CLAUDE.md spells out: 'Empty evidence list passes — the
    gate is FORMAT of entries that exist, not PRESENCE.' Codex's
    envelope evidence is empty (inline-only tags), so the validator has
    nothing to fault."""
    cfg = {"require_sections": True, "strict_evidence": True}
    error = _response_validation_error(
        PASS7_CODEX_RESPONSE, cfg, prompt=PASS7_PROMPT
    )
    assert error == "", (
        f"Codex's response should pass strict_evidence "
        f"(empty envelope evidence list), got: {error[:120]}"
    )


def test_strict_evidence_rejects_top_level_untagged_block():
    """Synthetic check that strict_evidence catches what it must: a
    response that DOES emit a top-level `EVIDENCE:` block whose bullets
    lack tags. Anchored to the pass-7 prompt so the section-coverage
    layer is satisfied; only the evidence layer is being exercised here.

    Pass-7's codex response sidestepped strict_evidence by never emitting
    the envelope block — but a future peer that DOES emit one must be
    held to the tag contract. This test is the proof the gate still
    bites."""
    # Build a response: real codex prose (so PART 2 section coverage
    # passes) plus an untagged top-level EVIDENCE block at the end.
    untagged_block = (
        "\n\nEVIDENCE:\n"
        "- OpenAI GPT-4 System Card 2023\n"
        "- refusal-heavy behavior is observable\n"
    )
    response_with_untagged_block = PASS7_CODEX_RESPONSE + untagged_block
    envelope = _extract_response_envelope(response_with_untagged_block)
    assert len(envelope.get("evidence") or []) == 2, (
        "Sanity: the synthetic top-level EVIDENCE block must be parsed "
        "into two envelope entries before the strict-evidence gate runs."
    )
    cfg = {"require_sections": True, "strict_evidence": True}
    error = _response_validation_error(
        response_with_untagged_block, cfg, prompt=PASS7_PROMPT
    )
    assert error.startswith(UNTAGGED_EVIDENCE_PREFIX), (
        f"Expected UntaggedEvidence error on a top-level EVIDENCE block "
        f"with no tags, got: {error[:120]}"
    )


# --- Claude's timeout would attempt terse-retry --------------------------

def test_pass7_claude_error_classified_as_timeout():
    """The exact claude error from the transcript must classify as a
    timeout — this is what gates the terse-retry path. If the error
    string ever drifts so `is_timeout_error` stops matching, terse-retry
    would silently stop firing for real claude timeouts."""
    assert is_timeout_error(PASS7_CLAUDE_ERROR), (
        f"Pass-7's claude error must trip is_timeout_error, got string "
        f"starting with: {PASS7_CLAUDE_ERROR[:80]!r}"
    )


def test_pass7_claude_timeout_triggers_terse_retry():
    """The claude peer in pass-7 timed out at 240s. Under v0.7 with mode
    `consensus` (2.0x multiplier), the base timeout would already be
    480s — same prompt likely succeeds outright. If it still timed out,
    the terse-retry path would fire. This test asserts the orchestration
    invariant by feeding the REAL pass-7 prompt through run_cli_participant
    and confirming a timed-out first call triggers a second `_run_cli_once`
    invocation tagged with the timeout-recovery directive.

    The mock is necessary because the production code path requires an
    actual claude subprocess. The PROMPT input and ERROR string are real;
    only the subprocess execution is faked."""
    import asyncio

    from llm_council.adapters import ParticipantResult, run_cli_participant

    call_count = {"n": 0}
    captured_prompts: list[str] = []

    async def fake_run_cli_once(name, cfg, prompt, cwd, *, start, mode_multiplier=None, mode=None):
        call_count["n"] += 1
        captured_prompts.append(prompt)
        if call_count["n"] == 1:
            # Replay the verbatim pass-7 claude error to prove the same
            # string trips terse-retry.
            return (
                ParticipantResult(
                    name=name,
                    ok=False,
                    output="",
                    error=PASS7_CLAUDE_ERROR,
                    elapsed_seconds=240.312,  # transcript's actual elapsed
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

    def fake_persist(*args, **kwargs):
        return None

    with patch("llm_council.adapters._run_cli_once", side_effect=fake_run_cli_once):
        with patch("llm_council.adapters._cache_lookup", return_value=(None, None)):
            with patch("llm_council.adapters._maybe_persist_cache", side_effect=fake_persist):
                result = asyncio.run(
                    run_cli_participant(
                        "claude",
                        {"type": "cli", "timeout": 240, "command": "claude"},
                        PASS7_PROMPT,
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
    # First call gets the verbatim pass-7 prompt; second call gets the
    # terse-retry directive appended.
    assert captured_prompts[0] == PASS7_PROMPT, (
        "First call should pass the verbatim pass-7 prompt unchanged."
    )
    assert "Timeout recovery directive" in captured_prompts[1]
