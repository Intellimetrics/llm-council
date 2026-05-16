"""Abdication detection across the adapter pipeline.

Covers:
- Repair-retry interaction (`_envelope_parse_source`, `_with_envelope`
  with `repair_retry_recovered=True` results).
- Cache round-trip: abdications cached as raw output, re-flagged
  `ok=False` on read via `_with_envelope`.
- `_is_label_only_failure` interaction with `EFFORT: blocked` — a
  self-reported blocked peer is terminal, not eligible for the
  label-only repair retry.
- `deliberation.recommendation_line` placeholder for fenced-only labels.

For envelope-shape parsing (no abdication interaction) see
``test_effort_contract.py``. For chair gating and synthesis flow see
``test_synthesis_gating.py``.
"""

from __future__ import annotations

from pathlib import Path

from llm_council.adapters import (
    ABDICATED_ERROR_PREFIX,
    CacheContext,
    ParticipantResult,
    _envelope_parse_source,
    _is_label_only_failure,
    _maybe_persist_cache,
    _with_envelope,
    classify_error,
)
from llm_council.deliberation import recommendation_line


# --- Repair-retry envelope parsing ---------------------------------------

def test_envelope_parse_source_strips_original_attempt_section():
    """``_format_retry_transcript`` produces [Repaired] then [Original].
    The envelope parser must only see the [Repaired] section so an
    ``EFFORT: blocked`` in the original section does not leak through."""
    output = (
        "[recovered after retry] First attempt was missing the required "
        "RECOMMENDATION label; second attempt is shown below.\n"
        "\n"
        "--- Repaired response ---\n"
        "RECOMMENDATION: yes - looks good\n"
        "EFFORT: full\n"
        "\n"
        "--- Original response (first attempt) ---\n"
        "EFFORT: blocked\n"
        "Some original prose without a label.\n"
    )
    parse_source = _envelope_parse_source(output)
    assert "RECOMMENDATION: yes" in parse_source
    assert "EFFORT: full" in parse_source
    assert "EFFORT: blocked" not in parse_source
    assert "Original response" not in parse_source


def test_with_envelope_recovered_repair_with_valid_repaired_response():
    """A successful repair-retry whose original had ``EFFORT: blocked``
    must keep ``ok=True``. The parse-source strip removes the original
    section so abdication detection only sees the (valid) repaired text."""
    output = (
        "[recovered after retry]\n"
        "--- Repaired response ---\n"
        "RECOMMENDATION: yes - on closer look this is fine\n"
        "\n"
        "--- Original response (first attempt) ---\n"
        "EFFORT: blocked\n"
        "I cannot evaluate this.\n"
    )
    r = ParticipantResult(
        "peer", True, output, "", 1.0, repair_retry_recovered=True
    )
    out = _with_envelope(r)
    assert out.ok is True
    assert out.error == ""


def test_with_envelope_abdicates_when_repaired_response_itself_abdicates():
    """Abdication detection still fires on the repaired response itself.
    The ``repair_retry_recovered`` flag is metadata, not a skip condition."""
    output = (
        "[recovered after retry] First attempt was missing the required "
        "RECOMMENDATION label; second attempt is shown below.\n"
        "\n"
        "--- Repaired response ---\n"
        "RECOMMENDATION: no - I genuinely cannot evaluate this\n"
        "EFFORT: blocked\n"
        "\n"
        "--- Original response (first attempt) ---\n"
        "Some original prose without a label.\n"
    )
    r = ParticipantResult(
        "peer", True, output, "", 1.0, repair_retry_recovered=True
    )
    out = _with_envelope(r)
    assert out.ok is False
    assert out.error.startswith(ABDICATED_ERROR_PREFIX)
    assert classify_error(out.error) == "abdicated"


# --- Cache round-trip ----------------------------------------------------

def test_abdication_cached_then_rederived_on_hit(tmp_path: Path):
    """Abdication output IS written to the cache. Read-side
    ``_with_envelope`` re-derivation flips ``ok=False``, preserving the
    user-visible "failed runs are not counted" invariant without paying
    the peer twice for the same abdication."""
    abdication_output = (
        "RECOMMENDATION: no - too complex to evaluate\n"
        "EFFORT: blocked\n"
    )
    # Adapter writes ok=True (the flip happens later via _with_envelope).
    r = ParticipantResult("peer", True, abdication_output, "", 1.0)
    cache_ctx = CacheContext(cwd=tmp_path, cache_mode="on", cache_disabled=False)
    _maybe_persist_cache("peer", "the prompt", "fake-key", r, cache_ctx)
    cache_dir = tmp_path / ".llm-council" / "cache"
    cached_files = list(cache_dir.glob("*.json"))
    assert len(cached_files) == 1

    # Simulate run_participant pulling from cache and applying _with_envelope.
    rehydrated = ParticipantResult(
        "peer", True, abdication_output, "", 1.0, from_cache=True
    )
    out = _with_envelope(rehydrated)
    assert out.ok is False
    assert classify_error(out.error) == "abdicated"


def test_maybe_persist_cache_writes_normal_success(tmp_path: Path):
    """Sanity: a non-abdication success still gets cached."""
    output = "RECOMMENDATION: yes - looks fine\nEFFORT: full"
    r = ParticipantResult("peer", True, output, "", 1.0)
    cache_ctx = CacheContext(cwd=tmp_path, cache_mode="on", cache_disabled=False)
    _maybe_persist_cache("peer", "the prompt", "fake-key", r, cache_ctx)
    files = list((tmp_path / ".llm-council" / "cache").glob("*.json"))
    assert len(files) == 1


# --- Label-only repair retry vs. EFFORT: blocked ------------------------

def test_is_label_only_failure_refuses_retry_when_effort_blocked():
    """A peer that self-reported blocked must not be eligible for the
    label-only repair retry — re-asking the same prompt produces another
    abdication for no new signal."""
    blocked_without_label = (
        "Some prose explaining why this can't be done.\n"
        "EFFORT: blocked\n"
    )
    assert _is_label_only_failure(blocked_without_label, {}) is False


def test_is_label_only_failure_retries_normal_missing_label():
    """Sanity: an honest missing-label response (no EFFORT field) still
    triggers the repair retry."""
    no_label = "Here is my analysis but I forgot the label line.\n"
    assert _is_label_only_failure(no_label, {}) is True


# --- recommendation_line placeholder for fenced-only labels --------------

def test_recommendation_line_placeholder_for_fenced_only_label():
    """A peer with no out-of-fence label has no usable vote.
    ``recommendation_line`` returns an explicit placeholder so the
    round-2 deliberation summary doesn't echo arbitrary intro prose."""
    text = (
        "Here is my analysis of the proposal:\n"
        "\n"
        "```\n"
        "RECOMMENDATION: yes - example syntax only\n"
        "```\n"
        "\n"
        "The actual conclusion is not labeled.\n"
    )
    assert recommendation_line(text) == "(no RECOMMENDATION label emitted)"


def test_recommendation_line_returns_real_label():
    """Sanity: out-of-fence labels are returned verbatim."""
    text = "Some intro.\n\nRECOMMENDATION: tradeoff - depends on data\n"
    assert "tradeoff" in recommendation_line(text)
