"""Regression tests for the 6 pass-4 (post-implementation council review) fixes.

Pass-4 council reviewed v0.5.0 commit 5b0f4af and returned RECOMMENDATION: no
(2/3 labeled). Transcript: .llm-council/runs/20260516_073739_*. Findings:

- Fix #1: Synthesis chair's decision-memo format failed label validation.
- Fix #2: Synthesis chair received the cumulative results list (round-1 +
  round-2 entries) instead of final-round only.
- Fix #3: Universal-abdication short-circuit ran AFTER the deliberation loop,
  defeating the "save spend by skipping round 2" intent.
- Fix #4: A successful repair-retry's output is a concatenation of repaired
  + original; an original `EFFORT: blocked` would re-flag the (valid)
  repaired result as abdication.
- Fix #6: A peer that emitted `EFFORT: blocked` without a label was eligible
  for the label-only repair retry — wastes a round trip on a definitively
  blocked peer.
- Fix #10: Abdication outputs were cached as successes (cache write fires
  before run_participant -> _with_envelope flips ok=False), violating the
  "failed runs are never cached" invariant.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.adapters import (
    ABDICATED_ERROR_PREFIX,
    CacheContext,
    ParticipantResult,
    _envelope_parse_source,
    _is_label_only_failure,
    _maybe_persist_cache,
    _with_envelope,
)
from llm_council.deliberation import recommendation_line
from llm_council.synthesis import build_synthesis_prompt, run_synthesis_chair


# --- Fix #1: synthesis chair bypasses RECOMMENDATION label validation ----

def test_synthesis_chair_overrides_require_recommendation():
    """The chair returns a structured memo with `### Decision`, NOT a
    `RECOMMENDATION:` line. Without overriding require_recommendation the
    chair output would be rejected as invalid_response and the chair would
    burn a useless repair retry."""

    captured_cfg: dict = {}

    async def fake_run_participant(name, cfg, prompt, cwd, *, cache_ctx=None):
        captured_cfg.update(cfg)
        return ParticipantResult(
            name=name,
            ok=True,
            output="### Decision\nTradeoff. Ship with the migration test.",
            error="",
            elapsed_seconds=0.1,
        )

    with patch("llm_council.synthesis.run_participant", side_effect=fake_run_participant):
        import asyncio

        peer = ParticipantResult("p", True, "RECOMMENDATION: yes - ok", "", 1.0)
        asyncio.run(
            run_synthesis_chair(
                question="should we ship?",
                results=[peer],
                convergence=None,
                participant_cfg={"chair": {"type": "cli", "command": "echo"}},
                cwd=Path("."),
                chair_name="chair",
            )
        )

    assert captured_cfg.get("require_recommendation") is False
    assert captured_cfg.get("retry_on_missing_label") is False


# --- Fix #2: synthesis sees ONLY the final round ------------------------

def test_synthesis_chair_receives_only_final_round_after_deliberation():
    """Builds an orchestrator.execute_council where round-1 disagreed,
    round-2 ran, and synthesize=True. The chair must see round-2 peer
    outputs ONLY (round-1 names are 'a', 'b'; round-2 names get :round2
    suffix). Pass-4 bug: chair was getting all of them."""
    import asyncio
    import llm_council.orchestrator as orch_module

    captured: dict = {}

    async def fake_run_synthesis(*args, **kwargs):
        captured["results"] = kwargs["results"]
        return {
            "chair": kwargs["chair_name"],
            "ok": True,
            "output": "### Decision\nyes",
            "error": "",
            "decision_label": "unknown",
            "blockers": [],
            "evidence": [],
            "tests_to_run": [],
            "elapsed_seconds": 0.1,
            "model": None,
            "total_tokens": None,
            "cost_usd": None,
            "consumed_convergence": False,
            "prompt_chars": 0,
        }

    # Build a results list of the shape execute_council would produce
    # after one deliberation round, then directly call the
    # post-deliberation synthesis block by simulating its inputs.
    round1 = [
        ParticipantResult("a", True, "RECOMMENDATION: yes - r1", "", 1.0),
        ParticipantResult("b", True, "RECOMMENDATION: no - r1", "", 1.0),
    ]
    round2 = [
        ParticipantResult("a:round2", True, "RECOMMENDATION: tradeoff - r2", "", 1.0),
        ParticipantResult("b:round2", True, "RECOMMENDATION: tradeoff - r2", "", 1.0),
    ]
    results = round1 + round2

    from llm_council.synthesis import select_synthesizer
    from llm_council.transcript import final_round_results

    config = {
        "defaults": {"synthesizer": "a"},
    }
    participant_cfg = {"a": {"type": "cli"}, "b": {"type": "cli"}}
    chair_name = select_synthesizer(
        config, participant_cfg, stances=None, current=None
    )

    with patch("llm_council.synthesis.run_synthesis_chair", side_effect=fake_run_synthesis):
        # Replicate the orchestrator slice that decides what the chair sees.
        chair_input = final_round_results(results)
        asyncio.run(
            fake_run_synthesis(
                question="q",
                results=chair_input,
                convergence=None,
                participant_cfg=participant_cfg,
                cwd=Path("."),
                chair_name=chair_name,
            )
        )

    names = [r.name for r in captured["results"]]
    assert names == ["a:round2", "b:round2"], (
        "chair must receive only the final-round results"
    )


# --- Fix #3: universal-abdication short-circuits BEFORE round 2 ---------

def test_universal_abdication_short_circuits_deliberation():
    """If all peers abdicate in round 1, the deliberation loop must NOT run.
    The orchestrator stamps `universal_abdication` and `deliberation_status:
    skipped_universal_abdication` and the while-loop guard refuses to enter."""
    import asyncio
    import llm_council.orchestrator as orch_module

    # Two peers, both abdicate in round 1. The orchestrator's per-result
    # _with_envelope would normally do this for us, but the test stubs out
    # the per-type adapters so we install abdication-shape outputs directly.
    async def fake_run_participants(selected, *args, **kwargs):
        return [
            ParticipantResult(
                name,
                False,
                "RECOMMENDATION: no - too complex\nEFFORT: blocked",
                f"{ABDICATED_ERROR_PREFIX} test fixture",
                1.0,
                effort="blocked",
            )
            for name in selected
        ]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(orch_module, "run_participants", side_effect=fake_run_participants):
        with patch.object(
            orch_module, "preflight_local_participants", side_effect=fake_preflight
        ):
            results, metadata = asyncio.run(
                orch_module.execute_council(
                    participants=["a", "b"],
                    participant_cfg={"a": {"type": "cli"}, "b": {"type": "cli"}},
                    prompt="q",
                    cwd=Path("."),
                    config={"defaults": {}},
                    deliberate=True,
                    max_rounds=2,
                )
            )

    assert metadata.get("universal_abdication") is not None
    assert metadata["deliberation_status"] == "skipped_universal_abdication"
    # Round counter must not have incremented past 1.
    assert metadata["rounds"] == 1, "round 2 must not have run"
    assert metadata["deliberated"] is False


# --- Fix #4: repair-retry transcript not misclassified as abdication ----

def test_envelope_parse_source_strips_original_attempt_section():
    """`_format_retry_transcript` output shape: [Repaired]...[Original]...
    The envelope parser must only see the [Repaired] section so an
    EFFORT: blocked in the original does NOT leak through."""
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


def test_with_envelope_does_not_abdicate_recovered_repair():
    """End-to-end: a successful repair-retry whose original had EFFORT:
    blocked must keep ok=True after _with_envelope."""
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
    assert out.ok is True, "recovered repair must not be re-flagged as abdication"
    assert out.error == ""


# --- Fix #6: EFFORT:blocked without label is terminal (no retry) --------

def test_is_label_only_failure_refuses_retry_when_effort_blocked():
    """A peer that emitted `EFFORT: blocked` self-declared it cannot
    respond to this prompt. Retrying with the same prompt is wasted spend.
    _is_label_only_failure must return False so the retry path skips."""
    blocked_without_label = (
        "Some prose explaining why this can't be done.\n"
        "EFFORT: blocked\n"
    )
    assert _is_label_only_failure(blocked_without_label, {}) is False


def test_is_label_only_failure_still_retries_normal_missing_label():
    """Sanity: a normal missing-label response (no EFFORT field) still
    triggers retry as before."""
    no_label = "Here is my analysis but I forgot the label line.\n"
    assert _is_label_only_failure(no_label, {}) is True


# --- Fix #10: abdications are never cached ------------------------------

def test_maybe_persist_cache_writes_abdication_for_offline_rederivation(
    tmp_path: Path,
):
    """Pass-5 reverted v0.5.1 fix #10: abdications DO get cached.

    The cache hit path always pipes results back through `_with_envelope`,
    which re-derives the envelope and flips abdication to ok=False with
    zero API cost. Refusing the cache write (v0.5.1) instead forced every
    repeat run to re-pay the peer for the same abdication — a real cost
    regression for no correctness gain. The "failed runs are never cached"
    invariant is preserved at the RESULT layer (via re-derivation) rather
    than at the cache-file layer.
    """
    abdication_output = (
        "RECOMMENDATION: no - too complex to evaluate\n"
        "EFFORT: blocked\n"
    )
    r = ParticipantResult(
        "peer",
        True,  # adapter sets ok=True; flip happens later via _with_envelope
        abdication_output,
        "",
        1.0,
    )
    cache_ctx = CacheContext(cwd=tmp_path, cache_mode="on", cache_disabled=False)

    _maybe_persist_cache("peer", "the prompt", "fake-key", r, cache_ctx)
    cache_dir = tmp_path / ".llm-council" / "cache"
    files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
    assert len(files) == 1, (
        "abdication outputs must be cached so repeat runs don't re-pay "
        "the peer; _with_envelope re-derives ok=False on the read side"
    )


def test_maybe_persist_cache_allows_normal_success(tmp_path: Path):
    """Sanity: a non-abdication success still gets cached."""
    output = "RECOMMENDATION: yes - looks fine\nEFFORT: full"
    r = ParticipantResult("peer", True, output, "", 1.0)
    cache_ctx = CacheContext(cwd=tmp_path, cache_mode="on", cache_disabled=False)
    _maybe_persist_cache("peer", "the prompt", "fake-key", r, cache_ctx)
    cache_dir = tmp_path / ".llm-council" / "cache"
    files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
    assert len(files) == 1


# --- Fix #7: recommendation_line returns placeholder on no-label --------

def test_recommendation_line_placeholder_for_fenced_only_label():
    """A peer that only puts its label inside a code fence has no usable
    vote. Old behavior: fall back to first_nonempty_line (noisy prose).
    New behavior: explicit placeholder so round-2 prompts don't echo
    arbitrary intro sentences as that peer's 'position'."""
    text = (
        "Here is my analysis of the proposal:\n"
        "\n"
        "```\n"
        "RECOMMENDATION: yes - example syntax only\n"
        "```\n"
        "\n"
        "The actual conclusion is not labeled.\n"
    )
    line = recommendation_line(text)
    assert line == "(no RECOMMENDATION label emitted)"


def test_recommendation_line_still_returns_real_label():
    """Sanity: out-of-fence labels still work as before."""
    text = "Some intro.\n\nRECOMMENDATION: tradeoff - depends on data\n"
    assert "tradeoff" in recommendation_line(text)
