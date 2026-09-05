"""Tests for the `fable` mode + `claude_fable` peer (Claude Fable 5 as a
read-only council peer).

Covers the two moving parts of the "reduce + detect" design:

  * REDUCE — `context.build_prompt(safe_context=True)` injects the defensive-
    review framing (only when the mode opts in) that lowers Fable's
    false-positive safety-classifier refusals.
  * DETECT — `require_pinned_model` on a `usage_from_json` peer drops the peer
    (ok=False, error_kind=model_substituted) when the CLI-reported served model
    doesn't match the pinned one (e.g. Fable refused → Claude Code silently fell
    back to Opus 4.8), so a substituted model's answer is never recorded as a
    Fable vote. The orchestrator surfaces the swap top-level.

Subprocess stubs are shared with tests/test_usage_from_json.py via
tests/proc_stubs.py; the execute_council harness mirrors
tests/test_independence_warning.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from llm_council.adapters import (
    ERROR_KIND_MODEL_SUBSTITUTED,
    KNOWN_ERROR_KINDS,
    MODEL_SUBSTITUTED_PREFIX,
    ParticipantResult,
    _build_cli_command,
    _model_pin_satisfied,
    classify_error,
    run_cli_participant,
)
from llm_council.config import validate_config
from llm_council.context import build_prompt
from llm_council.defaults import DEFAULT_CONFIG
from proc_stubs import TimingOutProc, fake_proc_returning, fake_proc_sequence


# --- unit: model-pin match --------------------------------------------------


def test_model_pin_satisfied_exact_and_variant_match():
    assert _model_pin_satisfied("claude-fable-5", "claude-fable-5") is True
    # Dated / minor-version variant of the requested id still counts.
    assert _model_pin_satisfied("claude-fable-5", "claude-fable-5-20260601") is True


def test_model_pin_satisfied_family_mismatch_is_false():
    # The Fable → Opus refusal fallback: served model is a different family.
    assert _model_pin_satisfied("claude-fable-5", "claude-opus-4-8") is False
    assert _model_pin_satisfied("claude-fable-5", "claude-opus-4-8-20260101") is False


def test_model_pin_satisfied_missing_id_does_not_flag():
    # Can't decide with a missing id → never a positive mismatch.
    assert _model_pin_satisfied("claude-fable-5", None) is True
    assert _model_pin_satisfied(None, "claude-opus-4-8") is True
    assert _model_pin_satisfied("", "claude-opus-4-8") is True


# --- unit: error taxonomy ---------------------------------------------------


def test_model_substituted_error_classifies_and_is_known():
    err = f"{MODEL_SUBSTITUTED_PREFIX} requested claude-fable-5, served opus"
    assert classify_error(err) == ERROR_KIND_MODEL_SUBSTITUTED
    assert ERROR_KIND_MODEL_SUBSTITUTED in KNOWN_ERROR_KINDS


def test_model_substituted_kind_advertised_in_mcp_schema():
    from llm_council.mcp_server import COUNCIL_RUN_VALID_ERROR_KINDS

    assert ERROR_KIND_MODEL_SUBSTITUTED in COUNCIL_RUN_VALID_ERROR_KINDS


# --- config: the claude_fable peer + fable mode -----------------------------


def test_claude_fable_participant_shape():
    cf = DEFAULT_CONFIG["participants"]["claude_fable"]
    assert cf["family"] == "claude"
    assert cf["model"] == "claude-fable-5-1"
    # Observability + guard for the silent Fable→Opus refusal fallback.
    assert cf["usage_from_json"] is True
    assert cf["require_pinned_model"] is True
    # Empty so no `--fallback-model` injection (second silent-swap path).
    assert cf["fallback_chain"] == []
    # Read-only flags preserved.
    assert "--permission-mode" in cf["args"]
    assert cf["args"][cf["args"].index("--permission-mode") + 1] == "manual"


def test_fable_mode_shape_and_config_valid():
    fm = DEFAULT_CONFIG["modes"]["fable"]
    assert fm["participants"] == ["claude_fable"]
    assert fm["safe_context"] is True
    assert fm["timeout_multiplier"] == 1.5
    # The new participant + mode keys must not break config validation.
    validate_config(DEFAULT_CONFIG)


def test_build_cli_command_claude_fable_pins_model_no_fallback(tmp_path):
    cf = DEFAULT_CONFIG["participants"]["claude_fable"]
    cmd = _build_cli_command("claude_fable", cf, "p", tmp_path)
    # Model pinned.
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-fable-5-1"
    # Empty fallback_chain → NO `--fallback-model` (would be a silent swap path).
    assert "--fallback-model" not in cmd
    # usage_from_json → JSON output flag; read-only flag preserved.
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert "--permission-mode" in cmd


# --- prompt: safe_context framing -------------------------------------------


def _framing_present(prompt: str) -> bool:
    return (
        "operator-invoked, read-only" in prompt
        and "second-opinion code review" in prompt
    )


def test_safe_context_framing_present_when_on():
    p = build_prompt(
        "Is this safe?",
        mode="fable",
        cwd=Path("."),
        context_paths=[],
        include_diff=False,
        stdin_text=None,
        safe_context=True,
    )
    assert _framing_present(p)


def test_safe_context_framing_absent_by_default():
    p = build_prompt(
        "Is this safe?",
        mode="review",
        cwd=Path("."),
        context_paths=[],
        include_diff=False,
        stdin_text=None,
    )
    assert not _framing_present(p)


# --- integration: require_pinned_model drop through a stubbed subprocess -----


def _claude_json(model_key: str) -> str:
    """A `claude -p --output-format json` object whose modelUsage names the
    model that actually served the turn."""
    return json.dumps(
        {
            "type": "result",
            "result": "RECOMMENDATION: yes - looks fine\n\nBody.",
            "total_cost_usd": 0.01,
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "modelUsage": {model_key: {"inputTokens": 100, "outputTokens": 20}},
        }
    )


def _fable_cfg(**extra) -> dict:
    cfg = {
        "type": "cli",
        "family": "claude",
        "command": "claude",
        "args": [],
        "model": "claude-fable-5",
        "usage_from_json": True,
        "stdin_prompt": False,
        "timeout": 30,
        "timeout_per_kb_chars": 0,
        "require_sections": False,
    }
    cfg.update(extra)
    return cfg


def _drive(cfg: dict, stdout: str, tmp_path: Path) -> ParticipantResult:
    async def _go():
        with fake_proc_returning(stdout):
            return await run_cli_participant("claude_fable", cfg, "prompt", tmp_path)

    return asyncio.run(_go())


def test_fable_served_by_opus_drops_the_peer(tmp_path: Path):
    # Pinned claude-fable-5, but the CLI reports it was served by Opus (the
    # silent refusal fallback). require_pinned_model → drop.
    result = _drive(
        _fable_cfg(require_pinned_model=True),
        _claude_json("claude-opus-4-8-20260101"),
        tmp_path,
    )
    assert result.ok is False
    assert result.error.startswith(MODEL_SUBSTITUTED_PREFIX)
    assert classify_error(result.error) == ERROR_KIND_MODEL_SUBSTITUTED
    # The REAL served model is still reported so the transcript shows the swap.
    assert result.model == "claude-opus-4-8-20260101"


def test_fable_served_by_fable_is_kept(tmp_path: Path):
    # Served by a dated Fable variant → pin satisfied → normal success.
    result = _drive(
        _fable_cfg(require_pinned_model=True),
        _claude_json("claude-fable-5-20260601"),
        tmp_path,
    )
    assert result.ok is True, result.error
    assert result.model == "claude-fable-5-20260601"
    assert result.output.startswith("RECOMMENDATION: yes")


def test_no_require_pinned_model_never_drops_on_substitution(tmp_path: Path):
    # Without the guard, a served-by-Opus turn is kept (today's default
    # behavior for every other peer) — the drop is strictly opt-in.
    result = _drive(
        _fable_cfg(require_pinned_model=False),
        _claude_json("claude-opus-4-8-20260101"),
        tmp_path,
    )
    assert result.ok is True, result.error
    assert result.model == "claude-opus-4-8-20260101"


# --- orchestrator: top-level surfacing of the substitution ------------------


# --- review fixes (v0.16.0 council review) ----------------------------------


def test_parse_claude_usage_json_multi_model_picks_answer_author():
    # A refusal-fallback turn logs usage for BOTH models; the refusing model
    # (near-zero output) can be listed first. The parser must report the
    # model that AUTHORED the answer (most outputTokens), not the first key.
    from llm_council.adapters import _parse_cli_usage_json

    fixture = json.dumps(
        {
            "result": "RECOMMENDATION: yes - fine",
            "usage": {"input_tokens": 100, "output_tokens": 300},
            "modelUsage": {
                "claude-fable-5": {"inputTokens": 100, "outputTokens": 0},
                "claude-opus-4-8": {"inputTokens": 100, "outputTokens": 300},
            },
        }
    )
    parsed = _parse_cli_usage_json("claude", fixture)
    assert parsed is not None
    assert parsed["model"] == "claude-opus-4-8"

    # Symmetric: a healthy Fable turn with a small helper model listed first
    # must NOT be misreported as the helper (would false-positive the guard).
    fixture2 = json.dumps(
        {
            "result": "RECOMMENDATION: yes - fine",
            "usage": {"input_tokens": 100, "output_tokens": 300},
            "modelUsage": {
                "claude-haiku-4-5": {"inputTokens": 20, "outputTokens": 5},
                "claude-fable-5": {"inputTokens": 100, "outputTokens": 300},
            },
        }
    )
    parsed2 = _parse_cli_usage_json("claude", fixture2)
    assert parsed2 is not None
    assert parsed2["model"] == "claude-fable-5"

    # Count-less payloads keep the legacy first-key behavior.
    fixture3 = json.dumps(
        {
            "result": "RECOMMENDATION: yes - fine",
            "modelUsage": {"claude-fable-5-20260601": {}},
        }
    )
    parsed3 = _parse_cli_usage_json("claude", fixture3)
    assert parsed3 is not None
    assert parsed3["model"] == "claude-fable-5-20260601"


def test_label_retry_substitution_is_not_swallowed(tmp_path: Path):
    # Round 1: valid JSON served by Fable but MISSING the label -> label-repair
    # retry fires. The retry is served by Opus (substituted). The merged
    # result must classify model_substituted, not invalid_response.
    unlabeled = json.dumps(
        {
            "result": "Looks fine to me (no label).",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "modelUsage": {"claude-fable-5": {"outputTokens": 5}},
        }
    )
    substituted = _claude_json("claude-opus-4-8-20260101")

    async def _go():
        with fake_proc_sequence(unlabeled, substituted):
            return await run_cli_participant(
                "claude_fable",
                _fable_cfg(require_pinned_model=True),
                "prompt",
                tmp_path,
            )

    result = asyncio.run(_go())
    assert result.ok is False
    assert classify_error(result.error) == ERROR_KIND_MODEL_SUBSTITUTED
    assert result.model == "claude-opus-4-8-20260101"
    # Combined transcript: the original Fable-authored attempt stays
    # auditable next to the substituted retry (v0.16.0 re-review fix).
    assert "Looks fine to me (no label)." in result.output


def test_terse_timeout_retry_substitution_is_not_swallowed(tmp_path: Path):
    # Round 1 times out; the terse retry trips the refusal fallback and is
    # served by Opus. The result must classify model_substituted (not
    # timeout) and record that the retry fired.
    substituted = _claude_json("claude-opus-4-8-20260101")

    async def _go():
        with fake_proc_sequence(TimingOutProc(), substituted):
            return await run_cli_participant(
                "claude_fable",
                _fable_cfg(
                    require_pinned_model=True,
                    timeout=0.2,
                    max_prompt_chars=100_000,
                ),
                "prompt",
                tmp_path,
            )

    result = asyncio.run(_go())
    assert result.ok is False
    assert classify_error(result.error) == ERROR_KIND_MODEL_SUBSTITUTED
    assert result.terse_retry_attempted is True
    assert result.model == "claude-opus-4-8-20260101"


def test_section_retry_substitution_is_not_swallowed(tmp_path: Path):
    # Round 1: labeled Fable response that misses a REQUIRED section →
    # section-repair retry fires. The retry is served by Opus. The merged
    # result must classify model_substituted (not incomplete_response) and
    # keep the original Fable text auditable in the combined transcript.
    prompt = "PART 1 — SECURITY ANALYSIS (REQUIRED)\n\nAssess the change."
    labeled_missing_sections = json.dumps(
        {
            "result": "RECOMMENDATION: yes - fine but skipped the sections",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "modelUsage": {"claude-fable-5": {"outputTokens": 5}},
        }
    )
    substituted = _claude_json("claude-opus-4-8-20260101")

    async def _go():
        with fake_proc_sequence(labeled_missing_sections, substituted):
            return await run_cli_participant(
                "claude_fable",
                _fable_cfg(require_pinned_model=True, require_sections=True),
                prompt,
                tmp_path,
            )

    result = asyncio.run(_go())
    assert result.ok is False
    assert classify_error(result.error) == ERROR_KIND_MODEL_SUBSTITUTED
    assert result.section_repair_attempted is True
    assert "skipped the sections" in result.output


_FINDINGS_OUTPUT = (
    "RECOMMENDATION: no - blocker\n\n"
    "FINDINGS:\n"
    "- id: F1\n"
    "  severity: blocker\n"
    "  claim: bad thing in a.py\n"
    "  evidence: [VERIFIED:a.py:1-2]\n"
)


def _run_council_with(results):
    # Drive the REAL orchestrator (not a local re-implementation of its
    # matrix filter) with a canned round-1 result set.
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        return list(results)

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ), patch.object(
        orch_module, "preflight_local_participants", side_effect=fake_preflight
    ):
        return asyncio.run(
            orch_module.execute_council(
                participants=[r.name for r in results],
                participant_cfg={
                    "claude_fable": {
                        "type": "cli",
                        "family": "claude",
                        "model": "claude-fable-5",
                    },
                    "codex": {"type": "cli", "family": "codex"},
                },
                prompt="review this",
                cwd=Path("."),
                config={},
                deliberate=False,
                max_rounds=1,
            )
        )


def test_substituted_output_excluded_from_finding_matrix():
    substituted = ParticipantResult(
        name="claude_fable",
        ok=False,
        output=_FINDINGS_OUTPUT,
        error=f"{MODEL_SUBSTITUTED_PREFIX} served by opus",
        elapsed_seconds=1.0,
        model="claude-opus-4-8",
    )
    healthy = ParticipantResult(
        name="codex",
        ok=True,
        output="RECOMMENDATION: yes - fine",
        error="",
        elapsed_seconds=1.0,
    )
    _, metadata = _run_council_with([substituted, healthy])
    # The substituted (Opus-served) FINDINGS block must NOT enter the matrix.
    assert "finding_matrix" not in metadata
    # ...and the swap is surfaced with the round it actually happened in.
    peers = metadata.get("model_substituted_peers")
    assert peers and peers[0]["peer"] == "claude_fable"
    assert peers[0]["requested"] == "claude-fable-5"
    assert peers[0]["served_by"] == "claude-opus-4-8"
    events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "peer_model_substituted"
    ]
    assert events and events[0]["round"] == 1


def test_healthy_findings_block_does_enter_finding_matrix():
    # Control for the exclusion test above: the same FINDINGS block from a
    # healthy peer DOES surface — proving the exclusion is the substitution
    # filter, not a parsing gap.
    healthy_with_findings = ParticipantResult(
        name="codex",
        ok=True,
        output=_FINDINGS_OUTPUT,
        error="",
        elapsed_seconds=1.0,
    )
    _, metadata = _run_council_with([healthy_with_findings])
    assert "finding_matrix" in metadata
    assert "model_substituted_peers" not in metadata


def test_config_validation_rejects_non_bool_new_keys():
    import pytest

    from llm_council.config import deep_merge

    bad_mode = deep_merge(
        DEFAULT_CONFIG, {"modes": {"custom": {"participants": ["claude"], "safe_context": "false"}}}
    )
    with pytest.raises(ValueError, match="safe_context must be a boolean"):
        validate_config(bad_mode)

    bad_peer = deep_merge(
        DEFAULT_CONFIG,
        {"participants": {"claude": {"require_pinned_model": "yes"}}},
    )
    with pytest.raises(ValueError, match="require_pinned_model must be a boolean"):
        validate_config(bad_peer)

    bad_usage = deep_merge(
        DEFAULT_CONFIG,
        {"participants": {"claude": {"usage_from_json": "false"}}},
    )
    with pytest.raises(ValueError, match="usage_from_json must be a boolean"):
        validate_config(bad_usage)


def test_estimate_prompt_includes_safe_context_framing():
    # Parity rule: an estimate that passes must not be rejected by the real
    # run's prompt-size guard, so the estimate prompt must include the
    # framing when the mode opts in.
    from llm_council.estimate import estimate_council

    est_fable = estimate_council(
        config=DEFAULT_CONFIG,
        cwd=Path("."),
        question="Is this change safe?",
        mode="fable",
        current="claude",
        allow_network=False,
    )
    est_review = estimate_council(
        config=DEFAULT_CONFIG,
        cwd=Path("."),
        question="Is this change safe?",
        mode="peer-only",
        current="codex",
        allow_network=False,
    )
    # The framing adds ~850 chars; assert a conservative floor.
    assert est_fable["prompt_chars"] > est_review["prompt_chars"] + 500


def test_mcp_output_schema_declares_model_substituted_peers():
    from llm_council.mcp_server import council_run_output_schema

    props = council_run_output_schema()["properties"]
    assert "model_substituted_peers" in props
    item_props = props["model_substituted_peers"]["items"]["properties"]
    assert {"peer", "requested", "served_by"} <= set(item_props)


def test_orchestrator_surfaces_model_substituted_peers():
    import llm_council.orchestrator as orch_module

    participant_cfg = {
        "claude_fable": {"type": "cli", "family": "claude", "model": "claude-fable-5"}
    }

    substituted = ParticipantResult(
        name="claude_fable",
        ok=False,
        output="",
        error=(
            f"{MODEL_SUBSTITUTED_PREFIX} `claude_fable` requested claude-fable-5 "
            f"but the CLI served claude-opus-4-8-20260101"
        ),
        elapsed_seconds=1.0,
        model="claude-opus-4-8-20260101",
    )

    async def fake_run_participants(selected, *args, **kwargs):
        return [substituted]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ), patch.object(
        orch_module, "preflight_local_participants", side_effect=fake_preflight
    ):
        _, metadata = asyncio.run(
            orch_module.execute_council(
                participants=["claude_fable"],
                participant_cfg=participant_cfg,
                prompt="q",
                cwd=Path("."),
                config={},
                deliberate=False,
                max_rounds=1,
                mode="fable",
            )
        )

    peers = metadata.get("model_substituted_peers")
    assert peers is not None and len(peers) == 1
    entry = peers[0]
    assert entry["peer"] == "claude_fable"
    assert entry["requested"] == "claude-fable-5"
    assert entry["served_by"] == "claude-opus-4-8-20260101"

    events = [
        e
        for e in metadata["progress_events"]
        if e.get("event") == "peer_model_substituted"
    ]
    assert len(events) == 1
    assert events[0]["served_by"] == "claude-opus-4-8-20260101"


def test_round1_substitution_surfaced_in_deliberating_run():
    # Round 1: a/b disagree (triggers deliberation) and claude_fable is
    # substituted. Round 2: everyone (including fable) answers ok. The
    # round-1 swap must STILL be surfaced — the scan covers all rounds,
    # not just the final one.
    import llm_council.orchestrator as orch_module

    participant_cfg = {
        "a": {"type": "cli", "family": "acme"},
        "b": {"type": "cli", "family": "globex"},
        "claude_fable": {
            "type": "cli",
            "family": "claude",
            "model": "claude-fable-5",
        },
    }

    def _vote(name, label):
        return ParticipantResult(
            name=name,
            ok=True,
            output=f"RECOMMENDATION: {label} - reason",
            error="",
            elapsed_seconds=1.0,
        )

    substituted = ParticipantResult(
        name="claude_fable",
        ok=False,
        output="",
        error=f"{MODEL_SUBSTITUTED_PREFIX} served by opus",
        elapsed_seconds=1.0,
        model="claude-opus-4-8-20260101",
    )

    calls = {"n": 0}

    async def fake_run_participants(selected, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [_vote("a", "yes"), _vote("b", "no"), substituted]
        return [_vote(name, "yes") for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ), patch.object(
        orch_module, "preflight_local_participants", side_effect=fake_preflight
    ):
        _, metadata = asyncio.run(
            orch_module.execute_council(
                participants=["a", "b", "claude_fable"],
                participant_cfg=participant_cfg,
                prompt="q",
                cwd=Path("."),
                config={},
                deliberate=True,
                max_rounds=2,
            )
        )

    assert calls["n"] == 2, "expected a deliberation round to have run"
    peers = metadata.get("model_substituted_peers")
    assert peers is not None and len(peers) == 1
    assert peers[0]["peer"] == "claude_fable"
    assert peers[0]["served_by"] == "claude-opus-4-8-20260101"
