"""Derived deliberation body budget.

The round-2 prompt a peer receives is ``body + per-peer directive suffix``.
The body budget is DERIVED from the cap that governs the run minus the
largest suffix that will be appended — never a fixed constant that must be
kept in sync with the cap by hand. Before this derivation existed, any
suffix-bearing peer (e.g. antigravity's read-tool hint) pushed the
worst-case bound past the default MCP cap and every ``deliberate: true``
MCP run rostering that peer was refused regardless of actual prompt size.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_council.adapters import ParticipantResult
from llm_council.budget import DEFAULT_MCP_MAX_PROMPT_CHARS
from llm_council.context import apply_per_peer_directives
from llm_council.deliberation import (
    DELIBERATION_TRUNCATION_SUFFIX,
    MAX_DELIBERATION_PROMPT_CHARS,
    build_deliberation_prompt,
    deliberation_body_budget,
)
from llm_council.estimate import deliberation_prompt_char_bounds


def _result(name: str, output: str) -> ParticipantResult:
    return ParticipantResult(
        name=name, ok=True, output=output, error="", elapsed_seconds=0.1
    )


def test_body_budget_derives_from_builder_ceiling_without_cap() -> None:
    assert (
        deliberation_body_budget(None, 300)
        == MAX_DELIBERATION_PROMPT_CHARS - 300
    )


def test_body_budget_respects_a_lower_effective_cap() -> None:
    assert deliberation_body_budget(50_000, 300) == 50_000 - 300


def test_body_budget_ignores_a_higher_effective_cap() -> None:
    # The builder ceiling still applies when the surface cap is looser.
    assert (
        deliberation_body_budget(200_000, 0) == MAX_DELIBERATION_PROMPT_CHARS
    )


def test_body_budget_floor_keeps_truncation_slice_valid() -> None:
    assert deliberation_body_budget(10, 1_000_000) == len(
        DELIBERATION_TRUNCATION_SUFFIX
    )


def test_build_deliberation_prompt_honors_max_chars() -> None:
    results = [_result("a", "RECOMMENDATION: yes\n" + "x" * 30_000)]
    prompt, _ = build_deliberation_prompt(
        "question" * 5_000, results, max_chars=9_000
    )
    assert len(prompt) <= 9_000
    assert prompt.endswith(DELIBERATION_TRUNCATION_SUFFIX)


def test_final_round2_prompt_fits_cap_for_suffix_bearing_peer() -> None:
    """body budget + the peer's own suffix never exceeds the effective cap."""
    cap = DEFAULT_MCP_MAX_PROMPT_CHARS
    cfg = {"family": "antigravity"}
    suffix = len(
        apply_per_peer_directives(
            "", mode="review", family="antigravity", tool_call_voting=False
        )
    )
    assert suffix > 0  # antigravity's read-tool hint must exist for this test
    body = deliberation_body_budget(cap, suffix)
    results = [_result("a", "RECOMMENDATION: yes\n" + "x" * 200_000)]
    prompt, _ = build_deliberation_prompt("q" * 100_000, results, max_chars=body)
    assert len(prompt) + suffix <= cap


def test_bounds_never_exceed_default_mcp_cap_with_antigravity() -> None:
    """Regression: the pre-derivation bound was 80_000 + suffix > 80_000,
    structurally refusing every deliberate MCP run that rostered agy."""
    participant_cfg = {
        "claude": {"family": "claude"},
        "antigravity": {"family": "antigravity"},
    }
    bounds = deliberation_prompt_char_bounds(
        participants=["claude", "antigravity"],
        participant_cfg=participant_cfg,
        mode="review",
        tool_call_voting=False,
        effective_prompt_cap=DEFAULT_MCP_MAX_PROMPT_CHARS,
    )
    assert max(bounds.values()) <= DEFAULT_MCP_MAX_PROMPT_CHARS


@pytest.mark.asyncio
async def test_mcp_deliberate_run_with_antigravity_passes_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end regression at DEFAULT caps: the exact refusal observed live
    (`MCP council_run budget exceeded: max_prompt_chars 80268 > 80000`)."""
    from llm_council import mcp_server

    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
participants:
  agy:
    type: cli
    family: antigravity
    command: agy
modes:
  custom:
    participants: [agy]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    payload = await mcp_server.run_council(
        {
            "question": "review",
            "working_directory": str(tmp_path),
            "deliberate": True,
            "dry_run": True,
        }
    )
    budget = payload["metadata"]["budget"]
    assert budget["within_budget"] is True
    assert budget["violations"] == []
    assert budget["max_call_prompt_chars"] <= DEFAULT_MCP_MAX_PROMPT_CHARS
