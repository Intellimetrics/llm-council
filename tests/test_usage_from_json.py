"""Tests for the opt-in `usage_from_json` per-peer feature (M7).

Covers:
  * `_build_cli_command` flag injection (claude / codex on; no-op family;
    default-off byte-identical; read-only flags preserved).
  * `_parse_cli_usage_json` for claude (single object) and codex (JSONL),
    including cache-token subtraction and fail-soft None returns.
  * An end-to-end `run_cli_participant` drive through a real subprocess whose
    stdout is the JSON fixture, asserting the token fields land on the result
    and the label check passes on the EXTRACTED text.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llm_council.adapters import (
    _build_cli_command,
    _parse_cli_usage_json,
    run_cli_participant,
)
from proc_stubs import fake_proc_returning as _fake_proc_returning


# --- fixtures ---------------------------------------------------------------


def _claude_json_fixture() -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "result": "RECOMMENDATION: yes - looks good\n\nThe change is sound.",
            "total_cost_usd": 0.0123,
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 320,
                "cache_read_input_tokens": 800,
            },
            "modelUsage": {
                "claude-opus-4-8-20260101": {"inputTokens": 1500, "outputTokens": 320}
            },
        }
    )


def _codex_jsonl_fixture() -> str:
    """A realistic codex `exec --json` JSONL stream (several event lines)."""
    lines = [
        {"type": "session.created", "session_id": "abc"},
        {"type": "agent_message", "text": "thinking out loud..."},
        {
            "type": "agent_message",
            "text": "RECOMMENDATION: tradeoff - acceptable\n\nFinal answer body.",
            "model": "gpt-5.5-codex",
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 2000,
                "cached_input_tokens": 500,
                "output_tokens": 450,
            },
        },
    ]
    return "\n".join(json.dumps(line) for line in lines)


# --- _build_cli_command -----------------------------------------------------


def _claude_cfg(**extra) -> dict:
    cfg = {
        "type": "cli",
        "family": "claude",
        "command": "claude",
        "args": ["-p", "--permission-mode", "default", "--no-session-persistence"],
        "stdin_prompt": True,
    }
    cfg.update(extra)
    return cfg


def _codex_cfg(**extra) -> dict:
    cfg = {
        "type": "cli",
        "family": "codex",
        "command": "codex",
        "args": ["exec", "--sandbox", "read-only", "--ephemeral", "-"],
        "stdin_prompt": True,
    }
    cfg.update(extra)
    return cfg


def test_build_cli_command_claude_usage_json_adds_flag_and_keeps_readonly(tmp_path):
    cmd = _build_cli_command("claude", _claude_cfg(usage_from_json=True), "p", tmp_path)
    assert "--output-format" in cmd
    # The flag value must follow the flag.
    assert cmd[cmd.index("--output-format") + 1] == "json"
    # Read-only flags preserved (purely additive).
    assert "--permission-mode" in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "default"


def test_build_cli_command_codex_usage_json_adds_flag_and_keeps_readonly(tmp_path):
    cmd = _build_cli_command("codex", _codex_cfg(usage_from_json=True), "p", tmp_path)
    assert "--json" in cmd
    # `--json` belongs to the exec subcommand: it must come right after `exec`.
    assert cmd[cmd.index("exec") + 1] == "--json"
    # Read-only sandbox flag preserved.
    assert "--sandbox" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    # No malformed double-exec.
    assert cmd.count("exec") == 1


def test_build_cli_command_codex_usage_json_with_model_keeps_single_exec(tmp_path):
    cfg = _codex_cfg(usage_from_json=True, model="gpt-5.5")
    cmd = _build_cli_command("codex", cfg, "p", tmp_path)
    assert cmd.count("exec") == 1
    assert "--json" in cmd
    assert cmd[cmd.index("exec") + 1] in {"--json", "-m"}  # both valid orderings
    assert "--json" in cmd and "-m" in cmd
    assert "gpt-5.5" in cmd
    # Read-only flag still present.
    assert "--sandbox" in cmd


def test_build_cli_command_default_off_is_byte_identical(tmp_path):
    # Without usage_from_json, neither JSON flag appears — byte-identical
    # command vs. the same cfg sans the key.
    claude_off = _build_cli_command("claude", _claude_cfg(), "p", tmp_path)
    claude_explicit_false = _build_cli_command(
        "claude", _claude_cfg(usage_from_json=False), "p", tmp_path
    )
    assert "--output-format" not in claude_off
    assert claude_off == claude_explicit_false

    codex_off = _build_cli_command("codex", _codex_cfg(), "p", tmp_path)
    assert "--json" not in codex_off


def test_build_cli_command_gemini_usage_json_is_noop(tmp_path):
    gemini_cfg = {
        "type": "cli",
        "family": "gemini",
        "command": "gemini",
        "args": ["--approval-mode", "plan"],
        "stdin_prompt": True,
        "usage_from_json": True,
    }
    cmd = _build_cli_command("gemini", gemini_cfg, "p", tmp_path)
    # No JSON output flag added for an unparsed family.
    assert "--output-format" not in cmd
    assert "--json" not in cmd
    # Identical to the same cfg without the key.
    cmd_off = _build_cli_command(
        "gemini", {k: v for k, v in gemini_cfg.items() if k != "usage_from_json"},
        "p", tmp_path,
    )
    assert cmd == cmd_off


# --- _parse_cli_usage_json --------------------------------------------------


def test_parse_claude_usage_json_happy_path():
    parsed = _parse_cli_usage_json("claude", _claude_json_fixture())
    assert parsed is not None
    assert parsed["text"].startswith("RECOMMENDATION: yes")
    assert parsed["prompt_tokens"] == 1500
    assert parsed["completion_tokens"] == 320
    assert parsed["total_tokens"] == 1820
    assert parsed["cost_usd"] == 0.0123
    assert parsed["model"] == "claude-opus-4-8-20260101"


def test_parse_codex_usage_json_subtracts_cached_tokens():
    parsed = _parse_cli_usage_json("codex", _codex_jsonl_fixture())
    assert parsed is not None
    assert parsed["text"].startswith("RECOMMENDATION: tradeoff")
    # input_tokens (2000) - cached_input_tokens (500) = 1500 billable prompt.
    assert parsed["prompt_tokens"] == 1500
    assert parsed["completion_tokens"] == 450
    assert parsed["total_tokens"] == 1950
    assert parsed["cost_usd"] is None  # codex reports no cost
    assert parsed["model"] == "gpt-5.5-codex"


def test_parse_codex_usage_json_takes_last_agent_message():
    parsed = _parse_cli_usage_json("codex", _codex_jsonl_fixture())
    assert parsed is not None
    # "thinking out loud..." is an earlier agent_message; the final one wins.
    assert "Final answer body." in parsed["text"]
    assert "thinking out loud" not in parsed["text"]


def test_parse_codex_usage_json_skips_non_json_lines():
    fixture = "this is a log line\n" + _codex_jsonl_fixture() + "\nanother stray line"
    parsed = _parse_cli_usage_json("codex", fixture)
    assert parsed is not None
    assert parsed["prompt_tokens"] == 1500


def test_parse_cli_usage_json_malformed_returns_none():
    assert _parse_cli_usage_json("claude", "not json at all") is None
    assert _parse_cli_usage_json("codex", "not json at all") is None
    assert _parse_cli_usage_json("claude", "") is None
    assert _parse_cli_usage_json("codex", "") is None


def test_parse_claude_usage_json_missing_result_returns_none():
    obj = json.dumps({"usage": {"input_tokens": 10, "output_tokens": 5}})
    assert _parse_cli_usage_json("claude", obj) is None


def test_parse_claude_usage_json_partial_usage_tolerated():
    # Missing output_tokens → completion None → total None, but still parses.
    obj = json.dumps(
        {"result": "RECOMMENDATION: no - nope", "usage": {"input_tokens": 100}}
    )
    parsed = _parse_cli_usage_json("claude", obj)
    assert parsed is not None
    assert parsed["prompt_tokens"] == 100
    assert parsed["completion_tokens"] is None
    assert parsed["total_tokens"] is None


def test_parse_cli_usage_json_unparsed_family_returns_none():
    assert _parse_cli_usage_json("gemini", _claude_json_fixture()) is None


# --- integration through a stubbed subprocess -------------------------------
#
# We stub `create_subprocess_exec` rather than driving a real binary: the
# real claude `--output-format json` flag would be appended to whatever
# command we point at, and a generic stub (e.g. `python -c`) can't accept
# arbitrary CLI flags. The fake proc returns the JSON fixture on stdout
# regardless of the (correctly-built) command — exactly the existing pattern
# in tests/test_timeout_policy.py.


def _stub_cfg(family: str) -> dict:
    return {
        "type": "cli",
        "family": family,
        "command": family,
        "args": [],
        "usage_from_json": True,
        "stdin_prompt": False,
        "timeout": 30,
        "timeout_per_kb_chars": 0,
        # Bare label output below is a valid success; skip the section gate.
        "require_sections": False,
    }


def test_run_cli_participant_claude_json_populates_usage(tmp_path: Path):
    async def _go():
        with _fake_proc_returning(_claude_json_fixture()):
            return await run_cli_participant(
                "claude", _stub_cfg("claude"), "prompt", tmp_path
            )

    result = asyncio.run(_go())
    assert result.ok is True, result.error  # label check passed on extracted text
    assert result.output.startswith("RECOMMENDATION: yes")
    assert result.prompt_tokens == 1500
    assert result.completion_tokens == 320
    assert result.total_tokens == 1820
    assert result.cost_usd == 0.0123
    # CLI-reported model preferred over cfg model (cfg had none here).
    assert result.model == "claude-opus-4-8-20260101"


def test_run_cli_participant_codex_json_populates_usage(tmp_path: Path):
    async def _go():
        with _fake_proc_returning(_codex_jsonl_fixture()):
            return await run_cli_participant(
                "codex", _stub_cfg("codex"), "prompt", tmp_path
            )

    result = asyncio.run(_go())
    assert result.ok is True, result.error
    assert "RECOMMENDATION: tradeoff" in result.output
    assert result.prompt_tokens == 1500  # cache-subtracted
    assert result.completion_tokens == 450
    assert result.cost_usd is None
    assert result.model == "gpt-5.5-codex"


def test_run_cli_participant_malformed_json_falls_back_to_raw(tmp_path: Path):
    # When JSON parsing fails, the raw stdout (which here contains a valid
    # RECOMMENDATION label) is used for the label check; no token fields set.
    raw = "RECOMMENDATION: yes - this is not json but has the label"

    async def _go():
        with _fake_proc_returning(raw):
            return await run_cli_participant(
                "claude", _stub_cfg("claude"), "prompt", tmp_path
            )

    result = asyncio.run(_go())
    assert result.ok is True, result.error
    assert result.output == raw
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.cost_usd is None


def test_run_cli_participant_default_off_no_usage_fields(tmp_path: Path):
    # usage_from_json absent → JSON ignored, raw text used, no token fields.
    cfg = _stub_cfg("claude")
    del cfg["usage_from_json"]

    async def _go():
        with _fake_proc_returning(_claude_json_fixture()):
            return await run_cli_participant("claude", cfg, "prompt", tmp_path)

    result = asyncio.run(_go())
    # The raw JSON has no top-level RECOMMENDATION line → label check fails,
    # proving we did NOT extract the embedded `result` text. Confirms off-path.
    assert result.prompt_tokens is None
    assert result.cost_usd is None
