"""Nested-council recursion guard.

A council participant must never be able to start another council: each
level would spawn its own peers, so recursion grows exponentially (the
observed real-world path is a peer CLI whose global MCP config registers
llm-council itself). Two independent layers are covered here:

1. Every CLI child env carries ``LLM_COUNCIL_NESTED=1`` and
   ``execute_council`` refuses to run when it sees the marker.
2. The codex baseline args clear the peer's MCP server table entirely
   (``-c mcp_servers={}``), so the nested server never even boots — and
   ``_build_cli_command`` re-enforces the starvation for ANY codex-family
   arg list. The exact-match migration only recognizes pristine old
   baselines; a field config with one extra flag
   (``--skip-git-repo-check``) slipped through it and booted the
   operator's global MCP servers (headless browsers + a nested
   llm-council) on every council run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llm_council.adapters import _build_cli_command, clean_subprocess_env
from llm_council.config import (
    OLD_CODEX_EPHEMERAL_ARGS,
    migrate_known_cli_defaults,
)
from llm_council.defaults import DEFAULT_CONFIG
from llm_council.orchestrator import execute_council


def test_clean_subprocess_env_injects_nested_marker_sieve() -> None:
    env = clean_subprocess_env()
    assert env["LLM_COUNCIL_NESTED"] == "1"


def test_clean_subprocess_env_injects_nested_marker_strict() -> None:
    env = clean_subprocess_env(strict=True)
    assert env["LLM_COUNCIL_NESTED"] == "1"


@pytest.mark.parametrize("marker", ["1", "0", ""])
def test_execute_council_refuses_on_any_marker_value(
    marker: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Presence semantics: any value — including empty or \"0\" — means this
    process descends from a council participant. Unsetting the variable is
    the only escape hatch; a truthiness check would make the empty string a
    silent bypass."""
    monkeypatch.setenv("LLM_COUNCIL_NESTED", marker)
    with pytest.raises(ValueError, match="NestedCouncilRefused"):
        asyncio.run(
            execute_council(
                participants=["claude"],
                participant_cfg={},
                prompt="question",
                cwd=tmp_path,
                config={},
            )
        )


def test_cli_run_refuses_when_nested_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI surfaces the refusal as a clean SystemExit, not a traceback."""
    monkeypatch.setenv("LLM_COUNCIL_NESTED", "1")
    from llm_council.cli import build_parser, cmd_run

    args = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "quick", "question"]
    )
    with pytest.raises(SystemExit, match="NestedCouncilRefused"):
        cmd_run(args)


def test_cli_run_other_value_errors_still_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the nested refusal converts to SystemExit; other ValueErrors
    keep their traceback so real bugs stay debuggable."""
    from llm_council import cli as cli_module
    from llm_council.cli import build_parser, cmd_run

    async def boom(*_args, **_kwargs):
        raise ValueError("SomeOtherError: not the nested guard")

    monkeypatch.setattr(cli_module, "execute_council", boom)
    args = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "quick", "question"]
    )
    with pytest.raises(ValueError, match="SomeOtherError"):
        cmd_run(args)


def test_codex_default_args_disable_mcp_servers() -> None:
    args = DEFAULT_CONFIG["participants"]["codex"]["args"]
    idx = args.index("-c")
    assert args[idx + 1] == "mcp_servers={}"
    # The read-only sandbox flag must survive alongside the MCP override.
    assert "read-only" in args


def test_migration_upgrades_pre_v21_codex_args() -> None:
    config = {
        "participants": {
            "codex": {
                "type": "cli",
                "family": "codex",
                "args": list(OLD_CODEX_EPHEMERAL_ARGS),
            }
        }
    }
    migrate_known_cli_defaults(config)
    assert (
        config["participants"]["codex"]["args"]
        == DEFAULT_CONFIG["participants"]["codex"]["args"]
    )


def test_codex_customized_args_get_mcp_starvation_injected() -> None:
    """Field case: one extra flag defeats the exact-match migration, so the
    command builder itself must starve the MCP table."""
    cfg = {
        "family": "codex",
        "command": "codex",
        "args": [
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--cd",
            "{cwd}",
            "-",
        ],
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    idx = cmd.index("-c")
    assert cmd[idx + 1] == "mcp_servers={}"
    # Attached to the exec subcommand, before the stdin marker.
    assert cmd.index("exec") < idx < cmd.index("-")
    # Read-only sandbox flag untouched.
    assert "read-only" in cmd


def test_codex_baseline_args_not_double_injected() -> None:
    cfg = dict(DEFAULT_CONFIG["participants"]["codex"])
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    assert cmd.count("mcp_servers={}") == 1


def test_codex_operator_mcp_override_suppresses_injection() -> None:
    """An explicit per-server mcp_servers override is an operator opt-in;
    the blanket starvation must not clobber it."""
    cfg = {
        "family": "codex",
        "command": "codex",
        "args": ["exec", "-c", "mcp_servers.tools.command=x", "-"],
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    assert "mcp_servers={}" not in cmd


def test_codex_model_pinned_injection_lands_after_exec() -> None:
    """With a pinned model, `exec` migrates into the command head; the
    starvation must follow it there, once."""
    cfg = {
        "family": "codex",
        "command": "codex",
        "model": "gpt-5.4",
        "args": ["exec", "--sandbox", "read-only", "-"],
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    assert cmd.count("mcp_servers={}") == 1
    assert cmd.index("exec") < cmd.index("mcp_servers={}")


def test_non_exec_codex_args_left_alone() -> None:
    """No `exec` token → unknown invocation shape → no injection."""
    cfg = {"family": "codex", "command": "codex", "args": ["--version"]}
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    assert "mcp_servers={}" not in cmd


def test_non_codex_families_never_injected() -> None:
    cfg = {"family": "claude", "command": "claude", "args": ["-p", "exec"]}
    cmd = _build_cli_command("claude", cfg, "q", Path("/tmp"))
    assert "mcp_servers={}" not in cmd
