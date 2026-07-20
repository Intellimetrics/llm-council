"""Nested-council recursion guard.

A council participant must never be able to start another council: each
level would spawn its own peers, so recursion grows exponentially (the
observed real-world path is a peer CLI whose global MCP config registers
llm-council itself). Two independent layers are covered here:

1. Every CLI child env carries ``LLM_COUNCIL_NESTED=1`` and
   ``execute_council`` refuses to run when it sees the marker.
2. The codex baseline args clear the peer's MCP server table entirely
   (``-c mcp_servers={}``), so the nested server never even boots.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llm_council.adapters import clean_subprocess_env
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
