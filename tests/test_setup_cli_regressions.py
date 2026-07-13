"""Regressions for setup routing and operator-facing CLI safety."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from llm_council import cli as cli_module
from llm_council.cli import (
    _auto_setup_preset,
    build_parser,
    cmd_config,
    cmd_doctor,
    cmd_install_hook,
)
from llm_council.config import is_local_participant, select_participants
from llm_council.doctor import Check
from llm_council.setup_wizard import mcp_config, project_config, write_setup_files
from llm_council.mcp_server import list_modes


def _native_config() -> dict:
    return project_config(
        include_native=True,
        include_openrouter=False,
        include_local=False,
    )


def _which_for(*available: str):
    commands = set(available)
    return lambda command: f"/usr/bin/{command}" if command in commands else None


def test_stats_help_uses_participant_and_hides_legacy_peer_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    canonical = parser.parse_args(["stats", "--participant", "claude"])
    legacy = parser.parse_args(["stats", "--peer", "claude"])
    assert canonical.participant == "claude"
    assert legacy.participant == "claude"

    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["stats", "--help"])
    help_text = capsys.readouterr().out
    assert "--participant" in help_text
    assert "--peer" not in help_text


def test_transcript_prune_help_uses_delete_and_hides_legacy_apply_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    canonical = parser.parse_args(
        ["transcripts", "prune", "--keep-last", "1", "--delete"]
    )
    legacy = parser.parse_args(
        ["transcripts", "prune", "--keep-last", "1", "--apply"]
    )
    assert canonical.apply is True
    assert legacy.apply is True

    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["transcripts", "prune", "--help"])
    help_text = capsys.readouterr().out
    assert "--delete" in help_text
    assert "--apply" not in help_text


@pytest.mark.parametrize("command", ["run", "estimate"])
def test_cost_cap_help_says_unknown_pricing_is_refused(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build_parser().parse_args([command, "--help"])
    help_text = capsys.readouterr().out
    assert "unknown pricing are refused" in help_text
    assert "informational only" not in help_text


def test_auto_route_and_selection_work_with_claude_and_gemini_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    which = _which_for("claude", "gemini")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", which)

    assert _auto_setup_preset(which) == "tri-cli"
    assert select_participants(_native_config(), "quick", current=None) == [
        "claude",
        "gemini",
    ]


def test_native_selection_prefers_available_antigravity_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", _which_for("claude", "codex", "gemini", "agy"))

    selected = select_participants(_native_config(), "quick", current=None)

    assert selected == ["claude", "codex", "antigravity"]
    assert "gemini" not in selected


def test_active_host_requires_its_subprocess_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", _which_for("agy"))

    with pytest.raises(
        ValueError,
        match="No configured Claude- or Codex-family CLI is available on PATH",
    ):
        select_participants(_native_config(), "quick", current="claude")


def test_missing_active_host_uses_other_available_primary_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", _which_for("codex", "agy"))

    assert select_participants(_native_config(), "quick", current="claude") == [
        "codex",
        "antigravity",
    ]


def test_antigravity_host_does_not_make_missing_gemini_look_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", _which_for("claude", "agy"))

    selected = select_participants(
        _native_config(), "quick", current="antigravity"
    )

    assert selected == ["claude", "antigravity"]
    assert "gemini" not in selected


def test_unconfigured_antigravity_binary_does_not_replace_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _native_config()
    config["participants"].pop("antigravity")
    monkeypatch.setattr("shutil.which", _which_for("claude", "codex", "gemini", "agy"))

    assert select_participants(config, "quick", current=None) == [
        "claude",
        "codex",
        "gemini",
    ]


def test_local_private_scaffold_defaults_to_strict_local_only() -> None:
    config = project_config(
        include_native=False,
        include_openrouter=False,
        include_local=True,
    )

    assert config["defaults"]["mode"] == "private-local"
    assert set(config["participants"]) == {"local_qwen_coder"}
    assert "private-local" in config["modes"]
    assert "local-only" not in config["modes"]
    assert "local-private" not in config["modes"]
    selected = select_participants(config, "private-local", current=None)
    assert selected == ["local_qwen_coder"]
    assert all(is_local_participant(config["participants"][name]) for name in selected)


def test_legacy_local_only_mode_input_resolves_to_private_local() -> None:
    config = project_config(
        include_native=False,
        include_openrouter=False,
        include_local=True,
    )

    assert select_participants(config, "local-only", current=None) == [
        "local_qwen_coder"
    ]


def test_private_local_excludes_ollama_on_public_endpoint() -> None:
    config = project_config(
        include_native=False,
        include_openrouter=False,
        include_local=True,
    )
    config["participants"]["hosted_ollama"] = {
        "type": "ollama",
        "model": "remote-model",
        "base_url": "https://ollama.example.com",
    }
    config["participants"]["lan_ollama"] = {
        "type": "ollama",
        "model": "remote-model",
        "base_url": "http://10.0.0.5:11434",
    }

    selected = select_participants(config, "private-local", current=None)

    assert "local_qwen_coder" in selected
    assert "hosted_ollama" not in selected
    assert "lan_ollama" not in selected


def test_legacy_local_only_config_migrates_to_canonical_name(tmp_path: Path) -> None:
    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        """
replace_defaults: true
defaults:
  mode: local-only
participants:
  local:
    type: ollama
    model: qwen
modes:
  local-only:
    strategy: local_only_peers
""".lstrip(),
        encoding="utf-8",
    )

    from llm_council.config import load_config

    config = load_config(config_path)
    assert config["defaults"]["mode"] == "private-local"
    assert set(config["modes"]) == {"private-local"}
    assert select_participants(config, "local-only", current=None) == ["local"]


def test_mcp_mode_listing_exposes_only_private_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_setup_files(
        tmp_path,
        include_native=False,
        include_openrouter=False,
        include_local=True,
        write_mcp=False,
        write_instructions=False,
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    modes = list_modes({"working_directory": str(tmp_path)})["modes"]

    assert "private-local" in modes
    assert "local-only" not in modes
    assert "local-private" not in modes


def test_setup_merge_preserves_explicit_private_local_roster(tmp_path: Path) -> None:
    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "replace_defaults": True,
                "defaults": {"mode": "private-local"},
                "participants": {
                    "local_pinned": {
                        "type": "ollama",
                        "family": "qwen",
                        "origin": "China / Alibaba Qwen",
                        "base_url": "http://127.0.0.1:11434",
                        "model": "qwen3:latest",
                        "read_only": True,
                    }
                },
                "modes": {
                    "private-local": {
                        "participants": ["local_pinned"],
                        "description": "Pinned local roster.",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    write_setup_files(
        tmp_path,
        include_native=False,
        include_openrouter=False,
        include_local=True,
        write_mcp=False,
        write_instructions=False,
        force=False,
    )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    private_local = raw["modes"]["private-local"]
    assert private_local["participants"] == ["local_pinned"]
    assert "strategy" not in private_local

    resolved = cli_module.load_config(config_path)
    assert select_participants(resolved, "private-local", current=None) == [
        "local_pinned"
    ]


def test_generated_mcp_config_does_not_import_shadow_installed_package(
    tmp_path: Path,
) -> None:
    env = mcp_config(tmp_path)["mcpServers"]["llm-council"]["env"]

    assert env == {"LLM_COUNCIL_MCP_ROOT": str(tmp_path.resolve())}
    assert "PYTHONPATH" not in env


def test_local_private_instructions_name_generated_default_mode(tmp_path: Path) -> None:
    write_setup_files(
        tmp_path,
        include_native=False,
        include_openrouter=False,
        include_local=True,
        write_mcp=False,
        write_instructions=True,
    )

    project_instructions = (
        tmp_path / ".llm-council" / "instructions" / "codex.md"
    ).read_text(encoding="utf-8")
    host_instructions = (
        tmp_path / ".llm-council" / "skills" / "codex-cli" / "AGENTS.md"
    ).read_text(encoding="utf-8")
    assert "configured default (`private-local`)" in project_instructions
    assert "Omit `mode` to use the configured project default" in host_instructions
    assert "Default `mode` is `quick`" not in project_instructions
    assert "Default `mode` is `quick`" not in host_instructions
    assert "`private-local` for loopback Ollama-only review" in host_instructions
    assert "does not firewall the Ollama daemon" in host_instructions


def test_doctor_fails_and_explains_unrunnable_default_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "defaults": {"mode": "private-local"},
        "participants": {},
        "modes": {"private-local": {"strategy": "local_only_peers"}},
    }
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_a, **_k: [])
    monkeypatch.setattr(cli_module, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(
        cli_module,
        "check_environment",
        lambda *_a, **_k: [Check("python:mcp", True, "installed")],
    )
    args = argparse.Namespace(
        config=None,
        json=False,
        probe_openrouter=False,
        probe_ollama=False,
        probe_local_openai=None,
        check_update=False,
    )

    assert cmd_doctor(args) == 1
    output = capsys.readouterr().out
    assert "route:default-mode" in output
    assert "has no matching participants" in output


def test_config_set_refuses_invalid_value_without_changing_file(tmp_path: Path) -> None:
    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "replace_defaults": True,
                "participants": {
                    "peer": {"type": "cli", "command": "true", "family": "test"}
                },
                "modes": {"custom": {"participants": ["peer"]}},
                "defaults": {"mode": "custom"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    before = config_path.read_bytes()
    args = build_parser().parse_args(
        [
            "config",
            "set",
            "modes.custom.min_quorum",
            "0",
            "--cwd",
            str(tmp_path),
        ]
    )

    with pytest.raises(SystemExit, match="Refusing to write invalid configuration"):
        cmd_config(args)
    assert config_path.read_bytes() == before


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)


def _hook_args(root: Path, *, mode: str = "consensus", force: bool = False):
    return argparse.Namespace(
        root=str(root),
        hook_type="pre-commit",
        mode=mode,
        force=force,
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows runners do not consistently grant symlink privileges",
)
def test_install_hook_refuses_existing_and_force_replaces_symlink_safely(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    hook = repo / ".git" / "hooks" / "pre-commit"
    target = tmp_path / "outside"
    target.write_text("do not overwrite", encoding="utf-8")
    hook.symlink_to(target)

    assert cmd_install_hook(_hook_args(repo)) == 1
    assert target.read_text(encoding="utf-8") == "do not overwrite"
    assert hook.is_symlink()

    assert cmd_install_hook(_hook_args(repo, force=True)) == 0
    assert target.read_text(encoding="utf-8") == "do not overwrite"
    assert not hook.is_symlink()
    assert "--mode consensus" in hook.read_text(encoding="utf-8")
    assert stat.S_IMODE(hook.stat().st_mode) & stat.S_IXUSR


def test_install_hook_validates_and_shell_quotes_mode(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    config_path = repo / ".llm-council.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "modes": {
                    "safe mode; touch nope": {"participants": ["claude"]},
                }
            }
        ),
        encoding="utf-8",
    )

    assert cmd_install_hook(_hook_args(repo, mode="unknown")) == 1
    assert cmd_install_hook(_hook_args(repo, mode="safe mode; touch nope")) == 0
    hook_text = (repo / ".git" / "hooks" / "pre-commit").read_text(encoding="utf-8")
    assert "--mode 'safe mode; touch nope'" in hook_text


def test_install_hook_accepts_legacy_local_only_alias(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    assert cmd_install_hook(_hook_args(repo, mode="local-only")) == 0

    hook_text = (repo / ".git" / "hooks" / "pre-commit").read_text(
        encoding="utf-8"
    )
    assert "--mode private-local" in hook_text
    assert "--mode local-only" not in hook_text


def test_install_hook_supports_linked_git_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    repo.mkdir()
    _init_git_repo(repo)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-m", "base"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", "-b", "linked", str(linked)],
        check=True,
    )

    assert cmd_install_hook(_hook_args(linked)) == 0
    hook_path = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "--git-path", "hooks/pre-commit"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hook = Path(hook_path)
    if not hook.is_absolute():
        hook = linked / hook
    assert hook.is_file()


def test_list_prints_experimental_marker_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_a, **_k: [])
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_a, **_k: {
            "participants": {},
            "modes": {
                "preview": {
                    "experimental": True,
                    "description": "EXPERIMENTAL — Inspect with tools.",
                }
            },
        },
    )

    assert cli_module.cmd_list(argparse.Namespace(config=None)) == 0
    assert capsys.readouterr().out.count("EXPERIMENTAL") == 1
