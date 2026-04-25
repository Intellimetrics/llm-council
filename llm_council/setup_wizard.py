"""Setup helpers for llm-council."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_council.config import deep_merge
from llm_council.defaults import DEFAULT_CONFIG


INSTRUCTION_TEXT = """# LLM Council

When the user says "go to council", "ask council", or "take this to council",
call the `llm-council` MCP tool `council_run`.

Default behavior:
- Use `quick` mode unless the user names a mode.
- Treat "with deepseek" as including `deepseek_v4_pro`.
- Treat "with qwen" as including `qwen_coder_plus`.
- Treat "with glm" as including `glm_5_1`.
- Treat "cheap" as `review-cheap`.
- Treat "private" or "local" as `private-local`.
- Treat "on the diff" as `include_diff: true`.

Use council when the task is architectural, risky, cross-cutting, ambiguous, has
failed multiple times, or needs an independent code review. Do not use council
for trivial edits, formatting, obvious syntax fixes, or when the user gave exact
implementation steps.

Council is advisory and read-only by default. Do not apply edits from council
without explicit user direction.
"""


def project_config(
    include_openrouter: bool = True,
    include_local: bool = True,
    us_only_default: bool = False,
) -> dict:
    participant_names = {"claude", "codex", "gemini"}
    if include_openrouter:
        participant_names.update(
            {
                "deepseek_v4_pro",
                "deepseek_v4_flash",
                "qwen_coder_plus",
                "qwen_coder_free",
                "glm_5_1",
                "glm_4_7_flash",
                "kimi_k2_6",
            }
        )
    if include_local:
        participant_names.add("local_qwen_coder")

    modes = {
        name: mode
        for name, mode in DEFAULT_CONFIG["modes"].items()
        if _mode_participants(mode).issubset(participant_names)
    }
    config = {
        "version": 1,
        "transcripts_dir": ".llm-council/runs",
        "defaults": DEFAULT_CONFIG["defaults"],
        "participants": {
            "claude": {"model": None},
            "codex": {"model": None},
            "gemini": {"model": None},
        },
        "modes": modes,
    }
    if us_only_default:
        config["defaults"] = dict(config["defaults"])
        config["defaults"]["origin_policy"] = "us"
    if include_openrouter:
        for name in (
            "deepseek_v4_pro",
            "deepseek_v4_flash",
            "qwen_coder_plus",
            "qwen_coder_free",
            "glm_5_1",
            "glm_4_7_flash",
            "kimi_k2_6",
        ):
            config["participants"][name] = DEFAULT_CONFIG["participants"][name]
    if include_local:
        config["participants"]["local_qwen_coder"] = DEFAULT_CONFIG["participants"][
            "local_qwen_coder"
        ]
    return config


def _mode_participants(mode: dict[str, Any]) -> set[str]:
    names = set(mode.get("participants", []))
    names.update(mode.get("add", []))
    return names


def mcp_config(project_root: Path) -> dict[str, Any]:
    local_python = project_root / ".venv" / "bin" / "python"
    installed = shutil.which("llm-council")
    if installed:
        command = installed
        args = ["mcp-server"]
    elif local_python.exists():
        command = str(local_python)
        args = ["-m", "llm_council.mcp_server"]
    else:
        command = sys.executable
        args = ["-m", "llm_council.mcp_server"]
    return {
        "mcpServers": {
            "llm-council": {
                "type": "stdio",
                "command": command,
                "args": args,
                "env": {
                    "PYTHONPATH": str(project_root.resolve()),
                    "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}",
                },
            }
        }
    }


def write_setup_files(
    root: Path,
    *,
    include_openrouter: bool = True,
    include_local: bool = True,
    us_only_default: bool = False,
    write_mcp: bool = True,
    write_instructions: bool = True,
    force: bool = False,
) -> list[Path]:
    written: list[Path] = []
    root.mkdir(parents=True, exist_ok=True)

    config_path = root / ".llm-council.yaml"
    desired_config = project_config(
        include_openrouter=include_openrouter,
        include_local=include_local,
        us_only_default=us_only_default,
    )
    if config_path.exists() and not force:
        existing_config = yaml.safe_load(config_path.read_text()) or {}
        merged_config = deep_merge(desired_config, existing_config)
    else:
        merged_config = desired_config
    if force or not config_path.exists() or merged_config != (
        yaml.safe_load(config_path.read_text()) if config_path.exists() else None
    ):
        config_path.write_text(
            yaml.safe_dump(merged_config, sort_keys=False)
        )
        written.append(config_path)

    if write_mcp:
        mcp_path = root / ".mcp.json"
        if mcp_path.exists() and not force:
            existing = json.loads(mcp_path.read_text())
            servers = existing.setdefault("mcpServers", {})
            servers["llm-council"] = mcp_config(root)["mcpServers"]["llm-council"]
            mcp_path.write_text(json.dumps(existing, indent=2) + "\n")
        else:
            mcp_path.write_text(json.dumps(mcp_config(root), indent=2) + "\n")
        written.append(mcp_path)

    if write_instructions:
        instructions_dir = root / ".llm-council" / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        for name in ("claude", "codex", "gemini"):
            path = instructions_dir / f"{name}.md"
            if force or not path.exists():
                path.write_text(INSTRUCTION_TEXT)
                written.append(path)

    runtime_gitignore = root / ".llm-council" / ".gitignore"
    runtime_gitignore.parent.mkdir(parents=True, exist_ok=True)
    desired_gitignore = "runs/\n*.log\n"
    if force or not runtime_gitignore.exists() or runtime_gitignore.read_text() != desired_gitignore:
        runtime_gitignore.write_text(desired_gitignore)
        written.append(runtime_gitignore)

    return written
