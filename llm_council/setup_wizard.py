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


OPENROUTER_PARTICIPANTS = (
    "deepseek_v4_pro",
    "deepseek_v4_flash",
    "qwen_coder_plus",
    "qwen_coder_flash",
    "glm_5_1",
    "glm_4_7_flash",
    "kimi_k2_6",
)


INSTRUCTION_TEXT = """# LLM Council

This project has LLM Council installed. Use it as a read-only second-opinion
system when the user wants more than this single agent's judgment.

Natural triggers:
- "use council"
- "go to council"
- "ask council"
- "take this to council"
- "get another model's opinion"
- "have Claude/Gemini/Codex review this"

When triggered, call the `llm-council` MCP tool `council_run`.

Routing rules:
- Always pass `current` as `{current}` so transcripts show which host will
  synthesize and act on the council output.
- Use `quick` mode unless the user names a mode. `quick` asks Claude, Codex,
  and Gemini as explicit read-only participants. Use `peer-only` only when the
  user specifically wants to exclude this host CLI from subprocess review.
- Treat "on the diff", "current diff", or "review my changes" as
  `include_diff: true`.
- Treat "cheap" or "budget" as `review-cheap`.
- Treat "private", "local", or "offline" as `private-local`.
- Treat "with deepseek" as including `deepseek_v4_pro`.
- Treat "with qwen" as including `qwen_coder_plus`.
- Treat "with glm" as including `glm_5_1`.

Use council for:
- architecture or design decisions
- risky or cross-cutting refactors
- security-sensitive code paths
- database migrations
- release-gate reviews
- stubborn bugs after a failed attempt
- plans where independent disagreement would be useful

Do not use council for trivial formatting, obvious syntax fixes, or exact
mechanical edits the user already specified.

Before acting on council feedback:
- Summarize the main agreements, disagreements, and concrete risks.
- Identify which recommendations you will follow.
- Ask before making large or risky edits unless the user already authorized
  implementation.

Data boundary:
- Do not send classified, CUI, regulated, production, secret, credential, or
  customer data to council unless the user explicitly confirms the configured
  participants are approved for that data.
- Do not include files, diffs, logs, or environment content marked secret,
  sensitive, private, or deployment-only.
- US-origin participants mean model/company origin only; that is not the same
  as GovCloud, FedRAMP, or an enterprise data-handling approval.

Council is advisory and read-only by default. Council participants should not
edit files; this agent remains responsible for deciding what to do next.
"""


def project_config(
    include_native: bool = True,
    include_openrouter: bool = True,
    include_local: bool = True,
    us_only_default: bool = False,
) -> dict:
    participant_names: set[str] = set()
    if include_native:
        participant_names.update({"claude", "codex", "gemini"})
    if include_openrouter:
        participant_names.update(OPENROUTER_PARTICIPANTS)
    if include_local:
        participant_names.add("local_qwen_coder")

    modes = {
        name: mode
        for name, mode in DEFAULT_CONFIG["modes"].items()
        if _mode_available(mode, participant_names, include_native=include_native)
    }
    if not include_native and include_openrouter:
        modes.update(_openrouter_only_modes())
    config = {
        "version": 1,
        "replace_defaults": True,
        "transcripts_dir": ".llm-council/runs",
        "defaults": dict(DEFAULT_CONFIG["defaults"]),
        "participants": {},
        "modes": modes,
    }
    if not include_native and include_openrouter:
        config["defaults"]["mode"] = "quick"
    if us_only_default:
        config["defaults"]["origin_policy"] = "us"
        modes.pop("us-only", None)
    if include_native:
        for name in ("claude", "codex", "gemini"):
            config["participants"][name] = DEFAULT_CONFIG["participants"][name]
    if include_openrouter:
        for name in OPENROUTER_PARTICIPANTS:
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


def _mode_available(
    mode: dict[str, Any], participant_names: set[str], *, include_native: bool
) -> bool:
    if mode.get("strategy") == "other_cli_peers" and not include_native:
        return False
    return _mode_participants(mode).issubset(participant_names)


def _openrouter_only_modes() -> dict[str, dict[str, Any]]:
    return {
        "quick": {
            "participants": [
                "deepseek_v4_flash",
                "qwen_coder_flash",
                "glm_4_7_flash",
            ],
            "description": "Hosted OpenRouter breadth reviewers.",
        },
        "plan": {
            "participants": [
                "deepseek_v4_pro",
                "qwen_coder_plus",
                "glm_5_1",
            ],
            "description": "Hosted OpenRouter planning reviewers.",
        },
        "review": {
            "participants": [
                "deepseek_v4_flash",
                "qwen_coder_plus",
                "glm_4_7_flash",
            ],
            "description": "Hosted OpenRouter code review.",
        },
    }


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
                    "LLM_COUNCIL_MCP_ROOT": str(project_root.resolve()),
                },
            }
        }
    }


def write_setup_files(
    root: Path,
    *,
    include_native: bool = True,
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
        include_native=include_native,
        include_openrouter=include_openrouter,
        include_local=include_local,
        us_only_default=us_only_default,
    )
    if config_path.exists() and not force:
        existing_config = _read_yaml_mapping(config_path)
        merged_config = deep_merge(desired_config, existing_config)
    else:
        merged_config = desired_config
    if force or not config_path.exists() or merged_config != (
        _read_yaml_mapping(config_path) if config_path.exists() else None
    ):
        config_path.write_text(
            yaml.safe_dump(merged_config, sort_keys=False),
            encoding="utf-8",
        )
        written.append(config_path)

    if write_mcp:
        mcp_path = root / ".mcp.json"
        if mcp_path.exists() and not force:
            existing = _read_json_mapping(mcp_path)
            servers = existing.setdefault("mcpServers", {})
            servers["llm-council"] = mcp_config(root)["mcpServers"]["llm-council"]
            mcp_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        else:
            mcp_path.write_text(
                json.dumps(mcp_config(root), indent=2) + "\n", encoding="utf-8"
            )
        written.append(mcp_path)

    if write_instructions:
        instructions_dir = root / ".llm-council" / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        for name in ("claude", "codex", "gemini"):
            path = instructions_dir / f"{name}.md"
            if force or not path.exists():
                path.write_text(
                    INSTRUCTION_TEXT.format(current=name),
                    encoding="utf-8",
                )
                written.append(path)

    runtime_gitignore = root / ".llm-council" / ".gitignore"
    runtime_gitignore.parent.mkdir(parents=True, exist_ok=True)
    desired_gitignore = "runs/\n*.log\n"
    if (
        force
        or not runtime_gitignore.exists()
        or runtime_gitignore.read_text(encoding="utf-8") != desired_gitignore
    ):
        runtime_gitignore.write_text(desired_gitignore, encoding="utf-8")
        written.append(runtime_gitignore)

    project_gitignore = root / ".gitignore"
    if write_mcp:
        updated = _ensure_project_gitignore(project_gitignore)
        if updated:
            written.append(project_gitignore)

    return written


def _ensure_project_gitignore(path: Path) -> bool:
    desired_lines = [
        "# LLM Council local machine config and runtime data",
        ".mcp.json",
        ".llm-council/runs/",
        ".llm-council/*.log",
        ".llm-council.env",
    ]
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    missing = [line for line in desired_lines if line not in existing]
    if not missing:
        return False
    lines = existing[:]
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(missing)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse {path}. Fix it or rerun setup with --force.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping. Fix it or rerun setup with --force.")
    return data


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {path}. Fix it or rerun setup with --force.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object. Fix it or rerun setup with --force.")
    return data
