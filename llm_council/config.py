"""Configuration loading and participant selection."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from llm_council.defaults import DEFAULT_CONFIG


BASELINE_CLIS = ("claude", "codex", "gemini")
CONFIG_NAMES = (
    ".llm-council.yaml",
    ".llm-council.yml",
    "llm-council.yaml",
    "llm-council.yml",
)
OLD_CLAUDE_PLAN_ARGS = [
    "-p",
    "--permission-mode",
    "plan",
    "--tools",
    "Read,Grep,Glob,LS",
    "--no-session-persistence",
]
OLD_CODEX_APPROVAL_ARGS = [
    "exec",
    "--sandbox",
    "read-only",
    "--ask-for-approval",
    "never",
    "--ephemeral",
    "--cd",
    "{cwd}",
    "-",
]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of override into base."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def find_config(start: Path | None = None) -> Path | None:
    """Find a project config by walking upward from start."""

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in CONFIG_NAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def load_config(path: str | Path | None = None, *, search: bool = True) -> dict[str, Any]:
    """Load config, merging project values over built-in defaults."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        config_path = Path(path).expanduser()
    else:
        config_path = find_config() if search else None
    if not config_path:
        validate_config(config)
        return config
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    if data.get("replace_defaults"):
        config["participants"] = {}
        config["modes"] = {}
        data = {key: value for key, value in data.items() if key != "replace_defaults"}
    merged = deep_merge(config, data)
    migrate_known_cli_defaults(merged)
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate the small config surface before any participant is invoked."""

    participants = config.get("participants")
    if not isinstance(participants, dict) or not participants:
        raise ValueError("Config must define a non-empty participants mapping")
    for name, participant in participants.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Participant names must be non-empty strings")
        if not isinstance(participant, dict):
            raise ValueError(f"Participant '{name}' must be a mapping")
        ptype = participant.get("type")
        if ptype not in {"cli", "openrouter", "ollama"}:
            raise ValueError(
                f"Participant '{name}' has unsupported type '{ptype}'. "
                "Expected cli, openrouter, or ollama."
            )
        if ptype == "cli":
            if not participant.get("command"):
                raise ValueError(f"CLI participant '{name}' must define command")
            _validate_string_list(participant, "args", f"CLI participant '{name}'")
            _validate_string_list(
                participant, "env_passthrough", f"CLI participant '{name}'"
            )
        if ptype in {"openrouter", "ollama"} and not participant.get("model"):
            raise ValueError(f"Participant '{name}' must define model")
        _validate_positive_int(participant, "timeout", f"participant '{name}'")
        _validate_positive_int(participant, "max_prompt_chars", f"participant '{name}'")

    modes = config.get("modes")
    if not isinstance(modes, dict) or not modes:
        raise ValueError("Config must define a non-empty modes mapping")
    for name, mode in modes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Mode names must be non-empty strings")
        if not isinstance(mode, dict):
            raise ValueError(f"Mode '{name}' must be a mapping")
        has_participants = "participants" in mode
        has_strategy = mode.get("strategy") is not None
        if has_participants == has_strategy:
            raise ValueError(
                f"Mode '{name}' must define exactly one of participants or strategy"
            )
        referenced = list(mode.get("participants") or []) + list(mode.get("add") or [])
        if not all(isinstance(item, str) for item in referenced):
            raise ValueError(f"Mode '{name}' participants/add must contain strings")
        for participant in referenced:
            if participant not in participants:
                raise ValueError(
                    f"Mode '{name}' references unknown participant '{participant}'"
                )
        if has_strategy and mode.get("strategy") != "other_cli_peers":
            raise ValueError(f"Mode '{name}' has unsupported strategy '{mode.get('strategy')}'")
        if mode.get("origin_policy") not in (None, "any", "us"):
            raise ValueError(f"Mode '{name}' origin_policy must be 'any' or 'us'")
        _validate_positive_int(mode, "max_rounds", f"mode '{name}'")

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Config defaults must be a mapping")
    if defaults.get("origin_policy") not in (None, "any", "us"):
        raise ValueError("defaults.origin_policy must be 'any' or 'us'")
    if defaults.get("mode") and defaults["mode"] not in modes:
        raise ValueError(f"defaults.mode references unknown mode '{defaults['mode']}'")
    _validate_positive_int(defaults, "max_concurrency", "defaults")
    _validate_positive_int(defaults, "max_deliberation_rounds", "defaults")
    _validate_positive_int(defaults, "max_prompt_chars", "defaults")
    _validate_positive_int(defaults, "mcp_max_prompt_chars", "defaults")
    _validate_positive_number(defaults, "mcp_max_estimated_cost_usd", "defaults")


def migrate_known_cli_defaults(config: dict[str, Any]) -> None:
    """Apply compatibility fixes for previously generated unsafe defaults."""

    claude = config.get("participants", {}).get("claude")
    if isinstance(claude, dict) and (
        claude.get("type") == "cli"
        and claude.get("family") == "claude"
        and claude.get("args") == OLD_CLAUDE_PLAN_ARGS
    ):
        claude["args"] = list(DEFAULT_CONFIG["participants"]["claude"]["args"])
    codex = config.get("participants", {}).get("codex")
    if isinstance(codex, dict) and (
        codex.get("type") == "cli"
        and codex.get("family") == "codex"
        and codex.get("args") == OLD_CODEX_APPROVAL_ARGS
    ):
        codex["args"] = list(DEFAULT_CONFIG["participants"]["codex"]["args"])


def _validate_positive_int(mapping: dict[str, Any], key: str, label: str) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}.{key} must be a positive integer")


def _validate_string_list(mapping: dict[str, Any], key: str, label: str) -> None:
    value = mapping.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} {key} must be a string list")


def _validate_positive_number(mapping: dict[str, Any], key: str, label: str) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive number")


def detect_current_agent() -> str | None:
    """Best-effort detection of the CLI we are currently running under."""

    explicit = os.environ.get("LLM_COUNCIL_CURRENT") or os.environ.get(
        "LLM_COUNCIL_AGENT"
    )
    if explicit:
        normalized = explicit.strip().lower()
        return normalized if normalized in BASELINE_CLIS else None

    # Linux-specific parent process walk. If it fails, caller can use all peers.
    try:
        pid = os.getppid()
        seen: set[int] = set()
        while pid > 1 and pid not in seen:
            seen.add(pid)
            cmdline_path = Path("/proc") / str(pid) / "cmdline"
            stat_path = Path("/proc") / str(pid) / "stat"
            raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(
                errors="ignore"
            )
            lowered = raw.lower()
            for name in BASELINE_CLIS:
                if f"/{name}" in lowered or lowered.startswith(name):
                    return name
            stat = stat_path.read_text(errors="ignore")
            pid = int(stat.split()[3])
    except Exception:
        return None
    return None


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def select_participants(
    config: dict[str, Any],
    mode: str,
    current: str | None,
    explicit: list[str] | None = None,
    include: list[str] | None = None,
    origin_policy: str | None = None,
) -> list[str]:
    """Resolve participant names for a run."""

    participants = config.get("participants", {})
    modes = config.get("modes", {})

    mode_cfg = modes.get(mode, {})
    effective_origin_policy = (
        origin_policy
        or mode_cfg.get("origin_policy")
        or config.get("defaults", {}).get("origin_policy")
        or "any"
    )

    explicit_requested = bool(explicit)
    if explicit_requested:
        selected = list(explicit)
    else:
        if not mode_cfg:
            raise ValueError(f"Unknown mode '{mode}'. Known modes: {', '.join(modes)}")
        if "participants" in mode_cfg:
            selected = list(mode_cfg["participants"])
        elif mode_cfg.get("strategy") == "other_cli_peers":
            selected = [name for name in BASELINE_CLIS if name != current]
            if not current:
                selected = list(BASELINE_CLIS)
            selected.extend(mode_cfg.get("add", []))
        else:
            raise ValueError(f"Mode '{mode}' has no participants or known strategy")

    if include:
        selected.extend(include)

    deduped: list[str] = []
    for name in selected:
        if name not in participants:
            raise ValueError(f"Unknown participant '{name}'")
        if effective_origin_policy == "us":
            origin = str(participants[name].get("origin", ""))
            if not origin.startswith("US /"):
                continue
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        raise ValueError(
            "No participants selected"
            + (
                f" after applying origin_policy '{effective_origin_policy}'"
                if effective_origin_policy != "any"
                else ""
            )
        )
    return deduped
