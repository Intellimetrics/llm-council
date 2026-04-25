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


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config, merging project values over built-in defaults."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path).expanduser() if path else find_config()
    if not config_path:
        return config

    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return deep_merge(config, data)


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

    if explicit:
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
    return deduped
