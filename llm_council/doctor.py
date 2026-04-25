"""Environment diagnostics for llm-council."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def check_environment(
    config: dict[str, Any],
    *,
    probe_openrouter: bool = False,
    probe_ollama: bool = False,
) -> list[Check]:
    checks: list[Check] = []
    participants = config.get("participants", {})

    for name in ("claude", "codex", "gemini"):
        cfg = participants.get(name, {})
        command = cfg.get("command", name)
        resolved = shutil.which(command)
        checks.append(
            Check(
                name=f"cli:{name}",
                ok=bool(resolved),
                detail=resolved or f"{command} not found on PATH",
            )
        )

    if any(cfg.get("type") == "openrouter" for cfg in participants.values()):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        checks.append(
            Check(
                name="env:OPENROUTER_API_KEY",
                ok=bool(api_key),
                detail="set" if api_key else "not set",
            )
        )
        if probe_openrouter:
            checks.append(_probe_openrouter(api_key))

    if any(cfg.get("type") == "ollama" for cfg in participants.values()):
        resolved = shutil.which("ollama")
        checks.append(
            Check(
                name="cli:ollama",
                ok=bool(resolved),
                detail=resolved or "ollama not found on PATH",
            )
        )
        if probe_ollama:
            ollama_cfgs = [
                cfg for cfg in participants.values() if cfg.get("type") == "ollama"
            ]
            base_url = str(
                (ollama_cfgs[0] if ollama_cfgs else {}).get("base_url")
                or "http://localhost:11434"
            )
            checks.append(_probe_ollama(base_url))

    try:
        import mcp  # noqa: F401

        checks.append(Check(name="python:mcp", ok=True, detail="installed"))
    except Exception as exc:
        checks.append(
            Check(name="python:mcp", ok=False, detail=f"{type(exc).__name__}: {exc}")
        )

    return checks


def _probe_openrouter(api_key: str | None) -> Check:
    if not api_key:
        return Check(
            name="probe:openrouter",
            ok=False,
            detail="skipped because OPENROUTER_API_KEY is not set",
        )
    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if response.status_code == 200:
            return Check(name="probe:openrouter", ok=True, detail="auth ok")
        return Check(
            name="probe:openrouter",
            ok=False,
            detail=f"HTTP {response.status_code}",
        )
    except Exception as exc:
        return Check(
            name="probe:openrouter",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _probe_ollama(base_url: str) -> Check:
    root = base_url.rstrip("/")
    try:
        response = httpx.get(f"{root}/api/tags", timeout=5)
        if response.status_code == 200:
            count = len((response.json() or {}).get("models", []))
            return Check(name="probe:ollama", ok=True, detail=f"{count} models")
        return Check(
            name="probe:ollama", ok=False, detail=f"HTTP {response.status_code}"
        )
    except Exception as exc:
        return Check(
            name="probe:ollama",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def checks_to_dict(checks: list[Check]) -> list[dict[str, object]]:
    return [
        {"name": check.name, "ok": check.ok, "detail": check.detail}
        for check in checks
    ]
