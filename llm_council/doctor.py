"""Environment diagnostics for llm-council."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from llm_council.model_catalog import (
    openrouter_cache_age_seconds,
    openrouter_cache_path,
)


CATALOG_STALE_SECONDS_DEFAULT = 14 * 24 * 60 * 60


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _is_openrouter_participant(cfg: dict[str, Any]) -> bool:
    if cfg.get("type") == "openrouter":
        return True
    if cfg.get("type") != "openai_compatible":
        return False
    base_url = str(cfg.get("base_url") or "")
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def check_environment(
    config: dict[str, Any],
    *,
    probe_openrouter: bool = False,
    probe_ollama: bool = False,
) -> list[Check]:
    checks: list[Check] = []
    participants = config.get("participants", {})

    for name, cfg in participants.items():
        if cfg.get("type") != "cli":
            continue
        command = cfg.get("command", name)
        resolved = shutil.which(command)
        checks.append(
            Check(
                name=f"cli:{name}",
                ok=bool(resolved),
                detail=resolved or f"{command} not found on PATH",
            )
        )

    openrouter_envs = sorted(
        {
            str(cfg.get("api_key_env") or "OPENROUTER_API_KEY")
            for cfg in participants.values()
            if _is_openrouter_participant(cfg)
        }
    )
    for key_env in openrouter_envs:
        api_key = os.environ.get(key_env)
        checks.append(
            Check(
                name=f"env:{key_env}",
                ok=bool(api_key),
                detail="set" if api_key else "not set",
            )
        )
    if openrouter_envs:
        checks.append(_check_openrouter_catalog_age(config))
        if probe_openrouter:
            key_env = openrouter_envs[0]
            checks.append(_probe_openrouter(os.environ.get(key_env), key_env=key_env))

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


def _check_openrouter_catalog_age(config: dict[str, Any]) -> Check:
    threshold = int(
        (config.get("defaults", {}) or {}).get(
            "catalog_stale_seconds", CATALOG_STALE_SECONDS_DEFAULT
        )
    )
    age = openrouter_cache_age_seconds()
    if age is None:
        return Check(
            name="catalog:openrouter",
            ok=False,
            detail=(
                f"missing ({openrouter_cache_path()}) — "
                "run `llm-council models refresh`"
            ),
        )
    days = age / 86400.0
    if age > threshold:
        return Check(
            name="catalog:openrouter",
            ok=False,
            detail=(
                f"stale ({days:.1f} days old > "
                f"{threshold / 86400.0:.0f}-day threshold) — "
                "run `llm-council models refresh`"
            ),
        )
    return Check(
        name="catalog:openrouter",
        ok=True,
        detail=f"fresh ({days:.1f} days old)",
    )


def _probe_openrouter(
    api_key: str | None, *, key_env: str = "OPENROUTER_API_KEY"
) -> Check:
    if not api_key:
        return Check(
            name="probe:openrouter",
            ok=False,
            detail=f"skipped because {key_env} is not set",
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
