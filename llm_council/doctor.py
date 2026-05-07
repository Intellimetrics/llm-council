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

# Well-known local OpenAI-compatible inference servers. Listed in order of
# rough popularity; the doctor port-scan iterates this list.
#
# Each entry is (port, label) where the label is purely cosmetic (shows up in
# the Check name so users can tell which server responded). The probe still
# validates the response shape — port answering is necessary but not
# sufficient. `:8000` and `:8080` are common dev-server ports (Django,
# FastAPI, http.server, Tomcat) so the JSON-shape check is load-bearing.
WELL_KNOWN_LOCAL_OPENAI_PORTS: list[tuple[int, str]] = [
    (8000, "vLLM/sglang"),
    (1234, "LM Studio"),
    (8080, "llama.cpp/TGI"),
    (11434, "Ollama /v1"),
    (5000, "MLX"),
]


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
    if age > threshold:
        return Check(
            name="catalog:openrouter",
            ok=False,
            detail=(
                f"stale ({_format_duration(age)} old > "
                f"{_format_duration(threshold)} threshold) — "
                "run `llm-council models refresh`"
            ),
        )
    return Check(
        name="catalog:openrouter",
        ok=True,
        detail=f"fresh ({_format_duration(age)} old)",
    )


def _format_duration(seconds: float) -> str:
    """Render a duration in the smallest sensible unit.

    Avoids the `0.0 days old > 0-day threshold` confusion when a user
    configures sub-day thresholds — picks whichever unit produces a
    readable, non-zero number.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


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


def normalize_local_openai_base_url(url: str) -> str:
    """Canonicalize a user-provided base URL to point at `/v1`.

    Accepts `http://host:port`, `http://host:port/`, `http://host:port/v1`,
    or `http://host:port/v1/` and returns the trimmed `/v1` form. The probe
    appends `/models` itself, so the canonical form must end in `/v1` with
    no trailing slash.
    """
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


@dataclass
class LocalOpenAIProbe:
    """Structured result from probing a single local OpenAI-compatible endpoint.

    Carries everything the wizard needs to scaffold a participant block
    (canonical `base_url`, full list of served `models`) plus the
    human-readable `Check` for the doctor command. Replaces the prior
    pattern of having the wizard reverse-engineer model ids by parsing
    `Check.detail` strings.
    """

    label: str          # well-known port label or full URL
    base_url: str       # canonical http://host:port/v1 form
    ok: bool
    detail: str         # human-readable status (matches check.detail)
    models: tuple[str, ...]  # full served model id list (NOT truncated)

    def to_check(self) -> Check:
        return Check(
            name=f"probe:local-openai:{self.label}",
            ok=self.ok,
            detail=self.detail,
        )


def _probe_one_local_openai(
    base_url: str, *, timeout: float, label: str | None = None
) -> LocalOpenAIProbe:
    """Probe a single OpenAI-compatible endpoint.

    Validates the JSON shape of `/v1/models`, not just that the port answers.
    `:8000` is a common dev-server port; without shape validation the probe
    would happily report a Django app as a "local model server."
    """
    root = normalize_local_openai_base_url(base_url)
    effective_label = label or root

    def fail(detail: str) -> LocalOpenAIProbe:
        return LocalOpenAIProbe(
            label=effective_label,
            base_url=root,
            ok=False,
            detail=detail,
            models=(),
        )

    try:
        response = httpx.get(f"{root}/models", timeout=timeout)
    except Exception as exc:
        return fail(f"{type(exc).__name__}: {exc}")
    if response.status_code == 404:
        return fail(
            "HTTP 404 — server reachable but `/v1/models` not implemented "
            "(some llama.cpp builds; can still be usable via "
            "`/v1/chat/completions` if the model id is known)"
        )
    if response.status_code != 200:
        return fail(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except Exception:
        return fail(
            "HTTP 200 but body is not JSON — almost certainly not an "
            "OpenAI-compatible endpoint (probably a generic web server)"
        )
    # Canonical OpenAI shape: {"object": "list", "data": [{"id": "...", ...}]}.
    # Some servers omit `object` but the data shape is the load-bearing check.
    if not isinstance(body, dict):
        return fail(
            "HTTP 200 JSON but not an object (not OpenAI-compatible)"
        )
    data = body.get("data")
    if not isinstance(data, list):
        return fail(
            "HTTP 200 JSON but missing OpenAI-compatible `data` array "
            "(probably a different API on the same port)"
        )
    model_ids: tuple[str, ...] = tuple(
        str(entry.get("id"))
        for entry in data
        if isinstance(entry, dict) and entry.get("id")
    )
    if not model_ids:
        return LocalOpenAIProbe(
            label=effective_label,
            base_url=root,
            ok=True,
            detail="endpoint reachable but no models listed",
            models=(),
        )
    preview = ", ".join(model_ids[:3])
    if len(model_ids) > 3:
        preview = f"{preview}, … (+{len(model_ids) - 3})"
    return LocalOpenAIProbe(
        label=effective_label,
        base_url=root,
        ok=True,
        detail=f"{len(model_ids)} model(s): {preview}",
        models=model_ids,
    )


def discover_local_openai(base_url: str | None) -> list[LocalOpenAIProbe]:
    """Probe local OpenAI-compatible inference servers, returning structured results.

    Same scan-or-explicit semantics as :func:`probe_local_openai` (which is
    a thin wrapper that adapts these records to `Check` objects), but
    exposes the canonical `base_url` and the full served-models list.
    Used by the setup wizard to scaffold participant blocks without
    reverse-engineering anything from human-readable strings.
    """
    if base_url:
        return [_probe_one_local_openai(base_url, timeout=5.0)]

    probes: list[LocalOpenAIProbe] = []
    for port, label in WELL_KNOWN_LOCAL_OPENAI_PORTS:
        url = f"http://127.0.0.1:{port}"
        probe = _probe_one_local_openai(url, timeout=0.5, label=label)
        # Suppress noise from ports nothing is listening on. Connection
        # failures show up as ConnectError / ConnectionRefusedError in the
        # detail; everything else (timeouts, 404s, wrong-shape responses)
        # is informative and worth surfacing.
        if not probe.ok and (
            "ConnectError" in probe.detail
            or "ConnectionRefused" in probe.detail
            or "Connect call failed" in probe.detail
        ):
            continue
        probes.append(probe)
    return probes


def probe_local_openai(base_url: str | None) -> list[Check]:
    """Probe local OpenAI-compatible inference servers.

    With `base_url=None`, scans the well-known ports in
    :data:`WELL_KNOWN_LOCAL_OPENAI_PORTS` on `127.0.0.1` with short
    per-port timeouts. Silent ports (connection refused) are omitted from
    the output to keep the report scannable; only ports that responded —
    successfully or otherwise — are included.

    With an explicit `base_url`, probes that endpoint with a longer timeout
    and always emits one check.

    Wraps :func:`discover_local_openai` for cmd_doctor; if you need
    structured access to URL + served models (e.g. for scaffolding), use
    `discover_local_openai` directly.
    """
    probes = discover_local_openai(base_url)
    checks = [probe.to_check() for probe in probes]
    if not checks and base_url is None:
        ports_list = ", ".join(
            str(port) for port, _ in WELL_KNOWN_LOCAL_OPENAI_PORTS
        )
        checks.append(
            Check(
                name="probe:local-openai",
                ok=False,
                detail=(
                    f"no local OpenAI-compatible endpoints found on common "
                    f"ports ({ports_list}). Pass an explicit URL to probe a "
                    f"non-default port."
                ),
            )
        )
    return checks


def checks_to_dict(checks: list[Check]) -> list[dict[str, object]]:
    return [
        {"name": check.name, "ok": check.ok, "detail": check.detail}
        for check in checks
    ]
