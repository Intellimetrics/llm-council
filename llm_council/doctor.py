"""Environment diagnostics for llm-council."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from llm_council.adapters import (
    ERROR_KIND_CLIENT_INELIGIBLE,
    _build_cli_command,
    classify_error,
    clean_subprocess_env,
)
from llm_council.config import participant_api_key_env
from llm_council.env import env_get
from llm_council.model_catalog import (
    openrouter_cache_age_seconds,
    openrouter_cache_path,
    refresh_openrouter_cache,
)


CATALOG_STALE_SECONDS_DEFAULT = 14 * 24 * 60 * 60
# Tighter than `refresh_openrouter_cache`'s 30s default so a slow connection
# can't stall the doctor for half a minute. On failure we fall through to a
# stale-warning Check, so the user still gets a usable report.
CATALOG_AUTO_REFRESH_TIMEOUT_SECONDS = 10.0
NATIVE_CLI_PROBE_TIMEOUT_SECONDS = 15.0
NATIVE_CLI_PROBE_MAX_TIMEOUT_SECONDS = 30.0

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
    error_kind: str | None = None
    suggested_fallback: str | None = None


def _native_probe_requested(
    probe_native: bool | Collection[str], name: str
) -> bool:
    if probe_native is True:
        return True
    if not probe_native:
        return False
    if isinstance(probe_native, str):
        return probe_native == name
    return name in probe_native


def _probe_error_excerpt(text: str, *, limit: int = 240) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def probe_native_cli(
    name: str,
    cfg: dict[str, Any],
    *,
    cwd: Path | None = None,
    suggested_fallback: str | None = None,
) -> Check:
    """Run a tiny, explicitly requested native-CLI auth/readiness probe.

    The normal doctor remains PATH-only and makes no model calls. This probe
    intentionally invokes the configured command because Gemini's known
    ``UNSUPPORTED_CLIENT`` state is account-dependent and is not exposed by
    ``--version``. Runtime is hard-capped even when config requests more.
    """

    command_name = str(cfg.get("command") or name)
    if not shutil.which(command_name):
        return Check(
            name=f"probe:cli:{name}",
            ok=False,
            detail=f"skipped because {command_name} is not on PATH",
            error_kind="cli_not_found",
        )
    prompt = "Readiness probe. Reply with only: OK"
    probe_cwd = (cwd or Path.cwd()).resolve()
    command = _build_cli_command(name, cfg, prompt, probe_cwd)
    timeout = min(
        max(
            float(
                cfg.get("doctor_probe_timeout")
                or NATIVE_CLI_PROBE_TIMEOUT_SECONDS
            ),
            1.0,
        ),
        NATIVE_CLI_PROBE_MAX_TIMEOUT_SECONDS,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(probe_cwd),
            env=clean_subprocess_env(
                cfg.get("env_passthrough"),
                strict=bool(cfg.get("env_strict", False)),
            ),
            input=prompt if cfg.get("stdin_prompt") else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Check(
            name=f"probe:cli:{name}",
            ok=False,
            detail=f"readiness probe timed out after {timeout:g}s",
            error_kind="timeout",
        )
    except Exception as exc:  # noqa: BLE001 - doctor must report, not abort
        return Check(
            name=f"probe:cli:{name}",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            error_kind="probe_failed",
        )

    if completed.returncode == 0:
        return Check(
            name=f"probe:cli:{name}",
            ok=True,
            detail="authenticated invocation succeeded",
        )

    error_text = (completed.stderr or completed.stdout or "").strip()
    error_kind = classify_error(error_text) or "unknown"
    if error_kind == ERROR_KIND_CLIENT_INELIGIBLE:
        fallback_note = (
            f"; configured fallback {suggested_fallback!r} is available"
            if suggested_fallback
            else ""
        )
        detail = (
            "Gemini authentication rejected this CLI client "
            f"(UNSUPPORTED_CLIENT){fallback_note}"
        )
    else:
        excerpt = _probe_error_excerpt(error_text) or "no stderr output"
        detail = f"exit {completed.returncode}: {excerpt}"
    return Check(
        name=f"probe:cli:{name}",
        ok=False,
        detail=detail,
        error_kind=error_kind,
        suggested_fallback=suggested_fallback,
    )


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
    probe_native: bool | Collection[str] = False,
    probe_cwd: Path | None = None,
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
                detail=(
                    f"{resolved} (executable found; authentication not probed)"
                    if resolved
                    else f"{command} not found on PATH"
                ),
            )
        )
        if resolved and _native_probe_requested(probe_native, name):
            suggested_fallback = None
            if cfg.get("family") == "gemini":
                fallback_cfg = participants.get("antigravity") or {}
                fallback_command = str(
                    fallback_cfg.get("command") or "antigravity"
                )
                if (
                    fallback_cfg.get("type") == "cli"
                    and shutil.which(fallback_command)
                ):
                    suggested_fallback = "antigravity"
            checks.append(
                probe_native_cli(
                    name,
                    cfg,
                    cwd=probe_cwd,
                    suggested_fallback=suggested_fallback,
                )
            )

    api_key_envs = sorted(
        {
            key_env
            for cfg in participants.values()
            if (key_env := participant_api_key_env(cfg)) is not None
        }
    )
    for key_env in api_key_envs:
        api_key = env_get(key_env)
        checks.append(
            Check(
                name=f"env:{key_env}",
                ok=bool(api_key),
                detail="set" if api_key else "not set",
            )
        )
    openrouter_envs = sorted(
        {
            key_env
            for cfg in participants.values()
            if _is_openrouter_participant(cfg)
            and (key_env := participant_api_key_env(cfg)) is not None
        }
    )
    if openrouter_envs:
        checks.append(_check_openrouter_catalog_age(config))
        if probe_openrouter:
            key_env = openrouter_envs[0]
            checks.append(_probe_openrouter(env_get(key_env), key_env=key_env))

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
            expected_models = [
                str(cfg.get("model"))
                for cfg in ollama_cfgs
                if cfg.get("model")
                and str(cfg.get("base_url") or "http://localhost:11434").rstrip("/")
                == base_url.rstrip("/")
            ]
            checks.append(_probe_ollama(base_url, expected_models=expected_models))

    try:
        import mcp  # noqa: F401

        checks.append(Check(name="python:mcp", ok=True, detail="installed"))
    except Exception as exc:
        checks.append(
            Check(name="python:mcp", ok=False, detail=f"{type(exc).__name__}: {exc}")
        )

    return checks


def _check_openrouter_catalog_age(config: dict[str, Any]) -> Check:
    defaults = config.get("defaults", {}) or {}
    threshold = int(
        defaults.get("catalog_stale_seconds", CATALOG_STALE_SECONDS_DEFAULT)
    )
    auto_refresh = bool(defaults.get("catalog_auto_refresh", True))

    age = openrouter_cache_age_seconds()
    needs_refresh = age is None or age > threshold

    if not needs_refresh:
        return Check(
            name="catalog:openrouter",
            ok=True,
            detail=f"fresh ({_format_duration(age)} old)",
        )

    if auto_refresh:
        # Best-effort inline refresh. The catalog is just OpenRouter's public
        # model list — fetching it requires no auth and the failure mode is
        # network-only. Fail-soft so a disconnected user still gets a useful
        # diagnostic instead of an error.
        try:
            summary = refresh_openrouter_cache(
                timeout=CATALOG_AUTO_REFRESH_TIMEOUT_SECONDS
            )
        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {exc}"
            if age is None:
                return Check(
                    name="catalog:openrouter",
                    ok=False,
                    detail=f"missing — auto-refresh failed: {error_detail}",
                )
            return Check(
                name="catalog:openrouter",
                ok=False,
                detail=(
                    f"stale ({_format_duration(age)} old) — "
                    f"auto-refresh failed: {error_detail}"
                ),
            )
        return Check(
            name="catalog:openrouter",
            ok=True,
            detail=f"auto-refreshed ({summary['model_count']} models)",
        )

    if age is None:
        return Check(
            name="catalog:openrouter",
            ok=False,
            detail=(
                f"missing ({openrouter_cache_path()}) — "
                "run `llm-council models refresh`"
            ),
        )
    return Check(
        name="catalog:openrouter",
        ok=False,
        detail=(
            f"stale ({_format_duration(age)} old > "
            f"{_format_duration(threshold)} threshold) — "
            "run `llm-council models refresh`"
        ),
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


def _probe_ollama(
    base_url: str, *, expected_models: list[str] | None = None
) -> Check:
    root = base_url.rstrip("/")
    try:
        response = httpx.get(f"{root}/api/tags", timeout=5)
        if response.status_code == 200:
            records = (response.json() or {}).get("models", [])
            installed = {
                str(record.get("name") or record.get("model"))
                for record in records
                if isinstance(record, dict)
                and (record.get("name") or record.get("model"))
            }
            missing = [
                model
                for model in expected_models or []
                if model not in installed
                and not (
                    ":" not in model
                    and any(name.startswith(f"{model}:") for name in installed)
                )
            ]
            if missing:
                return Check(
                    name="probe:ollama",
                    ok=False,
                    detail=(
                        "configured model(s) not installed: "
                        + ", ".join(sorted(set(missing)))
                        + f"; server reports {len(installed)} model(s)"
                    ),
                )
            return Check(
                name="probe:ollama", ok=True, detail=f"{len(installed)} models"
            )
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
    rendered: list[dict[str, object]] = []
    for check in checks:
        item: dict[str, object] = {
            "name": check.name,
            "ok": check.ok,
            "detail": check.detail,
        }
        if check.error_kind is not None:
            item["error_kind"] = check.error_kind
        if check.suggested_fallback is not None:
            item["suggested_fallback"] = check.suggested_fallback
        rendered.append(item)
    return rendered
