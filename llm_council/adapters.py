"""Participant adapters."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class ParticipantResult:
    name: str
    ok: bool
    output: str
    error: str
    elapsed_seconds: float
    command: list[str] | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


def _format_arg(value: str, *, prompt: str, cwd: Path) -> str:
    return value.replace("{prompt}", prompt).replace("{cwd}", str(cwd))


def _build_cli_command(name: str, cfg: dict[str, Any], prompt: str, cwd: Path) -> list[str]:
    command = [cfg.get("command", name)]
    args = [_format_arg(str(arg), prompt=prompt, cwd=cwd) for arg in cfg.get("args", [])]

    model = cfg.get("model")
    family = cfg.get("family", name)
    if model:
        if family == "codex":
            command.extend(["exec", "-m", str(model)])
            # If args already start with exec, remove duplicate below.
            if args and args[0] == "exec":
                args = args[1:]
        elif family == "claude":
            command.extend(["--model", str(model)])
        elif family == "gemini":
            command.extend(["--model", str(model)])
        else:
            command.extend(["--model", str(model)])

    return command + args


async def run_cli_participant(
    name: str, cfg: dict[str, Any], prompt: str, cwd: Path
) -> ParticipantResult:
    start = time.monotonic()
    timeout = int(cfg.get("timeout") or 240)
    command = _build_cli_command(name, cfg, prompt, cwd)
    stdin_prompt = bool(cfg.get("stdin_prompt"))
    stdin_data = prompt if stdin_prompt else None
    env = clean_subprocess_env()

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_data.encode() if stdin_data is not None else None),
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        return ParticipantResult(
            name=name,
            ok=proc.returncode == 0,
            output=out,
            error=err,
            elapsed_seconds=elapsed,
            command=redact_prompt_args(command, prompt),
            model=cfg.get("model"),
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return ParticipantResult(
            name=name,
            ok=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
            command=redact_prompt_args(command, prompt),
            model=cfg.get("model"),
        )


async def run_openrouter_participant(
    name: str, cfg: dict[str, Any], prompt: str
) -> ParticipantResult:
    start = time.monotonic()
    key_env = cfg.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(key_env)
    model = cfg.get("model")
    if not api_key:
        return ParticipantResult(
            name=name,
            ok=False,
            output="",
            error=f"Missing {key_env}",
            elapsed_seconds=0,
            model=model,
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a read-only coding council participant.",
            },
            {"role": "user", "content": prompt},
        ],
        "usage": {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Intellimetrics/llm-council",
        "X-Title": "llm-council",
    }
    timeout = float(cfg.get("timeout") or 180)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await _request_with_retries(
                client,
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                retries=int(cfg.get("retries") or 2),
                headers=headers,
                json=payload,
            )
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return ParticipantResult(
            name=name,
            ok=True,
            output=content.strip(),
            error="",
            elapsed_seconds=time.monotonic() - start,
            model=data.get("model") or model,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            cost_usd=_float_or_none(usage.get("cost")),
        )
    except Exception as exc:
        return ParticipantResult(
            name=name,
            ok=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - start,
            model=model,
        )


async def run_ollama_participant(
    name: str, cfg: dict[str, Any], prompt: str
) -> ParticipantResult:
    start = time.monotonic()
    model = cfg.get("model")
    base_url = str(cfg.get("base_url") or "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=float(cfg.get("timeout") or 180)) as client:
            response = await _request_with_retries(
                client,
                "POST",
                f"{base_url}/api/chat",
                retries=int(cfg.get("retries") or 1),
                json=payload,
            )
            data = response.json()
        content = data.get("message", {}).get("content", "")
        return ParticipantResult(
            name=name,
            ok=True,
            output=content.strip(),
            error="",
            elapsed_seconds=time.monotonic() - start,
            model=model,
        )
    except Exception as exc:
        return ParticipantResult(
            name=name,
            ok=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - start,
            model=model,
        )


async def run_participant(
    name: str, cfg: dict[str, Any], prompt: str, cwd: Path
) -> ParticipantResult:
    ptype = cfg.get("type")
    if ptype == "cli":
        return await run_cli_participant(name, cfg, prompt, cwd)
    if ptype == "openrouter":
        return await run_openrouter_participant(name, cfg, prompt)
    if ptype == "ollama":
        return await run_ollama_participant(name, cfg, prompt)
    return ParticipantResult(
        name=name,
        ok=False,
        output="",
        error=f"Unsupported participant type: {ptype}",
        elapsed_seconds=0,
        model=cfg.get("model"),
    )


async def _request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retries: int,
    **kwargs: Any,
) -> httpx.Response:
    delay = 0.75
    last_exc: Exception | None = None
    retries = max(0, retries)
    for attempt in range(retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
            if response.status_code < 400:
                return response
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            if attempt == retries:
                response.raise_for_status()
            last_exc = httpx.HTTPStatusError(
                f"Retryable HTTP status {response.status_code}",
                request=response.request,
                response=response,
            )
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt == retries:
                raise
        await asyncio.sleep(delay)
        delay *= 2
    assert last_exc is not None
    raise last_exc


async def run_participants(
    selected: list[str],
    participant_cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    *,
    max_concurrency: int = 4,
) -> list[ParticipantResult]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_one(name: str) -> ParticipantResult:
        async with semaphore:
            return await run_participant(name, participant_cfg[name], prompt, cwd)

    return await asyncio.gather(*[run_one(name) for name in selected])


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def command_for_display(command: list[str] | None) -> str:
    if not command:
        return ""
    return " ".join(shlex.quote(part) for part in command)


def redact_prompt_args(command: list[str], prompt: str) -> list[str]:
    if not prompt:
        return command
    return ["[prompt]" if part == prompt else part.replace(prompt, "[prompt]") for part in command]


def clean_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    for key in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        env.pop(key, None)
    return env


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)
