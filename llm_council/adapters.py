"""Participant adapters."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from llm_council.context import IMAGE_MIME_ALLOWLIST


RECOMMENDATION_RE = re.compile(
    r"""
    ^\s*
    (?:>\s*)?
    (?:[-*]\s+)?
    (?:\#{1,6}\s*)?
    (?:\*\*)?
    recommendation
    (?:\*\*)?
    \s*[:\-]\s*
    (?:\*\*)?
    \s*
    (yes|no|tradeoff)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

REPAIR_RETRY_INSTRUCTION = (
    "Your previous response was missing the required label. "
    "Please re-emit your response beginning with a single line of the form "
    "`RECOMMENDATION: yes|no|tradeoff - <one-line rationale>` followed by your "
    "reasoning. Do not change your reasoning, only add the missing label."
)

CLI_LAUNCH_RETRY_STDERR_LIMIT = 4096


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
    recovered_after_launch_retry: bool = False


def _format_arg(value: str, *, prompt: str, cwd: Path) -> str:
    return value.replace("{prompt}", prompt).replace("{cwd}", str(cwd))


def _build_cli_command(name: str, cfg: dict[str, Any], prompt: str, cwd: Path) -> list[str]:
    command = [cfg.get("command", name)]
    args = [_format_arg(str(arg), prompt=prompt, cwd=cwd) for arg in cfg.get("args", [])]

    model = cfg.get("model")
    family = cfg.get("family", name)
    if model:
        if family == "codex":
            # Codex's exec subcommand takes the model via `-m`; the default
            # args list starts with `exec` so we drop the duplicate when we
            # synthesize `exec -m <model>`. If a custom config drops `exec`,
            # we still emit the canonical pair, no double-`exec`.
            command.extend(["exec", "-m", str(model)])
            if args and args[0] == "exec":
                args = args[1:]
        else:
            # claude, gemini, and any other family use the standard
            # `--model <id>` flag.
            command.extend(["--model", str(model)])

    return command + args


async def run_cli_participant(
    name: str, cfg: dict[str, Any], prompt: str, cwd: Path
) -> ParticipantResult:
    start = time.monotonic()
    result, meta = await _run_cli_once(name, cfg, prompt, cwd, start=start)
    if _should_launch_retry(meta, cfg):
        await asyncio.sleep(_launch_retry_backoff(0))
        retry_result, retry_meta = await _run_cli_once(
            name, cfg, prompt, cwd, start=start
        )
        if retry_meta.get("exited") and not retry_meta.get("nonzero_exit"):
            retry_result.recovered_after_launch_retry = True
        result, meta = retry_result, retry_meta
    if (
        not result.ok
        and _retry_enabled(cfg)
        and result.error.startswith("InvalidParticipantResponse: missing required")
        and _is_label_only_failure(result.output, cfg)
    ):
        retry_prompt = _build_cli_retry_prompt(prompt, result.output)
        max_prompt_chars = cfg.get("max_prompt_chars")
        if max_prompt_chars is not None and len(retry_prompt) > int(max_prompt_chars):
            return result
        retry_result, _retry_meta = await _run_cli_once(
            name, cfg, retry_prompt, cwd, start=start
        )
        merged = _merge_cli_retry(result, retry_result)
        if result.recovered_after_launch_retry:
            merged.recovered_after_launch_retry = True
        return merged
    return result


async def _run_cli_once(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    *,
    start: float,
) -> tuple[ParticipantResult, dict[str, Any]]:
    timeout = int(cfg.get("timeout") or 240)
    command = _build_cli_command(name, cfg, prompt, cwd)
    max_prompt_chars = cfg.get("max_prompt_chars")
    if max_prompt_chars is not None and len(prompt) > int(max_prompt_chars):
        return (
            ParticipantResult(
                name=name,
                ok=False,
                output="",
                error=(
                    "PromptTooLarge: participant skipped before launch; "
                    f"prompt has {len(prompt)} chars, limit is {int(max_prompt_chars)}"
                ),
                elapsed_seconds=time.monotonic() - start,
                command=redact_prompt_args(command, prompt),
                model=cfg.get("model"),
            ),
            {"nonzero_exit": False, "stderr": "", "exited": False},
        )
    stdin_prompt = bool(cfg.get("stdin_prompt"))
    stdin_data = prompt if stdin_prompt else None
    env = clean_subprocess_env(cfg.get("env_passthrough"))

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate = asyncio.create_task(
            proc.communicate(stdin_data.encode() if stdin_data is not None else None)
        )
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communicate), timeout)
        except TimeoutError:
            await _cleanup_timed_out_process(proc, communicate)
            raise
        elapsed = time.monotonic() - start
        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        ok = proc.returncode == 0
        validation_error = _response_validation_error(out, cfg) if ok else ""
        return (
            ParticipantResult(
                name=name,
                ok=ok and not validation_error,
                output=out,
                error=validation_error or (err if not ok else ""),
                elapsed_seconds=elapsed,
                command=redact_prompt_args(command, prompt),
                model=cfg.get("model"),
            ),
            {"nonzero_exit": not ok, "stderr": err, "exited": True},
        )
    except TimeoutError:
        elapsed = time.monotonic() - start
        return (
            ParticipantResult(
                name=name,
                ok=False,
                output="",
                error=_format_timeout_error(name, timeout, len(prompt)),
                elapsed_seconds=elapsed,
                command=redact_prompt_args(command, prompt),
                model=cfg.get("model"),
            ),
            {"nonzero_exit": False, "stderr": "", "exited": False},
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return (
            ParticipantResult(
                name=name,
                ok=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=elapsed,
                command=redact_prompt_args(command, prompt),
                model=cfg.get("model"),
            ),
            {"nonzero_exit": False, "stderr": "", "exited": False},
        )


def _should_launch_retry(meta: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if not meta.get("nonzero_exit"):
        return False
    stderr = meta.get("stderr") or ""
    if len(stderr) > CLI_LAUNCH_RETRY_STDERR_LIMIT:
        return False
    patterns = cfg.get("cli_retry_stderr_patterns") or []
    if not patterns:
        return False
    for pattern in patterns:
        try:
            if re.search(pattern, stderr):
                return True
        except re.error:
            continue
    return False


def _launch_retry_backoff(attempt: int) -> float:
    return float(min(2 * (1 + attempt), 8))


def _build_cli_retry_prompt(original_prompt: str, prior_response: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "--- Your previous response (first attempt) ---\n"
        f"{prior_response.strip()}\n\n"
        f"{REPAIR_RETRY_INSTRUCTION}"
    )


def _merge_cli_retry(
    original: ParticipantResult, retry: ParticipantResult
) -> ParticipantResult:
    if retry.ok:
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=True,
        )
        return ParticipantResult(
            name=retry.name,
            ok=True,
            output=merged_output,
            error="",
            elapsed_seconds=retry.elapsed_seconds,
            command=retry.command,
            model=retry.model,
        )
    if retry.error.startswith("InvalidParticipantResponse: missing required") and retry.output:
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=False,
        )
        return ParticipantResult(
            name=retry.name,
            ok=False,
            output=merged_output,
            error=(
                "InvalidParticipantResponse: missing required RECOMMENDATION label "
                "after one repair retry"
            ),
            elapsed_seconds=retry.elapsed_seconds,
            command=retry.command,
            model=retry.model,
        )
    return original


async def _cleanup_timed_out_process(
    proc: asyncio.subprocess.Process,
    communicate: asyncio.Task[tuple[bytes, bytes]],
    *,
    terminate_grace_seconds: float = 2.0,
) -> None:
    if proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=terminate_grace_seconds)
        except TimeoutError:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.wait()

    try:
        await communicate
    except (BrokenPipeError, ConnectionResetError):
        pass


async def run_openrouter_participant(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
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

    user_content = await _build_user_content_async(prompt, image_manifest, cfg)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a read-only coding council participant.",
            },
            {"role": "user", "content": user_content},
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
                retries=_coerce_retries(cfg.get("retries"), default=2),
                headers=headers,
                json=payload,
            )
            data = response.json()
        usage = data.get("usage") or {}
        if data.get("error"):
            return ParticipantResult(
                name=name,
                ok=False,
                output="",
                error=f"OpenRouterError: {data['error']}",
                elapsed_seconds=time.monotonic() - start,
                model=data.get("model") or model,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                cost_usd=_float_or_none(usage.get("cost")),
            )
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        content = _message_content_text(message.get("content"))
        if not content:
            content = _message_content_text(message.get("reasoning"))
        if not content or not content.strip():
            detail = choice.get("finish_reason") or "missing message content"
            return ParticipantResult(
                name=name,
                ok=False,
                output="",
                error=f"OpenRouterEmptyResponse: {detail}",
                elapsed_seconds=time.monotonic() - start,
                model=data.get("model") or model,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                cost_usd=_float_or_none(usage.get("cost")),
            )
        finish_reason = choice.get("finish_reason")
        validation_error = _response_validation_error(content, cfg)
        if validation_error:
            should_retry = (
                _retry_enabled(cfg)
                and finish_reason != "length"
                and _is_label_only_failure(content, cfg)
            )
            if should_retry:
                retry_messages = list(payload["messages"]) + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": REPAIR_RETRY_INSTRUCTION},
                ]
                retry_serialized = _serialize_openrouter_messages(retry_messages)
                max_prompt_chars = cfg.get("max_prompt_chars")
                if (
                    max_prompt_chars is None
                    or len(retry_serialized) <= int(max_prompt_chars)
                ):
                    retry_payload = dict(payload)
                    retry_payload["messages"] = retry_messages
                    try:
                        async with httpx.AsyncClient(timeout=timeout) as retry_client:
                            retry_response = await _request_with_retries(
                                retry_client,
                                "POST",
                                "https://openrouter.ai/api/v1/chat/completions",
                                retries=_coerce_retries(cfg.get("retries"), default=2),
                                headers=headers,
                                json=retry_payload,
                            )
                            retry_data = retry_response.json()
                    except Exception:
                        retry_data = None
                    if retry_data is not None:
                        return _resolve_openrouter_retry(
                            name=name,
                            original_content=content,
                            original_usage=usage,
                            retry_data=retry_data,
                            cfg=cfg,
                            start=start,
                            fallback_model=model,
                        )
            return ParticipantResult(
                name=name,
                ok=False,
                output=content.strip(),
                error=validation_error,
                elapsed_seconds=time.monotonic() - start,
                model=data.get("model") or model,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                cost_usd=_float_or_none(usage.get("cost")),
            )
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


def _serialize_openrouter_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return "\n".join(parts)


def _resolve_openrouter_retry(
    *,
    name: str,
    original_content: str,
    original_usage: dict[str, Any],
    retry_data: dict[str, Any],
    cfg: dict[str, Any],
    start: float,
    fallback_model: Any,
) -> ParticipantResult:
    retry_usage = retry_data.get("usage") or {}
    combined_usage = _combine_openrouter_usage(original_usage, retry_usage)
    retry_choices = retry_data.get("choices") or []
    retry_choice = retry_choices[0] if retry_choices else {}
    retry_message = retry_choice.get("message") or {}
    retry_content = _message_content_text(retry_message.get("content"))
    if not retry_content:
        retry_content = _message_content_text(retry_message.get("reasoning"))
    retry_finish = retry_choice.get("finish_reason")
    model_id = retry_data.get("model") or fallback_model
    if retry_data.get("error") or not retry_content or not retry_content.strip():
        return ParticipantResult(
            name=name,
            ok=False,
            output=original_content.strip(),
            error=(
                "InvalidParticipantResponse: missing required RECOMMENDATION label "
                "after one repair retry"
            ),
            elapsed_seconds=time.monotonic() - start,
            model=model_id,
            prompt_tokens=_int_or_none(combined_usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(combined_usage.get("completion_tokens")),
            total_tokens=_int_or_none(combined_usage.get("total_tokens")),
            cost_usd=_float_or_none(combined_usage.get("cost")),
        )
    if retry_finish == "length":
        merged_output = _format_retry_transcript(
            original_output=original_content,
            retry_output=retry_content,
            recovered=False,
        )
        return ParticipantResult(
            name=name,
            ok=False,
            output=merged_output,
            error=(
                "InvalidParticipantResponse: retry response was truncated "
                "(finish_reason=length); cannot trust label"
            ),
            elapsed_seconds=time.monotonic() - start,
            model=model_id,
            prompt_tokens=_int_or_none(combined_usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(combined_usage.get("completion_tokens")),
            total_tokens=_int_or_none(combined_usage.get("total_tokens")),
            cost_usd=_float_or_none(combined_usage.get("cost")),
        )
    retry_validation = _response_validation_error(retry_content, cfg)
    if retry_validation:
        merged_output = _format_retry_transcript(
            original_output=original_content,
            retry_output=retry_content,
            recovered=False,
        )
        return ParticipantResult(
            name=name,
            ok=False,
            output=merged_output,
            error=(
                "InvalidParticipantResponse: missing required RECOMMENDATION label "
                "after one repair retry"
            ),
            elapsed_seconds=time.monotonic() - start,
            model=model_id,
            prompt_tokens=_int_or_none(combined_usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(combined_usage.get("completion_tokens")),
            total_tokens=_int_or_none(combined_usage.get("total_tokens")),
            cost_usd=_float_or_none(combined_usage.get("cost")),
        )
    merged_output = _format_retry_transcript(
        original_output=original_content,
        retry_output=retry_content,
        recovered=True,
    )
    return ParticipantResult(
        name=name,
        ok=True,
        output=merged_output,
        error="",
        elapsed_seconds=time.monotonic() - start,
        model=model_id,
        prompt_tokens=_int_or_none(combined_usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(combined_usage.get("completion_tokens")),
        total_tokens=_int_or_none(combined_usage.get("total_tokens")),
        cost_usd=_float_or_none(combined_usage.get("cost")),
    )


def _combine_openrouter_usage(
    a: dict[str, Any], b: dict[str, Any]
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        first = _int_or_none(a.get(key))
        second = _int_or_none(b.get(key))
        if first is None and second is None:
            continue
        combined[key] = (first or 0) + (second or 0)
    cost_a = _float_or_none(a.get("cost"))
    cost_b = _float_or_none(b.get("cost"))
    if cost_a is not None or cost_b is not None:
        combined["cost"] = (cost_a or 0.0) + (cost_b or 0.0)
    return combined


async def run_ollama_participant(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
) -> ParticipantResult:
    start = time.monotonic()
    model = cfg.get("model")
    base_url = str(cfg.get("base_url") or "http://localhost:11434").rstrip("/")
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    if cfg.get("vision") and image_manifest:
        user_message["images"] = [
            await asyncio.to_thread(_read_image_base64, entry)
            for entry in image_manifest
        ]
    payload = {
        "model": model,
        "messages": [user_message],
        "stream": False,
    }
    try:
        ollama_timeout = float(cfg.get("timeout") or 180)
        async with httpx.AsyncClient(timeout=ollama_timeout) as client:
            response = await _request_with_retries(
                client,
                "POST",
                f"{base_url}/api/chat",
                retries=_coerce_retries(cfg.get("retries"), default=1),
                json=payload,
            )
            data = response.json()
        content = data.get("message", {}).get("content", "")
        finish_reason = data.get("done_reason")
        validation_error = _response_validation_error(content, cfg)
        if validation_error:
            should_retry = (
                _retry_enabled(cfg)
                and finish_reason != "length"
                and _is_label_only_failure(content, cfg)
            )
            if should_retry:
                retry_messages = list(payload["messages"]) + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": REPAIR_RETRY_INSTRUCTION},
                ]
                retry_serialized = "\n".join(
                    str(m.get("content") or "") for m in retry_messages
                )
                max_prompt_chars = cfg.get("max_prompt_chars")
                if (
                    max_prompt_chars is None
                    or len(retry_serialized) <= int(max_prompt_chars)
                ):
                    retry_payload = dict(payload)
                    retry_payload["messages"] = retry_messages
                    try:
                        async with httpx.AsyncClient(timeout=ollama_timeout) as retry_client:
                            retry_response = await _request_with_retries(
                                retry_client,
                                "POST",
                                f"{base_url}/api/chat",
                                retries=_coerce_retries(cfg.get("retries"), default=1),
                                json=retry_payload,
                            )
                            retry_data = retry_response.json()
                    except Exception:
                        retry_data = None
                    if retry_data is not None:
                        retry_content = (
                            retry_data.get("message", {}).get("content", "") or ""
                        )
                        retry_done_reason = retry_data.get("done_reason")
                        if retry_content.strip():
                            if retry_done_reason == "length":
                                merged = _format_retry_transcript(
                                    original_output=content,
                                    retry_output=retry_content,
                                    recovered=False,
                                )
                                return ParticipantResult(
                                    name=name,
                                    ok=False,
                                    output=merged,
                                    error=(
                                        "InvalidParticipantResponse: retry response "
                                        "was truncated (done_reason=length); cannot "
                                        "trust label"
                                    ),
                                    elapsed_seconds=time.monotonic() - start,
                                    model=model,
                                )
                            retry_validation = _response_validation_error(
                                retry_content, cfg
                            )
                            if retry_validation:
                                merged = _format_retry_transcript(
                                    original_output=content,
                                    retry_output=retry_content,
                                    recovered=False,
                                )
                                return ParticipantResult(
                                    name=name,
                                    ok=False,
                                    output=merged,
                                    error=(
                                        "InvalidParticipantResponse: missing required "
                                        "RECOMMENDATION label after one repair retry"
                                    ),
                                    elapsed_seconds=time.monotonic() - start,
                                    model=model,
                                )
                            merged = _format_retry_transcript(
                                original_output=content,
                                retry_output=retry_content,
                                recovered=True,
                            )
                            return ParticipantResult(
                                name=name,
                                ok=True,
                                output=merged,
                                error="",
                                elapsed_seconds=time.monotonic() - start,
                                model=model,
                            )
            return ParticipantResult(
                name=name,
                ok=False,
                output=content.strip(),
                error=validation_error,
                elapsed_seconds=time.monotonic() - start,
                model=model,
            )
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
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
) -> ParticipantResult:
    ptype = cfg.get("type")
    if ptype == "cli":
        # CLI participants intentionally don't receive image_manifest at the
        # adapter layer: they share the project filesystem with the host and
        # open staged images themselves via the file paths listed in the
        # ## Images prompt section. Adding `vision: true` to a CLI cfg
        # therefore has no effect — the orchestrator's images_skipped check
        # treats CLI as always image-aware (orchestrator.py).
        return await run_cli_participant(name, cfg, prompt, cwd)
    if ptype == "openrouter":
        return await run_openrouter_participant(
            name, cfg, prompt, image_manifest=image_manifest
        )
    if ptype == "ollama":
        return await run_ollama_participant(
            name, cfg, prompt, image_manifest=image_manifest
        )
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
    progress: Callable[[dict[str, Any]], None] | None = None,
    round_number: int = 1,
    image_manifest: list[dict[str, Any]] | None = None,
) -> list[ParticipantResult]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_one(name: str) -> ParticipantResult:
        async with semaphore:
            cfg = participant_cfg[name]
            timeout = int(cfg.get("timeout") or 240)
            override = cfg.get("slow_warn_after_seconds")
            if override is not None:
                slow_after = float(override)
            else:
                slow_after = max(30.0, timeout * 0.75)

            slow_task = None
            try:
                if progress and slow_after < timeout:
                    async def _emit_slow() -> None:
                        try:
                            await asyncio.sleep(slow_after)
                        except asyncio.CancelledError:
                            return
                        progress(
                            {
                                "event": "participant_slow",
                                "participant": name,
                                "round": round_number,
                                "elapsed_seconds": slow_after,
                                "timeout_seconds": timeout,
                            }
                        )

                    slow_task = asyncio.create_task(_emit_slow())
                if progress:
                    progress({"event": "participant_start", "participant": name, "round": round_number})
                result = await run_participant(
                    name,
                    cfg,
                    prompt,
                    cwd,
                    image_manifest=image_manifest,
                )
            finally:
                if slow_task is not None and not slow_task.done():
                    slow_task.cancel()
                    try:
                        await slow_task
                    except asyncio.CancelledError:
                        pass
            status = "ok" if result.ok else "error"
            if result.error.startswith("PromptTooLarge:"):
                status = "skipped"
            elif is_timeout_error(result.error):
                status = "timeout"
            if progress:
                progress(
                    {
                        "event": "participant_finish",
                        "participant": name,
                        "round": round_number,
                        "status": status,
                        "ok": result.ok,
                        "elapsed_seconds": round(result.elapsed_seconds, 3),
                        "error": result.error,
                        "model": result.model,
                        "total_tokens": result.total_tokens,
                        "cost_usd": result.cost_usd,
                    }
                )
            return result

    return await asyncio.gather(*[run_one(name) for name in selected])


def _coerce_retries(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _response_validation_error(output: str, cfg: dict[str, Any]) -> str:
    if cfg.get("require_recommendation") is False:
        return ""
    if _has_recommendation_label(output):
        return ""
    excerpt = _first_output_excerpt(output)
    if excerpt:
        return (
            "InvalidParticipantResponse: missing required RECOMMENDATION label. "
            f"First output: {excerpt}"
        )
    return "InvalidParticipantResponse: empty response"


def _retry_enabled(cfg: dict[str, Any]) -> bool:
    return cfg.get("retry_on_missing_label", True) is not False


def _is_label_only_failure(output: str, cfg: dict[str, Any]) -> bool:
    if not output or not output.strip():
        return False
    error = _response_validation_error(output, cfg)
    return error.startswith("InvalidParticipantResponse: missing required")


def _format_retry_transcript(
    *, original_output: str, retry_output: str, recovered: bool
) -> str:
    header = (
        "[recovered after retry] "
        "First attempt was missing the required RECOMMENDATION label; "
        "second attempt is shown below."
        if recovered
        else "[retry exhausted] "
        "Both attempts were missing the required RECOMMENDATION label."
    )
    return (
        f"{header}\n\n"
        "--- Repaired response ---\n"
        f"{retry_output.strip()}\n\n"
        "--- Original response (first attempt) ---\n"
        f"{original_output.strip()}"
    )


def _has_recommendation_label(output: str) -> bool:
    in_fence = False
    fenced_match = False
    for line in output.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not RECOMMENDATION_RE.match(line):
            continue
        if not in_fence:
            return True
        fenced_match = True
    return fenced_match


def _first_output_excerpt(output: str, max_chars: int = 240) -> str:
    cleaned = " ".join(output.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _format_timeout_error(name: str, timeout: int, prompt_chars: int) -> str:
    return (
        f"Timeout: `{name}` did not respond within {timeout}s "
        f"(prompt was {prompt_chars} chars). "
        "To raise the limit, set `participants."
        f"{name}.timeout: <seconds>` in `.llm-council.yaml`. "
        "To skip this participant for one run, pass an explicit participant "
        "list that omits it. To shorten the prompt, drop large `--context` "
        "files or use `--diff` more selectively."
    )


def is_timeout_error(error: str) -> bool:
    return error.startswith("Timeout:") or error.startswith("TimeoutError:")


def _read_image_base64(entry: dict[str, Any]) -> str:
    mime = entry.get("mime")
    if mime not in IMAGE_MIME_ALLOWLIST:
        raise ValueError(f"Image mime '{mime}' is not allowed for council attachments")
    path = Path(entry.get("path") or entry.get("relative_path") or "")
    if not path.exists():
        raise ValueError(f"Image path does not exist: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


async def _build_user_content_async(
    prompt: str,
    image_manifest: list[dict[str, Any]] | None,
    cfg: dict[str, Any],
) -> Any:
    if not (image_manifest and cfg.get("vision")):
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for entry in image_manifest:
        b64 = await asyncio.to_thread(_read_image_base64, entry)
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{entry['mime']};base64,{b64}"},
            }
        )
    return parts


def command_for_display(command: list[str] | None) -> str:
    if not command:
        return ""
    return " ".join(shlex.quote(part) for part in command)


def redact_prompt_args(command: list[str], prompt: str) -> list[str]:
    if not prompt:
        return list(command)
    return [_redact_prompt_arg(part, prompt) for part in command]


def _redact_prompt_arg(part: str, prompt: str) -> str:
    if part == prompt:
        return "[prompt]"
    redacted = part.replace(prompt, "[prompt]")
    for fragment in _prompt_fragments(prompt):
        redacted = redacted.replace(fragment, "[prompt]")
    if _contains_prompt_substring(redacted, prompt):
        return _redact_arg_value(part)
    return redacted


def _prompt_fragments(prompt: str) -> list[str]:
    min_length = 64
    if len(prompt) < min_length:
        return []

    fragments = {
        prompt[: min(len(prompt), 256)],
        prompt[-min(len(prompt), 256) :],
    }
    for length in (64, 128):
        if len(prompt) >= length:
            fragments.add(prompt[:length])
            fragments.add(prompt[-length:])
    for line in prompt.splitlines():
        line = line.strip()
        if len(line) >= min_length:
            fragments.add(line)
            for length in (64, 128, 256):
                if len(line) >= length:
                    fragments.add(line[:length])
                    fragments.add(line[-length:])

    return sorted(fragments, key=len, reverse=True)


def _contains_prompt_substring(part: str, prompt: str) -> bool:
    min_length = 64
    if len(prompt) < min_length or len(part) < min_length:
        return False
    step = 32
    last_start = max(0, len(prompt) - min_length)
    starts = range(0, last_start + 1, step)
    return any(prompt[start : start + min_length] in part for start in starts) or (
        prompt[last_start:] in part
    )


def _redact_arg_value(part: str) -> str:
    prefix, separator, _value = part.partition("=")
    if separator and prefix:
        return f"{prefix}=[prompt]"
    return "[prompt]"


def clean_subprocess_env(env_passthrough: list[str] | None = None) -> dict[str, str]:
    allowed = {key.upper() for key in (env_passthrough or [])}
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key == "CLAUDECODE":
            continue
        if _is_secret_env_name(key) and key.upper() not in allowed:
            continue
        env[key] = value
    return env


def _is_secret_env_name(key: str) -> bool:
    upper = key.upper()
    if upper in _SAFE_ENV_NAMES or upper.startswith("LC_"):
        return False
    parts = [part for part in upper.replace("-", "_").split("_") if part]
    secret_parts = {
        "AUTH",
        "CREDENTIAL",
        "CREDENTIALS",
        "KEY",
        "OAUTH",
        "PASS",
        "PASSWD",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    }
    if any(part in secret_parts for part in parts):
        return True
    return any(word in upper for word in ("CREDENTIAL", "PASSWORD"))


_SAFE_ENV_NAMES = {
    "APPDATA",
    "COLORTERM",
    "HOME",
    "LANG",
    "LOCALAPPDATA",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "XDG_STATE_HOME",
}
