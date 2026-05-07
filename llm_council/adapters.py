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
from urllib.parse import urlparse

import httpx

from llm_council.cache import (
    build_payload as cache_build_payload,
    cache_path as cache_path_for,
    compute_key as cache_compute_key,
    is_caching_disabled_for_mode,
    read_cache as cache_read,
    write_cache as cache_write,
)
from llm_council.context import IMAGE_MIME_ALLOWLIST


OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Intellimetrics/llm-council",
    "X-Title": "llm-council",
}


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

CONTEXT_OVERFLOW_ERROR_PREFIX = "ContextOverflowExcluded:"


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
    repair_retry_recovered: bool = False
    from_cache: bool = False
    # Wall-clock seconds the cache lookup itself took. None for non-cached
    # runs. `elapsed_seconds` always reports the original run's timing
    # (preserved across cache hits) so callers can see "true cost"; this
    # field documents how fast the cache hit actually returned.
    cache_hit_seconds: float | None = None
    stance: str | None = None


@dataclass
class CacheContext:
    """Per-run cache settings threaded through the adapter dispatchers.

    `mode` is one of "on", "off", "refresh". `cache_disabled` is a
    pre-computed kill-switch the orchestrator flips for things like
    consensus mode or deliberation rounds beyond the first.
    """

    cwd: Path
    cache_mode: str = "on"
    ttl_seconds: int = 86400
    cache_disabled: bool = False

    def can_read(self) -> bool:
        return (
            not self.cache_disabled
            and self.cache_mode == "on"
        )

    def can_write(self) -> bool:
        return (
            not self.cache_disabled
            and self.cache_mode in ("on", "refresh")
        )


def _participant_recommendation_label(output: str) -> str | None:
    for line in (output or "").splitlines():
        match = RECOMMENDATION_RE.match(line)
        if match:
            return match.group(1).lower()
    return None


def _result_from_cache_payload(
    name: str, payload: dict[str, Any]
) -> ParticipantResult:
    return ParticipantResult(
        name=name,
        ok=True,
        output=str(payload.get("output") or ""),
        error="",
        elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
        command=list(payload.get("command")) if payload.get("command") else None,
        model=payload.get("model"),
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        total_tokens=payload.get("total_tokens"),
        cost_usd=payload.get("cost_usd"),
        from_cache=True,
    )


def _cache_lookup(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    cache_ctx: CacheContext | None,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
) -> tuple[str | None, ParticipantResult | None]:
    if cache_ctx is None or cache_ctx.cache_disabled:
        return None, None
    lookup_start = time.monotonic()
    key = cache_compute_key(name, cfg, prompt, image_manifest=image_manifest)
    if not cache_ctx.can_read():
        return key, None
    path = cache_path_for(cache_ctx.cwd, name, key)
    payload = cache_read(path, expected_key=key)
    if payload is None:
        return key, None
    result = _result_from_cache_payload(name, payload)
    result.cache_hit_seconds = round(time.monotonic() - lookup_start, 6)
    return key, result


def _maybe_persist_cache(
    name: str,
    prompt: str,
    key: str | None,
    result: ParticipantResult,
    cache_ctx: CacheContext | None,
) -> None:
    if cache_ctx is None or key is None:
        return
    if not cache_ctx.can_write():
        return
    if not result.ok:
        return
    if result.from_cache:
        return
    payload = cache_build_payload(
        participant_name=name,
        prompt=prompt,
        key=key,
        output=result.output,
        recommendation_label=_participant_recommendation_label(result.output),
        elapsed_seconds=result.elapsed_seconds,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cost_usd=result.cost_usd,
        model=result.model,
        command=result.command,
    )
    try:
        cache_write(
            cache_path_for(cache_ctx.cwd, name, key),
            payload,
            cache_ctx.ttl_seconds,
        )
    except OSError:
        pass


def _context_overflow_result(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
) -> ParticipantResult | None:
    raw_limit = cfg.get("max_context_tokens")
    if raw_limit is None:
        return None
    limit = int(raw_limit)
    from llm_council.estimate import IMAGE_TOKEN_HEURISTIC, estimate_tokens

    estimated = estimate_tokens(prompt)
    if image_manifest and cfg.get("vision"):
        estimated += len(image_manifest) * IMAGE_TOKEN_HEURISTIC
    if estimated <= limit:
        return None
    return ParticipantResult(
        name=name,
        ok=False,
        output="",
        error=(
            f"{CONTEXT_OVERFLOW_ERROR_PREFIX} estimated {estimated} prompt tokens "
            f"(approximate; chars/4) exceed max_context_tokens={limit}"
        ),
        elapsed_seconds=0.0,
        model=cfg.get("model"),
        prompt_tokens=estimated,
    )


def is_context_overflow_error(error: str) -> bool:
    return error.startswith(CONTEXT_OVERFLOW_ERROR_PREFIX)


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
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    *,
    cache_ctx: CacheContext | None = None,
) -> ParticipantResult:
    overflow = _context_overflow_result(name, cfg, prompt)
    if overflow is not None:
        return overflow
    cache_key, cached = _cache_lookup(name, cfg, prompt, cache_ctx)
    if cached is not None:
        return cached
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
            _maybe_persist_cache(name, prompt, cache_key, result, cache_ctx)
            return result
        retry_result, _retry_meta = await _run_cli_once(
            name, cfg, retry_prompt, cwd, start=start
        )
        merged = _merge_cli_retry(result, retry_result)
        if result.recovered_after_launch_retry:
            merged.recovered_after_launch_retry = True
        _maybe_persist_cache(name, prompt, cache_key, merged, cache_ctx)
        return merged
    _maybe_persist_cache(name, prompt, cache_key, result, cache_ctx)
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
    env = clean_subprocess_env(
        cfg.get("env_passthrough"),
        strict=bool(cfg.get("env_strict", False)),
    )

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
        # Silent CLI failures (nonzero exit but empty stderr) used to land
        # with `error=""`, which made `classify_error` return None — a
        # taxonomy hole. Always synthesize a stable error string when ok is
        # false so downstream callers can branch on `error_kind`.
        if not ok and not validation_error and not err:
            err = (
                f"CliExitNonZero: `{name}` exited with status "
                f"{proc.returncode} and no stderr output"
            )
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
            repair_retry_recovered=True,
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


async def run_openai_compatible_participant(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
    cache_ctx: CacheContext | None = None,
) -> ParticipantResult:
    overflow = _context_overflow_result(
        name, cfg, prompt, image_manifest=image_manifest
    )
    if overflow is not None:
        return overflow
    cache_key, cached = _cache_lookup(
        name, cfg, prompt, cache_ctx, image_manifest=image_manifest
    )
    if cached is not None:
        return cached
    result = await _run_openai_compatible_inner(
        name, cfg, prompt, image_manifest=image_manifest
    )
    _maybe_persist_cache(name, prompt, cache_key, result, cache_ctx)
    return result


async def _run_openai_compatible_inner(
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

    base_url = str(cfg.get("base_url") or OPENROUTER_DEFAULT_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    is_openrouter = _is_openrouter_endpoint(base_url)

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
    headers = _build_openai_compatible_headers(api_key, cfg, is_openrouter=is_openrouter)
    timeout = float(cfg.get("timeout") or 180)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await _request_with_retries(
                client,
                "POST",
                endpoint,
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
                        async with httpx.AsyncClient(
                            timeout=timeout, follow_redirects=False
                        ) as retry_client:
                            retry_response = await _request_with_retries(
                                retry_client,
                                "POST",
                                endpoint,
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


async def run_openrouter_participant(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
    cache_ctx: CacheContext | None = None,
) -> ParticipantResult:
    return await run_openai_compatible_participant(
        name, cfg, prompt, image_manifest=image_manifest, cache_ctx=cache_ctx
    )


_RESERVED_HEADER_LOWER = frozenset(
    {"authorization", "content-type", "http-referer", "x-title"}
)


def _build_openai_compatible_headers(
    api_key: str, cfg: dict[str, Any], *, is_openrouter: bool
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers.update(OPENROUTER_HEADERS)
    extra_headers = cfg.get("extra_headers") or {}
    if isinstance(extra_headers, dict):
        for key, value in extra_headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            if key.lower() in _RESERVED_HEADER_LOWER:
                continue
            headers[key] = value
    return headers


def _is_openrouter_endpoint(base_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


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
        repair_retry_recovered=True,
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
    cache_ctx: CacheContext | None = None,
) -> ParticipantResult:
    overflow = _context_overflow_result(
        name, cfg, prompt, image_manifest=image_manifest
    )
    if overflow is not None:
        return overflow
    cache_key, cached = _cache_lookup(
        name, cfg, prompt, cache_ctx, image_manifest=image_manifest
    )
    if cached is not None:
        return cached
    result = await _run_ollama_inner(name, cfg, prompt, image_manifest=image_manifest)
    _maybe_persist_cache(name, prompt, cache_key, result, cache_ctx)
    return result


async def _run_ollama_inner(
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
                                repair_retry_recovered=True,
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
    cache_ctx: CacheContext | None = None,
) -> ParticipantResult:
    ptype = cfg.get("type")
    if ptype == "cli":
        # CLI participants intentionally don't receive image_manifest at the
        # adapter layer: they share the project filesystem with the host and
        # open staged images themselves via the file paths listed in the
        # ## Images prompt section. Adding `vision: true` to a CLI cfg
        # therefore has no effect — the orchestrator's images_skipped check
        # treats CLI as always image-aware (orchestrator.py).
        return await run_cli_participant(name, cfg, prompt, cwd, cache_ctx=cache_ctx)
    if ptype in ("openrouter", "openai_compatible"):
        return await run_openai_compatible_participant(
            name, cfg, prompt, image_manifest=image_manifest, cache_ctx=cache_ctx
        )
    if ptype == "ollama":
        return await run_ollama_participant(
            name, cfg, prompt, image_manifest=image_manifest, cache_ctx=cache_ctx
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
    cache_ctx: CacheContext | None = None,
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
                    cache_ctx=cache_ctx,
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
            elif is_context_overflow_error(result.error):
                status = "excluded"
                if progress:
                    progress(
                        {
                            "event": "context_overflow_excluded",
                            "participant": name,
                            "round": round_number,
                            "estimated_tokens": result.prompt_tokens,
                            "max_context_tokens": int(cfg.get("max_context_tokens"))
                            if cfg.get("max_context_tokens") is not None
                            else None,
                        }
                    )
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
                        "from_cache": result.from_cache,
                        "cache_hit_seconds": result.cache_hit_seconds,
                        "recovered_after_launch_retry": result.recovered_after_launch_retry,
                        "repair_retry_recovered": result.repair_retry_recovered,
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
    if cfg.get("retry_on_missing_label", True) is False:
        return False
    # An explicit `retries: 0` is the user saying "no extra calls of any
    # kind"; respect that for the application-level repair retry too,
    # otherwise the cost regression undoes commit 45b44ee.
    if "retries" in cfg and _coerce_retries(cfg.get("retries"), default=1) == 0:
        return False
    return True


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


# Stable, machine-readable classification of result errors. Callers can
# branch on `error_kind` instead of pattern-matching the human-facing
# `error` string. Keep the enum closed so consumers can rely on a fixed
# set of values; add new kinds explicitly when a new failure path is
# introduced rather than letting strings drift.
ERROR_KIND_TIMEOUT = "timeout"
ERROR_KIND_CONTEXT_OVERFLOW = "context_overflow"
ERROR_KIND_PROMPT_TOO_LARGE = "prompt_too_large"
ERROR_KIND_INVALID_RESPONSE = "invalid_response"
ERROR_KIND_DOWNSTREAM = "downstream_error"
ERROR_KIND_CLI_NONZERO = "cli_nonzero_exit"
ERROR_KIND_PREFLIGHT_FAILED = "preflight_failed"
ERROR_KIND_UNKNOWN = "unknown"

KNOWN_ERROR_KINDS = frozenset(
    {
        ERROR_KIND_TIMEOUT,
        ERROR_KIND_CONTEXT_OVERFLOW,
        ERROR_KIND_PROMPT_TOO_LARGE,
        ERROR_KIND_INVALID_RESPONSE,
        ERROR_KIND_DOWNSTREAM,
        ERROR_KIND_CLI_NONZERO,
        ERROR_KIND_PREFLIGHT_FAILED,
        ERROR_KIND_UNKNOWN,
    }
)

PREFLIGHT_FAILED_PREFIX = "PreflightFailed:"


def classify_error(error: str) -> str | None:
    """Return a stable error_kind for non-empty error strings, else None.

    Empty errors (success) map to None so result_to_dict can omit the
    field. Any non-empty error that does not match a known prefix falls
    through to ``unknown`` rather than silently returning None — that lets
    consumers detect "we need to add a new kind" instead of mistaking an
    unclassified failure for success.
    """
    if not error:
        return None
    if error.startswith("Timeout:") or error.startswith("TimeoutError:"):
        return ERROR_KIND_TIMEOUT
    if error.startswith(CONTEXT_OVERFLOW_ERROR_PREFIX):
        return ERROR_KIND_CONTEXT_OVERFLOW
    if error.startswith("PromptTooLarge:"):
        return ERROR_KIND_PROMPT_TOO_LARGE
    if error.startswith("InvalidParticipantResponse:"):
        return ERROR_KIND_INVALID_RESPONSE
    if error.startswith("CliExitNonZero:"):
        return ERROR_KIND_CLI_NONZERO
    if error.startswith(PREFLIGHT_FAILED_PREFIX):
        return ERROR_KIND_PREFLIGHT_FAILED
    # httpx + downstream-API errors funnel through f"{type(exc).__name__}: ..."
    # in the openrouter / ollama paths; we don't try to introspect those
    # further here, just classify them as `downstream_error` so callers can
    # distinguish "their service blew up" from "our validation rejected it".
    downstream_markers = (
        "HTTPStatusError",
        "ConnectError",
        "ReadTimeout",
        "RemoteProtocolError",
        "ReadError",
        "WriteError",
        "ProxyError",
    )
    if any(marker in error for marker in downstream_markers):
        return ERROR_KIND_DOWNSTREAM
    return ERROR_KIND_UNKNOWN


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


def clean_subprocess_env(
    env_passthrough: list[str] | None = None,
    *,
    strict: bool = False,
) -> dict[str, str]:
    """Build the subprocess environment for a CLI participant.

    Two modes:

    - **Sieve (default, `strict=False`):** Inherit the parent environment
      with secrets-by-name (KEY, AUTH, SECRET, TOKEN, …) stripped unless
      they are explicitly listed in `env_passthrough`. This is the
      historical behavior — preserves PATH/LANG/TERM and other harmless
      configuration without requiring per-CLI allowlisting.

    - **Strict (`strict=True`):** Allowlist-only. The child gets nothing
      but the names in :data:`_SAFE_ENV_NAMES` (PATH/HOME/LANG/etc.) plus
      whatever is in `env_passthrough`. Use this for CLI participants
      that auto-detect provider configuration from env vars and could
      mis-route given an unrelated `GEMINI_MODEL` or `OPENAI_BASE_URL`
      leaking from the parent shell — the qwen-code (gemini-cli fork)
      class of bug.

    `LC_*` locale vars and `TERM` always pass through regardless of mode
    so the child renders correctly. `CLAUDECODE` is always stripped.
    """
    allowed = {key.upper() for key in (env_passthrough or [])}
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key == "CLAUDECODE":
            continue
        upper = key.upper()
        if strict:
            # Allowlist mode: only essentials + explicit pass-through.
            if upper in _SAFE_ENV_NAMES or upper.startswith("LC_"):
                env[key] = value
            elif upper in allowed:
                env[key] = value
            # everything else: dropped
            continue
        # Sieve mode (legacy): inherit non-secrets, allowlist secrets.
        if _is_secret_env_name(key) and upper not in allowed:
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
