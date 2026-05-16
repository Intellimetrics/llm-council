"""Participant adapters."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
import time
from dataclasses import dataclass, field
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

# When a peer times out, the adapter retries once with this directive and a
# tight TERSE_RETRY_TIMEOUT_SECONDS budget. The mode multiplier does NOT
# apply to the retry — the 60s ceiling is fixed by design: a peer that
# can't produce a terse answer in a minute is in a state no further
# multiplier-scaling will rescue, and the retry has to stay bounded so the
# wall-clock cost ceiling of "one extra round" holds.
SECTION_REPAIR_RETRY_INSTRUCTION = (
    "Your previous response satisfied the RECOMMENDATION label but skipped "
    "one or more REQUIRED sections from the prompt: {missing}. Re-emit your "
    "full response with those sections present. Keep your existing reasoning; "
    "add the missing sections as additional content. Each required section "
    "must use a heading that matches the prompt's `PART N` title (or includes "
    "the salient title tokens within a few lines)."
)

# Strict-evidence repair-retry. Fires when defaults.strict_evidence is true
# AND one or more EVIDENCE bullets lack a canonical epistemic tag. The
# instruction names the four legal tags explicitly so the peer can fix the
# response without re-reading the original prompt. Mirrors the
# section-repair pattern: one shot, no chaining with terse-retry, label-
# retry, or section-repair (each gate's retry is independent so the
# cumulative wall-clock cost ceiling of "one extra round" holds).
STRICT_EVIDENCE_REPAIR_RETRY_INSTRUCTION = (
    "Your previous response satisfied the RECOMMENDATION label but one or "
    "more EVIDENCE bullets lacked an epistemic tag. Re-emit your full "
    "response and tag EVERY EVIDENCE bullet with exactly one of "
    "`[PUBLISHED]` (cited in published literature), `[OBSERVABLE]` "
    "(directly observable behavior), `[INFERRED]` (reasoned from priors), "
    "or `[SPECULATIVE]` (informed guess). Tags may appear at the start or "
    "end of each bullet. Keep your existing reasoning; only add the tags."
)

TERSE_RETRY_INSTRUCTION = (
    "Your previous response timed out. Re-answer the question with the same "
    "RECOMMENDATION discipline, but be terse: cover every REQUIRED section "
    "concisely. Single sentence per field where applicable. Skip elaboration. "
    "Keep the RECOMMENDATION: yes|no|tradeoff label and, at minimum, the "
    "BLOCKERS or ASSUMPTIONS list if you would otherwise abdicate."
)
TERSE_RETRY_TIMEOUT_SECONDS = 60

# Suffix appended to the original timeout error when the terse-retry fired
# but also failed. Pass-8 dogfood surfaced the silent-failure mode: a
# user reading the transcript could not tell whether the retry attempt
# happened (terse-retry budget burned silently, ~60s extra wall-clock not
# visible in `elapsed_seconds`) or never fired at all (config gate /
# code bug). The annotation makes the attempt visible and tells the
# operator the next mitigation lever — raising `participants.<name>.timeout`
# or the mode multiplier — since the 60s terse window also failed.
TERSE_RETRY_FAILED_SUFFIX = (
    " Terse-retry-on-timeout was attempted with a {budget}s budget and also "
    "failed ({retry_kind}); the recorded `elapsed_seconds` reflects only the "
    "original call, but wall-clock cost includes the failed retry. Raise "
    "`participants.{name}.timeout` or the mode's `timeout_multiplier` if "
    "this prompt-size band keeps tripping the timeout wall."
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
    # Response envelope fields (Pick A v1 — all optional). Parsed from the
    # peer's text via _extract_response_envelope. Free-form for now; the
    # only enforced contract is `RECOMMENDATION:` itself. `evidence`'s
    # element type is `Any` (not `str`) so it can hold either plain
    # strings (legacy / untagged) or `{text, tag}` dicts from the v0.7
    # evidence-tag parser. The other list fields stay `list[str]` because
    # tag-parsing only applies to evidence claims.
    effort: str | None = None
    confidence: str | None = None
    risk: str | None = None
    blockers: list[str] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    tests_to_run: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    # Set True when the original call timed out and the terse-retry
    # recovered with a valid response. Separate from
    # `recovered_after_launch_retry` (launch-fail retry) and
    # `repair_retry_recovered` (label-only retry) so stats and transcripts
    # can attribute recoveries to the right mechanism.
    recovered_after_timeout: bool = False
    # Set True whenever the terse-retry-on-timeout path was attempted,
    # regardless of outcome. Distinct from `recovered_after_timeout` which
    # only flips when the retry SUCCEEDED. Pass-8 dogfood surfaced a
    # silent-failure mode: the original timeout result was returned with
    # `recovered_after_timeout=False`, indistinguishable from "retry never
    # fired" — the only signal was the wall-clock cost (~60s extra past
    # the original elapsed_seconds), which the transcript does not record.
    # This field makes the retry attempt visible so operators can tell the
    # two failure shapes apart and decide whether the terse window itself
    # needs raising for the affected prompt-size bucket.
    terse_retry_attempted: bool = False
    # Length of the prompt this call was made against. Populated on every
    # result produced by an adapter call attempt (success or failure, all
    # paths) so `stats.aggregate` can bucket telemetry by prompt size
    # without re-parsing the error string. In particular this lets
    # `timeout_recoveries` (success after terse-retry) be cross-tab-able
    # with prompt size — without it the recovery counter cannot tell
    # whether bigger prompts disproportionately trip the timeout wall.
    # Stays `None` only on results that never represented a real call
    # attempt (e.g. cache hits, the unsupported-type fallback).
    prompt_chars: int | None = None
    # Set True whenever the section-coverage repair retry path FIRED
    # (regardless of outcome), as opposed to `repair_retry_recovered`
    # which only flips on success of any repair retry. Pass-9 dogfood
    # surfaced a coupled failure mode: a section-repair retry can fix
    # the missing sections but surface a NEW UntaggedEvidence error
    # (label → sections → evidence validation order, see
    # `_response_validation_error`). Without this flag the
    # strict-evidence wrapper would chain a SECOND repair retry on top
    # of the section-repair merge result, breaking the "one extra call
    # per peer per round" invariant. The merge functions
    # (`_merge_cli_section_retry`, `_merge_hosted_section_retry`) set
    # this flag so the strict-evidence wrapper can refuse to fire.
    # Distinct from `repair_retry_recovered`: BOTH can be True in the
    # case where sections recovered but the now-visible evidence is
    # untagged (handled by the new third merge branch).
    section_repair_attempted: bool = False


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
    # `recovered_after_timeout` and `prompt_chars` default to
    # False/None when absent so older v3 payloads written before
    # these receipts landed (and any hand-rolled fixtures) rehydrate
    # cleanly. The CACHE_SCHEMA_VERSION bump to 3 already invalidates
    # pre-v0.7.0 payloads; the defaults are belt-and-braces.
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
        recovered_after_timeout=bool(payload.get("recovered_after_timeout", False)),
        prompt_chars=payload.get("prompt_chars"),
        section_repair_attempted=bool(
            payload.get("section_repair_attempted", False)
        ),
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
    # Abdication outputs are cached normally. The correctness mechanism
    # is read-side: `run_participant` always pipes cache hits back through
    # `_with_envelope`, which re-derives the envelope from `output` and
    # flips ok=False for abdication shapes — offline, no API cost. The
    # "failed runs are not counted" invariant lives at the RESULT layer
    # via re-derivation, not at the cache-file layer. Adding a write-side
    # abdication guard here would force every repeat run to re-pay the
    # peer for the same abdication.
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
        recovered_after_timeout=result.recovered_after_timeout,
        prompt_chars=result.prompt_chars,
        section_repair_attempted=result.section_repair_attempted,
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
        prompt_chars=len(prompt),
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
    mode_multiplier: float | None = None,
    mode: str | None = None,
) -> ParticipantResult:
    overflow = _context_overflow_result(name, cfg, prompt)
    if overflow is not None:
        return overflow
    cache_key, cached = _cache_lookup(name, cfg, prompt, cache_ctx)
    if cached is not None:
        return cached
    start = time.monotonic()
    result, meta = await _run_cli_once(
        name, cfg, prompt, cwd, start=start,
        mode_multiplier=mode_multiplier, mode=mode,
    )
    # Terse-retry on timeout. Mutually exclusive with launch-retry: a
    # timeout never returns nonzero_exit+stderr (launch-retry's gate).
    # The retry uses TERSE_RETRY_TIMEOUT_SECONDS regardless of the mode
    # multiplier so the total wall-clock cost stays bounded.
    if (
        not result.ok
        and is_timeout_error(result.error)
        and _terse_retry_enabled(cfg)
    ):
        terse_prompt = _build_terse_retry_prompt(prompt)
        max_prompt_chars = cfg.get("max_prompt_chars")
        if max_prompt_chars is None or len(terse_prompt) <= int(max_prompt_chars):
            terse_cfg = dict(cfg)
            terse_cfg["timeout"] = TERSE_RETRY_TIMEOUT_SECONDS
            terse_result, _terse_meta = await _run_cli_once(
                name, terse_cfg, terse_prompt, cwd, start=start,
                mode_multiplier=None,  # fixed 60s budget; no scaling
                mode=mode,
            )
            if terse_result.ok:
                from dataclasses import replace as _replace
                # `prompt_chars` reports the ORIGINAL prompt size (the one
                # that tripped the timeout wall), not the terse retry's
                # size. That keeps `timeout_recoveries` cross-tab-able with
                # `timeout_by_prompt_size`: both report the prompt that
                # caused the recovery path to fire.
                merged = _replace(
                    terse_result,
                    recovered_after_timeout=True,
                    terse_retry_attempted=True,
                    prompt_chars=len(prompt),
                )
                _maybe_persist_cache(name, prompt, cache_key, merged, cache_ctx)
                return merged
            # Terse retry also failed (re-timed, label-missing, abdication).
            # Annotate the original result so the retry attempt is visible
            # in transcripts/stats — without `terse_retry_attempted=True`
            # the failure looks identical to "retry never fired". Keep the
            # original error prefix intact so `classify_error` / quorum
            # math still see "timeout". `elapsed_seconds` stays the
            # original (multiplier-scaled) budget by design — wall-clock
            # includes the retry, but the budget that gated the timeout
            # decision was the original one.
            result = _annotate_timeout_retry_failure(result, terse_result, name)
    if _should_launch_retry(meta, cfg):
        await asyncio.sleep(_launch_retry_backoff(0))
        retry_result, retry_meta = await _run_cli_once(
            name, cfg, prompt, cwd, start=start,
            mode_multiplier=mode_multiplier, mode=mode,
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
            name, cfg, retry_prompt, cwd, start=start,
            mode_multiplier=mode_multiplier, mode=mode,
        )
        merged = _merge_cli_retry(result, retry_result)
        if result.recovered_after_launch_retry:
            merged.recovered_after_launch_retry = True
        _maybe_persist_cache(name, prompt, cache_key, merged, cache_ctx)
        return merged
    # Section-coverage repair retry (label was present but required sections
    # were missing). Distinct from the label-repair path above — different
    # error prefix, different directive. Cap of one retry stands: a peer
    # that misses sections twice has a deeper issue than this mechanism
    # can fix. Skip when terse-retry already ran (cumulative call ceiling).
    if _should_section_repair(result, cfg):
        from llm_council.sections import required_sections_missing
        missing = required_sections_missing(prompt, result.output)
        if missing:
            retry_prompt = _build_section_repair_prompt(
                prompt, result.output, missing
            )
            max_prompt_chars = cfg.get("max_prompt_chars")
            if (
                max_prompt_chars is None
                or len(retry_prompt) <= int(max_prompt_chars)
            ):
                retry_result, _retry_meta = await _run_cli_once(
                    name, cfg, retry_prompt, cwd, start=start,
                    mode_multiplier=mode_multiplier, mode=mode,
                )
                merged = _merge_cli_section_retry(result, retry_result)
                if result.recovered_after_launch_retry:
                    merged.recovered_after_launch_retry = True
                _maybe_persist_cache(name, prompt, cache_key, merged, cache_ctx)
                return merged
    # Strict-evidence repair retry (label present, sections satisfied, but
    # one or more EVIDENCE bullets lacked an epistemic tag). Cap of one
    # retry. No chaining with terse-retry, label-retry, or section-repair —
    # they're independent error-paths, and chaining would push past the
    # documented "one extra round" wall-clock ceiling. The
    # `section_repair_attempted` guard is the pass-9 fix: when the
    # section-repair retry already fired and surfaced new
    # `UntaggedEvidence:`, chaining a strict-evidence retry on top would
    # be the third outer-visible call per peer.
    if (
        not result.ok
        and _retry_enabled(cfg)
        and result.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
        and not getattr(result, "recovered_after_timeout", False)
        and not getattr(result, "section_repair_attempted", False)
    ):
        retry_prompt = (
            f"{prompt}\n\n"
            "--- Your previous response (first attempt) ---\n"
            f"{result.output.strip()}\n\n"
            f"{STRICT_EVIDENCE_REPAIR_RETRY_INSTRUCTION}"
        )
        max_prompt_chars = cfg.get("max_prompt_chars")
        if (
            max_prompt_chars is None
            or len(retry_prompt) <= int(max_prompt_chars)
        ):
            retry_result, _retry_meta = await _run_cli_once(
                name, cfg, retry_prompt, cwd, start=start,
                mode_multiplier=mode_multiplier, mode=mode,
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
    mode_multiplier: float | None = None,
    mode: str | None = None,
) -> tuple[ParticipantResult, dict[str, Any]]:
    base_timeout = int(cfg.get("timeout") or 240)
    timeout = _resolve_effective_timeout(cfg, mode_multiplier)
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
                prompt_chars=len(prompt),
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
        validation_error = (
            _response_validation_error(out, cfg, prompt=prompt) if ok else ""
        )
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
                prompt_chars=len(prompt),
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
                error=_format_timeout_error(
                    name, timeout, len(prompt),
                    mode_multiplier=mode_multiplier, mode=mode,
                    base_timeout=base_timeout,
                ),
                elapsed_seconds=elapsed,
                command=redact_prompt_args(command, prompt),
                model=cfg.get("model"),
                prompt_chars=len(prompt),
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
                prompt_chars=len(prompt),
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


def _build_section_repair_prompt(
    original_prompt: str, prior_response: str, missing: list[str]
) -> str:
    """Compose the section-coverage repair prompt.

    Used by all three adapter paths (CLI, openai_compatible, ollama) so
    the directive wording stays consistent. The missing-sections list
    comes from `sections.required_sections_missing` and is rendered as
    a comma-joined string in the directive.
    """
    section_directive = SECTION_REPAIR_RETRY_INSTRUCTION.format(
        missing=", ".join(missing)
    )
    return (
        f"{original_prompt}\n\n"
        "--- Your previous response (first attempt) ---\n"
        f"{prior_response.strip()}\n\n"
        f"{section_directive}"
    )


def _should_section_repair(result: ParticipantResult, cfg: dict[str, Any]) -> bool:
    """Whether a result is eligible for the section-coverage repair retry.

    Three conditions, all required:
    1. The result failed with the `IncompleteResponse:` prefix (label
       was present but one or more REQUIRED sections were missing).
    2. The retry-on-missing-label kill-switch (`retry_on_missing_label`
       / `retries: 0`) is NOT engaged — same gate as label-repair so an
       opt-out user gets a single behavior across all repair retries.
    3. The result is NOT a terse-retry recovery. The CLI/openai_compatible/
       ollama wrappers each give every peer at most one extra round to
       recover (terse-retry on timeout OR section-repair, never both),
       so a peer that already exhausted its retry budget via terse-retry
       does not get a third call here.
    """
    if result.ok:
        return False
    if not result.error.startswith(INCOMPLETE_RESPONSE_PREFIX):
        return False
    if not _retry_enabled(cfg):
        return False
    if getattr(result, "recovered_after_timeout", False):
        return False
    return True


def _build_terse_retry_prompt(original_prompt: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "--- Timeout recovery directive ---\n"
        f"{TERSE_RETRY_INSTRUCTION}"
    )


def _annotate_timeout_retry_failure(
    original: "ParticipantResult",
    terse: "ParticipantResult",
    name: str,
) -> "ParticipantResult":
    """Return a copy of ``original`` annotated with terse-retry attempt info.

    Invoked when the terse-retry path fired but also failed. Sets
    ``terse_retry_attempted=True`` so transcripts/stats can distinguish
    "retry fired and failed" from "retry never fired", and appends a
    suffix to the error string naming the next mitigation lever. The
    base error prefix stays unchanged so ``classify_error`` still
    returns the original kind (typically ``timeout``).
    """
    from dataclasses import replace
    retry_kind = classify_error(terse.error) or "unknown"
    suffix = TERSE_RETRY_FAILED_SUFFIX.format(
        budget=TERSE_RETRY_TIMEOUT_SECONDS,
        retry_kind=retry_kind,
        name=name,
    )
    # Avoid double-annotation if the suffix is already present (defensive
    # against future code paths that might call this twice).
    annotated_error = original.error
    if suffix not in annotated_error:
        annotated_error = original.error + suffix
    return replace(
        original,
        terse_retry_attempted=True,
        error=annotated_error,
    )


def _build_strict_evidence_retry_prompt(
    original_prompt: str, prior_response: str
) -> str:
    """Compose the strict-evidence repair retry prompt for hosted/local
    transports. Shape mirrors `_build_cli_retry_prompt`: original prompt +
    prior response excerpt + repair directive. The CLI path constructs
    the equivalent string inline at the retry site rather than calling
    this helper because the section-repair branch wants the same
    structure; both end up with the directive trailing the prior output.
    """
    return (
        f"{original_prompt}\n\n"
        "--- Your previous response (first attempt) ---\n"
        f"{prior_response.strip()}\n\n"
        f"{STRICT_EVIDENCE_REPAIR_RETRY_INSTRUCTION}"
    )


def _merge_hosted_strict_evidence_retry(
    original: ParticipantResult, retry: ParticipantResult
) -> ParticipantResult:
    """Merge a strict-evidence retry result for openai_compatible / ollama.

    Success: return the retry with `repair_retry_recovered=True` plus a
    formatted transcript that preserves both attempts. Failure: keep the
    original `UntaggedEvidence:` error string so downstream error_kind
    classification stays stable. If the retry itself surfaced a different
    failure (e.g. timeout, downstream HTTP error), we still prefer the
    ORIGINAL error — the retry is a best-effort repair, not a replacement
    for the original verdict.
    """
    if retry.ok:
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=True,
        )
        from dataclasses import replace as _replace
        return _replace(
            retry,
            ok=True,
            output=merged_output,
            error="",
            repair_retry_recovered=True,
        )
    # Retry failed (still untagged, or a different error). Keep the
    # original verdict so error_kind stays `untagged_evidence`.
    return original



def _merge_cli_retry(
    original: ParticipantResult, retry: ParticipantResult
) -> ParticipantResult:
    # The label-repair / section-repair retry replays the SAME prompt, so
    # `prompt_chars` on the merged result reflects the same length as both
    # the original and the retry. Prefer `original.prompt_chars` so the
    # value is stable even if a future retry path reshapes the prompt; fall
    # back to the retry's value otherwise.
    merged_prompt_chars = original.prompt_chars or retry.prompt_chars
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
            prompt_chars=merged_prompt_chars,
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
            prompt_chars=merged_prompt_chars,
        )
    return original


def _merge_cli_section_retry(
    original: ParticipantResult, retry: ParticipantResult
) -> ParticipantResult:
    """Merge a CLI section-repair retry attempt with the original failure.

    Mirrors `_merge_cli_retry` but for the section-coverage path:
    - retry succeeded → ok=True with section-themed recovery header
    - retry came back missing sections again → ok=False with the
      `IncompleteResponse:` prefix preserved (so error_kind stays
      `incomplete_response`) and a section-themed exhausted-retry header
    - retry came back with sections OK but EVIDENCE untagged
      (`UntaggedEvidence:` — strict-evidence is the gate AFTER
      sections in `_response_validation_error`) → ok=False with the
      retry's `UntaggedEvidence:` error preserved so `classify_error`
      returns `untagged_evidence`. Pass-9 dogfood found this case used
      to fall through to `return original`, silently discarding the
      section-fixed retry and undercounting `untagged_evidence`.
    - retry failed for an unrelated reason (downstream/timeout/etc.) →
      preserve the original failure so the operator sees the section-
      coverage error, not the retry's incidental failure

    ALL non-fall-through branches set `section_repair_attempted=True`
    so the strict-evidence wrapper can refuse to fire on the
    sections-recovered-but-evidence-untagged result (without that
    guard, the wrapper would chain a third call, breaking the "one
    extra call per peer per round" invariant).
    """
    if retry.ok:
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=True,
            header_kind="sections",
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
            section_repair_attempted=True,
        )
    if retry.error.startswith(INCOMPLETE_RESPONSE_PREFIX) and retry.output:
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=False,
            header_kind="sections",
        )
        return ParticipantResult(
            name=retry.name,
            ok=False,
            output=merged_output,
            error=(
                f"{INCOMPLETE_RESPONSE_PREFIX} response had the RECOMMENDATION "
                "label but missed one or more REQUIRED sections after one "
                "repair retry"
            ),
            elapsed_seconds=retry.elapsed_seconds,
            command=retry.command,
            model=retry.model,
            section_repair_attempted=True,
        )
    if retry.error.startswith(UNTAGGED_EVIDENCE_PREFIX) and retry.output:
        # Pass-9 fix: section-repair retry recovered the missing sections
        # but the now-visible EVIDENCE bullets lack epistemic tags.
        # Preserve the retry's `UntaggedEvidence:` error so error_kind
        # routes correctly, and surface BOTH attempts in the merged
        # output so the operator can see what happened. The
        # strict-evidence wrapper checks `section_repair_attempted`
        # and refuses to chain another retry.
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=False,
            header_kind="sections_then_evidence",
        )
        return ParticipantResult(
            name=retry.name,
            ok=False,
            output=merged_output,
            error=retry.error,
            elapsed_seconds=retry.elapsed_seconds,
            command=retry.command,
            model=retry.model,
            section_repair_attempted=True,
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
    mode_multiplier: float | None = None,
    mode: str | None = None,
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
        name, cfg, prompt, image_manifest=image_manifest,
        mode_multiplier=mode_multiplier, mode=mode,
    )
    # Terse-retry on timeout. Hooks at the wrapper layer because the inner
    # has many return paths; one outer retry against the bounded
    # TERSE_RETRY_TIMEOUT_SECONDS budget is simpler than threading retry
    # state through every branch.
    if (
        not result.ok
        and is_timeout_error(result.error)
        and _terse_retry_enabled(cfg)
    ):
        terse_prompt = _build_terse_retry_prompt(prompt)
        max_prompt_chars = cfg.get("max_prompt_chars")
        if max_prompt_chars is None or len(terse_prompt) <= int(max_prompt_chars):
            terse_cfg = dict(cfg)
            terse_cfg["timeout"] = TERSE_RETRY_TIMEOUT_SECONDS
            terse_result = await _run_openai_compatible_inner(
                name, terse_cfg, terse_prompt,
                image_manifest=image_manifest,
                mode_multiplier=None, mode=mode,
            )
            if terse_result.ok:
                from dataclasses import replace as _replace
                # `prompt_chars` reports the ORIGINAL prompt size (the one
                # that tripped the timeout wall), not the terse retry's
                # size. Keeps `timeout_recoveries` cross-tab-able with
                # `timeout_by_prompt_size`.
                result = _replace(
                    terse_result,
                    recovered_after_timeout=True,
                    terse_retry_attempted=True,
                    prompt_chars=len(prompt),
                )
            else:
                # Annotate the original result so the retry attempt is
                # visible in transcripts/stats (matches CLI path).
                result = _annotate_timeout_retry_failure(
                    result, terse_result, name
                )
    # Retry layering invariant: terse-retry → section-repair → strict-evidence.
    # Section-repair runs first because a section-fixed retry can introduce
    # new untagged EVIDENCE bullets, which strict-evidence should then catch.
    # Each gate caps at one extra round, and _should_section_repair /
    # the strict-evidence guard both refuse to fire after terse-retry so
    # a single peer never burns more than one extra call per turn.
    #
    # Section-coverage repair retry. Mirrors the CLI path: when the
    # label is present but one or more REQUIRED sections are missing,
    # re-ask once with `SECTION_REPAIR_RETRY_INSTRUCTION` appended.
    # The retry runs through the SAME inner with `retry_on_missing_label:
    # False` so the inner's own label-repair branch can't fire a chained
    # third call.
    result = await _maybe_section_repair_openai_compatible(
        name=name,
        cfg=cfg,
        prompt=prompt,
        result=result,
        image_manifest=image_manifest,
        mode_multiplier=mode_multiplier,
        mode=mode,
    )
    # Strict-evidence repair retry. The inner's label-retry path skips
    # untagged_evidence failures (gated by `_is_label_only_failure`), so we
    # apply the retry here at the wrapper layer. One shot, no chaining
    # with terse-retry above: a peer that already needed terse recovery
    # shouldn't get a third call. Mirrors the CLI strict-evidence path so
    # all three transports behave identically. The
    # `section_repair_attempted` guard is the pass-9 fix: when the
    # section-repair retry already fired and surfaced new
    # `UntaggedEvidence:`, chaining a strict-evidence retry on top would
    # be the third outer-visible call per peer.
    if (
        not result.ok
        and _retry_enabled(cfg)
        and result.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
        and not getattr(result, "recovered_after_timeout", False)
        and not getattr(result, "section_repair_attempted", False)
    ):
        retry_prompt = _build_strict_evidence_retry_prompt(prompt, result.output)
        max_prompt_chars = cfg.get("max_prompt_chars")
        if max_prompt_chars is None or len(retry_prompt) <= int(max_prompt_chars):
            # Disable the inner's own label-repair retry for this call so a
            # strict-evidence retry response that drops the RECOMMENDATION
            # label cannot fire a chained third outer call. Mirrors the
            # section-repair pattern above.
            retry_cfg = dict(cfg)
            retry_cfg["retry_on_missing_label"] = False
            retry_result = await _run_openai_compatible_inner(
                name, retry_cfg, retry_prompt, image_manifest=image_manifest,
                mode_multiplier=mode_multiplier, mode=mode,
            )
            result = _merge_hosted_strict_evidence_retry(result, retry_result)
    _maybe_persist_cache(name, prompt, cache_key, result, cache_ctx)
    return result


async def _maybe_section_repair_openai_compatible(
    *,
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    result: ParticipantResult,
    image_manifest: list[dict[str, Any]] | None,
    mode_multiplier: float | None,
    mode: str | None,
) -> ParticipantResult:
    if not _should_section_repair(result, cfg):
        return result
    from llm_council.sections import required_sections_missing
    missing = required_sections_missing(prompt, result.output)
    if not missing:
        return result
    retry_prompt = _build_section_repair_prompt(prompt, result.output, missing)
    max_prompt_chars = cfg.get("max_prompt_chars")
    if max_prompt_chars is not None and len(retry_prompt) > int(max_prompt_chars):
        return result
    # Disable the inner's own label-repair retry for this call. The
    # section-repair prompt explicitly tells the peer to keep its
    # existing reasoning and add the missing sections, so a label-only
    # failure on the retry is terminal — no chained third call.
    retry_cfg = dict(cfg)
    retry_cfg["retry_on_missing_label"] = False
    retry_result = await _run_openai_compatible_inner(
        name, retry_cfg, retry_prompt,
        image_manifest=image_manifest,
        mode_multiplier=mode_multiplier, mode=mode,
    )
    return _merge_hosted_section_retry(result, retry_result)


def _merge_hosted_section_retry(
    original: ParticipantResult, retry: ParticipantResult
) -> ParticipantResult:
    """Merge a hosted/local section-repair retry with the original failure.

    Same semantics as `_merge_cli_section_retry` but written against
    the hosted/local `ParticipantResult` shape (no `command`, has
    `prompt_tokens`/`completion_tokens`/`total_tokens`/`cost_usd`).
    See `_merge_cli_section_retry` for the third-branch rationale
    (sections recovered but EVIDENCE untagged → preserve the retry's
    `UntaggedEvidence:` error and set `section_repair_attempted=True`).
    """
    if retry.ok:
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=True,
            header_kind="sections",
        )
        return ParticipantResult(
            name=retry.name,
            ok=True,
            output=merged_output,
            error="",
            elapsed_seconds=retry.elapsed_seconds,
            model=retry.model,
            prompt_tokens=retry.prompt_tokens,
            completion_tokens=retry.completion_tokens,
            total_tokens=retry.total_tokens,
            cost_usd=retry.cost_usd,
            repair_retry_recovered=True,
            section_repair_attempted=True,
        )
    if retry.error.startswith(INCOMPLETE_RESPONSE_PREFIX) and retry.output:
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=False,
            header_kind="sections",
        )
        return ParticipantResult(
            name=retry.name,
            ok=False,
            output=merged_output,
            error=(
                f"{INCOMPLETE_RESPONSE_PREFIX} response had the RECOMMENDATION "
                "label but missed one or more REQUIRED sections after one "
                "repair retry"
            ),
            elapsed_seconds=retry.elapsed_seconds,
            model=retry.model,
            prompt_tokens=retry.prompt_tokens,
            completion_tokens=retry.completion_tokens,
            total_tokens=retry.total_tokens,
            cost_usd=retry.cost_usd,
            section_repair_attempted=True,
        )
    if retry.error.startswith(UNTAGGED_EVIDENCE_PREFIX) and retry.output:
        # Pass-9 fix: section-repair retry recovered the missing sections
        # but the now-visible EVIDENCE bullets lack epistemic tags. See
        # `_merge_cli_section_retry` for the full rationale.
        merged_output = _format_retry_transcript(
            original_output=original.output,
            retry_output=retry.output,
            recovered=False,
            header_kind="sections_then_evidence",
        )
        return ParticipantResult(
            name=retry.name,
            ok=False,
            output=merged_output,
            error=retry.error,
            elapsed_seconds=retry.elapsed_seconds,
            model=retry.model,
            prompt_tokens=retry.prompt_tokens,
            completion_tokens=retry.completion_tokens,
            total_tokens=retry.total_tokens,
            cost_usd=retry.cost_usd,
            section_repair_attempted=True,
        )
    # Retry failed for an unrelated reason (downstream/timeout/empty
    # response/etc.). Preserve the original failure so the operator
    # sees the section-coverage error rather than the retry's
    # incidental failure.
    return original


async def _run_openai_compatible_inner(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
    mode_multiplier: float | None = None,
    mode: str | None = None,
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
            prompt_chars=len(prompt),
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
    timeout = float(_resolve_effective_timeout(cfg, mode_multiplier, base_default=180))
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
                prompt_chars=len(prompt),
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
                prompt_chars=len(prompt),
            )
        finish_reason = choice.get("finish_reason")
        validation_error = _response_validation_error(content, cfg, prompt=prompt)
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
                            prompt=prompt,
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
                prompt_chars=len(prompt),
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
            prompt_chars=len(prompt),
        )
    except Exception as exc:
        return ParticipantResult(
            name=name,
            ok=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - start,
            model=model,
            prompt_chars=len(prompt),
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
    prompt: str | None = None,
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
    # The label-repair retry replays the same logical prompt; record its
    # length so telemetry stays consistent with non-retry paths.
    prompt_chars = len(prompt) if prompt is not None else None
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
            prompt_chars=prompt_chars,
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
            prompt_chars=prompt_chars,
        )
    retry_validation = _response_validation_error(retry_content, cfg, prompt=prompt)
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
            prompt_chars=prompt_chars,
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
        prompt_chars=prompt_chars,
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
    mode_multiplier: float | None = None,
    mode: str | None = None,
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
    result = await _run_ollama_inner(
        name, cfg, prompt, image_manifest=image_manifest,
        mode_multiplier=mode_multiplier, mode=mode,
    )
    if (
        not result.ok
        and is_timeout_error(result.error)
        and _terse_retry_enabled(cfg)
    ):
        terse_prompt = _build_terse_retry_prompt(prompt)
        max_prompt_chars = cfg.get("max_prompt_chars")
        if max_prompt_chars is None or len(terse_prompt) <= int(max_prompt_chars):
            terse_cfg = dict(cfg)
            terse_cfg["timeout"] = TERSE_RETRY_TIMEOUT_SECONDS
            terse_result = await _run_ollama_inner(
                name, terse_cfg, terse_prompt,
                image_manifest=image_manifest,
                mode_multiplier=None, mode=mode,
            )
            if terse_result.ok:
                from dataclasses import replace as _replace
                # `prompt_chars` reports the ORIGINAL prompt size (the one
                # that tripped the timeout wall), not the terse retry's
                # size. Keeps `timeout_recoveries` cross-tab-able with
                # `timeout_by_prompt_size`.
                result = _replace(
                    terse_result,
                    recovered_after_timeout=True,
                    terse_retry_attempted=True,
                    prompt_chars=len(prompt),
                )
            else:
                # Annotate the original result so the retry attempt is
                # visible in transcripts/stats (matches CLI path).
                result = _annotate_timeout_retry_failure(
                    result, terse_result, name
                )
    # Retry layering invariant (see openai_compatible wrapper for full notes):
    # terse-retry → section-repair → strict-evidence, each capped at one
    # extra round per peer.
    #
    # Section-coverage repair retry. Same contract as the CLI and
    # openai_compatible paths: label present + REQUIRED sections
    # missing → one extra call with `SECTION_REPAIR_RETRY_INSTRUCTION`.
    # `_should_section_repair` forbids chaining after terse-retry.
    result = await _maybe_section_repair_ollama(
        name=name,
        cfg=cfg,
        prompt=prompt,
        result=result,
        image_manifest=image_manifest,
        mode_multiplier=mode_multiplier,
        mode=mode,
    )
    # Strict-evidence repair retry (parallels CLI + openai_compatible).
    # The inner's label-retry path skips untagged_evidence failures, so we
    # apply the retry here. One shot, no chaining with terse-retry. The
    # `section_repair_attempted` guard is the pass-9 fix: when the
    # section-repair retry already fired and surfaced new
    # `UntaggedEvidence:`, chaining a strict-evidence retry on top would
    # be the third outer-visible call per peer.
    if (
        not result.ok
        and _retry_enabled(cfg)
        and result.error.startswith(UNTAGGED_EVIDENCE_PREFIX)
        and not getattr(result, "recovered_after_timeout", False)
        and not getattr(result, "section_repair_attempted", False)
    ):
        retry_prompt = _build_strict_evidence_retry_prompt(prompt, result.output)
        max_prompt_chars = cfg.get("max_prompt_chars")
        if max_prompt_chars is None or len(retry_prompt) <= int(max_prompt_chars):
            # Disable the inner's own label-repair retry for this call so a
            # strict-evidence retry response that drops the RECOMMENDATION
            # label cannot fire a chained third outer call. Mirrors the
            # section-repair pattern above.
            retry_cfg = dict(cfg)
            retry_cfg["retry_on_missing_label"] = False
            retry_result = await _run_ollama_inner(
                name, retry_cfg, retry_prompt, image_manifest=image_manifest,
                mode_multiplier=mode_multiplier, mode=mode,
            )
            result = _merge_hosted_strict_evidence_retry(result, retry_result)
    _maybe_persist_cache(name, prompt, cache_key, result, cache_ctx)
    return result


async def _maybe_section_repair_ollama(
    *,
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    result: ParticipantResult,
    image_manifest: list[dict[str, Any]] | None,
    mode_multiplier: float | None,
    mode: str | None,
) -> ParticipantResult:
    if not _should_section_repair(result, cfg):
        return result
    from llm_council.sections import required_sections_missing
    missing = required_sections_missing(prompt, result.output)
    if not missing:
        return result
    retry_prompt = _build_section_repair_prompt(prompt, result.output, missing)
    max_prompt_chars = cfg.get("max_prompt_chars")
    if max_prompt_chars is not None and len(retry_prompt) > int(max_prompt_chars):
        return result
    retry_cfg = dict(cfg)
    retry_cfg["retry_on_missing_label"] = False
    retry_result = await _run_ollama_inner(
        name, retry_cfg, retry_prompt,
        image_manifest=image_manifest,
        mode_multiplier=mode_multiplier, mode=mode,
    )
    return _merge_hosted_section_retry(result, retry_result)


async def _run_ollama_inner(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
    mode_multiplier: float | None = None,
    mode: str | None = None,
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
        ollama_timeout = float(_resolve_effective_timeout(cfg, mode_multiplier, base_default=180))
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
        validation_error = _response_validation_error(content, cfg, prompt=prompt)
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
                                    prompt_chars=len(prompt),
                                )
                            retry_validation = _response_validation_error(
                                retry_content, cfg, prompt=prompt
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
                                    prompt_chars=len(prompt),
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
                                prompt_chars=len(prompt),
                            )
            return ParticipantResult(
                name=name,
                ok=False,
                output=content.strip(),
                error=validation_error,
                elapsed_seconds=time.monotonic() - start,
                model=model,
                prompt_chars=len(prompt),
            )
        return ParticipantResult(
            name=name,
            ok=True,
            output=content.strip(),
            error="",
            elapsed_seconds=time.monotonic() - start,
            model=model,
            prompt_chars=len(prompt),
        )
    except Exception as exc:
        return ParticipantResult(
            name=name,
            ok=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - start,
            model=model,
            prompt_chars=len(prompt),
        )


async def run_participant(
    name: str,
    cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    *,
    image_manifest: list[dict[str, Any]] | None = None,
    cache_ctx: CacheContext | None = None,
    mode_multiplier: float | None = None,
    mode: str | None = None,
) -> ParticipantResult:
    ptype = cfg.get("type")
    if ptype == "cli":
        # CLI participants intentionally don't receive image_manifest at the
        # adapter layer: they share the project filesystem with the host and
        # open staged images themselves via the file paths listed in the
        # ## Images prompt section. Adding `vision: true` to a CLI cfg
        # therefore has no effect — the orchestrator's images_skipped check
        # treats CLI as always image-aware (orchestrator.py).
        result = await run_cli_participant(
            name, cfg, prompt, cwd, cache_ctx=cache_ctx,
            mode_multiplier=mode_multiplier, mode=mode,
        )
    elif ptype in ("openrouter", "openai_compatible"):
        result = await run_openai_compatible_participant(
            name, cfg, prompt, image_manifest=image_manifest, cache_ctx=cache_ctx,
            mode_multiplier=mode_multiplier, mode=mode,
        )
    elif ptype == "ollama":
        result = await run_ollama_participant(
            name, cfg, prompt, image_manifest=image_manifest, cache_ctx=cache_ctx,
            mode_multiplier=mode_multiplier, mode=mode,
        )
    else:
        result = ParticipantResult(
            name=name,
            ok=False,
            output="",
            error=f"Unsupported participant type: {ptype}",
            elapsed_seconds=0,
            model=cfg.get("model"),
        )
    return _with_envelope(result)


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
    mode_multiplier: float | None = None,
    mode: str | None = None,
) -> list[ParticipantResult]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_one(name: str) -> ParticipantResult:
        async with semaphore:
            cfg = participant_cfg[name]
            timeout = _resolve_effective_timeout(cfg, mode_multiplier)
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
                    mode_multiplier=mode_multiplier,
                    mode=mode,
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
                        "recovered_after_timeout": result.recovered_after_timeout,
                        "terse_retry_attempted": result.terse_retry_attempted,
                        "section_repair_attempted": result.section_repair_attempted,
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


def _response_validation_error(
    output: str,
    cfg: dict[str, Any],
    *,
    prompt: str | None = None,
) -> str:
    if cfg.get("require_recommendation") is False:
        # Chair/synthesizer-style cfg: skip both label AND section checks.
        # The chair produces a decision memo that wouldn't satisfy either.
        return ""
    if not _has_recommendation_label(output):
        excerpt = _first_output_excerpt(output)
        if excerpt:
            return (
                "InvalidParticipantResponse: missing required RECOMMENDATION label. "
                f"First output: {excerpt}"
            )
        return "InvalidParticipantResponse: empty response"
    # Label is present. Check the structured-section contract: when the
    # prompt declared `PART N — TITLE (REQUIRED)` headers AND the per-run
    # config has section validation enabled (default: True), the response
    # must reference each required section. PART 6 (RECOMMENDATION) is
    # skipped — the label check above already covers it.
    if prompt and cfg.get("require_sections", True) is not False:
        from llm_council.sections import required_sections_missing
        missing = required_sections_missing(prompt, output)
        if missing:
            return (
                f"{INCOMPLETE_RESPONSE_PREFIX} response had the RECOMMENDATION "
                f"label but missed required sections: {', '.join(missing)}"
            )
    # Strict evidence-tagging (default off — optional → required staging
    # mirrors v0.5.0 envelope rollout). When enabled, every EVIDENCE
    # bullet must carry one of [PUBLISHED]/[OBSERVABLE]/[INFERRED]/
    # [SPECULATIVE]. Untagged entries indicate the peer smuggled
    # speculation as fact. Empty EVIDENCE list passes — strict-evidence
    # gates the FORMAT of entries that exist, not their PRESENCE.
    if cfg.get("strict_evidence", False):
        envelope = _extract_response_envelope(output)
        untagged = [
            i for i, entry in enumerate(envelope.get("evidence") or [])
            if isinstance(entry, dict) and entry.get("tag") is None
            or isinstance(entry, str)
        ]
        if untagged:
            return (
                f"{UNTAGGED_EVIDENCE_PREFIX} {len(untagged)} EVIDENCE "
                "entry/entries lack a [PUBLISHED]/[OBSERVABLE]/[INFERRED]/"
                "[SPECULATIVE] tag while defaults.strict_evidence is true"
            )
    return ""


def _retry_enabled(cfg: dict[str, Any]) -> bool:
    if cfg.get("retry_on_missing_label", True) is False:
        return False
    # An explicit `retries: 0` is the user saying "no extra calls of any
    # kind"; respect that for the application-level repair retry too,
    # otherwise the cost regression undoes commit 45b44ee.
    if "retries" in cfg and _coerce_retries(cfg.get("retries"), default=1) == 0:
        return False
    return True


def _terse_retry_enabled(cfg: dict[str, Any]) -> bool:
    """Whether terse-retry-on-timeout is allowed for this participant.

    Respects the same `retries: 0` kill-switch as label-repair retry — a
    user who wrote "no extra calls of any kind" should not get an extra
    timeout-recovery call either. Also gated by an explicit per-participant
    `terse_retry_on_timeout: false` opt-out.
    """
    if cfg.get("terse_retry_on_timeout", True) is False:
        return False
    if "retries" in cfg and _coerce_retries(cfg.get("retries"), default=1) == 0:
        return False
    return True


def _is_label_only_failure(output: str, cfg: dict[str, Any]) -> bool:
    if not output or not output.strip():
        return False
    error = _response_validation_error(output, cfg)
    if not error.startswith("InvalidParticipantResponse: missing required"):
        return False
    # A peer that self-reports `EFFORT: blocked` is terminal — re-asking
    # the same prompt produces another abdication for no new signal. The
    # adapter records the original error, `_with_envelope` later flips
    # ok=False if a label was present, and the orchestrator drops the
    # peer from quorum. Label-only repair retry is for honest mistakes
    # (label forgotten), not declined-effort responses.
    envelope = _extract_response_envelope(output)
    if (envelope.get("effort") or "").lower() == "blocked":
        return False
    return True


def _format_retry_transcript(
    *,
    original_output: str,
    retry_output: str,
    recovered: bool,
    header_kind: str = "label",
) -> str:
    """Render a paired transcript of the original + retry attempts.

    `header_kind` distinguishes which validation failure triggered the
    repair retry so the human-readable header is accurate. "label" is
    the original label-missing path; "sections" is the section-coverage
    repair path (the response had the RECOMMENDATION label but skipped
    one or more `PART N — TITLE (REQUIRED)` sections);
    "sections_then_evidence" is the pass-9 case where the section-repair
    retry produced sections OK but EVIDENCE bullets without epistemic
    tags — the retry IS preserved (not silently dropped) but the result
    is still ok=False because strict_evidence flagged it.
    """
    if header_kind == "sections_then_evidence":
        header = (
            "[retry exhausted] "
            "Section-repair retry recovered the missing REQUIRED sections "
            "but the now-visible EVIDENCE bullets lack a [PUBLISHED]/"
            "[OBSERVABLE]/[INFERRED]/[SPECULATIVE] tag (strict_evidence is "
            "enabled). No further repair retry will fire — the cumulative "
            "wall-clock cost ceiling caps at one extra round per peer."
        )
    elif header_kind == "sections":
        header = (
            "[recovered after retry] "
            "First attempt was missing one or more REQUIRED sections; "
            "second attempt is shown below."
            if recovered
            else "[retry exhausted] "
            "Both attempts were missing one or more REQUIRED sections."
        )
    else:
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


_ENVELOPE_SINGLE_RE = re.compile(
    r"""
    ^\s*(?:>\s*)?(?:[-*]\s+)?(?:\*\*)?
    (?P<key>EFFORT|CONFIDENCE|RISK)
    (?:\*\*)?\s*:\s*(?:\*\*)?
    (?P<value>[A-Za-z][A-Za-z\-_]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ENVELOPE_LIST_HEADERS = {
    "BLOCKERS": "blockers",
    "EVIDENCE": "evidence",
    "TESTS_TO_RUN": "tests_to_run",
    "ASSUMPTIONS": "assumptions",
}

_ENVELOPE_LIST_HEADER_RE = re.compile(
    r"""
    ^\s*(?:>\s*)?(?:\*\*)?
    (?P<key>BLOCKERS|EVIDENCE|TESTS_TO_RUN|ASSUMPTIONS)
    (?:\*\*)?\s*:\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Inline single-entry form: ``EVIDENCE: <content>`` on one line. Peers
# in the wild (qwen, observed in pass-9) emit each evidence claim as its
# own ``EVIDENCE: ...`` line instead of a bare ``EVIDENCE:`` header
# followed by ``- bullet`` lines. Without this fallback those entries
# are silently dropped, which ALSO silently disables strict-evidence
# validation for the response — the validator only fires on parsed
# entries that lack a tag (empty list passes by design). Requires at
# least one non-whitespace character after the colon so it doesn't
# collide with the bare-header form already handled above.
_ENVELOPE_LIST_INLINE_RE = re.compile(
    r"""
    ^\s*(?:>\s*)?(?:[-*+]\s+)?(?:\*\*)?
    (?P<key>BLOCKERS|EVIDENCE|TESTS_TO_RUN|ASSUMPTIONS)
    (?:\*\*)?\s*:\s*
    (?P<item>\S.*?)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ENVELOPE_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<item>.+?)\s*$")

# Evidence-tag parser. Pass-7's R3 rule defined four tags for grading the
# epistemic status of each EVIDENCE bullet. Tags may appear at the start
# (`[PUBLISHED] foo`), end (`foo [PUBLISHED]`), or inline. Qualifiers
# after a separator (`[OBSERVABLE — behavioral]`) are accepted but
# stripped from the canonical tag.
EVIDENCE_TAG_RE = re.compile(
    r"""
    \[
        \s*
        (?P<tag>PUBLISHED|OBSERVABLE|INFERRED|SPECULATIVE)
        (?:\s*[—\-–:]\s*[^\]]+)?
        \s*
    \]
    """,
    re.VERBOSE | re.IGNORECASE,
)

KNOWN_EVIDENCE_TAGS = frozenset({"published", "observable", "inferred", "speculative"})

# Tag-parsing only applies to evidence claims today. BLOCKERS are
# concrete missing artifacts (file paths, command output), not
# empirical claims. ASSUMPTIONS are first-person ("we assume X").
# TESTS_TO_RUN are commands. Promoting tags to those fields would
# change their semantics — defer until a follow-up.
TAG_PARSED_LIST_FIELDS = frozenset({"evidence"})


def _parse_tagged_entry(raw: str) -> dict[str, Any]:
    """Extract a tag from an evidence bullet; structure as `{text, tag}`.

    Tag-less entries return `{"text": raw, "tag": None}` so the parsed
    shape is uniform and downstream consumers (stats, transcripts) can
    treat untagged entries as a distinct category.
    """
    if not raw:
        return {"text": raw, "tag": None}
    match = EVIDENCE_TAG_RE.search(raw)
    if not match:
        return {"text": raw, "tag": None}
    tag = match.group("tag").lower()
    cleaned = (raw[: match.start()] + raw[match.end():]).strip(" -—–:")
    return {"text": cleaned or raw.strip(), "tag": tag}


def _extract_response_envelope(output: str) -> dict[str, Any]:
    """Parse the optional Pick-A response envelope from a peer's output.

    Fields are all optional in v1. List fields collect bullet lines under a
    `FIELD:` header until the next blank line, header, or fence. Single
    fields match `FIELD: value` on one line. Matches inside fenced blocks
    are ignored — same rule as the recommendation label.
    """
    envelope: dict[str, Any] = {
        "effort": None,
        "confidence": None,
        "risk": None,
        "blockers": [],
        "evidence": [],
        "tests_to_run": [],
        "assumptions": [],
    }
    if not output:
        return envelope
    in_fence = False
    active_list: str | None = None
    for line in output.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            active_list = None
            continue
        if in_fence:
            continue
        if not line.strip():
            active_list = None
            continue
        list_match = _ENVELOPE_LIST_HEADER_RE.match(line)
        if list_match:
            active_list = _ENVELOPE_LIST_HEADERS[list_match.group("key").upper()]
            continue
        if active_list:
            bullet = _ENVELOPE_BULLET_RE.match(line)
            if bullet:
                item = bullet.group("item")
                if active_list in TAG_PARSED_LIST_FIELDS:
                    envelope[active_list].append(_parse_tagged_entry(item))
                else:
                    envelope[active_list].append(item)
                continue
            active_list = None
        inline = _ENVELOPE_LIST_INLINE_RE.match(line)
        if inline:
            # `EVIDENCE: <content>` (and the BLOCKERS / TESTS_TO_RUN /
            # ASSUMPTIONS analogues) as a single-line entry. Treat each
            # such line as one item under the named list, and set
            # active_list so any following `- bullet` lines accrue too.
            target = _ENVELOPE_LIST_HEADERS[inline.group("key").upper()]
            item = inline.group("item")
            if target in TAG_PARSED_LIST_FIELDS:
                envelope[target].append(_parse_tagged_entry(item))
            else:
                envelope[target].append(item)
            active_list = target
            continue
        single = _ENVELOPE_SINGLE_RE.match(line)
        if single:
            key = single.group("key").lower()
            if envelope.get(key) is None:
                envelope[key] = single.group("value").lower()
    return envelope


def _is_abdication(envelope: dict[str, Any], output: str) -> bool:
    """A peer abdicates when it self-reports ``EFFORT: blocked`` while
    naming no concrete missing artifact. Substantive ``RECOMMENDATION: no``
    answers that omit EFFORT entirely are NOT abdications — only the
    explicit-blocked-without-blockers shape qualifies. Empty ASSUMPTIONS
    is required too: a peer that lists what it assumed has at least named
    the unknowns and is not abdicating."""
    if (envelope.get("effort") or "").lower() != "blocked":
        return False
    if envelope.get("blockers"):
        return False
    if envelope.get("assumptions"):
        return False
    # Abdication only makes sense when there is also a label to vote with;
    # without a label the response is already an invalid_response.
    return _has_recommendation_label(output)


def _with_envelope(result: ParticipantResult) -> ParticipantResult:
    """Populate envelope fields on a result; mark abdications as terminal failures.

    Abdication detection runs once here so quorum math (in the orchestrator)
    sees ``ok=False`` and excludes the peer. Abdication is intentionally NOT
    eligible for the label-only repair retry — see ``_is_label_only_failure``.

    Invariant: envelope parsing always runs on `_envelope_parse_source(output)`,
    not on raw output. For repair-retry results this strips the original
    attempt section out so an `EFFORT: blocked` in the (failed) original
    cannot leak into the (valid) repaired response's envelope. The strip
    is the lone correctness mechanism — do not gate the abdication check
    on `repair_retry_recovered` here, because that would let a legitimately
    abdicating *repaired* response slip through as ok=True.
    """
    from dataclasses import replace

    parse_source = _envelope_parse_source(result.output)
    envelope = _extract_response_envelope(parse_source)
    updated = replace(result, **envelope)
    if updated.ok and _is_abdication(envelope, parse_source):
        return replace(
            updated,
            ok=False,
            error=(
                f"{ABDICATED_ERROR_PREFIX} peer reported EFFORT: blocked with no "
                "concrete missing artifact in BLOCKERS or ASSUMPTIONS. "
                "Abdication is treated as a non-vote; the council will not retry "
                "the same prompt — re-run with more context or escalate the mode."
            ),
        )
    return updated


def _envelope_parse_source(output: str) -> str:
    """Strip the original-attempt section out of a repair-retry transcript.

    ``_format_retry_transcript`` produces output of the shape::

        [recovered after retry] ...
        --- Repaired response ---
        <repaired>
        --- Original response (first attempt) ---
        <original>

    We want the envelope to reflect only the repaired (valid) attempt, so
    the original section is dropped. Non-retry outputs are returned as-is.
    """
    if not output or "--- Repaired response ---" not in output:
        return output
    head, _, tail = output.partition("--- Repaired response ---")
    repaired_section, _, _original_section = tail.partition(
        "--- Original response (first attempt) ---"
    )
    return repaired_section.strip()


def _has_recommendation_label(output: str) -> bool:
    in_fence = False
    for line in output.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if RECOMMENDATION_RE.match(line):
            return True
    return False


def _first_output_excerpt(output: str, max_chars: int = 240) -> str:
    cleaned = " ".join(output.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _resolve_effective_timeout(
    cfg: dict[str, Any],
    mode_multiplier: float | None,
    *,
    base_default: int = 240,
) -> int:
    """Resolve the per-call timeout from per-participant base and mode multiplier.

    Per-participant ``cfg["timeout"]`` stays the source of truth for the base
    value; the multiplier is layered on top so users who raised the base for
    a stubborn host CLI also benefit on consensus/deliberate runs. Pass-7's
    14K-char prompt timed claude out at 240s; consensus mode's 2.0x raises
    that to 480s without forcing the user to override the per-participant
    config. Rounds up so 1.5x of 240 lands at 360, not 359.
    """
    base = int(cfg.get("timeout") or base_default)
    if mode_multiplier is None or mode_multiplier <= 1.0:
        return base
    import math
    return int(math.ceil(base * float(mode_multiplier)))


def _format_timeout_error(
    name: str,
    timeout: int,
    prompt_chars: int,
    *,
    mode_multiplier: float | None = None,
    mode: str | None = None,
    base_timeout: int | None = None,
) -> str:
    multiplier_note = ""
    if (
        mode_multiplier is not None
        and mode_multiplier > 1.0
        and base_timeout is not None
        and mode
    ):
        multiplier_note = (
            f" (mode '{mode}' applied {mode_multiplier}x multiplier, "
            f"raising base {base_timeout}s -> {timeout}s)"
        )
    return (
        f"Timeout: `{name}` did not respond within {timeout}s"
        f"{multiplier_note} "
        f"(prompt was {prompt_chars} chars). "
        "To raise the limit, set `participants."
        f"{name}.timeout: <seconds>` in `.llm-council.yaml`. "
        "To skip this participant for one run, pass an explicit participant "
        "list that omits it. To shorten the prompt, drop large `--context` "
        "files or use `--diff` more selectively."
    )


# Single source of truth for "this error string is a timeout". Used by
# both `is_timeout_error` (gates the terse-retry path) and
# `classify_error` (produces the stable `error_kind=timeout` consumed by
# stats / transcripts / MCP --json). Keeping these in sync matters
# because `stats.aggregate` only buckets timed-out runs into
# `timeout_by_prompt_size` / `timeout_recoveries` when
# `error_kind == "timeout"` — drift here loses telemetry on hosted
# httpx timeouts.
#
# CLI subprocess timeouts use "Timeout:" / "TimeoutError:" prefixes.
# httpx-backed peers (openai_compatible, ollama) raise ReadTimeout /
# ConnectTimeout / WriteTimeout / PoolTimeout / TimeoutException which
# the generic `except Exception` in those adapters serializes as
# "ReadTimeout: ..." etc.
_TIMEOUT_PREFIXES: tuple[str, ...] = (
    "Timeout:",
    "TimeoutError:",
    "ReadTimeout:",
    "ConnectTimeout:",
    "WriteTimeout:",
    "PoolTimeout:",
    "TimeoutException:",
)


def is_timeout_error(error: str) -> bool:
    if not error:
        return False
    return any(error.startswith(p) for p in _TIMEOUT_PREFIXES)


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
ERROR_KIND_ABDICATED = "abdicated"
ERROR_KIND_INCOMPLETE_RESPONSE = "incomplete_response"
ERROR_KIND_UNTAGGED_EVIDENCE = "untagged_evidence"
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
        ERROR_KIND_ABDICATED,
        ERROR_KIND_INCOMPLETE_RESPONSE,
        ERROR_KIND_UNTAGGED_EVIDENCE,
        ERROR_KIND_UNKNOWN,
    }
)

PREFLIGHT_FAILED_PREFIX = "PreflightFailed:"
ABDICATED_ERROR_PREFIX = "AbdicatedResponse:"
INCOMPLETE_RESPONSE_PREFIX = "IncompleteResponse:"
UNTAGGED_EVIDENCE_PREFIX = "UntaggedEvidence:"


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
    # Timeout check must precede the `downstream_markers` substring scan
    # below — otherwise "ReadTimeout: ..." would match the
    # "ReadTimeout" downstream marker and get classified as
    # downstream_error instead of timeout, breaking
    # `stats.timeout_by_prompt_size` / `timeout_recoveries`.
    if is_timeout_error(error):
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
    if error.startswith(ABDICATED_ERROR_PREFIX):
        return ERROR_KIND_ABDICATED
    if error.startswith(INCOMPLETE_RESPONSE_PREFIX):
        return ERROR_KIND_INCOMPLETE_RESPONSE
    if error.startswith(UNTAGGED_EVIDENCE_PREFIX):
        return ERROR_KIND_UNTAGGED_EVIDENCE
    # httpx + downstream-API errors funnel through f"{type(exc).__name__}: ..."
    # in the openrouter / ollama paths; we don't try to introspect those
    # further here, just classify them as `downstream_error` so callers can
    # distinguish "their service blew up" from "our validation rejected it".
    # NOTE: httpx timeout class names (ReadTimeout, ConnectTimeout, etc.)
    # are intentionally NOT in this list — they're classified as `timeout`
    # above so timeout telemetry counts hosted timeouts uniformly.
    downstream_markers = (
        "HTTPStatusError",
        "ConnectError",
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
