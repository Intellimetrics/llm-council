"""Council run orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from llm_council.adapters import (
    CacheContext,
    ERROR_KIND_MODEL_SUBSTITUTED,
    ParticipantResult,
    PREFLIGHT_FAILED_PREFIX,
    classify_error,
    is_quota_exhausted_error,
    is_timeout_error,
    run_participants,
)
from llm_council.cache import (
    is_caching_disabled_for_mode,
    resolve_ttl_seconds,
)
from llm_council.citations import verify_evidence_citations
from llm_council.config import (
    is_local_base_url,
    is_loopback_base_url,
    participant_api_key_env,
)
from llm_council.convergence import (
    MIN_TOKENS_FOR_CLASSIFICATION,
    classify,
    jaccard_similarity,
    resolve_thresholds,
    tokenize,
)
from llm_council.deliberation import (
    build_deliberation_prompt,
    default_min_quorum,
    has_disagreement,
    labeled_quorum_count,
    recommendation_counts,
    recommendation_label,
)
from llm_council.env import env_get


PREFLIGHT_TIMEOUT_SECONDS = 1.0


def _utc_progress_timestamp() -> str:
    """Return a compact, timezone-explicit timestamp for progress telemetry."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# Embedded-credential regex shared by `_redact_base_url` (for the rendered
# base_url) and `_redact_credentials_in_text` (defense-in-depth pass over
# arbitrary strings — e.g. an httpx exception that echoes the URL it tried
# to reach). Matches `<scheme>://<userinfo>@<host>` and replaces userinfo
# with `***`. Conservative: the scheme/host shapes stay intact.
_EMBEDDED_CRED_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)[^/@\s]+@")


def _redact_base_url(base_url: str) -> str:
    """Strip embedded credentials (user:pass@host) from a URL before
    rendering it into user-facing error messages or transcripts.

    `allow_private: true` skips the embedded-credentials validator (so
    that local participants with `http://user:pass@127.0.0.1` are
    permitted), which means a careless config could otherwise leak the
    creds into transcripts via the preflight error message.
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return base_url
    if not parsed.username and not parsed.password:
        return base_url
    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username or parsed.password:
        netloc = f"***@{netloc}"
    return parsed._replace(netloc=netloc).geturl()


def _redact_credentials_in_text(text: str) -> str:
    """Defense-in-depth scrub of an arbitrary string for `scheme://user:pass@host`
    patterns. Used on the raw exception text from httpx, which may echo
    back the URL it tried to reach. `_redact_base_url` already handles
    the rendered `base_url`; this pass catches anything else that might
    quote the user-info portion.
    """
    return _EMBEDDED_CRED_RE.sub(lambda m: f"{m.group('scheme')}***@", text)


async def _preflight_one(name: str, cfg: dict[str, Any]) -> str | None:
    """Probe one local participant. Returns an error message on failure, None on success.

    For `type: ollama` we hit `/api/tags` (matches the doctor probe). For
    `type: openai_compatible` we hit `/v1/models` (matches the
    --probe-local-openai probe). 1-second timeout — anything slower than
    that on a loopback endpoint indicates a hung server, not a slow one.
    """
    ptype = cfg.get("type")
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        return None  # nothing to probe; let the run-time path produce the real error
    if ptype == "ollama":
        url = f"{base_url}/api/tags"
    elif ptype == "openai_compatible":
        # `/v1/models` lives under whatever path the user configured. The
        # base_url canonical form ends at `/v1`; tolerate both shapes.
        if base_url.endswith("/v1"):
            url = f"{base_url}/models"
        else:
            url = f"{base_url}/v1/models"
    else:
        return None
    redacted = _redact_base_url(base_url)
    try:
        # Local/on-prem probes must never inherit HTTP(S)_PROXY: doing so could
        # disclose the endpoint path (and, on some proxies, the request) beyond
        # the machine/private network before the real run even starts.
        async with httpx.AsyncClient(
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
            trust_env=False,
        ) as client:
            response = await client.get(url)
    except Exception as exc:  # noqa: BLE001 — surface every failure mode legibly
        # Defense-in-depth: httpx errors sometimes quote the URL they tried
        # to reach, which would re-introduce embedded creds even after the
        # base_url is redacted. Run the same scrub over the exception text.
        exc_text = _redact_credentials_in_text(f"{type(exc).__name__}: {exc}")
        return (
            f"{PREFLIGHT_FAILED_PREFIX} local endpoint unreachable for "
            f"{name!r} (base_url={redacted!r}): {exc_text}"
        )
    if response.status_code >= 500:
        return (
            f"{PREFLIGHT_FAILED_PREFIX} local endpoint at {redacted!r} returned "
            f"HTTP {response.status_code} for {name!r}"
        )
    # 2xx, 3xx, 4xx: server is up enough for the run path to make progress
    # or produce its own meaningful error. Don't pre-judge 4xx — some
    # llama.cpp builds 404 on /v1/models but serve /v1/chat/completions fine.
    return None


async def preflight_local_participants(
    participants: list[str],
    participant_cfg: dict[str, Any],
) -> dict[str, str]:
    """Quick probe of every selected local participant.

    Returns a mapping from participant name to a `PreflightFailed:` error
    string for every participant whose endpoint is unreachable. Hosted
    participants and any participant with `pre_flight_check: false` are
    skipped silently.

    Probes run concurrently — total wall time is bounded by the single-
    probe timeout, not the participant count. Pre-flight is best-effort:
    if the probe library itself raises an unexpected error, we let the
    real run path report it.
    """
    candidates = [
        (name, participant_cfg.get(name) or {})
        for name in participants
    ]
    # Default-on for loopback (`127.0.0.1`, `localhost`, `[::1]`)
    # where a 1s timeout is reasonable. Default-off for RFC1918 (`10.x`,
    # `192.168.x`, `172.16-31.x`) where a homelab/VPN endpoint might
    # legitimately take longer to respond. Users wanting to ping their LAN
    # vLLM can opt in with `pre_flight_check: true`. Users wanting to skip
    # an unreliable loopback endpoint can opt out with `pre_flight_check:
    # false`.
    todo = []
    for name, cfg in candidates:
        participant_type = cfg.get("type")
        if participant_type not in {"ollama", "openai_compatible"}:
            continue
        base_url = str(
            cfg.get("base_url")
            or ("http://localhost:11434" if participant_type == "ollama" else "")
        )
        # This operational probe may also be explicitly enabled for a LAN
        # inference server, so it intentionally uses the broader on-prem
        # classifier. `private-local` selection uses is_local_participant's
        # stricter loopback-only boundary instead.
        if not is_local_base_url(base_url):
            continue
        opted_in = cfg.get("pre_flight_check")  # tri-state: True / False / None
        is_loopback = is_loopback_base_url(base_url)
        if opted_in is False:
            continue  # explicit opt-out always wins
        if opted_in is True:
            todo.append((name, cfg))
            continue
        # opted_in is None — use the default policy
        if is_loopback:
            todo.append((name, cfg))
    if not todo:
        return {}
    results = await asyncio.gather(
        *(_preflight_one(name, cfg) for name, cfg in todo),
        return_exceptions=False,
    )
    return {
        name: error
        for (name, _cfg), error in zip(todo, results)
        if error is not None
    }


def _synth_preflight_failure(
    name: str, error: str, *, model: str | None = None
) -> ParticipantResult:
    """Construct a ParticipantResult that mirrors what a failed run looks like.

    Synthesizing this here (rather than letting the participant attempt to
    run and fail at the timeout) keeps the failure visible early and
    uses our explicit `preflight_failed` error_kind instead of the
    catch-all `downstream_error`. Carrying `model` through means
    transcripts and the summary table still identify which model was
    targeted, even though no call was made.
    """
    return ParticipantResult(
        name=name,
        ok=False,
        output="",
        error=error,
        elapsed_seconds=0.0,
        model=model,
    )


def _resolve_convergence_thresholds(
    config: dict[str, Any], mode: str | None
) -> dict[str, float]:
    defaults = config.get("defaults", {}) or {}
    base = defaults.get("convergence_thresholds") if isinstance(defaults, dict) else None
    override: dict[str, float] | None = None
    if mode:
        modes = config.get("modes", {}) or {}
        mode_cfg = modes.get(mode) or {}
        if isinstance(mode_cfg, dict):
            override = mode_cfg.get("convergence_thresholds")
    merged: dict[str, float] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return resolve_thresholds(merged or None)


def _base_name(name: str) -> str:
    return name.split(":round")[0]


def _is_labeled_vote(r: ParticipantResult) -> bool:
    return r.ok and recommendation_label(r.output) in {"yes", "no", "tradeoff"}


def _index_by_base_name(
    results: list[ParticipantResult],
) -> dict[str, ParticipantResult]:
    return {_base_name(result.name): result for result in results}


def _compute_round_convergence(
    prior: list[ParticipantResult],
    current: list[ParticipantResult],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    prior_index = _index_by_base_name(prior)
    records: list[dict[str, Any]] = []
    for result in current:
        base = _base_name(result.name)
        prior_result = prior_index.get(base)
        if prior_result is None or not prior_result.ok or not result.ok:
            continue
        prior_tokens = tokenize(prior_result.output or "")
        current_tokens = tokenize(result.output or "")
        token_floor = min(len(prior_tokens), len(current_tokens))
        if token_floor < MIN_TOKENS_FOR_CLASSIFICATION:
            records.append(
                {
                    "participant": base,
                    "similarity": None,
                    "state": "insufficient",
                    "prior_tokens": len(prior_tokens),
                    "current_tokens": len(current_tokens),
                }
            )
            continue
        similarity = jaccard_similarity(prior_tokens, current_tokens)
        state = classify(similarity, thresholds)
        records.append(
            {
                "participant": base,
                "similarity": round(similarity, 4),
                "state": state,
            }
        )
    return records


def _failed_for_deliberation(results: list[ParticipantResult]) -> set[str]:
    excluded: set[str] = set()
    for result in results:
        if result.ok:
            continue
        if is_timeout_error(result.error) or result.error.startswith("PromptTooLarge:"):
            excluded.add(_base_name(result.name))
    return excluded


def _detect_quota_throttled(
    results: list[ParticipantResult],
    participant_cfg: dict[str, Any],
    *,
    already_emitted: set[str],
) -> list[dict[str, Any]]:
    """Find newly quota-throttled peers in this round's results.

    Returns one record per peer whose error matches a quota-exhausted
    pattern AND whose base name is not yet in ``already_emitted``. The
    caller appends the base names it acts on to ``already_emitted`` so a
    peer throttled in round 1 doesn't produce a duplicate event in
    round 2's pass.
    """
    new_entries: list[dict[str, Any]] = []
    for result in results:
        if result.ok or not result.error:
            continue
        if not is_quota_exhausted_error(result.error):
            continue
        base = _base_name(result.name)
        if base in already_emitted:
            continue
        cfg = participant_cfg.get(base) or {}
        new_entries.append(
            {
                "peer": base,
                "family": cfg.get("family") or base,
                "model": result.model or cfg.get("model"),
                # Trim the raw error to one line for the metadata summary —
                # the full string is still on ParticipantResult.error for
                # consumers who want it.
                "message": result.error.splitlines()[0][:200],
            }
        )
    return new_entries


def _drop_missing_key_participants(
    participants: list[str],
    participant_cfg: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Skip hosted peers whose `api_key_env` env var is unset.

    Returns ``(active_participants, dropped_records)``. Each dropped
    record carries the peer name, family, and the env var name that was
    missing so the orchestrator can emit a `peer_missing_api_key`
    progress event AND so the operator sees what to set.

    v0.12.0 behavior: dropped peers are removed from the run roster
    BEFORE the quorum denominator is computed, so a missing key on one
    peer cannot flip the entire run to `degraded`. A run that ends with
    one hosted peer missing its key looks identical (for degraded /
    min_quorum purposes) to a run that never listed that peer at all.
    The dropped list still surfaces in metadata as `missing_key_peers`
    so the situation isn't invisible.

    Only `type: openrouter` and `type: openai_compatible` participants
    are checked here. Both adapters default an omitted ``api_key_env`` to
    ``OPENROUTER_API_KEY``. CLI peers authenticate via the host CLI's own
    session; Ollama has no adapter-level key contract. Unknown types are left
    alone (the run-time adapter will surface its own error).
    """
    active: list[str] = []
    dropped: list[dict[str, Any]] = []
    for name in participants:
        cfg = participant_cfg.get(name) or {}
        ptype = cfg.get("type")
        if ptype not in {"openrouter", "openai_compatible"}:
            active.append(name)
            continue
        key_env = participant_api_key_env(cfg)
        assert key_env is not None  # guarded by ptype above
        if env_get(key_env):
            active.append(name)
            continue
        dropped.append(
            {
                "peer": name,
                "family": cfg.get("family") or name,
                "api_key_env": key_env,
            }
        )
    return active, dropped


def _detect_quota_recoveries(
    results: list[ParticipantResult],
    participant_cfg: dict[str, Any],
    *,
    already_emitted: set[str],
) -> list[dict[str, Any]]:
    """Find newly recovered peers (quota hit → fallback succeeded) in this round.

    Mirror of `_detect_quota_throttled` for the success case. Distinct
    `already_emitted` set from the throttled path so a peer that
    recovered in round 1 then re-failed in round 2 emits BOTH a
    recovery event (round 1) and a throttle event (round 2)."""
    new_entries: list[dict[str, Any]] = []
    for result in results:
        if not result.recovered_after_quota:
            continue
        base = _base_name(result.name)
        if base in already_emitted:
            continue
        cfg = participant_cfg.get(base) or {}
        new_entries.append(
            {
                "peer": base,
                "family": cfg.get("family") or base,
                "fallback_model": result.model_fallback_used,
                "model": result.model,
            }
        )
    return new_entries


def apply_contextual_persona_recruitment(
    participants: list[str],
    participant_cfg: dict[str, Any],
    cwd: Path,
    *,
    stances: dict[str, str] | None = None,
) -> str | None:
    """Apply the changed-file persona assignment used by runtime prompts.

    The function deliberately mutates ``participant_cfg`` just as the
    historical inline implementation did. Preflight callers can invoke it
    before constructing per-peer prompts, and :func:`execute_council` invokes
    it again harmlessly so every entry point shares the same recruitment and
    target-selection rules.

    Returns the participant receiving the persona, or ``None`` when the diff
    does not match a contextual specialty (or no selected config is usable).
    """

    changed_files: list[str] = []
    from llm_council.context import _git_output

    try:
        git_staged = _git_output(cwd, ["diff", "--cached", "--name-only"])
        if git_staged:
            changed_files.extend(git_staged.splitlines())
        git_unstaged = _git_output(cwd, ["diff", "--name-only"])
        if git_unstaged:
            changed_files.extend(git_unstaged.splitlines())
    except Exception:
        return None

    persona: str | None = None
    persona_prompt: str | None = None
    for filename in changed_files:
        lowered = filename.lower()
        if any(
            marker in lowered
            for marker in (".sql", "db/", "migrations/", "models.py")
        ):
            persona = "database_architect"
            persona_prompt = (
                "Role: DATABASE ARCHITECT. Focus on schema design, query "
                "efficiency, indexes, migration safety, race conditions, and "
                "transaction safety."
            )
            break
        if any(
            marker in lowered
            for marker in (
                "auth",
                "login",
                "security",
                "perm",
                "crypt",
                ".env",
                "key",
            )
        ):
            persona = "security_auditor"
            persona_prompt = (
                "Role: SECURITY AUDITOR. Focus on vulnerability detection, "
                "authentication, input validation, encryption, secrets "
                "leakage, and authorization bypasses."
            )
            break
        if any(
            marker in lowered
            for marker in (".css", ".html", ".scss", "styles/", "components/")
        ):
            persona = "frontend_specialist"
            persona_prompt = (
                "Role: FRONTEND & UX SPECIALIST. Focus on semantic HTML, "
                "accessibility (a11y), responsive styling, layout shifts, "
                "bundle size, and browser compatibility."
            )
            break
        if any(
            marker in lowered
            for marker in (
                "dockerfile",
                "workflow",
                ".github",
                "yaml",
                "yml",
                "toml",
            )
        ):
            persona = "devops_engineer"
            persona_prompt = (
                "Role: DEVOPS & CI/CD ENGINEER. Focus on build pipelines, "
                "container safety, environment variable management, "
                "dependencies, resource limits, and deployment sanity."
            )
            break

    if not persona or not persona_prompt:
        return None

    target: str | None = None
    for name in participants:
        cfg = participant_cfg.get(name)
        if not isinstance(cfg, dict):
            continue
        assigned_stance = (
            stances.get(name)
            if isinstance(stances, dict) and name in stances
            else cfg.get("stance")
        )
        if assigned_stance in ("for", "against"):
            target = name
            break
    if target is None:
        target = next(
            (
                name
                for name in participants
                if isinstance(participant_cfg.get(name), dict)
            ),
            None,
        )
    if target is None:
        return None

    participant_cfg[target]["persona"] = persona
    participant_cfg[target]["persona_prompt"] = persona_prompt
    return target


async def execute_council(
    participants: list[str],
    participant_cfg: dict[str, Any],
    prompt: str,
    cwd: Path,
    config: dict[str, Any],
    *,
    deliberate: bool = False,
    max_rounds: int = 2,
    progress: Callable[[dict[str, Any]], None] | None = None,
    image_manifest: list[dict[str, Any]] | None = None,
    min_quorum: int | None = None,
    mode: str | None = None,
    cache_mode: str = "on",
    stances: dict[str, str] | None = None,
    synthesize: bool | None = None,
    synthesizer_name: str | None = None,
    current: str | None = None,
    question: str | None = None,
    cross_rank: bool = False,
    focus: list[Any] | None = None,
) -> tuple[list[ParticipantResult], dict[str, Any]]:
    run_started_monotonic = time.monotonic()
    run_started_at = _utc_progress_timestamp()
    max_concurrency = int(config.get("defaults", {}).get("max_concurrency") or 4)

    # Operator-authored review-focus bundles (review_skills.ReviewSkill).
    # Rendered once into an inert prompt directive appended to EVERY round
    # (round 1, the ranking pass, round-2 deliberation) so the focus
    # persists across rounds. When ``focus`` is None the directive is ""
    # and behavior is unchanged. Imported lazily to avoid an import cycle.
    focus_directive = ""
    if focus:
        from llm_council.review_skills import render_focus_directive

        focus_directive = render_focus_directive(focus)
    convergence_thresholds = _resolve_convergence_thresholds(config, mode)

    # Push run-wide validation toggles from config["defaults"] into each
    # participant cfg unless the participant already specifies its own
    # override. The adapter-level `_response_validation_error` reads off
    # `cfg`, not the global defaults block, so this propagation is what
    # makes the global `defaults.require_sections` / `defaults.strict_evidence`
    # actually affect peer responses.
    _defaults_for_peers = config.get("defaults") or {}
    for _propagated_key in ("require_sections", "strict_evidence"):
        _propagated_val = _defaults_for_peers.get(_propagated_key)
        if _propagated_val is None:
            continue
        for _peer_name in participants:
            _peer_cfg = participant_cfg.get(_peer_name)
            if isinstance(_peer_cfg, dict) and _propagated_key not in _peer_cfg:
                _peer_cfg[_propagated_key] = _propagated_val

    if stances:
        for _peer_name in participants:
            _peer_cfg = participant_cfg.get(_peer_name)
            if isinstance(_peer_cfg, dict) and _peer_name in stances:
                _peer_cfg["stance"] = stances[_peer_name]

    apply_contextual_persona_recruitment(
        participants,
        participant_cfg,
        cwd,
        stances=stances,
    )

    cache_disabled_for_mode = is_caching_disabled_for_mode(mode)
    cache_ctx_round1 = CacheContext(
        cwd=cwd,
        cache_mode=cache_mode,
        ttl_seconds=resolve_ttl_seconds(config, mode),
        cache_disabled=cache_disabled_for_mode,
    )
    cache_ctx_deliberation = CacheContext(
        cwd=cwd,
        cache_mode=cache_mode,
        ttl_seconds=resolve_ttl_seconds(config, mode),
        cache_disabled=True,
    )

    progress_events: list[dict[str, Any]] = []
    phase_starts: dict[tuple[Any, ...], float] = {
        ("council",): run_started_monotonic
    }

    def _phase_key(event: dict[str, Any], *, finish: bool) -> tuple[Any, ...] | None:
        kind = str(event.get("event") or "")
        if kind in {"participant_start", "participant_finish"}:
            return (
                "participant",
                event.get("participant"),
                event.get("round"),
            )
        if kind in {"council_start", "council_finish"}:
            return ("council",)
        if kind in {"cross_rank_start", "cross_rank_complete"}:
            return ("cross_rank",)
        if kind in {"synthesis_start", "synthesis_finish", "synthesis_error"}:
            return ("synthesis", event.get("chair"))
        if kind in {
            "deliberation_pending",
            "deliberation_round_start",
            "deliberation_finish",
        }:
            return ("deliberation",)
        return None

    def emit(event: dict[str, Any]) -> None:
        # Every event receives an absolute UTC timestamp plus elapsed time from
        # execute_council entry. Start/finish pairs additionally get a bounded
        # phase duration. Existing event-specific `elapsed_seconds` retains its
        # historical meaning (for example, participant attempt time).
        stamped = dict(event)
        now = time.monotonic()
        stamped.setdefault("timestamp", _utc_progress_timestamp())
        stamped.setdefault(
            "run_elapsed_seconds", round(now - run_started_monotonic, 3)
        )
        kind = str(stamped.get("event") or "")
        is_finish = kind.endswith("_finish") or kind in {
            "participant_finish",
            "cross_rank_complete",
            "synthesis_error",
        }
        phase_key = _phase_key(stamped, finish=is_finish)
        if phase_key is not None:
            if is_finish:
                phase_start = phase_starts.get(phase_key)
                if phase_start is not None:
                    stamped.setdefault(
                        "duration_seconds", round(now - phase_start, 3)
                    )
            else:
                phase_starts.setdefault(phase_key, now)
        progress_events.append(stamped)
        if progress:
            progress(stamped)

    # v0.12.0: drop hosted peers with missing api_key_env BEFORE preflight
    # / council_start, so the missing key never inflates the quorum
    # denominator. The dropped peers are surfaced in `missing_key_peers`
    # metadata + per-peer `peer_missing_api_key` events so the operator
    # sees the situation without it counting as degraded.
    participants, missing_key_records = _drop_missing_key_participants(
        participants, participant_cfg
    )

    # Resolve an opt-in synthesis chair before any peer is invoked. Restrict
    # resolution to the actual active roster (after missing-key drops) so a
    # configured-but-unselected peer cannot appear late as an unbudgeted call.
    # MCP may pass the already-preflighted concrete name; re-resolving here
    # verifies it still matches after runtime availability filtering.
    synthesize_flag = bool(
        config.get("defaults", {}).get("synthesize")
        if synthesize is None
        else synthesize
    )
    resolved_synthesizer: str | None = None
    if synthesize_flag:
        from llm_council.synthesis import select_synthesizer

        active_participant_cfg = {
            name: participant_cfg[name]
            for name in participants
            if name in participant_cfg
        }
        resolved_synthesizer = select_synthesizer(
            config,
            active_participant_cfg,
            stances=stances,
            current=current,
        )
        if (
            synthesizer_name is not None
            and synthesizer_name != resolved_synthesizer
        ):
            raise ValueError(
                "Preflight synthesizer no longer matches the active roster: "
                f"expected '{synthesizer_name}', resolved "
                f"'{resolved_synthesizer}'."
            )
    for record in missing_key_records:
        emit({"event": "peer_missing_api_key", **record})

    emit(
        {
            "event": "council_start",
            "participants": participants,
            "round": 1,
            "max_rounds": max_rounds,
            "deliberate": deliberate,
            "image_count": len(image_manifest or []),
        }
    )
    if image_manifest:
        for name in participants:
            cfg = participant_cfg.get(name, {})
            ptype = cfg.get("type")
            if ptype == "cli":
                continue
            if not cfg.get("vision"):
                emit(
                    {
                        "event": "images_skipped",
                        "participant": name,
                        "round": 1,
                        "reason": "non_vision",
                        "image_count": len(image_manifest),
                    }
                )
    # Pre-flight ping for local participants. Turns opaque "downstream_error"
    # at full timeout into a fast, legible "PreflightFailed: …" with the
    # base_url named, so the user sees what's actually wrong (server not
    # running, port wrong, model not loaded) without waiting through the
    # participant timeout.
    preflight_failures = await preflight_local_participants(
        participants, participant_cfg
    )
    if preflight_failures:
        for name, error in preflight_failures.items():
            emit(
                {
                    "event": "preflight_failed",
                    "participant": name,
                    "round": 1,
                    "error": error,
                }
            )
        run_targets = [name for name in participants if name not in preflight_failures]
    else:
        run_targets = participants

    # Mode-aware timeout: consensus/deliberate/diverse modes do harder work
    # and can blow through the default 240s. Each mode's `timeout_multiplier`
    # is layered on top of the per-participant base, so users who already
    # raised `claude.timeout` to 600s get the same multiplicative bump.
    mode_cfg_for_timeout = config.get("modes", {}).get(mode) if mode else None
    mode_multiplier = (
        float(mode_cfg_for_timeout.get("timeout_multiplier"))
        if isinstance(mode_cfg_for_timeout, dict)
        and mode_cfg_for_timeout.get("timeout_multiplier") is not None
        else None
    )
    # v0.9.0 Feature 3: opt-in tool-call voting flag, scoped to whichever
    # mode the operator enables it on (today only `review-with-tools`
    # ships the key; default `False`). Resolution mirrors
    # `timeout_multiplier` — peer-level config does NOT override the mode
    # flag.
    tool_call_voting = bool(
        isinstance(mode_cfg_for_timeout, dict)
        and mode_cfg_for_timeout.get("tool_call_voting")
    )
    # M-fable safe-context framing: resolved from the mode config here (like
    # `timeout_multiplier` / `tool_call_voting` above) rather than threaded in
    # by each caller — deriving it internally makes a caller-side desync
    # (round-1 prompt framed, ranking prompt bare) impossible. Callers still
    # resolve the same key themselves for `build_prompt`, which runs before
    # this function.
    safe_context = bool(
        isinstance(mode_cfg_for_timeout, dict)
        and mode_cfg_for_timeout.get("safe_context")
    )

    if run_targets:
        run_results = await run_participants(
            run_targets,
            participant_cfg,
            prompt,
            cwd,
            max_concurrency=max_concurrency,
            progress=emit,
            round_number=1,
            image_manifest=image_manifest,
            cache_ctx=cache_ctx_round1,
            mode_multiplier=mode_multiplier,
            mode=mode,
            tool_call_voting=tool_call_voting,
            focus_directive=focus_directive,
        )
        verify_evidence_citations(run_results, cwd)
    else:
        run_results = []
    # Merge pre-flight failures back in, preserving the original participant
    # order so transcripts and the summary table stay deterministic.
    by_name: dict[str, ParticipantResult] = {result.name: result for result in run_results}
    for name, error in preflight_failures.items():
        cfg = participant_cfg.get(name) or {}
        by_name[name] = _synth_preflight_failure(
            name, error, model=cfg.get("model")
        )
    results = [by_name[name] for name in participants if name in by_name]
    # `round_results` is a separate list from `results` so the v0.9.0
    # cross-rank pass (which appends ranking-round entries to `results`
    # via `results.extend(ranking_results)`) does NOT pollute the
    # round-1 view used by `has_disagreement` / `build_deliberation_prompt`.
    # Before v0.9.0 these were aliases; the cross-rank pass is the
    # only mutation site that requires the split.
    round_results = list(results)
    round_number = 1
    # Track peers that hit a quota wall during this run. Accumulated across
    # rounds (round 1 + optional deliberation) with per-peer dedup so a
    # throttled peer doesn't emit a duplicate event when round 2 re-fires
    # the same failure.
    quota_throttled: list[dict[str, Any]] = []
    quota_throttled_seen: set[str] = set()
    # Phase 2: parallel tracking for peers that hit quota but recovered
    # via the `fallback_chain` retry. Separate dedup set so a peer that
    # recovered round-1 and then re-failed round-2 emits BOTH events.
    quota_recoveries: list[dict[str, Any]] = []
    quota_recoveries_seen: set[str] = set()

    def _detect_and_emit_quota(round_outputs: list[ParticipantResult]) -> None:
        """Detect quota-throttled / recovered peers in ``round_outputs``,
        append them to the run-level accumulators (with per-peer dedup via
        the ``*_seen`` sets), and emit one progress event per new peer.

        Closes over the run-level accumulators / seen-sets and the current
        ``round_number`` so both the round-1 and round-2 sites stamp the
        right round. Reads ``round_number`` at call time, matching the
        original inline behavior (round 1 fires before the increment,
        round 2 after).
        """
        for entry in _detect_quota_throttled(
            round_outputs, participant_cfg, already_emitted=quota_throttled_seen
        ):
            quota_throttled.append(entry)
            quota_throttled_seen.add(entry["peer"])
            emit({"event": "peer_quota_throttled", "round": round_number, **entry})
        for entry in _detect_quota_recoveries(
            round_outputs, participant_cfg, already_emitted=quota_recoveries_seen
        ):
            quota_recoveries.append(entry)
            quota_recoveries_seen.add(entry["peer"])
            emit({"event": "peer_quota_recovered", "round": round_number, **entry})

    # Pinned-model substitutions (M-fable), mirrored on the quota pattern.
    # A `require_pinned_model` peer served by a different model (e.g. Claude
    # Fable 5 refused -> Claude Code silently fell back to Opus) drops with
    # `error_kind=model_substituted`; accumulate those across rounds + the
    # --cross-rank ranking pass with (peer, served_by) dedup. `served_by` is
    # the REAL model the CLI reported.
    model_substituted_peers: list[dict[str, Any]] = []
    substitution_seen: set[tuple[str, str | None]] = set()

    def _detect_and_emit_substitutions(
        round_outputs: list[ParticipantResult],
    ) -> None:
        """Detect pinned-model substitutions in ``round_outputs``, append them
        to the run-level accumulator (deduped on (peer, served_by)), and emit
        one progress event per new entry.

        Reads ``round_number`` at call time like ``_detect_and_emit_quota``,
        so each event carries the round the swap actually happened in and
        fires live rather than in an end-of-run scan.
        """
        for r in round_outputs:
            if classify_error(r.error) != ERROR_KIND_MODEL_SUBSTITUTED:
                continue
            # Strip both the ":roundN" deliberation suffix and the ":rank"
            # ranking-pass suffix back to the configured peer name.
            base = _base_name(r.name).split(":")[0]
            key = (base, r.model)
            if key in substitution_seen:
                continue
            substitution_seen.add(key)
            entry = {
                "peer": base,
                "requested": (participant_cfg.get(base, {}) or {}).get("model"),
                "served_by": r.model,
            }
            if getattr(r, "is_ranking_round", False):
                entry["ranking_round"] = True
            model_substituted_peers.append(entry)
            emit({"event": "peer_model_substituted", "round": round_number, **entry})

    _detect_and_emit_quota(results)
    _detect_and_emit_substitutions(results)
    initial_disagreement = has_disagreement(round_results)
    metadata = {
        "rounds": round_number,
        "max_rounds": max_rounds,
        "deliberation_requested": deliberate,
        "deliberated": False,
        "disagreement_detected": initial_disagreement,
        "final_disagreement_detected": initial_disagreement,
        "deliberation_status": "not_requested",
        "progress_events": progress_events,
    }
    # M11 provenance: record which focus bundles shaped this run (name +
    # short content hash). Omitted entirely when no focus was applied.
    if focus:
        metadata["applied_focus"] = [
            {"name": s.name, "sha256": s.sha256} for s in focus
        ]
    if deliberate:
        if not initial_disagreement:
            metadata["deliberation_status"] = "skipped_no_labeled_disagreement"
            emit(
                {
                    "event": "deliberation_skip",
                    "reason": "no_labeled_disagreement",
                    "round": round_number,
                }
            )
        elif max_rounds <= 1:
            metadata["deliberation_status"] = "skipped_max_rounds"
            emit({"event": "deliberation_skip", "reason": "max_rounds", "round": round_number})
        else:
            metadata["deliberation_status"] = "pending"
            emit({"event": "deliberation_pending", "round": round_number + 1})

    # Universal-abdication short-circuit: when every round-1 peer abdicates,
    # round 2 would re-abdicate against the same prompt for the same reasons.
    # Stamp the merged-blockers payload now; the while-loop guard below sees
    # `universal_abdication` and refuses to enter. `deliberation_status` only
    # changes when deliberation was actually on the table — for
    # `deliberate=False` runs the status stays `"not_requested"` so the
    # field is never misleading metadata.
    from llm_council.synthesis import universal_abdication as _universal_abdication

    _early_abdication = _universal_abdication(round_results)
    if _early_abdication:
        metadata["universal_abdication"] = _early_abdication
        if deliberate:
            metadata["deliberation_status"] = "skipped_universal_abdication"
        emit(
            {
                "event": "universal_abdication",
                "round": round_number,
                "blockers": _early_abdication["blockers"],
                "abdicated_peers": _early_abdication["abdicated_peers"],
            }
        )

    # v0.9.0 Feature 2 — Anonymized cross-ranking. Opt-in flag composable
    # with any mode. After round 1, peers with usable RECOMMENDATION
    # labels rank each other's anonymized outputs. The ranking outputs
    # are tagged `is_ranking_round=True` and intentionally NOT fed back
    # to round-2 deliberation (MAD literature risk — see
    # `deliberation.build_deliberation_prompt`). Skipped when:
    #   - the flag is off (default),
    #   - fewer than `CROSS_RANK_MIN_PEERS` peers produced usable labels
    #     (you cannot rank one response against itself),
    #   - the universal-abdication short-circuit fired (no signal to rank).
    if (
        cross_rank
        and not metadata.get("universal_abdication")
    ):
        from llm_council.context import (
            CROSS_RANK_MIN_PEERS,
            build_anonymization_map,
            build_ranking_prompt,
            compute_rank_position_means,
            parse_final_ranking,
        )

        labeled_results = [r for r in round_results if _is_labeled_vote(r)]
        if len(labeled_results) >= CROSS_RANK_MIN_PEERS:
            labeled_names = [r.name for r in labeled_results]
            anonymization_map = build_anonymization_map(labeled_names)
            metadata["anonymization_map"] = dict(anonymization_map)
            # Reverse map for de-anonymization: "A" -> "claude".
            metadata["anonymization_map_reverse"] = {
                full_label.replace("Response ", "").strip(): name
                for name, full_label in anonymization_map.items()
            }
            valid_label_set = {
                full_label.replace("Response ", "").strip()
                for full_label in anonymization_map.values()
            }
            response_by_peer = {r.name: r.output for r in labeled_results}
            ranking_question = question or prompt
            emit(
                {
                    "event": "cross_rank_start",
                    "round": round_number,
                    "peer_count": len(labeled_names),
                }
            )

            async def _rank_one(peer_name: str) -> ParticipantResult:
                ranking_prompt = build_ranking_prompt(
                    peer_name=peer_name,
                    own_response=response_by_peer.get(peer_name, ""),
                    other_peers={
                        name: text
                        for name, text in response_by_peer.items()
                        if name != peer_name
                    },
                    anonymization_map=anonymization_map,
                    question=ranking_question,
                )
                # Carry the operator's review focus into the ranking pass so
                # the scrutiny lens persists across rounds. The ranking pass
                # uses run_participant (singular) with a freshly-built prompt
                # rather than run_participants, so we append the inert focus
                # directive directly. No-op when focus is unset.
                if focus_directive:
                    ranking_prompt = ranking_prompt + "\n\n" + focus_directive
                # Same persistence rule for the defensive-review framing: the
                # ranking prompt quotes other peers' (possibly security-heavy)
                # findings verbatim, making it the most refusal-prone request
                # of the run — without the framing, a safe_context mode's
                # ranking turn would be MORE likely to trip the pinned peer's
                # refusal fallback than round 1 was. No-op when unset.
                if safe_context:
                    from llm_council.context import SAFE_CONTEXT_DIRECTIVE

                    ranking_prompt = (
                        SAFE_CONTEXT_DIRECTIVE + "\n\n" + ranking_prompt
                    )
                from llm_council.adapters import run_participant

                # Ranking pass intentionally bypasses tool-call voting:
                # the response shape is `FINAL RANKING:`, not a
                # `record_recommendation` tool call. We also turn off
                # cache writes for ranking-round results (cache_ctx
                # mirrors `cache_ctx_deliberation`) to avoid cross-run
                # bleed when the operator re-runs without --cross-rank.
                cfg_for_peer = participant_cfg.get(peer_name) or {}
                # Disable label/section/strict-evidence validation for
                # the ranking pass: the response is intentionally a
                # one-line FINAL RANKING with no RECOMMENDATION envelope.
                ranking_cfg = dict(cfg_for_peer)
                ranking_cfg["require_recommendation"] = False
                ranking_cfg["require_sections"] = False
                ranking_cfg["strict_evidence"] = False
                rank_result = await run_participant(
                    peer_name,
                    ranking_cfg,
                    ranking_prompt,
                    cwd,
                    image_manifest=None,
                    cache_ctx=cache_ctx_deliberation,
                    mode_multiplier=mode_multiplier,
                    mode=mode,
                    tool_call_voting=False,
                )
                return replace(
                    rank_result,
                    name=f"{peer_name}:rank",
                    is_ranking_round=True,
                )

            # The ranking pass launches one run_participant (subprocess / HTTP
            # call) per labeled peer. Cap it with the same max_concurrency the
            # primary rounds use so a wide council doesn't fan out N unbounded
            # subprocesses (each of which can itself walk the quota fallback
            # chain). return_exceptions keeps a single ranking failure from
            # aborting the whole council run — cross-rank is post-deliberation
            # telemetry, so a peer that errors here simply casts no ranking vote.
            rank_semaphore = asyncio.Semaphore(max(1, max_concurrency))

            async def _rank_one_guarded(peer_name: str) -> ParticipantResult:
                async with rank_semaphore:
                    return await _rank_one(peer_name)

            ranking_results_raw = await asyncio.gather(
                *[_rank_one_guarded(name) for name in labeled_names],
                return_exceptions=True,
            )
            ranking_results = [
                r for r in ranking_results_raw if isinstance(r, ParticipantResult)
            ]
            for raw in ranking_results_raw:
                if isinstance(raw, BaseException):
                    emit(
                        {
                            "event": "cross_rank_peer_error",
                            "round": round_number,
                            "error": f"{type(raw).__name__}: {raw}",
                        }
                    )
            results.extend(ranking_results)
            _detect_and_emit_substitutions(ranking_results)

            rankings_by_peer: dict[str, list[str]] = {}
            for r in ranking_results:
                base = r.name.split(":rank", 1)[0]
                if not r.ok:
                    continue
                # Each peer's valid labels = ALL bare labels MINUS its own.
                own_label = (
                    anonymization_map.get(base, "")
                    .replace("Response ", "")
                    .strip()
                )
                peer_valid = {lbl for lbl in valid_label_set if lbl != own_label}
                parsed = parse_final_ranking(r.output, peer_valid)
                if parsed is not None:
                    rankings_by_peer[base] = parsed

            cross_rank_scores = compute_rank_position_means(
                anonymization_map, rankings_by_peer
            )
            if cross_rank_scores:
                metadata["cross_rank_scores"] = cross_rank_scores
            metadata["cross_rank_rankings"] = rankings_by_peer
            emit(
                {
                    "event": "cross_rank_complete",
                    "round": round_number,
                    "scores": cross_rank_scores,
                    "ranker_count": len(rankings_by_peer),
                }
            )

    # v0.8.1 CONTINUE_DEBATE unanimity short-circuit. After round 1, if
    # every label-producing peer voted ``CONTINUE_DEBATE: no``, skip the
    # optional round-2 deliberation. Denominator MIRRORS the abdication
    # exclusions used elsewhere: peers without a usable RECOMMENDATION
    # label (no labeled vote, abdicated, invalid_response, etc.) are
    # excluded from BOTH numerator and denominator. The ``>= 2`` floor
    # avoids a degenerate single-peer council voting itself out of
    # deliberation. Unanimity (not 66%) per council recommendation —
    # conservative until corpus data audits gaming risk. Only fires when
    # deliberation was actually on the table (pending status).
    continue_debate_unanimous_no = False
    if (
        deliberate
        and metadata.get("deliberation_status") == "pending"
        and not metadata.get("universal_abdication")
    ):
        denominator = [r for r in round_results if _is_labeled_vote(r)]
        no_votes = sum(
            1 for r in denominator
            if (r.continue_debate or "").lower() == "no"
        )
        if no_votes >= len(denominator) and len(denominator) >= 2:
            continue_debate_unanimous_no = True
            metadata["deliberation_status"] = "skipped_continue_debate_unanimous"
            emit(
                {
                    "event": "deliberation_skipped",
                    "reason": "continue_debate_unanimous",
                    "no_votes": no_votes,
                    "denominator": len(denominator),
                }
            )

    # A peer that failed the local-server preflight was never launched in
    # round 1. Keep it out of every later round as well: rebuilding the roster
    # from the original participant list would otherwise silently reintroduce
    # it during deliberation and pay the full participant timeout for the same
    # endpoint that already failed fast.
    cumulative_excluded: set[str] = set(preflight_failures)
    aborted_all_excluded = False
    early_stopped = False
    convergence_by_round: dict[int, list[dict[str, Any]]] = {}
    deliberation_prompts: dict[int, str] = {}
    # No-new-movement early-stop (opt-in, default OFF). A mode-explicit value
    # overrides the default with None-aware precedence (so an explicit `false`
    # on a mode can disable a `true` default — not an `or` chain).
    early_stop_enabled = (
        (config.get("modes", {}) or {}).get(mode or "", {}) or {}
    ).get("deliberation_early_stop")
    if early_stop_enabled is None:
        early_stop_enabled = (config.get("defaults", {}) or {}).get(
            "deliberation_early_stop"
        )
    while (
        deliberate
        and max_rounds > round_number
        and has_disagreement(round_results)
        and not metadata.get("universal_abdication")
        and not continue_debate_unanimous_no
    ):
        cumulative_excluded.update(_failed_for_deliberation(round_results))
        deliberation_participants = [
            name for name in participants if name not in cumulative_excluded
        ]
        if cumulative_excluded:
            emit(
                {
                    "event": "deliberation_skip_participants",
                    "round": round_number + 1,
                    "skipped": sorted(cumulative_excluded),
                    "reason": (
                        "preflight_failed_or_timed_out_or_prompt_too_large"
                        if preflight_failures
                        else "timed_out_or_prompt_too_large"
                    ),
                }
            )
        if not deliberation_participants:
            metadata["deliberation_status"] = "skipped_all_excluded"
            aborted_all_excluded = True
            emit(
                {
                    "event": "deliberation_skip",
                    "reason": "no_remaining_participants",
                    "round": round_number + 1,
                }
            )
            break
        next_prompt, truncated_peers = build_deliberation_prompt(prompt, round_results)
        deliberation_prompts[round_number + 1] = next_prompt
        for peer_name in truncated_peers:
            emit(
                {
                    "event": "truncated_for_deliberation",
                    "round": round_number + 1,
                    "participant": peer_name,
                }
            )
        emit({"event": "deliberation_round_start", "round": round_number + 1})
        next_results = await run_participants(
            deliberation_participants,
            participant_cfg,
            next_prompt,
            cwd,
            max_concurrency=max_concurrency,
            progress=emit,
            round_number=round_number + 1,
            image_manifest=image_manifest,
            cache_ctx=cache_ctx_deliberation,
            mode_multiplier=mode_multiplier,
            mode=mode,
            tool_call_voting=tool_call_voting,
            focus_directive=focus_directive,
        )
        verify_evidence_citations(next_results, cwd)
        prior_round_results = list(round_results)
        round_number += 1
        round_results = [
            replace(result, name=f"{result.name}:round{round_number}")
            for result in next_results
        ]
        results.extend(round_results)
        metadata["rounds"] = round_number
        metadata["deliberated"] = True
        # Round-2 quota detection. Skips peers already in
        # `quota_throttled_seen` from round 1, so a peer throttled once
        # doesn't emit a second event when round 2 hits the same wall.
        _detect_and_emit_quota(round_results)
        _detect_and_emit_substitutions(round_results)

        round_convergence = _compute_round_convergence(
            prior_round_results, round_results, convergence_thresholds
        )
        if round_convergence:
            convergence_by_round[round_number] = round_convergence
            for record in round_convergence:
                emit(
                    {
                        "event": "convergence",
                        "round": round_number,
                        "participant": record["participant"],
                        "state": record["state"],
                        "similarity": record["similarity"],
                    }
                )

        # No-new-movement early-stop. Requires BOTH a non-diverging
        # convergence signal AND an unchanged vote tally — a "converged"
        # similarity can co-exist with a still-split vote, so the tally
        # comparison is the required corroboration of the Jaccard signal.
        # Only meaningful when a FURTHER round would actually run — i.e.
        # `round_number < max_rounds`. On a `max_rounds=2` run the loop is
        # about to exit anyway after this single deliberation round, so
        # firing here would relabel a normally-completed run as
        # `stopped_no_new_movement` without skipping anything (codex WU9
        # review). Gating on a remaining round keeps the "deep-audit
        # (max_rounds>=3) only" contract honest.
        if early_stop_enabled and round_number < max_rounds:
            no_divergence = not any(
                (rec.get("state") == "diverging")
                for rec in (round_convergence or [])
            )
            curr_counts = recommendation_counts(round_results)
            tally_unchanged = (
                recommendation_counts(prior_round_results) == curr_counts
            )
            if no_divergence and tally_unchanged:
                early_stopped = True
                emit(
                    {
                        "event": "deliberation_early_stop",
                        "round": round_number,
                        "reason": "no_new_movement",
                        "counts": curr_counts,
                    }
                )
                break

    if metadata["deliberated"] and not aborted_all_excluded:
        final_disagreement = has_disagreement(round_results)
        metadata["final_disagreement_detected"] = final_disagreement
        if early_stopped:
            metadata["deliberation_status"] = "stopped_no_new_movement"
        else:
            metadata["deliberation_status"] = (
                "ran_max_rounds_unresolved"
                if final_disagreement
                else "ran_no_labeled_disagreement"
            )
        emit(
            {
                "event": "deliberation_finish",
                "rounds": metadata["rounds"],
                "status": metadata["deliberation_status"],
            }
        )
    elif aborted_all_excluded:
        emit(
            {
                "event": "deliberation_finish",
                "rounds": metadata["rounds"],
                "status": metadata["deliberation_status"],
            }
        )

    if convergence_by_round:
        metadata["convergence"] = {
            str(round_no): records for round_no, records in sorted(convergence_by_round.items())
        }
        metadata["convergence_thresholds"] = convergence_thresholds

    if deliberation_prompts:
        metadata["deliberation_prompts"] = {
            str(round_no): text
            for round_no, text in sorted(deliberation_prompts.items())
        }

    effective_min_quorum = (
        int(min_quorum) if min_quorum is not None else default_min_quorum(len(participants))
    )
    effective_min_quorum = max(1, effective_min_quorum)
    final_labeled = labeled_quorum_count(round_results)
    degraded = final_labeled < effective_min_quorum
    metadata["min_quorum"] = effective_min_quorum
    metadata["labeled_quorum"] = final_labeled
    metadata["degraded"] = degraded
    if degraded:
        emit(
            {
                "event": "degraded_consensus",
                "labeled_quorum": final_labeled,
                "min_quorum": effective_min_quorum,
                "participant_count": len(participants),
                "round": metadata["rounds"],
            }
        )

    # --- Independence warning (H2) ----------------------------------------
    # OPTIONAL, advisory-only signal: when every labeled vote in the final
    # round comes from the same vendor family, correlated same-vendor
    # agreement can masquerade as independent corroboration. We surface a
    # warning; we do NOT drop a peer or touch quorum/degraded. Default OFF
    # (threshold unset). NEVER overload `metadata["degraded"]`.
    distinct_vendor_threshold = (
        (config.get("modes", {}) or {}).get(mode or "", {}) or {}
    ).get("require_distinct_vendors")
    if distinct_vendor_threshold is None:
        distinct_vendor_threshold = (config.get("defaults", {}) or {}).get(
            "min_distinct_vendors"
        )
    if distinct_vendor_threshold is not None:
        labeled = [r for r in round_results if _is_labeled_vote(r)]
        families = sorted(
            {
                (participant_cfg.get(_base_name(r.name), {}) or {}).get("family")
                or _base_name(r.name)
                for r in labeled
            }
        )
        distinct = len(families)
        # Only warn when there is actual labeled agreement whose vendor
        # diversity is in question. With zero labeled votes there is no
        # consensus to mistake for independent corroboration (the run is
        # already `degraded`), so a "single-vendor" warning there is a false
        # signal — codex review, WU2.
        if labeled and distinct < int(distinct_vendor_threshold):
            metadata["independence_warning"] = {
                "distinct_vendors": distinct,
                "required": int(distinct_vendor_threshold),
                "families": families,
                "labeled_quorum": final_labeled,
            }
            emit(
                {
                    "event": "single_vendor_quorum",
                    "distinct_vendors": distinct,
                    "required": int(distinct_vendor_threshold),
                    "families": families,
                    "round": metadata["rounds"],
                }
            )

    if stances:
        metadata["stances"] = dict(stances)
        for idx, result in enumerate(results):
            base_name = _base_name(result.name)
            assigned = stances.get(base_name)
            if assigned is not None:
                results[idx] = replace(result, stance=assigned)

    # --- Synthesis chair (Pick B) -----------------------------------------
    # Runs at most once per council run. Chair output is metadata; the
    # headline `recommendation` (computed in mcp_server.run_council / cli)
    # stays derived from peer votes only — see synthesis.run_synthesis_chair.
    from llm_council.synthesis import (
        run_synthesis_chair,
        should_synthesize,
    )
    from llm_council.transcript import final_round_results
    from llm_council.findings import build_matrix_from_results, matrix_to_dict

    # The chair sees ONLY final-round peer outputs. After deliberation,
    # `results` contains both round-1 entries (plain names) and round-2
    # entries (`:round2` suffix); `final_round_results()` returns the
    # latest round only. Passing the full cumulative list would feed the
    # chair stale positions the peers have since moved off.
    #
    # Build the per-finding agreement matrix ONCE over the final-round
    # results (Phase F). The matrix is post-deliberation, post-results-
    # merge by design — peers in round 2 must NEVER see it, or we recreate
    # the convergence-forcing pattern MAD literature warns against.
    # Compute `final_round_results(results)` once and share between the
    # finding-matrix pass and the synthesis chair below. Both consumers
    # want the same view: only the latest-round entries per peer.
    final_results = final_round_results(results)
    # Exclude every failed result from the matrix input. In particular, a
    # pinned-model integrity failure retains captured output for audit, but
    # that text must never be attributed to the configured peer as a finding.
    # Deliberation and synthesis already skip not-ok results; the matrix
    # builder does not, hence the explicit filter here.
    finding_matrix = build_matrix_from_results([r for r in final_results if r.ok])
    if not finding_matrix.is_empty():
        metadata["finding_matrix"] = matrix_to_dict(finding_matrix)

    # Surface quota-throttled peers as a top-level metadata field so the
    # transcript JSON + MCP `structured_results` can lift them out of the
    # per-result `error` strings. Omit the key entirely when empty so the
    # common case (no quota issues) leaves the schema unchanged.
    if quota_throttled:
        metadata["quota_throttled_peers"] = list(quota_throttled)
    if quota_recoveries:
        metadata["quota_recoveries"] = list(quota_recoveries)
    if missing_key_records:
        metadata["missing_key_peers"] = list(missing_key_records)

    if synthesize_flag and should_synthesize(synthesize_flag, metadata):
        assert resolved_synthesizer is not None
        chair_name = resolved_synthesizer
        chair_preflight_error = preflight_failures.get(chair_name)
        if chair_preflight_error is not None:
            # A local chair that failed the same endpoint preflight as its
            # round-1 vote must never be silently reintroduced as a second,
            # full-timeout synthesis invocation. Keep the preselected chair
            # (and therefore the preflight budget) stable; do not pick a
            # replacement after peer calls have started.
            synthesis_error = (
                f"Synthesis chair '{chair_name}' failed participant preflight "
                f"and was not invoked: {chair_preflight_error}"
            )
            metadata["synthesis_error"] = synthesis_error
            emit(
                {
                    "event": "synthesis_error",
                    "chair": chair_name,
                    "reason": "preflight_failed",
                    "error": synthesis_error,
                }
            )
        else:
            try:
                emit({"event": "synthesis_start", "chair": chair_name})
                convergence_for_chair = metadata.get("convergence")
                synthesis_payload = await run_synthesis_chair(
                    question=(question or prompt),
                    results=final_results,
                    convergence=convergence_for_chair,
                    participant_cfg=participant_cfg,
                    cwd=cwd,
                    chair_name=chair_name,
                    finding_matrix=finding_matrix,
                )
                metadata["synthesis"] = synthesis_payload
                # The chair turn never enters `results`, so the per-round
                # substitution detector cannot see it — scan its payload here.
                # A substituted chair memo (e.g. Fable refused the synthesis
                # prompt and Opus served it) is flagged on the payload and
                # surfaced through the same channel as in-round substitutions
                # so it is never consumed as the pinned chair's memo.
                if (
                    classify_error(str(synthesis_payload.get("error") or ""))
                    == ERROR_KIND_MODEL_SUBSTITUTED
                ):
                    synthesis_payload["model_substituted"] = True
                    chair_entry = {
                        "peer": chair_name,
                        "requested": (
                            participant_cfg.get(chair_name, {}) or {}
                        ).get("model"),
                        "served_by": synthesis_payload.get("model"),
                        "synthesis": True,
                    }
                    model_substituted_peers.append(chair_entry)
                    emit(
                        {
                            "event": "peer_model_substituted",
                            "round": metadata["rounds"],
                            **chair_entry,
                        }
                    )
                emit(
                    {
                        "event": "synthesis_finish",
                        "chair": chair_name,
                        "ok": synthesis_payload.get("ok"),
                        "decision_label": synthesis_payload.get("decision_label"),
                    }
                )
            except ValueError as exc:
                # Configuration/roster errors were already rejected before peer
                # launch. A ValueError raised while constructing or invoking the
                # chair happens after valid peer votes, so preserve those votes
                # and surface only the synthesis failure as metadata.
                metadata["synthesis_error"] = str(exc)
                emit({"event": "synthesis_error", "error": str(exc)})

    # Surface pinned-model substitutions (M-fable) top-level, mirroring
    # `quota_throttled_peers`. Entries were accumulated live per round
    # (round 1, the --cross-rank ranking pass, round-2 deliberation) plus
    # the synthesis-chair scan above; lift them out of the per-result error
    # strings so the transcript JSON / MCP `structured_results` show the
    # swap plainly. Omit the key entirely when empty so the common case
    # leaves the schema unchanged.
    if model_substituted_peers:
        metadata["model_substituted_peers"] = model_substituted_peers

    run_wall_elapsed = time.monotonic() - run_started_monotonic
    run_finished_at = _utc_progress_timestamp()
    participant_elapsed_aggregate = sum(
        float(result.elapsed_seconds) for result in results
    )
    participant_wall_elapsed_aggregate = sum(
        float(
            result.wall_elapsed_seconds
            if result.wall_elapsed_seconds is not None
            else result.elapsed_seconds
        )
        for result in results
    )
    metadata.update(
        {
            "run_started_at": run_started_at,
            "run_finished_at": run_finished_at,
            "run_wall_elapsed_seconds": round(run_wall_elapsed, 6),
            "participant_elapsed_seconds_aggregate": round(
                participant_elapsed_aggregate, 6
            ),
            "participant_wall_elapsed_seconds_aggregate": round(
                participant_wall_elapsed_aggregate, 6
            ),
        }
    )
    emit(
        {
            "event": "council_finish",
            "rounds": metadata["rounds"],
            "ok": sum(1 for result in results if result.ok),
            "total": len(results),
            "timestamp": run_finished_at,
            "run_elapsed_seconds": round(run_wall_elapsed, 3),
            "duration_seconds": round(run_wall_elapsed, 3),
            "participant_elapsed_seconds_aggregate": round(
                participant_elapsed_aggregate, 3
            ),
            "participant_wall_elapsed_seconds_aggregate": round(
                participant_wall_elapsed_aggregate, 3
            ),
        }
    )

    return results, metadata
