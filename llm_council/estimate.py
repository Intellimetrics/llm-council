"""Council prompt and hosted-model cost estimates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping

from llm_council.budget import (
    ESTIMATED_CHARS_PER_TOKEN,
    image_attachment_violations,
)
from llm_council.config import (
    balance_stances,
    canonical_mode_name,
    is_local_participant,
    select_participants,
)
from llm_council.context import (
    CROSS_RANK_MIN_PEERS,
    MAX_PROMPT_CHARS,
    SAFE_CONTEXT_DIRECTIVE,
    apply_per_peer_directives,
    build_anonymization_map,
    build_image_manifest,
    build_prompt,
    build_ranking_prompt,
)
from llm_council.deliberation import MAX_DELIBERATION_PROMPT_CHARS
from llm_council.model_catalog import fetch_openrouter_models
from llm_council.synthesis import MAX_SYNTHESIS_PROMPT_CHARS_DEFAULT

# litellm is an OPTIONAL pricing fallback for hosted peers whose model id
# is missing from the OpenRouter catalog. It is NOT a hard dependency — when
# it isn't installed (the default), `_litellm_price_per_million` returns
# (None, None) and the hosted peer stays unpriced exactly as before. No
# network is ever hit: only litellm's bundled local `model_cost` map is read.
try:  # pragma: no cover - import guard exercised by absence in default env
    import litellm
except ImportError:  # pragma: no cover - the default env has no litellm
    litellm = None


IMAGE_TOKEN_HEURISTIC = 1500
DEFAULT_COMPLETION_TOKENS = 1500

# L7 cost_class heuristic thresholds (USD). Derived from
# `known_total_with_retry_safety_usd` so the class reflects worst-case spend.
# Advisory only — never gates a run. Thresholds are deliberately coarse:
# sub-nickel runs are "low", sub-50-cents "moderate", anything more "high".
COST_CLASS_LOW_MAX_USD = 0.05
COST_CLASS_MODERATE_MAX_USD = 0.50

# Load-bearing sentinel for the per-participant estimate row's `model` field.
# Native CLI peers ship `model: None` (the engine cannot observe which concrete
# model a CLI ran — the "Per-CLI model identity is not observable" invariant),
# so the estimate row substitutes this constant. It is PRODUCED here and
# COMPARED in cli.py (`cmd_estimate`) to decide whether to render the peer NAME
# instead of a model id in the estimate table — keep the two in sync via this
# single constant. Must never be None (the row's `.endswith(":free")` check and
# the cli.py equality compare both depend on it being a real string).
CLI_DEFAULT_MODEL_LABEL = "cli default"


def estimate_tokens(text: str) -> int:
    """Approximate token count for a prompt string.

    Uses ``len(text) / 4`` rounded up, matching ``ESTIMATED_CHARS_PER_TOKEN``
    elsewhere. This is a coarse lower-bound suitable for budget guards; real
    tokenizers vary per model and may produce higher counts (especially for
    code, CJK, and long identifiers). Empty input returns 0.
    """
    if not text:
        return 0
    return math.ceil(len(text) / ESTIMATED_CHARS_PER_TOKEN)


def _bounded_prompt_chars(candidate_chars: int, cfg: dict[str, Any]) -> int:
    """Bound a dynamic prompt estimate by the runtime participant cap.

    When the candidate exceeds the cap the runtime skips the call entirely;
    charging the full cap is therefore conservative while keeping the estimate
    finite and tied to configured policy.
    """

    cap = cfg.get("max_prompt_chars")
    if cap is None:
        return max(0, int(candidate_chars))
    return min(max(0, int(candidate_chars)), max(0, int(cap)))


def cross_rank_prompt_char_bounds(
    *,
    participants: list[str],
    participant_cfg: dict[str, Any],
    question: str,
    completion_tokens: int,
    focus_directive: str | None,
    safe_context: bool,
) -> dict[str, int]:
    """Return conservative per-peer ranking prompt character bounds.

    The exact fixed framing comes from ``build_ranking_prompt``. Dynamic peer
    bodies are modeled arithmetically so a very large completion assumption
    cannot force an equally large temporary string allocation.
    """

    anonymization_map = build_anonymization_map(participants)
    assumed_output_chars = max(0, int(completion_tokens)) * ESTIMATED_CHARS_PER_TOKEN
    placeholder = "x" if assumed_output_chars else ""
    bounds: dict[str, int] = {}
    for name in participants:
        other_peers = {
            peer: placeholder for peer in participants if peer != name
        }
        ranking_prompt = build_ranking_prompt(
            peer_name=name,
            own_response=placeholder,
            other_peers=other_peers,
            anonymization_map=anonymization_map,
            question=question,
        )
        if focus_directive:
            ranking_prompt += "\n\n" + focus_directive
        if safe_context:
            ranking_prompt = SAFE_CONTEXT_DIRECTIVE + "\n\n" + ranking_prompt
        candidate_chars = len(ranking_prompt)
        if assumed_output_chars > 1:
            candidate_chars += len(other_peers) * (assumed_output_chars - 1)
        bounds[name] = _bounded_prompt_chars(
            candidate_chars,
            participant_cfg.get(name, {}),
        )
    return bounds


def synthesis_prompt_char_bound(chair_cfg: dict[str, Any]) -> int:
    """Worst-case synthesis prompt chars under runtime builder/peer caps."""

    return _bounded_prompt_chars(
        MAX_SYNTHESIS_PROMPT_CHARS_DEFAULT,
        chair_cfg,
    )


def deliberation_prompt_char_bounds(
    *,
    participants: list[str],
    participant_cfg: dict[str, Any],
    mode: str,
    tool_call_voting: bool,
    stances: Mapping[str, str] | None = None,
    focus_directive: str | None = None,
) -> dict[str, int]:
    """Worst-case successful prompt size for every extra council round.

    The shared deliberation builder caps its dynamic task/peer body at 80k
    characters. ``run_participants`` then appends peer-specific directives,
    so those exact suffixes must sit outside that cap in the estimate. A lower
    participant prompt cap bounds the largest call that could actually run.
    """

    bounds: dict[str, int] = {}
    for name in participants:
        cfg = participant_cfg.get(name, {}) or {}
        assigned_stance = (
            stances.get(name)
            if stances is not None and name in stances
            else cfg.get("stance")
        )
        directive_suffix = apply_per_peer_directives(
            "",
            mode=mode,
            family=cfg.get("family"),
            tool_call_voting=tool_call_voting,
            stance=assigned_stance,
            persona=cfg.get("persona"),
            persona_prompt=cfg.get("persona_prompt"),
            focus_directive=focus_directive,
        )
        bounds[name] = _bounded_prompt_chars(
            MAX_DELIBERATION_PROMPT_CHARS + len(directive_suffix),
            cfg,
        )
    return bounds


def estimate_council(
    *,
    config: dict[str, Any],
    cwd: Path,
    question: str,
    mode: str,
    current: str | None,
    explicit: list[str] | None = None,
    include: list[str] | None = None,
    origin_policy: str | None = None,
    context_paths: list[str] | None = None,
    include_diff: bool = False,
    stdin_text: str | None = None,
    allow_outside_cwd: bool = False,
    deliberate: bool = False,
    max_rounds: int | None = None,
    completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
    openrouter_models: list[str] | None = None,
    use_cache: bool = True,
    allow_network: bool = True,
    image_paths: list[str] | None = None,
    prepared_prompt: str | None = None,
    prepared_participants: list[str] | None = None,
    participant_prompts: Mapping[str, str] | None = None,
    cross_rank: bool = False,
    synthesize: bool = False,
    synthesizer_name: str | None = None,
    focus_directive: str | None = None,
    cross_rank_prompt_chars: Mapping[str, int] | None = None,
    synthesis_prompt_chars: int | None = None,
    deliberation_prompt_chars: Mapping[str, int] | None = None,
    chunk_strategy: str = "fail",
    chunk_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Return a best-effort preflight estimate for a council run.

    `allow_network=False` prevents the OpenRouter catalog fetch from falling
    back to live HTTP if the disk cache is missing or stale; the result is
    a fast-fail estimate suitable for the pre-flight budget gate (which
    needs to refuse before any network call so a too-low cap doesn't cost
    a multi-second wait). Hosted peers without a cached price come back
    with `estimated_total_cost_usd: None`, which the budget gate already
    treats as a refusal condition.
    """
    mode = canonical_mode_name(config, mode)
    participants = (
        list(prepared_participants)
        if prepared_participants is not None
        else select_participants(
            config,
            mode,
            current,
            explicit=explicit,
            include=include,
            origin_policy=origin_policy,
        )
    )
    if synthesize and not synthesizer_name:
        raise ValueError(
            "A concrete synthesizer_name is required to estimate an opt-in "
            "synthesis call."
        )
    if synthesize and synthesizer_name not in participants:
        raise ValueError(
            f"Synthesizer '{synthesizer_name}' is not in the selected roster."
        )
    # Match the runtime image budget so an estimate that passes can't be
    # rejected by the actual run.
    image_manifest = (
        build_image_manifest(
            image_paths, cwd=cwd, allow_outside_cwd=allow_outside_cwd
        )
        if image_paths
        else []
    )
    if image_manifest:
        violations = image_attachment_violations(image_manifest)
        if violations:
            raise ValueError(
                "Image attachment budget exceeded: "
                + ", ".join(
                    f"{v['limit']} {v.get('actual')} > {v.get('maximum')}"
                    for v in violations
                )
            )
    mode_cfg = config.get("modes", {}).get(mode, {})
    participant_cfg = config.get("participants", {})
    if not isinstance(participant_cfg, dict):
        participant_cfg = {}
    mode_stances = mode_cfg.get("stances") if isinstance(mode_cfg, dict) else None
    mode_stances = balance_stances(
        participants, mode_stances if isinstance(mode_stances, dict) else None
    )
    prompt = prepared_prompt
    if prompt is None:
        default_max = (
            config.get("defaults", {}).get("max_prompt_chars") or MAX_PROMPT_CHARS
        )
        peer_caps = [
            int(participant_cfg.get(name, {}).get("max_prompt_chars"))
            for name in participants
            if isinstance(participant_cfg.get(name), dict)
            and participant_cfg.get(name, {}).get("max_prompt_chars")
        ]
        effective_max = (
            min([int(default_max), *peer_caps]) if peer_caps else int(default_max)
        )
        prompt = build_prompt(
            question,
            mode=mode,
            cwd=cwd,
            context_paths=context_paths or [],
            include_diff=include_diff,
            stdin_text=stdin_text,
            allow_outside_cwd=allow_outside_cwd,
            max_prompt_chars=effective_max,
            image_manifest=image_manifest or None,
            # Match the runtime prompt so an estimate that passes can't be
            # rejected by the actual run's per-participant prompt-size guard
            # (same parity rule as the image budget above) — the framing adds
            # ~850 chars.
            safe_context=bool(mode_cfg.get("safe_context")),
            stances=mode_stances,
            participants=participant_cfg or None,
            chunk_strategy=chunk_strategy,
            chunk_progress=chunk_progress,
        )
    deliberate = bool(deliberate or mode_cfg.get("deliberate"))
    rounds = int(
        max_rounds
        or mode_cfg.get("max_rounds")
        or config.get("defaults", {}).get("max_deliberation_rounds")
        or 2
    )
    budgeted_rounds = max(1, rounds) if deliberate else 1
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = max(0, int(completion_tokens))

    if participant_prompts is None:
        # Match `run`: changed-file persona recruitment and every static
        # per-peer directive contribute to input tokens and hosted cost.
        from llm_council.orchestrator import apply_contextual_persona_recruitment

        apply_contextual_persona_recruitment(
            participants,
            participant_cfg,
            cwd,
            stances=mode_stances,
        )
        participant_prompts = {}
        for name in participants:
            cfg = participant_cfg.get(name) or {}
            participant_prompts[name] = apply_per_peer_directives(
                prompt,
                mode=mode,
                family=cfg.get("family"),
                tool_call_voting=bool(mode_cfg.get("tool_call_voting")),
                stance=(
                    mode_stances.get(name)
                    if isinstance(mode_stances, dict)
                    else cfg.get("stance")
                ),
                persona=cfg.get("persona"),
                persona_prompt=cfg.get("persona_prompt"),
                focus_directive=focus_directive,
            )
    extra_models = list(openrouter_models or [])
    needs_catalog = bool(extra_models) or any(
        _openrouter_needs_catalog(participant_cfg.get(name, {}))
        for name in participants
    )
    catalog_error: str | None = None
    catalog_by_id: dict[str, dict[str, Any]] = {}
    if needs_catalog:
        try:
            catalog_by_id = {
                model["id"]: model
                for model in fetch_openrouter_models(
                    use_cache=use_cache, allow_network=allow_network
                )
                if model.get("id")
            }
        except Exception as exc:  # pragma: no cover - depends on network state
            catalog_error = str(exc)

    image_count = len(image_paths or [])
    image_token_overhead = image_count * IMAGE_TOKEN_HEURISTIC
    rows = []
    for name in participants:
        peer_prompt = (
            participant_prompts.get(name, prompt)
            if participant_prompts is not None
            else prompt
        )
        peer_prompt_tokens = estimate_tokens(peer_prompt)
        rows.append(
            _estimate_participant_row(
                name=name,
                cfg=participant_cfg.get(name, {}),
                catalog_by_id=catalog_by_id,
                prompt_tokens=peer_prompt_tokens
                + (
                    image_token_overhead
                    if participant_cfg.get(name, {}).get("vision")
                    else 0
                ),
                completion_tokens=completion_tokens,
                rounds=1,
            )
        )

    # Round 1 uses the exact finalized peer prompts above. Every additional
    # deliberation round is dynamic: prior peer responses can fill the shared
    # 80k builder cap, after which run_participants appends the same per-peer
    # directives (and vision inputs) again. Represent that separately instead
    # of multiplying a tiny round-1 question across all rounds.
    if deliberate and budgeted_rounds > 1:
        mode_stances = mode_cfg.get("stances")
        deliberation_bounds = (
            dict(deliberation_prompt_chars)
            if deliberation_prompt_chars is not None
            else deliberation_prompt_char_bounds(
                participants=participants,
                participant_cfg=participant_cfg,
                mode=mode,
                tool_call_voting=bool(mode_cfg.get("tool_call_voting")),
                stances=(
                    mode_stances if isinstance(mode_stances, Mapping) else None
                ),
                focus_directive=focus_directive,
            )
        )
        extra_rounds = budgeted_rounds - 1
        for name in participants:
            if name not in deliberation_bounds:
                raise ValueError(
                    f"Missing deliberation prompt bound for participant '{name}'."
                )
            cfg = participant_cfg.get(name, {})
            row = _estimate_participant_row(
                name=f"{name}:deliberation",
                cfg=cfg,
                catalog_by_id=catalog_by_id,
                prompt_tokens=math.ceil(
                    deliberation_bounds[name] / ESTIMATED_CHARS_PER_TOKEN
                )
                + (image_token_overhead if cfg.get("vision") else 0),
                completion_tokens=completion_tokens,
                rounds=extra_rounds,
            )
            row["phase"] = "deliberation"
            row["participant"] = name
            row["prompt_bound_chars"] = deliberation_bounds[name]
            rows.append(row)

    # Cross-rank content depends on peer outputs that do not exist at
    # preflight time. Bound it conservatively by assuming every selected peer
    # produces a label and reaches the configured completion-token assumption;
    # the runtime prompt builder supplies the exact fixed framing. Participant
    # prompt caps bound each row because an over-cap runtime call is skipped.
    if cross_rank and len(participants) >= CROSS_RANK_MIN_PEERS:
        ranking_prompt_chars = (
            dict(cross_rank_prompt_chars)
            if cross_rank_prompt_chars is not None
            else cross_rank_prompt_char_bounds(
                participants=participants,
                participant_cfg=participant_cfg,
                question=question or prompt,
                completion_tokens=completion_tokens,
                focus_directive=focus_directive,
                safe_context=bool(mode_cfg.get("safe_context")),
            )
        )
        for name in participants:
            if name not in ranking_prompt_chars:
                raise ValueError(
                    f"Missing cross-rank prompt bound for participant '{name}'."
                )
            row = _estimate_participant_row(
                name=f"{name}:rank",
                cfg=participant_cfg.get(name, {}),
                catalog_by_id=catalog_by_id,
                prompt_tokens=math.ceil(
                    ranking_prompt_chars[name] / ESTIMATED_CHARS_PER_TOKEN
                ),
                completion_tokens=completion_tokens,
                rounds=1,
            )
            row["phase"] = "cross_rank"
            row["participant"] = name
            row["prompt_bound_chars"] = ranking_prompt_chars[name]
            rows.append(row)

    # The synthesis builder hard-caps its dynamic final-round envelope at
    # 60k chars. Count that full cap (or the chair's lower participant cap)
    # because the actual peer findings are unknowable before execution.
    if synthesize and synthesizer_name:
        chair_cfg = participant_cfg.get(synthesizer_name, {})
        bounded_synthesis_chars = (
            synthesis_prompt_char_bound(chair_cfg)
            if synthesis_prompt_chars is None
            else max(0, int(synthesis_prompt_chars))
        )
        row = _estimate_participant_row(
            name=f"{synthesizer_name}:synthesis",
            cfg=chair_cfg,
            catalog_by_id=catalog_by_id,
            prompt_tokens=math.ceil(
                bounded_synthesis_chars / ESTIMATED_CHARS_PER_TOKEN
            ),
            completion_tokens=completion_tokens,
            rounds=1,
        )
        row["phase"] = "synthesis"
        row["participant"] = synthesizer_name
        row["prompt_bound_chars"] = bounded_synthesis_chars
        rows.append(row)

    rows.extend(
        _estimate_openrouter_model_row(
            model_id=model_id,
            catalog_by_id=catalog_by_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            rounds=budgeted_rounds,
        )
        for model_id in extra_models
    )

    known_total = sum(
        row["estimated_total_cost_usd"]
        for row in rows
        if row["estimated_total_cost_usd"] is not None
    )
    # Outer-retry safety margin: the adapter may issue one extra model call for
    # label/section/evidence repair or timeout recovery. Ranking and synthesis
    # disable vote-envelope repair, but they still retain terse timeout recovery
    # unless explicitly disabled. Phase rows already aggregate their own round
    # counts, so doubling each eligible row models that worst case without
    # losing the extra deliberation rounds.
    safety_total = known_total
    for row in rows:
        if row["estimated_total_cost_usd"] is None:
            continue
        peer_name = row.get("participant") or row["name"]
        peer_cfg = participant_cfg.get(peer_name, {}) if isinstance(
            participant_cfg, dict
        ) else {}
        # An explicit retries: 0 is the adapter-wide no-extra-calls switch.
        retries_value = peer_cfg.get("retries")
        if retries_value is not None and int(retries_value) == 0:
            continue
        terse_retry = peer_cfg.get("terse_retry_on_timeout", True) is not False
        validation_retry = (
            row.get("phase") not in {"cross_rank", "synthesis"}
            and peer_cfg.get("retry_on_missing_label", True) is not False
        )
        if not terse_retry and not validation_retry:
            continue
        safety_total += row["estimated_total_cost_usd"]
    unknown_cost_rows = [
        row["name"] for row in rows if row["estimated_total_cost_usd"] is None
    ]
    notes = _estimate_notes(rows, catalog_error)
    if image_paths:
        notes.append(
            f"Image attachments add a heuristic {IMAGE_TOKEN_HEURISTIC} input "
            "tokens per image to vision-capable participants only; non-vision "
            "participants see images as text references."
        )
    if deliberate and budgeted_rounds > 1:
        notes.append(
            "Deliberation rows count round 1 exactly, then conservatively bound "
            "each additional round by the 80,000-character deliberation body "
            "plus exact per-peer directives, subject to a lower participant cap."
        )
    if cross_rank and len(participants) >= CROSS_RANK_MIN_PEERS:
        notes.append(
            "Cross-rank rows conservatively assume every selected peer labels, "
            "every peer response reaches the completion-token assumption, and "
            "the resulting ranking prompt runs up to any participant prompt cap."
        )
    if synthesize and synthesizer_name:
        notes.append(
            "The synthesis row uses the synthesis builder's 60,000-character "
            "dynamic prompt cap (or the chair's lower participant prompt cap) "
            "because final peer findings are unavailable at preflight time."
        )
    # L7: derived advisory signals. `cost_class` buckets the retry-safety
    # total; `paid_peer_count` / `free_peer_count` partition the roster.
    # A peer is potentially paid when it uses a non-local API endpoint and is
    # not a router-declared `:free` model. A catalog miss does NOT make a
    # remote peer free: remote Ollama is also paid/unpriced until explicit
    # pricing proves otherwise. Only loopback Ollama/openai_compatible peers
    # count as local $0 inference.

    def _row_is_paid(row: dict[str, Any]) -> bool:
        cfg = participant_cfg.get(row["name"], {}) or {}
        participant_type = cfg.get("type")
        if participant_type not in ("openrouter", "openai_compatible", "ollama"):
            return False
        if (
            participant_type != "ollama"
            and str(cfg.get("model") or "").endswith(":free")
        ):
            return False
        return not is_local_participant(cfg)

    safety_rounded = round(safety_total, 6)
    cost_class = _cost_class(safety_rounded)
    roster_rows = [row for row in rows if not row.get("phase")]
    paid_peer_count = sum(1 for row in roster_rows if _row_is_paid(row))
    free_peer_count = len(roster_rows) - paid_peer_count
    return {
        "mode": mode,
        "current": current,
        "participants": participants,
        "extra_openrouter_models": extra_models,
        "prompt_chars": len(prompt),
        "estimated_prompt_tokens": prompt_tokens,
        "budgeted_rounds": budgeted_rounds,
        "deliberate": deliberate,
        "completion_tokens_assumed_each": completion_tokens,
        "image_paths": list(image_paths or []),
        "known_total_usd": round(known_total, 6),
        "known_total_with_retry_safety_usd": safety_rounded,
        "unknown_cost_rows": unknown_cost_rows,
        "cost_class": cost_class,
        "paid_peer_count": paid_peer_count,
        "free_peer_count": free_peer_count,
        "catalog_error": catalog_error,
        "rows": rows,
        "notes": notes,
    }


def _cost_class(known_total_with_retry_safety_usd: float) -> str:
    """Bucket a worst-case USD total into a coarse advisory class."""
    if known_total_with_retry_safety_usd < COST_CLASS_LOW_MAX_USD:
        return "low"
    if known_total_with_retry_safety_usd < COST_CLASS_MODERATE_MAX_USD:
        return "moderate"
    return "high"


def compact_cost_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full `estimate_council` result to the compact advisory block
    echoed into run metadata as `cost_estimate` (L7).

    Pure projection — never raises, only reads keys already present on the
    estimate dict. Used by both the CLI and MCP run paths so a caller who
    skipped `council_estimate` still sees the cost signal without the full
    per-peer breakdown landing in metadata.
    """
    return {
        "known_total_usd": estimate.get("known_total_usd"),
        "retry_safety_usd": estimate.get("known_total_with_retry_safety_usd"),
        "cost_class": estimate.get("cost_class"),
        "paid_peer_count": estimate.get("paid_peer_count"),
        "free_peer_count": estimate.get("free_peer_count"),
    }


def _openrouter_needs_catalog(cfg: dict[str, Any]) -> bool:
    return (
        cfg.get("type") in ("openrouter", "openai_compatible")
        and _is_openrouter_cfg(cfg)
        and not str(cfg.get("model") or "").endswith(":free")
        and (
            cfg.get("input_per_million") is None
            or cfg.get("output_per_million") is None
        )
    )


def _is_openrouter_cfg(cfg: dict[str, Any]) -> bool:
    if cfg.get("type") == "openrouter":
        return True
    if cfg.get("type") != "openai_compatible":
        return False
    base_url = str(cfg.get("base_url") or "")
    if not base_url:
        return False
    try:
        from urllib.parse import urlparse
        host = (urlparse(base_url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _estimate_participant_row(
    *,
    name: str,
    cfg: dict[str, Any],
    catalog_by_id: dict[str, dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    rounds: int,
) -> dict[str, Any]:
    participant_type = cfg.get("type") or "unknown"
    model = cfg.get("model") or CLI_DEFAULT_MODEL_LABEL
    # Native CLI cost remains external/unobservable. API-style inference is
    # $0 only when the endpoint is proven loopback; a remote/LAN Ollama server
    # may be metered and must not inherit Ollama's usual local-runtime label.
    is_local_endpoint = (
        participant_type in ("ollama", "openai_compatible")
        and is_local_participant(cfg)
    )
    if (
        participant_type == "cli"
        or participant_type not in ("openrouter", "openai_compatible", "ollama")
        or is_local_endpoint
    ):
        note = (
            "Native CLI subscription or local runtime cost is external to "
            "llm-council."
        )
        # Force zero pricing for local endpoints so the budget gate's
        # "unpriced paid" check (cli.py / mcp_server.py) doesn't refuse
        # local-only runs as if they were unknown-cost hosted peers. The
        # cash cost to llm-council really is $0; the GPU/subscription
        # cost is the user's problem.
        if participant_type == "ollama" and is_local_endpoint:
            note = "Local Ollama runtime cost is external to llm-council."
            input_per_million_value: float | None = 0.0
            output_per_million_value: float | None = 0.0
            pricing_source_value: str | None = "local"
        elif is_local_endpoint:
            note = (
                "Local openai_compatible endpoint (base_url is loopback); "
                "GPU / subscription cost is external "
                "to llm-council."
            )
            input_per_million_value = 0.0
            output_per_million_value = 0.0
            pricing_source_value = "local"
        else:
            input_per_million_value = None
            output_per_million_value = None
            pricing_source_value = None
        return _row(
            name=name,
            participant_type=participant_type,
            model=str(model),
            pricing_source=pricing_source_value,
            input_per_million=input_per_million_value,
            output_per_million=output_per_million_value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            rounds=rounds,
            note=note,
        )

    input_per_million = _float_or_none(cfg.get("input_per_million"))
    output_per_million = _float_or_none(cfg.get("output_per_million"))
    pricing_source = "config" if (
        input_per_million is not None and output_per_million is not None
    ) else None
    if participant_type != "ollama" and str(model).endswith(":free"):
        input_per_million = input_per_million or 0.0
        output_per_million = output_per_million or 0.0
        pricing_source = pricing_source or "free route"
    elif (input_per_million is None or output_per_million is None) and _is_openrouter_cfg(cfg):
        catalog = catalog_by_id.get(str(model), {})
        input_per_million = (
            input_per_million
            if input_per_million is not None
            else _float_or_none(catalog.get("input_per_million"))
        )
        output_per_million = (
            output_per_million
            if output_per_million is not None
            else _float_or_none(catalog.get("output_per_million"))
        )
        if input_per_million is not None or output_per_million is not None:
            pricing_source = "catalog"

    # M8 litellm fallback (estimation only, hosted peers only). When the
    # OpenRouter catalog still yields no price, consult litellm's bundled
    # local cost map before leaving the row unpriced. No-op when litellm is
    # absent (the default) — the helper returns (None, None) and the peer
    # stays in unknown_cost_rows / unpriced_paid exactly as before.
    litellm_priced = False
    if (
        (input_per_million is None or output_per_million is None)
        and cfg.get("type") in ("openrouter", "openai_compatible")
    ):
        lit_in, lit_out = _litellm_price_per_million(str(model))
        if input_per_million is None and lit_in is not None:
            input_per_million = lit_in
            litellm_priced = True
        if output_per_million is None and lit_out is not None:
            output_per_million = lit_out
            litellm_priced = True
        if litellm_priced:
            pricing_source = "litellm"

    note = None
    if litellm_priced and input_per_million is not None and output_per_million is not None:
        note = (
            f"Pricing for {name} ({model}) came from litellm's local cost "
            "map, not the OpenRouter catalog."
        )
    elif input_per_million is None or output_per_million is None:
        if participant_type == "ollama":
            note = (
                "Remote Ollama endpoint pricing is not configured; set "
                "input_per_million and output_per_million on the participant."
            )
        elif _is_openrouter_cfg(cfg):
            note = "OpenRouter pricing unavailable; refresh catalog or configure prices."
        else:
            provider_label = cfg.get("provider_label") or "endpoint"
            note = (
                f"openai_compatible {provider_label} pricing not configured; "
                "set input_per_million and output_per_million on the participant."
            )
    return _row(
        name=name,
        participant_type=participant_type,
        model=str(model),
        pricing_source=pricing_source,
        input_per_million=input_per_million,
        output_per_million=output_per_million,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        rounds=rounds,
        note=note,
    )


def _estimate_openrouter_model_row(
    *,
    model_id: str,
    catalog_by_id: dict[str, dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    rounds: int,
) -> dict[str, Any]:
    catalog = catalog_by_id.get(model_id, {})
    input_per_million = _float_or_none(catalog.get("input_per_million"))
    output_per_million = _float_or_none(catalog.get("output_per_million"))
    pricing_source = "catalog" if catalog else None
    if model_id.endswith(":free"):
        input_per_million = input_per_million or 0.0
        output_per_million = output_per_million or 0.0
        pricing_source = pricing_source or "free route"
    note = None
    if input_per_million is None or output_per_million is None:
        note = "OpenRouter pricing unavailable; copy an exact ID from live models."
    return _row(
        name=f"openrouter:{model_id}",
        participant_type="openrouter",
        model=model_id,
        pricing_source=pricing_source,
        input_per_million=input_per_million,
        output_per_million=output_per_million,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        rounds=rounds,
        note=note,
    )


def _row(
    *,
    name: str,
    participant_type: str,
    model: str,
    pricing_source: str | None,
    input_per_million: float | None,
    output_per_million: float | None,
    prompt_tokens: int,
    completion_tokens: int,
    rounds: int,
    note: str | None,
) -> dict[str, Any]:
    estimated_input_tokens = prompt_tokens * rounds
    estimated_output_tokens = completion_tokens * rounds
    input_cost = _cost(estimated_input_tokens, input_per_million)
    output_cost = _cost(estimated_output_tokens, output_per_million)
    total = None
    if input_cost is not None and output_cost is not None:
        total = input_cost + output_cost
    return {
        "name": name,
        "type": participant_type,
        "model": model,
        "pricing_source": pricing_source,
        "input_per_million": input_per_million,
        "output_per_million": output_per_million,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_input_cost_usd": _round_cost(input_cost),
        "estimated_output_cost_usd": _round_cost(output_cost),
        "estimated_total_cost_usd": _round_cost(total),
        "rounds_assumed": rounds,
        "note": note,
    }


def _estimate_notes(rows: list[dict[str, Any]], catalog_error: str | None) -> list[str]:
    notes: list[str] = [
        "Token count is estimated from characters; provider billing may differ.",
        "Output cost assumes the configured completion token estimate per participant per round.",
    ]
    if any(row["type"] in {"cli", "ollama"} for row in rows):
        notes.append(
            "Native CLI and Ollama rows are not API-priced automatically here; "
            "check native subscriptions, treat loopback Ollama as local $0 "
            "inference, and configure explicit pricing for remote/LAN Ollama."
        )
    if any(row["model"].endswith(":free") for row in rows):
        notes.append(
            "OpenRouter :free routes can be account-gated and may fail with 402 Payment Required even when estimated as $0."
        )
    if catalog_error:
        notes.append(f"OpenRouter catalog lookup failed: {catalog_error}")
    elif any(row["type"] in {"openrouter", "openai_compatible"} for row in rows):
        notes.append(
            "OpenRouter prices come from config or the live cached catalog; rerun with --no-cache before expensive work."
        )
    # M8: surface every litellm-sourced price top-level so it's operator-
    # visible that the price wasn't from OpenRouter.
    for row in rows:
        if row.get("pricing_source") == "litellm":
            notes.append(
                f"Pricing for {row['name']} ({row['model']}) came from "
                "litellm's local cost map, not the OpenRouter catalog."
            )
    return notes


def _cost(tokens: int, per_million: float | None) -> float | None:
    if per_million is None:
        return None
    return tokens * per_million / 1_000_000


def _round_cost(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _litellm_price_per_million(model: str) -> tuple[float | None, float | None]:
    """Best-effort (input_per_million, output_per_million) USD from litellm.

    Returns (None, None) when litellm is absent or the model id isn't in
    litellm's bundled local cost map. NO network is performed — only the
    in-process `model_cost` map (or `get_model_info`, which also reads the
    same local map) is consulted. litellm stores cost PER TOKEN, so we
    multiply by 1_000_000 to match the per-million units used everywhere
    else in this module.
    """
    if litellm is None or not model:
        return None, None
    info: dict[str, Any] | None = None
    # Prefer the public accessor; fall back to the raw map. Both are local.
    try:
        get_info = getattr(litellm, "get_model_info", None)
        if callable(get_info):
            info = get_info(model)
    except Exception:  # pragma: no cover - litellm raises on unknown ids
        info = None
    if not isinstance(info, dict):
        model_cost = getattr(litellm, "model_cost", None)
        if isinstance(model_cost, dict):
            entry = model_cost.get(model)
            info = entry if isinstance(entry, dict) else None
    if not isinstance(info, dict):
        return None, None
    input_per_token = info.get("input_cost_per_token")
    output_per_token = info.get("output_cost_per_token")
    input_per_million = (
        float(input_per_token) * 1_000_000 if input_per_token is not None else None
    )
    output_per_million = (
        float(output_per_token) * 1_000_000 if output_per_token is not None else None
    )
    return input_per_million, output_per_million
