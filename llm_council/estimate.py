"""Council prompt and hosted-model cost estimates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from llm_council.budget import ESTIMATED_CHARS_PER_TOKEN
from llm_council.config import select_participants
from llm_council.context import MAX_PROMPT_CHARS, build_prompt
from llm_council.model_catalog import fetch_openrouter_models


IMAGE_TOKEN_HEURISTIC = 1500


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
    completion_tokens: int = 1500,
    openrouter_models: list[str] | None = None,
    use_cache: bool = True,
    image_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Return a best-effort preflight estimate for a council run."""
    participants = select_participants(
        config,
        mode,
        current,
        explicit=explicit,
        include=include,
        origin_policy=origin_policy,
    )
    prompt = build_prompt(
        question,
        mode=mode,
        cwd=cwd,
        context_paths=context_paths or [],
        include_diff=include_diff,
        stdin_text=stdin_text,
        allow_outside_cwd=allow_outside_cwd,
        max_prompt_chars=config.get("defaults", {}).get("max_prompt_chars")
        or MAX_PROMPT_CHARS,
        image_paths=image_paths,
    )
    mode_cfg = config.get("modes", {}).get(mode, {})
    deliberate = bool(deliberate or mode_cfg.get("deliberate"))
    rounds = int(
        max_rounds
        or mode_cfg.get("max_rounds")
        or config.get("defaults", {}).get("max_deliberation_rounds")
        or 2
    )
    budgeted_rounds = max(1, rounds) if deliberate else 1
    prompt_tokens = math.ceil(len(prompt) / ESTIMATED_CHARS_PER_TOKEN)
    completion_tokens = max(0, int(completion_tokens))

    participant_cfg = config.get("participants", {})
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
                for model in fetch_openrouter_models(use_cache=use_cache)
                if model.get("id")
            }
        except Exception as exc:  # pragma: no cover - depends on network state
            catalog_error = str(exc)

    image_count = len(image_paths or [])
    image_token_overhead = image_count * IMAGE_TOKEN_HEURISTIC
    rows = [
        _estimate_participant_row(
            name=name,
            cfg=participant_cfg.get(name, {}),
            catalog_by_id=catalog_by_id,
            prompt_tokens=prompt_tokens
            + (image_token_overhead if participant_cfg.get(name, {}).get("vision") else 0),
            completion_tokens=completion_tokens,
            rounds=budgeted_rounds,
        )
        for name in participants
    ]
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
        "unknown_cost_rows": unknown_cost_rows,
        "catalog_error": catalog_error,
        "rows": rows,
        "notes": notes,
    }


def _openrouter_needs_catalog(cfg: dict[str, Any]) -> bool:
    return (
        cfg.get("type") == "openrouter"
        and not str(cfg.get("model") or "").endswith(":free")
        and (
            cfg.get("input_per_million") is None
            or cfg.get("output_per_million") is None
        )
    )


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
    model = cfg.get("model") or "cli default"
    if participant_type != "openrouter":
        note = (
            "Native CLI subscription or local runtime cost is external to "
            "llm-council."
        )
        if participant_type == "ollama":
            note = "Local Ollama runtime cost is external to llm-council."
        return _row(
            name=name,
            participant_type=participant_type,
            model=str(model),
            pricing_source=None,
            input_per_million=None,
            output_per_million=None,
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
    if str(model).endswith(":free"):
        input_per_million = input_per_million or 0.0
        output_per_million = output_per_million or 0.0
        pricing_source = pricing_source or "free route"
    elif input_per_million is None or output_per_million is None:
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

    note = None
    if input_per_million is None or output_per_million is None:
        note = "OpenRouter pricing unavailable; refresh catalog or configure prices."
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
        "note": note,
    }


def _estimate_notes(rows: list[dict[str, Any]], catalog_error: str | None) -> list[str]:
    notes: list[str] = [
        "Token count is estimated from characters; provider billing may differ.",
        "Output cost assumes the configured completion token estimate per participant per round.",
    ]
    if any(row["type"] in {"cli", "ollama"} for row in rows):
        notes.append(
            "Native CLI and Ollama rows are not API-priced here; check your subscription, rate limit, or local runtime cost."
        )
    if any(row["model"].endswith(":free") for row in rows):
        notes.append(
            "OpenRouter :free routes can be account-gated and may fail with 402 Payment Required even when estimated as $0."
        )
    if catalog_error:
        notes.append(f"OpenRouter catalog lookup failed: {catalog_error}")
    elif any(row["type"] == "openrouter" for row in rows):
        notes.append(
            "OpenRouter prices come from config or the live cached catalog; rerun with --no-cache before expensive work."
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
