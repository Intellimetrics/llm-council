"""Conservative MCP budget checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from llm_council.model_catalog import _read_cache, openrouter_cache_path


DEFAULT_MCP_MAX_PROMPT_CHARS = 80_000
DEFAULT_MCP_MAX_ESTIMATED_COST_USD = 0.10
ESTIMATED_CHARS_PER_TOKEN = 4
DEFAULT_IMAGE_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_IMAGE_TOTAL_MAX_BYTES = 32 * 1024 * 1024


def summarize_preflight_caps(
    preflight: dict[str, Any],
) -> tuple[float, int, list[str]]:
    """Reduce an `estimate_council` result to the three numbers the run-level
    `--max-cost-usd` / `--max-tokens` caps gate on:

      - cost_total: the retry-safety total (worst-case repair-retry headroom)
        when present, else the plain known total. Using the retry-safety figure
        means a worst-case repair retry can't silently push spend past the cap.
      - token_total: summed estimated input+output tokens across all rows.
      - unpriced_paid: names of hosted (openrouter/openai_compatible) peers whose
        catalog price is unknown — a $-cap cannot be enforced against these, so
        callers refuse rather than trust an undercount.

    Shared by the CLI and MCP run pipelines (which previously duplicated this
    exact reduction) so the drift-prone arithmetic lives in one place. Callers
    keep their own surface-tailored refusal messages.
    """
    cost_total = float(
        preflight.get("known_total_with_retry_safety_usd")
        if preflight.get("known_total_with_retry_safety_usd") is not None
        else (preflight.get("known_total_usd") or 0.0)
    )
    token_rows = preflight.get("rows") or []
    token_total = sum(
        int(row.get("estimated_input_tokens") or 0)
        + int(row.get("estimated_output_tokens") or 0)
        for row in token_rows
    )
    unpriced_paid = [
        row.get("name")
        for row in token_rows
        if row.get("type") in {"openrouter", "openai_compatible"}
        and row.get("estimated_total_cost_usd") is None
    ]
    return cost_total, token_total, unpriced_paid


def enforce_preflight_caps(
    preflight: dict[str, Any],
    *,
    max_cost_usd: float | None,
    max_tokens: int | None,
    breakdown_hint: str,
) -> None:
    """Reproduce the three pre-flight budget-cap checks the CLI gates a run on.

    Factored out of `cmd_estimate` / `cmd_run_async` (which inlined byte-for-byte
    identical logic except for the cost-cap message's per-peer breakdown hint).
    Reduces `preflight` via `summarize_preflight_caps` and raises `ValueError`
    with the standard CLI message on the first violation:

      1. `--max-cost-usd` requested but hosted peers have no catalog price,
      2. estimated cost exceeds `--max-cost-usd`,
      3. estimated tokens exceed `--max-tokens`.

    `breakdown_hint` is interpolated into the cost-cap message — the only clause
    that differed between the two call sites. Callers convert the raised
    `ValueError` to `SystemExit`.
    """
    if max_cost_usd is None and max_tokens is None:
        return
    cost_total, token_total, unpriced_paid = summarize_preflight_caps(preflight)
    if max_cost_usd is not None and unpriced_paid:
        raise ValueError(
            "Pre-flight estimate cannot enforce --max-cost-usd: hosted "
            f"peer(s) without a catalog price: {', '.join(unpriced_paid)}. "
            "Run `llm-council models openrouter` to confirm the model id, "
            "or drop these peers, before relying on the cost cap."
        )
    if max_cost_usd is not None and cost_total > float(max_cost_usd):
        raise ValueError(
            f"Pre-flight estimate ${cost_total:.6f} (with worst-case "
            f"repair-retry headroom) exceeds --max-cost-usd "
            f"${float(max_cost_usd):.6f}. Free/local peers count as $0; "
            f"{breakdown_hint}"
        )
    if max_tokens is not None and token_total > int(max_tokens):
        raise ValueError(
            f"Pre-flight estimate {token_total} tokens exceeds --max-tokens "
            f"{int(max_tokens)}. Drop --diff/--context, narrow the question, "
            "or raise the cap."
        )


def image_attachment_violations(
    manifest: list[dict[str, Any]],
    *,
    max_per_file: int = DEFAULT_IMAGE_MAX_BYTES,
    max_total: int = DEFAULT_IMAGE_TOTAL_MAX_BYTES,
) -> list[dict[str, Any]]:
    """Return budget violations for staged image attachments before any encode."""

    violations: list[dict[str, Any]] = []
    total = 0
    for entry in manifest:
        size = int(entry.get("size") or 0)
        total += size
        if size > max_per_file:
            violations.append(
                {
                    "limit": "image_max_bytes",
                    "path": entry.get("relative_path") or entry.get("path"),
                    "actual": size,
                    "maximum": max_per_file,
                }
            )
    if total > max_total:
        violations.append(
            {
                "limit": "image_total_max_bytes",
                "actual": total,
                "maximum": max_total,
            }
        )
    return violations


def mcp_budget_report(
    *,
    config: dict[str, Any],
    participants: list[str],
    prompt_chars: int,
    deliberate: bool,
    max_rounds: int,
    cross_rank: bool = False,
    synthesize: bool = False,
) -> dict[str, Any]:
    participant_cfg = config.get("participants", {})
    paid_hosted = [
        name
        for name in participants
        if _is_paid_hosted_participant(participant_cfg.get(name, {}))
    ]
    budgeted_rounds = max(1, int(max_rounds or 1)) if deliberate else 1
    # `--cross-rank` runs one additional ranking pass (a full extra
    # run_participant per labeled peer) after the council rounds, so count it
    # as +1 round of paid-hosted prompts. Without this the pre-flight estimate
    # under-counts and a run that should trip mcp_max_estimated_cost_usd slips
    # through. (Conservative: the ranking prompt is actually larger, so this
    # is a lower bound on the true extra spend.)
    effective_rounds = budgeted_rounds + (1 if cross_rank else 0)
    # `synthesize` fires at most one extra chair call; it only adds hosted
    # spend when the configured synthesizer is itself a paid-hosted peer.
    synth_peer = config.get("defaults", {}).get("synthesizer")
    synth_billable = bool(
        synthesize
        and synth_peer
        and _is_paid_hosted_participant(participant_cfg.get(synth_peer, {}))
    )
    limits = _budget_limits(config)
    billable_prompt_chars = prompt_chars * effective_rounds * len(paid_hosted)
    if synth_billable:
        billable_prompt_chars += prompt_chars
    report = {
        "max_prompt_chars": limits["max_prompt_chars"],
        "max_estimated_cost_usd": limits["max_estimated_cost_usd"],
        "prompt_chars": prompt_chars,
        "budgeted_rounds": budgeted_rounds,
        "cross_rank": bool(cross_rank),
        "synthesize_billable": synth_billable,
        "paid_hosted_participants": paid_hosted,
        "estimated_billable_prompt_chars": billable_prompt_chars,
        "estimated_input_cost_usd": None,
        "cost_estimate_available": False,
        "violations": [],
    }

    guarded_by_prompt_size = (
        bool(paid_hosted) or deliberate or budgeted_rounds > 1 or cross_rank
    )
    if guarded_by_prompt_size and prompt_chars > limits["max_prompt_chars"]:
        report["violations"].append(
            {
                "limit": "max_prompt_chars",
                "actual": prompt_chars,
                "maximum": limits["max_prompt_chars"],
            }
        )

    catalog_path = openrouter_cache_path()
    cost = _estimate_input_cost_usd(
        paid_hosted,
        participant_cfg,
        prompt_chars=prompt_chars,
        rounds=effective_rounds,
        catalog_path=catalog_path,
    )
    unknown_paid_pricing = bool(paid_hosted) and cost is None
    if unknown_paid_pricing:
        report["violations"].append(
            {
                "limit": "known_paid_hosted_pricing",
                "actual": ", ".join(paid_hosted),
                "maximum": "configured input_per_million or cached catalog price",
                "participants": paid_hosted,
            }
        )
    else:
        # Fold in the synthesis chair call (one extra single-round call) when
        # it is a priceable paid-hosted peer. Skipped when we're already
        # blocking on unknown paid pricing above.
        if synth_billable:
            synth_cost = _estimate_input_cost_usd(
                [synth_peer],
                participant_cfg,
                prompt_chars=prompt_chars,
                rounds=1,
                catalog_path=catalog_path,
            )
            if synth_cost is not None:
                cost = (cost or 0.0) + synth_cost
        if cost is not None:
            report["estimated_input_cost_usd"] = round(cost, 6)
            report["cost_estimate_available"] = True
            if cost > limits["max_estimated_cost_usd"]:
                report["violations"].append(
                    {
                        "limit": "max_estimated_cost_usd",
                        "actual": round(cost, 6),
                        "maximum": limits["max_estimated_cost_usd"],
                    }
                )

    report["within_budget"] = not report["violations"]
    return report


def enforce_mcp_budget(report: dict[str, Any]) -> None:
    if report.get("within_budget", True):
        return
    details = ", ".join(
        f"{item['limit']} {item['actual']} > {item['maximum']}"
        for item in report.get("violations", [])
    )
    raise ValueError(f"MCP council_run budget exceeded: {details}")


def _budget_limits(config: dict[str, Any]) -> dict[str, float | int]:
    defaults = config.get("defaults", {})
    max_prompt_chars = _first_configured(
        defaults,
        "mcp_max_prompt_chars",
        fallback=DEFAULT_MCP_MAX_PROMPT_CHARS,
    )
    max_estimated_cost_usd = _first_configured(
        defaults,
        "mcp_max_estimated_cost_usd",
        fallback=DEFAULT_MCP_MAX_ESTIMATED_COST_USD,
    )
    return {
        "max_prompt_chars": int(max_prompt_chars),
        "max_estimated_cost_usd": float(max_estimated_cost_usd),
    }


def _first_configured(
    source: dict[str, Any], *keys: str, fallback: int | float
) -> int | float:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return fallback


def _is_paid_hosted_participant(cfg: dict[str, Any]) -> bool:
    if cfg.get("type") not in ("openrouter", "openai_compatible"):
        return False
    model = str(cfg.get("model") or "")
    return not model.endswith(":free")


def _estimate_input_cost_usd(
    participants: list[str],
    participant_cfg: dict[str, Any],
    *,
    prompt_chars: int,
    rounds: int,
    catalog_path: Path,
) -> float | None:
    if not participants:
        return None
    catalog = _read_cache(catalog_path) or []
    prices_by_model = {
        item.get("id"): item.get("input_per_million")
        for item in catalog
        if item.get("id") and item.get("input_per_million") is not None
    }

    total = 0.0
    any_price = False
    prompt_tokens = math.ceil(prompt_chars / ESTIMATED_CHARS_PER_TOKEN)
    for name in participants:
        cfg = participant_cfg.get(name, {})
        input_per_million = cfg.get("input_per_million")
        if input_per_million is None:
            input_per_million = prices_by_model.get(cfg.get("model"))
        if input_per_million is None:
            continue
        any_price = True
        total += (
            prompt_tokens
            * rounds
            * float(input_per_million)
            / 1_000_000
        )
    return total if any_price else None
