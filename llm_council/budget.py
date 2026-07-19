"""Conservative MCP budget checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from llm_council.config import is_local_participant
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

      - cost_total: the retry-safety total (worst-case outer-retry headroom)
        when present, else the plain known total. Using the retry-safety figure
        means a worst-case repair or timeout-recovery retry cannot silently
        push estimated spend past the cap.
      - token_total: summed estimated input+output tokens across all rows.
      - unpriced_paid: names of non-local API peers (OpenRouter,
        openai_compatible, or remote Ollama) whose price is unknown — a $-cap
        cannot be enforced against these, so callers refuse rather than trust
        an undercount.

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
        if row.get("type") in {"openrouter", "openai_compatible", "ollama"}
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
            "Pre-flight estimate cannot enforce --max-cost-usd: hosted peer(s) "
            "or other non-local endpoints without a catalog price: "
            f"{', '.join(unpriced_paid)}. "
            "Configure participant pricing (or, for OpenRouter, confirm the "
            "model id with `llm-council models openrouter`), or drop these "
            "peers before relying on the cost cap."
        )
    if max_cost_usd is not None and cost_total > float(max_cost_usd):
        raise ValueError(
            f"Pre-flight estimate ${cost_total:.6f} (with worst-case "
            f"outer-retry headroom) exceeds --max-cost-usd "
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
    synthesize: bool = False,
    synthesizer_name: str | None = None,
    participant_prompt_chars: Mapping[str, int] | None = None,
    deliberation_prompt_chars: Mapping[str, int] | None = None,
    synthesis_prompt_chars: int | None = None,
    image_prompt_tokens: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    participant_cfg = config.get("participants", {})
    paid_hosted = [
        name
        for name in participants
        if _is_paid_hosted_participant(participant_cfg.get(name, {}))
    ]
    budgeted_rounds = max(1, int(max_rounds or 1)) if deliberate else 1
    base_prompt_chars = {
        name: max(
            0,
            int(
                participant_prompt_chars.get(name, prompt_chars)
                if participant_prompt_chars is not None
                else prompt_chars
            ),
        )
        for name in participants
    }
    extra_deliberation_rounds = budgeted_rounds - 1
    if extra_deliberation_rounds:
        deliberation_chars = {
            name: max(
                0,
                int(
                    deliberation_prompt_chars.get(name, base_prompt_chars[name])
                    if deliberation_prompt_chars is not None
                    else base_prompt_chars[name]
                ),
            )
            for name in participants
        }
    else:
        deliberation_chars = {}
    per_call_image_tokens = {
        name: max(
            0,
            int(image_prompt_tokens.get(name, 0) if image_prompt_tokens else 0),
        )
        for name in participants
    }
    # `synthesize` fires at most one extra chair call; it only adds hosted
    # spend when the configured synthesizer is itself a paid-hosted peer.
    # Callers that already resolved the special ``current`` / ``neutral_peer``
    # aliases can supply the concrete chair name. Direct callers retain the
    # legacy behavior for explicitly named synthesizers.
    synth_peer = (
        synthesizer_name
        if synthesizer_name is not None
        else config.get("defaults", {}).get("synthesizer")
    )
    synth_billable = bool(
        synthesize
        and synth_peer
        and _is_paid_hosted_participant(participant_cfg.get(synth_peer, {}))
    )
    synth_call_chars = max(
        0,
        int(
            prompt_chars
            if synthesis_prompt_chars is None
            else synthesis_prompt_chars
        ),
    )
    limits = _budget_limits(config)
    billable_prompt_chars = sum(base_prompt_chars[name] for name in paid_hosted)
    if extra_deliberation_rounds:
        billable_prompt_chars += extra_deliberation_rounds * sum(
            deliberation_chars[name] for name in paid_hosted
        )
    if synth_billable:
        billable_prompt_chars += synth_call_chars
    billable_image_tokens = budgeted_rounds * sum(
        per_call_image_tokens[name] for name in paid_hosted
    )
    auxiliary_prompt_chars = [
        *base_prompt_chars.values(),
        *deliberation_chars.values(),
    ]
    if synthesize and synth_peer:
        auxiliary_prompt_chars.append(synth_call_chars)
    max_call_prompt_chars = max([prompt_chars, *auxiliary_prompt_chars])
    report = {
        "max_prompt_chars": limits["max_prompt_chars"],
        "max_estimated_cost_usd": limits["max_estimated_cost_usd"],
        "prompt_chars": prompt_chars,
        "max_call_prompt_chars": max_call_prompt_chars,
        "budgeted_rounds": budgeted_rounds,
        "synthesize_billable": synth_billable,
        "paid_hosted_participants": paid_hosted,
        "estimated_billable_prompt_chars": billable_prompt_chars,
        "estimated_billable_image_tokens": billable_image_tokens,
        "estimated_input_cost_usd": None,
        "cost_estimate_available": False,
        "violations": [],
    }

    guarded_by_prompt_size = (
        bool(paid_hosted)
        or deliberate
        or budgeted_rounds > 1
        or synthesize
    )
    if (
        guarded_by_prompt_size
        and max_call_prompt_chars > limits["max_prompt_chars"]
    ):
        report["violations"].append(
            {
                "limit": "max_prompt_chars",
                "actual": max_call_prompt_chars,
                "maximum": limits["max_prompt_chars"],
            }
        )

    catalog_path = openrouter_cache_path()
    cost: float | None = None
    unpriced_paid: list[str] = []
    for name in paid_hosted:
        base_cost, base_unpriced = _estimate_input_cost_usd(
            [name],
            participant_cfg,
            prompt_chars=base_prompt_chars[name],
            rounds=1,
            catalog_path=catalog_path,
            extra_prompt_tokens=per_call_image_tokens,
        )
        unpriced_paid.extend(base_unpriced)
        if base_cost is not None:
            cost = (cost or 0.0) + base_cost
        if extra_deliberation_rounds:
            deliberation_cost, deliberation_unpriced = _estimate_input_cost_usd(
                [name],
                participant_cfg,
                prompt_chars=deliberation_chars[name],
                rounds=extra_deliberation_rounds,
                catalog_path=catalog_path,
                extra_prompt_tokens=per_call_image_tokens,
            )
            unpriced_paid.extend(deliberation_unpriced)
            if deliberation_cost is not None:
                cost = (cost or 0.0) + deliberation_cost
    # Fold in the synthesis chair call (one extra single-round call) when it is
    # paid-hosted. Missing chair pricing is part of the same fail-closed check;
    # never silently price the base roster while omitting the chair.
    if synth_billable:
        synth_cost, synth_unpriced = _estimate_input_cost_usd(
            [synth_peer],
            participant_cfg,
            prompt_chars=synth_call_chars,
            rounds=1,
            catalog_path=catalog_path,
        )
        unpriced_paid.extend(synth_unpriced)
        if synth_cost is not None:
            cost = (cost or 0.0) + synth_cost

    if unpriced_paid:
        unique_unpriced = list(dict.fromkeys(unpriced_paid))
        report["violations"].append(
            {
                "limit": "known_paid_hosted_pricing",
                "actual": ", ".join(unique_unpriced),
                "maximum": "configured input_per_million or cached catalog price",
                "participants": unique_unpriced,
            }
        )
    else:
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


def apply_preflight_cost_to_mcp_budget(
    report: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    """Upgrade an MCP input-only report with the full run cost estimate.

    ``mcp_budget_report`` can cheaply guard prompt size and known input spend
    before the richer estimator runs.  The configured MCP cost ceiling is a
    run ceiling, though, so output tokens, outer-retry headroom, and
    synthesis must all be folded in before enforcement.  This mutates the
    report so dry-run and real-run callers expose and enforce the same result.
    """

    cost_total, _token_total, unpriced_paid = summarize_preflight_caps(preflight)
    unique_unpriced = list(dict.fromkeys(unpriced_paid))
    report["estimated_total_cost_usd"] = round(cost_total, 6)
    report["full_cost_estimate_available"] = not unique_unpriced

    violations = list(report.get("violations") or [])
    if unique_unpriced:
        pricing_violation = next(
            (
                item
                for item in violations
                if item.get("limit") == "known_paid_hosted_pricing"
            ),
            None,
        )
        if pricing_violation is None:
            violations.append(
                {
                    "limit": "known_paid_hosted_pricing",
                    "actual": ", ".join(unique_unpriced),
                    "maximum": (
                        "configured input_per_million and output_per_million "
                        "or cached catalog price"
                    ),
                    "participants": unique_unpriced,
                }
            )
        else:
            merged = list(
                dict.fromkeys(
                    [
                        *(pricing_violation.get("participants") or []),
                        *unique_unpriced,
                    ]
                )
            )
            pricing_violation["participants"] = merged
            pricing_violation["actual"] = ", ".join(merged)
            pricing_violation["maximum"] = (
                "configured input_per_million and output_per_million "
                "or cached catalog price"
            )
    else:
        # The cheap report may already contain an input-only cap violation.
        # Replace it with the authoritative whole-run comparison so the
        # reported ``actual`` is the same number the gate used.
        violations = [
            item
            for item in violations
            if item.get("limit") != "max_estimated_cost_usd"
        ]
        maximum = float(report["max_estimated_cost_usd"])
        if cost_total > maximum:
            violations.append(
                {
                    "limit": "max_estimated_cost_usd",
                    "actual": round(cost_total, 6),
                    "maximum": maximum,
                }
            )

    report["violations"] = violations
    report["within_budget"] = not violations


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
    participant_type = cfg.get("type")
    if participant_type not in ("openrouter", "openai_compatible", "ollama"):
        return False
    if is_local_participant(cfg):
        return False
    # `:free` is a hosted-router convention, not an Ollama model-tag promise.
    # A remote Ollama endpoint remains potentially billable unless explicit
    # pricing proves otherwise.
    if participant_type == "ollama":
        return True
    model = str(cfg.get("model") or "")
    return not model.endswith(":free")


def _estimate_input_cost_usd(
    participants: list[str],
    participant_cfg: dict[str, Any],
    *,
    prompt_chars: int,
    rounds: int,
    catalog_path: Path,
    extra_prompt_tokens: Mapping[str, int] | None = None,
) -> tuple[float | None, list[str]]:
    if not participants:
        return None, []
    catalog = _read_cache(catalog_path) or []
    prices_by_model = {
        item.get("id"): item.get("input_per_million")
        for item in catalog
        if item.get("id") and item.get("input_per_million") is not None
    }

    total = 0.0
    any_price = False
    unpriced: list[str] = []
    for name in participants:
        cfg = participant_cfg.get(name, {})
        input_per_million = cfg.get("input_per_million")
        if input_per_million is None:
            input_per_million = prices_by_model.get(cfg.get("model"))
        if input_per_million is None:
            unpriced.append(name)
            continue
        any_price = True
        prompt_tokens = math.ceil(prompt_chars / ESTIMATED_CHARS_PER_TOKEN)
        if extra_prompt_tokens is not None:
            prompt_tokens += max(0, int(extra_prompt_tokens.get(name, 0)))
        total += (
            prompt_tokens
            * rounds
            * float(input_per_million)
            / 1_000_000
        )
    return (total if any_price else None), unpriced
