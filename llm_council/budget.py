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
) -> dict[str, Any]:
    participant_cfg = config.get("participants", {})
    paid_hosted = [
        name
        for name in participants
        if _is_paid_hosted_participant(participant_cfg.get(name, {}))
    ]
    budgeted_rounds = max(1, int(max_rounds or 1)) if deliberate else 1
    limits = _budget_limits(config)
    report = {
        "max_prompt_chars": limits["max_prompt_chars"],
        "max_estimated_cost_usd": limits["max_estimated_cost_usd"],
        "prompt_chars": prompt_chars,
        "budgeted_rounds": budgeted_rounds,
        "paid_hosted_participants": paid_hosted,
        "estimated_billable_prompt_chars": prompt_chars * budgeted_rounds * len(paid_hosted),
        "estimated_input_cost_usd": None,
        "cost_estimate_available": False,
        "violations": [],
    }

    guarded_by_prompt_size = bool(paid_hosted) or deliberate or budgeted_rounds > 1
    if guarded_by_prompt_size and prompt_chars > limits["max_prompt_chars"]:
        report["violations"].append(
            {
                "limit": "max_prompt_chars",
                "actual": prompt_chars,
                "maximum": limits["max_prompt_chars"],
            }
        )

    cost = _estimate_input_cost_usd(
        paid_hosted,
        participant_cfg,
        prompt_chars=prompt_chars,
        rounds=budgeted_rounds,
        catalog_path=openrouter_cache_path(),
    )
    if paid_hosted and cost is None:
        report["violations"].append(
            {
                "limit": "known_paid_hosted_pricing",
                "actual": ", ".join(paid_hosted),
                "maximum": "configured input_per_million or cached catalog price",
                "participants": paid_hosted,
            }
        )
    elif cost is not None:
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
