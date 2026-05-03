"""Live model catalog helpers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = 3600

PROVIDER_ORIGINS = {
    "ai21": "Israel / AI21",
    "amazon": "US / Amazon",
    "anthropic": "US / Anthropic",
    "cohere": "Canada / Cohere",
    "deepseek": "China / DeepSeek",
    "google": "US / Google",
    "meta-llama": "US / Meta",
    "microsoft": "US / Microsoft",
    "minimax": "China / MiniMax",
    "moonshotai": "China / Moonshot AI",
    "mistralai": "France / Mistral AI",
    "nvidia": "US / NVIDIA",
    "openai": "US / OpenAI",
    "qwen": "China / Alibaba Qwen",
    "x-ai": "US / xAI",
    "z-ai": "China / Z.ai",
}


def price_per_million(raw_price: str | int | float | None) -> float | None:
    if raw_price in (None, ""):
        return None
    return float(raw_price) * 1_000_000


def infer_origin(model_id: str) -> str:
    provider = model_id.split("/", 1)[0].lstrip("~")
    return PROVIDER_ORIGINS.get(provider, "Unknown")


def normalize_openrouter_model(model: dict[str, Any]) -> dict[str, Any]:
    pricing = model.get("pricing") or {}
    return {
        "id": model.get("id", ""),
        "name": model.get("name") or model.get("id", ""),
        "origin": infer_origin(str(model.get("id", ""))),
        "context_length": model.get("context_length"),
        "input_per_million": price_per_million(pricing.get("prompt")),
        "output_per_million": price_per_million(pricing.get("completion")),
    }


def openrouter_cache_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return cache_root / "llm-council" / "openrouter-models.json"


def fetch_openrouter_models(
    timeout: float = 30, *, use_cache: bool = True, allow_network: bool = True
) -> list[dict[str, Any]]:
    """Fetch the OpenRouter model catalog.

    `use_cache=True` reads from the disk cache when fresh. `allow_network=False`
    additionally refuses to fall through to a live HTTP fetch on cache miss
    or stale cache, so callers that need a fast no-network path (e.g. the
    pre-flight budget gate) can fail cleanly instead of stalling on the
    network. Returns an empty list when both paths are denied.
    """
    cache_path = openrouter_cache_path()
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached
    if not allow_network:
        return []
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    models = [normalize_openrouter_model(model) for model in data.get("data", [])]
    if use_cache:
        _write_cache(cache_path, models)
    return models


def refresh_openrouter_cache(timeout: float = 30) -> dict[str, Any]:
    """Force-fetch the catalog and overwrite the disk cache.

    Returns a small summary so the CLI can print model count + cache age.
    Raises on network failure — callers (the explicit `models refresh`
    subcommand) want a clear error, not silent silence.
    """
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    models = [normalize_openrouter_model(model) for model in data.get("data", [])]
    cache_path = openrouter_cache_path()
    _write_cache(cache_path, models)
    return {
        "model_count": len(models),
        "cache_path": str(cache_path),
        "fetched_at": time.time(),
    }


def openrouter_cache_age_seconds() -> float | None:
    """Seconds since the catalog cache was last written, or None if missing."""
    path = openrouter_cache_path()
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _read_cache(path: Path) -> list[dict[str, Any]] | None:
    try:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) else None


def _write_cache(path: Path, models: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models, indent=2) + "\n")
