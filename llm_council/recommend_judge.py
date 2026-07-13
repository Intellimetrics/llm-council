"""Optional LLM difficulty judge for `council_recommend` (M9).

Default OFF and strictly fail-open: this module makes ONE small hosted
chat-completion call to grade a task's difficulty as a supplementary signal
alongside the always-on mechanical verdict from `policy.recommend`. It NEVER
raises out of `grade_difficulty` and NEVER routes through the full council
participant machinery (no RECOMMENDATION-label / envelope validation) — it is
a plain JSON request.

Enable by setting `defaults.recommend_judge: <peer-name>` in
`.llm-council.yaml`, where the named peer is a HOSTED participant
(`openrouter` or `openai_compatible`) with a `model` and a resolvable API key.
Anything missing => feature stays off (returns None).
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from llm_council.adapters import (
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_HEADERS,
    _is_openrouter_endpoint,
)
from llm_council.env import env_get

# Hosted participant types eligible to act as the judge.
_HOSTED_TYPES = frozenset({"openrouter", "openai_compatible"})

_JUDGE_MAX_TOKENS = 512
_JUDGE_TIMEOUT_SECONDS = 20.0
_JUDGE_VALID_DIFFICULTIES = frozenset({"TRIVIAL", "MODERATE", "HARD"})

_JUDGE_SYSTEM_PROMPT = (
    "You are a triage assistant grading how difficult a software task is. "
    "Respond with STRICT JSON only — no prose, no code fences — of the form: "
    '{"difficulty": "TRIVIAL"|"MODERATE"|"HARD", "rationale": "<one sentence>", '
    '"suggested_mode": "<short mode hint, e.g. quick, plan, review>"}'
)

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _resolve_judge_peer(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the resolved judge peer config, or None when the feature is
    off / misconfigured. Pure (no I/O beyond reading env for the key)."""
    defaults = config.get("defaults") or {}
    peer_name = defaults.get("recommend_judge")
    if not peer_name or not isinstance(peer_name, str):
        return None
    participants = config.get("participants") or {}
    peer = participants.get(peer_name)
    if not isinstance(peer, dict):
        return None
    if peer.get("type") not in _HOSTED_TYPES:
        return None
    model = peer.get("model")
    if not model or not isinstance(model, str):
        return None
    # Resolve the API key. openrouter defaults to OPENROUTER_API_KEY;
    # openai_compatible has no universal default, so require explicit
    # api_key_env (mirrors the missing-key pre-drop asymmetry).
    if peer.get("type") == "openrouter":
        key_env = peer.get("api_key_env") or "OPENROUTER_API_KEY"
    else:
        key_env = peer.get("api_key_env")
    if not key_env or not isinstance(key_env, str):
        return None
    api_key = env_get(key_env)
    if not api_key:
        return None
    return {"peer": peer, "model": model, "api_key": api_key}


def _parse_judge_json(content: str) -> dict[str, Any] | None:
    """Leniently parse the model's JSON reply. Strips code fences. Returns
    None on any parse failure or invalid difficulty."""
    if not content or not content.strip():
        return None
    cleaned = _CODE_FENCE_RE.sub("", content.strip())
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    difficulty = data.get("difficulty")
    if not isinstance(difficulty, str):
        return None
    difficulty = difficulty.strip().upper()
    if difficulty not in _JUDGE_VALID_DIFFICULTIES:
        return None
    rationale = data.get("rationale")
    suggested_mode = data.get("suggested_mode")
    return {
        "difficulty": difficulty,
        "rationale": rationale if isinstance(rationale, str) else "",
        "suggested_mode": (
            suggested_mode if isinstance(suggested_mode, str) else ""
        ),
    }


async def grade_difficulty(task: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Grade task difficulty via a single hosted chat-completion call.

    Returns a dict ``{"difficulty", "rationale", "suggested_mode"}`` on
    success, or None when the feature is off / misconfigured / any failure
    occurs. NEVER raises.
    """
    try:
        resolved = _resolve_judge_peer(config)
        if resolved is None:
            return None
        peer = resolved["peer"]
        model = resolved["model"]
        api_key = resolved["api_key"]

        base_url = str(
            peer.get("base_url") or OPENROUTER_DEFAULT_BASE_URL
        ).rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        is_openrouter = _is_openrouter_endpoint(base_url)

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": _JUDGE_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Grade the difficulty of this software task and "
                        "return STRICT JSON only:\n\n" + (task or "")
                    ),
                },
            ],
        }
        # Only OpenRouter reliably honors response_format json_object; for a
        # generic openai_compatible endpoint we rely on the prompt + lenient
        # parse instead.
        if is_openrouter:
            payload["response_format"] = {"type": "json_object"}

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if is_openrouter:
            headers.update(OPENROUTER_HEADERS)

        content = await _post_once_with_retry(endpoint, headers, payload)
        if content is None:
            return None
        return _parse_judge_json(content)
    except Exception:
        # Fail open on absolutely anything — the judge is supplementary.
        return None


async def _post_once_with_retry(
    endpoint: str, headers: dict[str, str], payload: dict[str, Any]
) -> str | None:
    """POST the chat-completion request with one retry on transient failure.
    Returns the assistant message text, or None on any failure."""
    for _attempt in range(2):  # original + one retry
        try:
            async with httpx.AsyncClient(
                timeout=_JUDGE_TIMEOUT_SECONDS, follow_redirects=False
            ) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
            if response.status_code < 200 or response.status_code >= 300:
                continue
            data = response.json()
            if data.get("error"):
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
        except Exception:  # noqa: BLE001 — fail open, try once more
            continue
    # Either no usable content or every attempt errored — fail open.
    return None
