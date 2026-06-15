"""Configuration loading and participant selection."""

from __future__ import annotations

import copy
import functools
import ipaddress
import os
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from llm_council.defaults import DEFAULT_CONFIG, KNOWN_ORIGIN_STRINGS, VALID_STANCES


BASELINE_CLIS = ("claude", "codex", "gemini", "antigravity")
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
PARTICIPANT_TYPES = frozenset({"cli", "openrouter", "openai_compatible", "ollama"})
OPENAI_COMPATIBLE_TYPES = frozenset({"openrouter", "openai_compatible"})
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})
_TRUSTED_PUBLIC_HOSTS = frozenset({"openrouter.ai"})
BUILTIN_FULL_TRIAD_MODES = frozenset(
    {"quick", "plan", "review", "diverse", "us-only", "deliberate"}
)
CONFIG_NAMES = (
    ".llm-council.yaml",
    ".llm-council.yml",
    "llm-council.yaml",
    "llm-council.yml",
)
OLD_CLAUDE_PLAN_ARGS = [
    "-p",
    "--permission-mode",
    "plan",
    "--tools",
    "Read,Grep,Glob,LS",
    "--no-session-persistence",
]
OLD_CODEX_APPROVAL_ARGS = [
    "exec",
    "--sandbox",
    "read-only",
    "--ask-for-approval",
    "never",
    "--ephemeral",
    "--cd",
    "{cwd}",
    "-",
]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of override into base."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def find_config(start: Path | None = None) -> Path | None:
    """Find a project config by walking upward from start."""

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in CONFIG_NAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def load_config(path: str | Path | None = None, *, search: bool = True) -> dict[str, Any]:
    """Load config, merging project values over built-in defaults."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        config_path = Path(path).expanduser()
    else:
        config_path = find_config() if search else None
    if not config_path:
        validate_config(config)
        return config
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    if data.get("replace_defaults"):
        config["participants"] = {}
        config["modes"] = {}
        data = {key: value for key, value in data.items() if key != "replace_defaults"}
    merged = deep_merge(config, data)
    migrate_known_cli_defaults(merged)
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate the small config surface before any participant is invoked."""

    participants = config.get("participants")
    if not isinstance(participants, dict) or not participants:
        raise ValueError("Config must define a non-empty participants mapping")
    for name, participant in participants.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Participant names must be non-empty strings")
        if not isinstance(participant, dict):
            raise ValueError(f"Participant '{name}' must be a mapping")
        ptype = participant.get("type")
        if ptype not in PARTICIPANT_TYPES:
            raise ValueError(
                f"Participant '{name}' has unsupported type '{ptype}'. "
                "Expected cli, openrouter, openai_compatible, or ollama."
            )
        if ptype == "cli":
            if not participant.get("command"):
                raise ValueError(f"CLI participant '{name}' must define command")
            _validate_string_list(participant, "args", f"CLI participant '{name}'")
            _validate_string_list(
                participant, "env_passthrough", f"CLI participant '{name}'"
            )
            if "env_strict" in participant and not isinstance(
                participant["env_strict"], bool
            ):
                raise ValueError(
                    f"CLI participant '{name}' env_strict must be a boolean"
                )
            _validate_regex_list(
                participant,
                "cli_retry_stderr_patterns",
                f"CLI participant '{name}'",
            )
            # fallback_chain is an ordered list of model ids walked on quota
            # errors. A bare string here (e.g. `fallback_chain: gpt-5.4`) is a
            # common mistake that previously sailed through load and was then
            # character-sliced into bogus single-char model ids on the quota
            # path — fail fast instead.
            if "fallback_chain" in participant and participant["fallback_chain"] is not None:
                _validate_string_list(
                    participant, "fallback_chain", f"CLI participant '{name}'"
                )
        if ptype in {"openrouter", "openai_compatible", "ollama"} and not participant.get("model"):
            raise ValueError(f"Participant '{name}' must define model")
        if ptype == "openai_compatible":
            _validate_openai_compatible_participant(name, participant)
        _validate_positive_int(participant, "timeout", f"participant '{name}'")
        _validate_positive_int(participant, "max_prompt_chars", f"participant '{name}'")
        _validate_positive_int(
            participant, "max_context_tokens", f"participant '{name}'"
        )
        _validate_positive_number(
            participant, "slow_warn_after_seconds", f"participant '{name}'"
        )
        # idle_timeout switches _run_cli_once into a streamed-read loop and is
        # consumed as a float in asyncio deadline arithmetic; timeout_per_kb_chars
        # scales the base timeout by prompt size. Both were previously unvalidated
        # and would crash at runtime (after subprocesses may have launched) on a
        # non-numeric value. timeout_per_kb_chars: 0 is a documented "disable
        # scaling" sentinel, so it is validated as non-negative rather than
        # strictly positive.
        _validate_positive_number(participant, "idle_timeout", f"participant '{name}'")
        _validate_nonnegative_number(
            participant, "timeout_per_kb_chars", f"participant '{name}'"
        )
        if "vision" in participant and not isinstance(participant["vision"], bool):
            raise ValueError(f"Participant '{name}' vision must be a boolean")
        if "retry_on_missing_label" in participant and not isinstance(
            participant["retry_on_missing_label"], bool
        ):
            raise ValueError(
                f"Participant '{name}' retry_on_missing_label must be a boolean"
            )
        if "stance" in participant and participant["stance"] is not None:
            stance_value = participant["stance"]
            if not isinstance(stance_value, str) or stance_value not in VALID_STANCES:
                raise ValueError(
                    f"Participant '{name}' stance must be one of "
                    f"{', '.join(VALID_STANCES)}"
                )
        if "stance_prompt" in participant and participant["stance_prompt"] is not None:
            if not isinstance(participant["stance_prompt"], str) or not participant[
                "stance_prompt"
            ].strip():
                raise ValueError(
                    f"Participant '{name}' stance_prompt must be a non-empty string"
                )

    modes = config.get("modes")
    if not isinstance(modes, dict) or not modes:
        raise ValueError("Config must define a non-empty modes mapping")
    for name, mode in modes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Mode names must be non-empty strings")
        if not isinstance(mode, dict):
            raise ValueError(f"Mode '{name}' must be a mapping")
        has_participants = "participants" in mode
        has_strategy = mode.get("strategy") is not None
        if has_participants == has_strategy:
            raise ValueError(
                f"Mode '{name}' must define exactly one of participants or strategy"
            )
        referenced = list(mode.get("participants") or []) + list(mode.get("add") or [])
        if not all(isinstance(item, str) for item in referenced):
            raise ValueError(f"Mode '{name}' participants/add must contain strings")
        for participant in referenced:
            if participant not in participants:
                raise ValueError(
                    f"Mode '{name}' references unknown participant '{participant}'"
                )
        if has_strategy and mode.get("strategy") not in (
            "other_cli_peers",
            "local_only_peers",
        ):
            raise ValueError(f"Mode '{name}' has unsupported strategy '{mode.get('strategy')}'")
        if mode.get("strategy") == "local_only_peers":
            # `include_current` is meaningless for local_only — the host CLI is
            # never a local participant in this sense (its inference is hosted).
            # Also reject `add` to keep the mode honest: local-only with a
            # hosted addition is contradictory; users wanting a hybrid should
            # use an explicit `participants:` list instead.
            if "include_current" in mode:
                raise ValueError(
                    f"Mode '{name}' (strategy local_only_peers) does not "
                    "support include_current"
                )
            if "add" in mode:
                raise ValueError(
                    f"Mode '{name}' (strategy local_only_peers) does not "
                    "support 'add' — use an explicit participants list for "
                    "hybrid modes"
                )
        if "include_current" in mode and not isinstance(mode["include_current"], bool):
            raise ValueError(f"Mode '{name}' include_current must be a boolean")
        # Independent-review isolation (advisory). Optional per-mode override
        # of the prior-context suppression on continuation runs; boolean when
        # present (absent = inherit defaults/off).
        if "independent_review" in mode and not isinstance(
            mode["independent_review"], bool
        ):
            raise ValueError(f"Mode '{name}' independent_review must be a boolean")
        # No-new-movement early-stop (advisory, opt-in). Boolean when present;
        # a mode-explicit value overrides the global default (absent = inherit).
        if "deliberation_early_stop" in mode and not isinstance(
            mode["deliberation_early_stop"], bool
        ):
            raise ValueError(
                f"Mode '{name}' deliberation_early_stop must be a boolean"
            )
        if mode.get("origin_policy") not in (None, "any", "us"):
            raise ValueError(f"Mode '{name}' origin_policy must be 'any' or 'us'")
        _validate_positive_int(mode, "max_rounds", f"mode '{name}'")
        _validate_positive_int(mode, "min_quorum", f"mode '{name}'")
        # H2 independence warning: optional per-mode override of the
        # distinct-vendor floor. Advisory-only; must be a positive integer
        # when present (absent = feature off for this mode).
        _validate_positive_int(mode, "require_distinct_vendors", f"mode '{name}'")
        # timeout_multiplier is layered onto the per-participant base timeout in
        # _resolve_effective_timeout. A non-numeric value (e.g. "fast") used to
        # pass load and then raise an uncaught ValueError mid-run, after
        # selection + prompt-build; reject it at load time instead.
        _validate_positive_number(mode, "timeout_multiplier", f"mode '{name}'")
        stances = mode.get("stances")
        if stances is not None:
            if not isinstance(stances, dict):
                raise ValueError(f"Mode '{name}' stances must be a mapping")
            for participant_name, stance_value in stances.items():
                if not isinstance(participant_name, str) or not participant_name:
                    raise ValueError(
                        f"Mode '{name}' stances keys must be non-empty strings"
                    )
                if participant_name not in participants:
                    raise ValueError(
                        f"Mode '{name}' stances references unknown "
                        f"participant '{participant_name}'"
                    )
                if not isinstance(stance_value, str) or stance_value not in VALID_STANCES:
                    raise ValueError(
                        f"Mode '{name}' stances['{participant_name}'] must be "
                        f"one of {', '.join(VALID_STANCES)}"
                    )

        # `model_overrides` pins a specific model id on a peer, but only
        # for runs that resolve to this mode. Validation is intentionally
        # light: a stale entry naming a peer not in the resolved roster
        # is a silent no-op at select_participants time, so we don't
        # require the peer to exist at config-load time. We do require
        # the shape to be `dict[str, str]` so a typo (list / int / empty
        # string) fails loudly rather than being silently ignored.
        model_overrides = mode.get("model_overrides")
        if model_overrides is not None:
            if not isinstance(model_overrides, dict):
                raise ValueError(
                    f"Mode '{name}' model_overrides must be a mapping of "
                    "peer-name to model-id string"
                )
            for peer_name, model_id in model_overrides.items():
                if not isinstance(peer_name, str) or not peer_name:
                    raise ValueError(
                        f"Mode '{name}' model_overrides keys must be "
                        "non-empty strings"
                    )
                if not isinstance(model_id, str) or not model_id:
                    raise ValueError(
                        f"Mode '{name}' model_overrides['{peer_name}'] must "
                        "be a non-empty model-id string"
                    )

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Config defaults must be a mapping")
    if defaults.get("origin_policy") not in (None, "any", "us"):
        raise ValueError("defaults.origin_policy must be 'any' or 'us'")
    if defaults.get("mode") and defaults["mode"] not in modes:
        raise ValueError(f"defaults.mode references unknown mode '{defaults['mode']}'")
    # Independent-review isolation (advisory). Optional global default for the
    # prior-context suppression on continuation runs; boolean when present
    # (absent = feature off).
    if "independent_review" in defaults and not isinstance(
        defaults["independent_review"], bool
    ):
        raise ValueError("defaults.independent_review must be a boolean")
    # M9 optional LLM difficulty judge: peer NAME of a hosted participant to
    # consult in `council_recommend`. String when present (absent = feature
    # off). Existence / hosted-ness / key resolution is checked lazily at
    # call time (recommend_judge.grade_difficulty), not here.
    if "recommend_judge" in defaults and not isinstance(
        defaults["recommend_judge"], str
    ):
        raise ValueError("defaults.recommend_judge must be a string")
    # No-new-movement early-stop for deliberation (advisory, opt-in). Boolean
    # when present (absent = feature off). Only meaningful for modes with
    # max_rounds >= 3; with the default max_rounds=2 it never triggers.
    if "deliberation_early_stop" in defaults and not isinstance(
        defaults["deliberation_early_stop"], bool
    ):
        raise ValueError("defaults.deliberation_early_stop must be a boolean")
    _validate_positive_int(defaults, "max_concurrency", "defaults")
    _validate_positive_int(defaults, "max_deliberation_rounds", "defaults")
    _validate_positive_int(defaults, "max_prompt_chars", "defaults")
    _validate_positive_int(defaults, "mcp_max_prompt_chars", "defaults")
    # H2 independence warning: optional global distinct-vendor floor.
    # Advisory-only; positive integer when present (absent = feature off).
    _validate_positive_int(defaults, "min_distinct_vendors", "defaults")
    _validate_positive_number(defaults, "mcp_max_estimated_cost_usd", "defaults")
    # M6 soft cost-warning threshold (advisory only — never gates a run).
    # Non-negative number when present (an explicit 0 means "warn on any
    # estimated spend"); absent = feature off.
    _validate_nonnegative_number(defaults, "cost_warn_usd", "defaults")
    _validate_convergence_thresholds(defaults, "defaults")
    for mode_name, mode in modes.items():
        if isinstance(mode, dict):
            _validate_convergence_thresholds(mode, f"mode '{mode_name}'")


def _validate_openai_compatible_participant(name: str, participant: dict[str, Any]) -> None:
    base_url = participant.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(
            f"openai_compatible participant '{name}' must define base_url "
            "(e.g. https://api.together.xyz/v1)"
        )
    extra_headers = participant.get("extra_headers")
    if extra_headers is not None:
        if not isinstance(extra_headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in extra_headers.items()
        ):
            raise ValueError(
                f"openai_compatible participant '{name}' extra_headers must be a "
                "mapping of string keys to string values"
            )
    allow_private = participant.get("allow_private", False)
    if not isinstance(allow_private, bool):
        raise ValueError(
            f"openai_compatible participant '{name}' allow_private must be a boolean"
        )
    provider_label = participant.get("provider_label")
    if provider_label is not None and (
        not isinstance(provider_label, str) or not provider_label.strip()
    ):
        raise ValueError(
            f"openai_compatible participant '{name}' provider_label must be a "
            "non-empty string"
        )
    parsed = urlparse(base_url.strip())
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(
            f"openai_compatible participant '{name}' base_url is not a valid URL: "
            f"{base_url!r}"
        )
    if allow_private:
        return
    _enforce_public_https_endpoint(name, parsed)


def _enforce_public_https_endpoint(name: str, parsed: Any) -> None:
    if parsed.scheme.lower() != "https":
        raise ValueError(
            f"openai_compatible participant '{name}' base_url must use https:// "
            f"(got scheme {parsed.scheme!r}). Set `allow_private: true` on the "
            "participant to opt in to private/non-https endpoints (e.g. local "
            "Ollama, vLLM, LM Studio)."
        )
    if parsed.username or parsed.password:
        raise ValueError(
            f"openai_compatible participant '{name}' base_url must not contain "
            "embedded credentials (user:pass@host). Use api_key_env and "
            "extra_headers instead."
        )
    host = parsed.hostname
    assert host is not None
    normalized = host.lower().rstrip(".")
    if not normalized:
        raise ValueError(
            f"openai_compatible participant '{name}' base_url has empty hostname"
        )
    if normalized in _LOOPBACK_HOSTNAMES:
        raise ValueError(
            f"openai_compatible participant '{name}' base_url host {host!r} is a "
            "loopback hostname. Set `allow_private: true` on the participant to "
            "opt in to private/non-https endpoints."
        )
    literal = _parse_ip_literal(host)
    if literal is not None and _is_private_ip(literal):
        raise ValueError(
            f"openai_compatible participant '{name}' base_url host {host!r} is a "
            "private/loopback/link-local IP literal. Set `allow_private: true` "
            "on the participant to opt in (e.g. local Ollama, vLLM, LM Studio, "
            "or other on-prem inference)."
        )
    if (
        literal is None
        and (
            normalized in _TRUSTED_PUBLIC_HOSTS
            or normalized.endswith("." + "openrouter.ai")
        )
    ):
        return
    addresses, resolution_error = _resolve_host_addresses(normalized)
    if literal is None and resolution_error is not None:
        raise ValueError(
            f"openai_compatible participant '{name}' base_url host {host!r} "
            f"could not be resolved to verify it is public ({resolution_error}); "
            "refusing to allow it. Set `allow_private: true` on the participant "
            "to skip this check."
        )
    for address in addresses:
        ip = _parse_ip_literal(address)
        if ip is not None and _is_private_ip(ip):
            raise ValueError(
                f"openai_compatible participant '{name}' base_url host {host!r} "
                f"resolves to a private/loopback/link-local address ({address}). "
                "Set `allow_private: true` on the participant to opt in (e.g. "
                "local Ollama, vLLM, LM Studio, or other on-prem inference)."
            )


@functools.lru_cache(maxsize=64)
def _resolve_host_addresses_cached(host: str) -> tuple[tuple[str, ...], str | None]:
    """Cached form of `getaddrinfo` for use in hot paths.

    `is_local_base_url` is called from `select_participants` (every run)
    and from preflight (every run); resolving the same hostname N times
    in close succession is wasteful. The OS resolver caches under the
    hood, but a small in-process cache eliminates the syscall + GIL
    round-trip too. The cache is intentionally small (64 entries) and
    process-lifetime — no TTL — because re-running the council inside
    the same process is the only path that hits this, and the host
    classification (loopback vs RFC1918 vs public) doesn't change
    mid-process for the same hostname.
    """
    if _parse_ip_literal(host) is not None:
        return (), None
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        return (), str(exc) or type(exc).__name__
    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            addresses.append(str(sockaddr[0]))
    return tuple(addresses), None


def _resolve_host_addresses(host: str) -> tuple[list[str], str | None]:
    addresses, error = _resolve_host_addresses_cached(host)
    return list(addresses), error


def _parse_ip_literal(value: str) -> ipaddress._BaseAddress | None:
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0].strip("[]"))
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_private_ip(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def _parse_base_url_host(
    base_url: str,
) -> tuple[str, ipaddress._BaseAddress | None] | None:
    """Shared prefix of :func:`is_loopback_base_url` / :func:`is_local_base_url`.

    Parses `base_url`, extracts and normalizes the host, and parses any IP
    literal. Returns `(normalized_host, ip_literal_or_None)` or `None` when the
    URL is not a non-empty string, fails to parse, or has no hostname. Callers
    apply their own loopback-hostname and IP classification on the result.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        return None
    try:
        parsed = urlparse(base_url.strip())
    except ValueError:
        return None
    host = parsed.hostname
    if not host:
        return None
    normalized = host.lower().rstrip(".")
    return normalized, _parse_ip_literal(host)


def is_loopback_base_url(base_url: str) -> bool:
    """True iff `base_url` points at the loopback interface.

    Stricter than :func:`is_local_base_url` — only matches `localhost`,
    `127.0.0.1`, `[::1]`, `0.0.0.0`, etc. RFC1918 addresses (`10.x`,
    `172.16-31.x`, `192.168.x`) return False. Used by the orchestrator
    pre-flight ping where a 1s timeout is reasonable for loopback but
    can false-fail homelab/VPN servers on a slower LAN link.
    """
    parsed = _parse_base_url_host(base_url)
    if parsed is None:
        return False
    normalized, literal = parsed
    if normalized in _LOOPBACK_HOSTNAMES:
        return True
    if literal is None:
        return False
    # is_loopback covers 127.0.0.0/8 and ::1; is_unspecified covers 0.0.0.0
    # which servers commonly bind to. Exclude is_private (RFC1918) — that's
    # what is_local_base_url handles.
    return literal.is_loopback or literal.is_unspecified


def is_local_base_url(base_url: str) -> bool:
    """True iff `base_url` points at a loopback or RFC1918-style address.

    Used by the `local_only_peers` mode strategy. Mirrors the host classification
    in :func:`_enforce_public_https_endpoint` but inverted — instead of erroring
    on private hosts, it identifies them so the local-only mode can select them.

    Hostnames that fail to resolve are treated as **not** local — better to
    omit a participant from a local-only run than to silently include a peer
    we cannot prove is on-prem.
    """
    parsed = _parse_base_url_host(base_url)
    if parsed is None:
        return False
    normalized, literal = parsed
    if not normalized:
        return False
    if normalized in _LOOPBACK_HOSTNAMES:
        return True
    if literal is not None:
        return _is_private_ip(literal)
    addresses, resolution_error = _resolve_host_addresses(normalized)
    if resolution_error is not None:
        return False
    for address in addresses:
        ip = _parse_ip_literal(address)
        if ip is not None and _is_private_ip(ip):
            return True
    return False


def is_local_participant(cfg: dict[str, Any]) -> bool:
    """True iff a participant runs on the user's machine / private network.

    - `type: ollama` is always local (the adapter only talks to localhost-ish
      servers and the type itself implies on-prem).
    - `type: openai_compatible` is local when its `base_url` resolves loopback
      or RFC1918.
    - `type: cli` and `type: openrouter` are never local for this purpose:
      the binary may run locally but the inference is hosted.
    """
    ptype = cfg.get("type")
    if ptype == "ollama":
        return True
    if ptype == "openai_compatible":
        return is_local_base_url(str(cfg.get("base_url") or ""))
    return False


def migrate_known_cli_defaults(config: dict[str, Any]) -> None:
    """Apply compatibility fixes for previously generated unsafe defaults."""

    claude = config.get("participants", {}).get("claude")
    if isinstance(claude, dict) and (
        claude.get("type") == "cli"
        and claude.get("family") == "claude"
        and claude.get("args") == OLD_CLAUDE_PLAN_ARGS
    ):
        claude["args"] = list(DEFAULT_CONFIG["participants"]["claude"]["args"])
    codex = config.get("participants", {}).get("codex")
    if isinstance(codex, dict) and (
        codex.get("type") == "cli"
        and codex.get("family") == "codex"
        and codex.get("args") == OLD_CODEX_APPROVAL_ARGS
    ):
        codex["args"] = list(DEFAULT_CONFIG["participants"]["codex"]["args"])
    participants = config.get("participants", {})
    if isinstance(participants, dict):
        for participant in participants.values():
            if not isinstance(participant, dict):
                continue
            if participant.get("type") != "openrouter":
                continue
            participant["type"] = "openai_compatible"
            if not participant.get("base_url"):
                participant["base_url"] = OPENROUTER_DEFAULT_BASE_URL
            if not participant.get("api_key_env"):
                participant["api_key_env"] = "OPENROUTER_API_KEY"
    modes = config.get("modes", {})
    if not isinstance(modes, dict):
        return
    if (
        isinstance(participants, dict)
        and all(name in participants for name in ("claude", "codex", "gemini"))
        and "peer-only" not in modes
    ):
        modes["peer-only"] = copy.deepcopy(DEFAULT_CONFIG["modes"]["peer-only"])
    for name in BUILTIN_FULL_TRIAD_MODES:
        mode = modes.get(name)
        if (
            isinstance(mode, dict)
            and mode.get("strategy") == "other_cli_peers"
            and "include_current" not in mode
        ):
            mode["include_current"] = True


def _validate_positive_int(mapping: dict[str, Any], key: str, label: str) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}.{key} must be a positive integer")


def _validate_string_list(mapping: dict[str, Any], key: str, label: str) -> None:
    value = mapping.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} {key} must be a string list")


def _validate_regex_list(mapping: dict[str, Any], key: str, label: str) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} {key} must be a list of regex strings")
    for pattern in value:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"{label} {key} contains invalid regex {pattern!r}: {exc}"
            ) from exc


def _validate_convergence_thresholds(mapping: dict[str, Any], label: str) -> None:
    value = mapping.get("convergence_thresholds")
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{label}.convergence_thresholds must be a mapping")
    allowed = {"converged", "refining"}
    for key, raw in value.items():
        if key not in allowed:
            raise ValueError(
                f"{label}.convergence_thresholds has unknown key '{key}'; "
                f"expected any of {sorted(allowed)}"
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"{label}.convergence_thresholds.{key} must be a number "
                "between 0.0 and 1.0"
            )
        if raw < 0.0 or raw > 1.0:
            raise ValueError(
                f"{label}.convergence_thresholds.{key} must be between 0.0 and 1.0"
            )
    converged = value.get("converged")
    refining = value.get("refining")
    if converged is not None and refining is not None and refining > converged:
        raise ValueError(
            f"{label}.convergence_thresholds.refining must be <= converged"
        )


def _validate_positive_number(mapping: dict[str, Any], key: str, label: str) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive number")


def _validate_nonnegative_number(mapping: dict[str, Any], key: str, label: str) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{label}.{key} must be a non-negative number")


def detect_current_agent() -> str | None:
    """Best-effort detection of the CLI we are currently running under."""

    explicit = os.environ.get("LLM_COUNCIL_CURRENT") or os.environ.get(
        "LLM_COUNCIL_AGENT"
    )
    if explicit:
        normalized = explicit.strip().lower()
        if normalized == "agy":
            normalized = "antigravity"
        return normalized if normalized in BASELINE_CLIS else None

    # Linux-specific parent process walk. If it fails, caller can use all peers.
    try:
        pid = os.getppid()
        seen: set[int] = set()
        while pid > 1 and pid not in seen:
            seen.add(pid)
            cmdline_path = Path("/proc") / str(pid) / "cmdline"
            stat_path = Path("/proc") / str(pid) / "stat"
            raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(
                errors="ignore"
            )
            lowered = raw.lower()
            for name in BASELINE_CLIS:
                if f"/{name}" in lowered or lowered.startswith(name):
                    return name
                if name == "antigravity" and (
                    "/agy" in lowered
                    or "/_agy" in lowered
                    or lowered.startswith("agy")
                ):
                    return "antigravity"
            stat = stat_path.read_text(errors="ignore")
            # /proc/<pid>/stat is: "pid (comm) state ppid ...". `comm` can
            # contain spaces and parentheses (e.g. "(tmux: server)"), so a
            # naive split()[3] mis-indexes and ppid parsing blows up — the
            # broad except then silently aborts the walk and the host CLI is
            # never excluded. Parse the fields AFTER the final ')': they are
            # [state, ppid, ...], so ppid is index 1.
            rest = stat[stat.rindex(")") + 1:].split()
            pid = int(rest[1])
    except Exception:
        return None
    return None


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def apply_tier_override(config: dict[str, Any], tier_name: str) -> list[str]:
    """Swap participant model ids for the named tier; returns swapped peers.

    `defaults.tiers.<tier_name>: {<peer>: <model_id>}` pins the per-peer
    model. Mutates `config` in place. Missing tier name raises ValueError so
    a typo doesn't silently fall through to the default models. Peers absent
    from the tier map are left untouched, so a tier can swap a subset of
    peers without redeclaring the rest.
    """
    defaults = config.get("defaults") or {}
    tiers = defaults.get("tiers") or {}
    if not isinstance(tiers, dict) or tier_name not in tiers:
        available = sorted(k for k in tiers.keys() if isinstance(k, str))
        if available:
            available_msg = f"available tiers: {', '.join(available)}"
        else:
            available_msg = (
                "no tiers configured — add `defaults.tiers.<name>: "
                "{<peer>: <model_id>}` to .llm-council.yaml"
            )
        raise ValueError(
            f"unknown tier '{tier_name}'; {available_msg}"
        )
    tier_map = tiers[tier_name]
    if not isinstance(tier_map, dict) or not tier_map:
        raise ValueError(
            f"tier '{tier_name}' is empty; expected mapping of peer -> model id"
        )
    participants = config.get("participants")
    if not isinstance(participants, dict):
        raise ValueError(
            f"tier '{tier_name}' configured but no participants in config"
        )
    swapped: list[str] = []
    for peer, model_id in tier_map.items():
        if peer not in participants or not isinstance(participants[peer], dict):
            continue
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                f"tier '{tier_name}' entry for peer '{peer}' must be a "
                f"non-empty model id"
            )
        participants[peer]["model"] = model_id
        swapped.append(peer)
    return swapped


def _normalize_origin(value: str) -> str:
    """Strip case, whitespace, and punctuation from an origin string.

    Used to detect near-miss typos in `origin` fields against
    :data:`KNOWN_ORIGIN_STRINGS`. The intent is purely typo detection, not
    fuzzy matching — `usanthropic` should match `US / Anthropic` but
    `usanthrpic` (a missed letter) should NOT, because edit-distance fuzzy
    matching is its own rabbit hole and the high-impact typo class is
    case/spacing/punctuation drift.
    """
    return re.sub(r"[^a-z0-9]", "", value.lower())


def config_warnings(config: dict[str, Any]) -> list[str]:
    """Return non-fatal advisories for a loaded config.

    Today this surfaces near-miss typos in participant `origin` strings —
    a participant whose origin normalizes to a canonical value in
    :data:`KNOWN_ORIGIN_STRINGS` but doesn't match it literally is almost
    certainly a typo (e.g., ``us/anthropic`` for ``US / Anthropic``).
    `origin_policy: us` does literal-prefix matching, so such participants
    silently fail to filter as the user intended.

    The returned strings are informational. Callers (CLI command handlers,
    MCP server) print them to stderr or include them in metadata; nothing
    here changes selection or exit codes.
    """
    warnings: list[str] = []
    canonical_by_normalized: dict[str, str] = {
        _normalize_origin(canonical): canonical
        for canonical in KNOWN_ORIGIN_STRINGS
    }
    participants = config.get("participants", {})
    if not isinstance(participants, dict):
        return warnings
    for name, cfg in participants.items():
        if not isinstance(cfg, dict):
            continue
        origin = cfg.get("origin")
        if not isinstance(origin, str) or not origin.strip():
            continue
        if origin in KNOWN_ORIGIN_STRINGS:
            continue
        normalized = _normalize_origin(origin)
        if not normalized:
            continue
        suggestion = canonical_by_normalized.get(normalized)
        if suggestion is None:
            continue
        warnings.append(
            f"Participant {name!r} has origin {origin!r}, which normalizes "
            f"to {suggestion!r} but does not match literally. "
            f"`origin_policy: us` uses literal-prefix matching ('US / '), "
            f"so this typo silently breaks US-only filtering. "
            f"Did you mean {suggestion!r}?"
        )
    import shutil
    if shutil.which("gemini") and not shutil.which("agy"):
        warnings.append(
            "Gemini CLI is installed but Antigravity CLI is replacing it. "
            "Please install/upgrade to Antigravity CLI (agy) for the best experience."
        )
    return warnings


def select_participants(
    config: dict[str, Any],
    mode: str,
    current: str | None,
    explicit: list[str] | None = None,
    include: list[str] | None = None,
    origin_policy: str | None = None,
) -> list[str]:
    """Resolve participant names for a run."""

    participants = config.get("participants", {})
    modes = config.get("modes", {})

    mode_cfg = modes.get(mode, {})
    effective_origin_policy = (
        origin_policy
        or mode_cfg.get("origin_policy")
        or config.get("defaults", {}).get("origin_policy")
        or "any"
    )

    explicit_requested = bool(explicit)
    if explicit_requested:
        selected = list(explicit)
    else:
        if not mode_cfg:
            raise ValueError(f"Unknown mode '{mode}'. Known modes: {', '.join(modes)}")
        if "participants" in mode_cfg:
            selected = list(mode_cfg["participants"])
        elif mode_cfg.get("strategy") == "other_cli_peers":
            import shutil
            has_agy = bool(shutil.which("agy"))
            has_gemini = bool(shutil.which("gemini"))
            if not has_agy and not has_gemini:
                raise ValueError(
                    "Neither Antigravity CLI (agy) nor Gemini CLI (gemini) "
                    "was found on your PATH. The quick triad mode requires "
                    "at least one Gemini-family CLI. Please install Antigravity CLI."
                )
            neutral_peer = "antigravity" if (has_agy or not has_gemini) else "gemini"
            triad = ["claude", "codex", neutral_peer]

            if mode_cfg.get("include_current", False):
                selected = list(triad)
            else:
                exclusions = {current} if current else set()
                if "gemini" in exclusions or "antigravity" in exclusions:
                    exclusions.add("gemini")
                    exclusions.add("antigravity")
                selected = [name for name in triad if name not in exclusions]
                if not current:
                    selected = list(triad)
            selected.extend(mode_cfg.get("add", []))
        elif mode_cfg.get("strategy") == "local_only_peers":
            selected = [
                name
                for name, cfg in participants.items()
                if is_local_participant(cfg)
            ]
            if not selected:
                raise ValueError(
                    f"Mode '{mode}' (strategy local_only_peers) has no "
                    "matching participants. Add at least one `type: ollama` "
                    "or `type: openai_compatible` participant whose "
                    "base_url is loopback or private (see "
                    "docs/local-models.md)."
                )
        else:
            raise ValueError(f"Mode '{mode}' has no participants or known strategy")

    if include:
        # Strict-mode posture for local_only_peers: refuse runtime --include
        # of hosted peers. Without this, `--mode local-only --include claude`
        # would smuggle a hosted CLI into a "local-only" run despite the
        # mode name and config-time strict checks. Matches the validator's
        # rejection of `add` and `include_current` at config-load time.
        if not explicit_requested and mode_cfg.get("strategy") == "local_only_peers":
            offenders = [
                name
                for name in include
                if name in participants
                and not is_local_participant(participants[name])
            ]
            if offenders:
                raise ValueError(
                    f"Mode '{mode}' (strategy local_only_peers) refuses "
                    f"--include of non-local participants: "
                    f"{', '.join(offenders)}. The mode's purpose is to "
                    "consult only on-prem inference. For a hybrid run, "
                    "use a different mode or pass --participants explicitly."
                )
        selected.extend(include)

    deduped: list[str] = []
    for name in selected:
        if name not in participants:
            raise ValueError(f"Unknown participant '{name}'")
        if effective_origin_policy == "us":
            origin = str(participants[name].get("origin", ""))
            if not origin.startswith("US /"):
                continue
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        raise ValueError(
            "No participants selected"
            + (
                f" after applying origin_policy '{effective_origin_policy}'"
                if effective_origin_policy != "any"
                else ""
            )
        )

    # Multiplex single participant if needed
    mode_cfg = config.get("modes", {}).get(mode, {})
    has_stances = isinstance(mode_cfg, dict) and mode_cfg.get("stances") is not None
    is_debate_mode = mode in ("consensus", "adversarial-red-team", "deliberate", "deep-audit")
    force_multiplex = isinstance(mode_cfg, dict) and mode_cfg.get("single_llm_multiplex") is True

    if mode == "single-llm" or force_multiplex:
        if deduped:
            deduped = [deduped[0]]

    if len(deduped) == 1 and (has_stances or is_debate_mode or force_multiplex or mode == "single-llm"):
        base_name = deduped[0]
        base_cfg = participants.get(base_name)
        if isinstance(base_cfg, dict):
            # Create three virtual peers
            deduped = [f"{base_name}_for", f"{base_name}_against", f"{base_name}_neutral"]
            
            # Add them to participants config
            for suffix, stance in [("_for", "for"), ("_against", "against"), ("_neutral", "neutral")]:
                virtual_name = f"{base_name}{suffix}"
                virtual_cfg = dict(base_cfg)
                virtual_cfg["stance"] = stance
                participants[virtual_name] = virtual_cfg
            
            # Seed stances in the mode config so balance_stances preserves them
            if "modes" not in config:
                config["modes"] = {}
            if mode not in config["modes"]:
                config["modes"][mode] = {}
            config["modes"][mode]["stances"] = {
                f"{base_name}_for": "for",
                f"{base_name}_against": "against",
                f"{base_name}_neutral": "neutral",
            }
            # Refresh mode_cfg reference
            mode_cfg = config["modes"][mode]

    # Per-mode model overrides. Highest priority in the resolution chain
    # (base participants.<peer>.model -> tiers.<tier>.<peer> swap, already
    # applied by apply_tier_override before we get here -> this). Mutates
    # config["participants"][peer]["model"] in place so downstream code
    # that reads the participant dict picks up the pin. Silent on stale
    # entries: an override naming a peer absent from the resolved roster
    # is a no-op rather than an error. Shape (dict[str, str] non-empty)
    # is already enforced by validate_config; we only re-check membership
    # in the resolved roster + that the participant entry exists.
    model_overrides = mode_cfg.get("model_overrides") if mode_cfg else None
    if isinstance(model_overrides, dict):
        for peer, model_id in model_overrides.items():
            if peer not in deduped:
                continue
            participant_entry = participants.get(peer)
            if isinstance(participant_entry, dict):
                participant_entry["model"] = model_id

    return deduped


def balance_stances(active_participants: list[str], mode_stances: dict[str, str] | None) -> dict[str, str] | None:
    """Dynamically balance assigned stances when a stance role is missing.

    If mode_stances contains stance assignments, we filter them to the active
    participants and ensure that the 'for', 'against', and 'neutral' stances
    are assigned as evenly as possible (e.g. exactly 1 of each if N=3, or 1 'for'
    and 1 'against' if N=2). Original stance preferences are preserved where
    possible.
    """
    if mode_stances is None:
        return None

    if not active_participants:
        return {}

    n = len(active_participants)

    # Target counts:
    target_counts = {
        "for": n // 3 + (1 if n % 3 >= 1 else 0),
        "against": n // 3 + (1 if n % 3 >= 2 else 0),
        "neutral": n // 3
    }

    assigned: dict[str, str] = {}
    remaining_targets = dict(target_counts)

    # First pass: assign to those whose original stance is still available under remaining_targets.
    unassigned = []
    for p in active_participants:
        orig = mode_stances.get(p, "neutral")
        if remaining_targets.get(orig, 0) > 0:
            assigned[p] = orig
            remaining_targets[orig] -= 1
        else:
            unassigned.append(p)

    # Second pass: assign the unassigned participants to whatever target stances are still remaining.
    for p in unassigned:
        for stance, count in remaining_targets.items():
            if count > 0:
                assigned[p] = stance
                remaining_targets[stance] -= 1
                break

    return assigned


def apply_smart_routing(config: dict[str, Any], mode: str, cwd: Path) -> None:
    """Scan the git diff and context files to dynamically downgrade premium models to cheaper alternatives if changes are low-risk."""
    smart_cfg = config.get("smart_routing", {})
    if isinstance(smart_cfg, dict) and smart_cfg.get("enabled", True) is False:
        return

    # Get changed files
    from llm_council.context import _git_output
    changed_files = []
    try:
        git_staged = _git_output(cwd, ["diff", "--cached", "--name-only"])
        if git_staged:
            changed_files.extend(git_staged.splitlines())
        git_unstaged = _git_output(cwd, ["diff", "--name-only"])
        if git_unstaged:
            changed_files.extend(git_unstaged.splitlines())
    except Exception:
        pass

    if not changed_files:
        return

    # Count lines changed
    lines_changed = 0
    try:
        diff_summary = _git_output(cwd, ["diff", "--shortstat"])
        if diff_summary:
            import re
            m = re.search(r"(\d+) insertion", diff_summary)
            if m:
                lines_changed += int(m.group(1))
            m = re.search(r"(\d+) deletion", diff_summary)
            if m:
                lines_changed += int(m.group(1))
    except Exception:
        pass

    low_risk_extensions = smart_cfg.get("low_risk_extensions") or [".md", ".txt", ".png", ".jpg", ".css", ".html", ".scss"]
    max_lines = smart_cfg.get("max_lines_changed", 30)

    all_low_risk_ext = all(
        any(f.endswith(ext) for ext in low_risk_extensions)
        for f in changed_files
    )
    
    is_low_risk = all_low_risk_ext or (0 < lines_changed <= max_lines)
    
    if is_low_risk:
        from llm_council.defaults import DEFAULT_CHEAPER_MODELS
        cheaper_mapping = dict(DEFAULT_CHEAPER_MODELS)
        if isinstance(smart_cfg.get("cheaper_models"), dict):
            cheaper_mapping.update(smart_cfg["cheaper_models"])
            
        participants = config.get("participants", {})
        if isinstance(participants, dict):
            swapped = []
            for p_name, p_cfg in participants.items():
                if isinstance(p_cfg, dict) and p_cfg.get("model"):
                    orig_model = p_cfg["model"]
                    if orig_model in cheaper_mapping:
                        cheaper_model = cheaper_mapping[orig_model]
                        p_cfg["model"] = cheaper_model
                        swapped.append(f"{p_name} ({orig_model} -> {cheaper_model})")
            if swapped:
                import sys
                print(
                    f"[Smart Routing] Swapped premium models for cheaper alternatives (low-risk diff detected): "
                    f"{', '.join(swapped)}",
                    file=sys.stderr,
                    flush=True
                )


