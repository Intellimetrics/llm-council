"""Tests for per-mode `model_overrides` resolution in select_participants.

`modes.<name>.model_overrides: {peer: model_id}` pins a peer's model
when that mode is active. Resolution chain (highest priority wins):
    participants.<peer>.model (base) ->
    tiers.<tier>.<peer> (apply_tier_override; --tier <name>) ->
    modes.<name>.model_overrides.<peer>  (this feature)

The override mutates `config["participants"][peer]["model"]` in place so
downstream code (orchestrator, adapters) sees the pinned model with no
schema change. Only the `model` field is touched.
"""

from copy import deepcopy

import pytest

from llm_council.config import (
    apply_tier_override,
    load_config,
    select_participants,
    validate_config,
)


def _base_config() -> dict:
    """Minimal config matching defaults' shape, with three CLI peers
    and the built-in `quick` and `review` modes so we can assert
    overrides apply per-mode rather than globally.
    """
    return {
        "defaults": {},
        "participants": {
            "claude": {
                "type": "cli",
                "family": "claude",
                "origin": "US / Anthropic",
                "command": "claude",
                "args": ["-p", "--permission-mode", "default"],
                "model": "anthropic/claude-sonnet-4-6",
                "timeout": 240,
            },
            "codex": {
                "type": "cli",
                "family": "codex",
                "origin": "US / OpenAI",
                "command": "codex",
                "args": ["exec", "--sandbox", "read-only"],
                "model": "openai/gpt-5",
                "timeout": 240,
            },
            "gemini": {
                "type": "cli",
                "family": "gemini",
                "origin": "US / Google",
                "command": "gemini",
                "args": ["--approval-mode", "plan"],
                "model": "google/gemini-2.5-pro",
                "timeout": 240,
            },
        },
        "modes": {
            "quick": {
                "participants": ["claude", "codex", "gemini"],
            },
            "review": {
                "participants": ["claude", "codex", "gemini"],
            },
        },
    }


def test_mode_without_model_overrides_leaves_base_model_unchanged():
    """No regression: a mode that does NOT declare model_overrides
    must leave each peer's base `model` field untouched."""
    config = _base_config()
    before = {
        name: cfg["model"] for name, cfg in config["participants"].items()
    }

    selected = select_participants(config, "review", current=None)

    assert selected == ["claude", "codex", "gemini"]
    after = {
        name: cfg["model"] for name, cfg in config["participants"].items()
    }
    assert after == before


def test_model_overrides_pins_model_for_named_peer():
    """A mode with `model_overrides: {claude: "..."}` swaps claude's
    model when that mode is active; other peers stay on their base."""
    config = _base_config()
    config["modes"]["review"]["model_overrides"] = {
        "claude": "anthropic/claude-opus-4-7",
    }

    selected = select_participants(config, "review", current=None)

    assert selected == ["claude", "codex", "gemini"]
    assert (
        config["participants"]["claude"]["model"]
        == "anthropic/claude-opus-4-7"
    )
    # Other peers untouched.
    assert config["participants"]["codex"]["model"] == "openai/gpt-5"
    assert config["participants"]["gemini"]["model"] == "google/gemini-2.5-pro"


def test_model_overrides_silently_ignores_peer_outside_roster():
    """An override naming a peer NOT in the resolved participant list
    is a no-op — the operator may have left a stale entry. No error,
    no warning, no mutation of the absent peer's model."""
    config = _base_config()
    # `quick` mode here resolves to claude/codex/gemini. An override
    # targeting `qwen` (never in this config) must be a silent no-op.
    config["modes"]["quick"]["model_overrides"] = {
        "qwen": "alibaba/qwen-coder-plus",
    }

    selected = select_participants(config, "quick", current=None)

    assert selected == ["claude", "codex", "gemini"]
    # No KeyError, no mutation of unrelated peers.
    assert config["participants"]["claude"]["model"] == "anthropic/claude-sonnet-4-6"
    assert "qwen" not in config["participants"]


def test_model_overrides_beats_tier_swap():
    """Resolution order requires `model_overrides` to win against a
    `--tier` swap. apply_tier_override mutates the base model first,
    then select_participants applies the mode override on top."""
    config = _base_config()
    config["defaults"]["tiers"] = {
        "deep": {
            "claude": "anthropic/claude-opus-4-6",
            "codex": "openai/gpt-5-pro",
        }
    }
    config["modes"]["review"]["model_overrides"] = {
        "claude": "anthropic/claude-opus-4-7",
    }

    # Mirror the CLI / MCP call order: tier override first, then
    # participant selection (which applies model_overrides).
    swapped = apply_tier_override(config, "deep")
    assert sorted(swapped) == ["claude", "codex"]

    select_participants(config, "review", current=None)

    # claude: tier swapped to 4-6, then mode override pins 4-7.
    assert (
        config["participants"]["claude"]["model"]
        == "anthropic/claude-opus-4-7"
    )
    # codex: tier swap stands (no per-mode override for codex).
    assert config["participants"]["codex"]["model"] == "openai/gpt-5-pro"
    # gemini: neither layer touches it.
    assert config["participants"]["gemini"]["model"] == "google/gemini-2.5-pro"


def test_model_overrides_only_touches_model_field():
    """Only the `model` key is mutated. args, timeout, type, family,
    origin, command, etc. must be unchanged after override applies."""
    config = _base_config()
    config["modes"]["review"]["model_overrides"] = {
        "claude": "anthropic/claude-opus-4-7",
    }

    before = deepcopy(config["participants"]["claude"])

    select_participants(config, "review", current=None)

    after = config["participants"]["claude"]
    # The model field is intentionally changed.
    assert after["model"] == "anthropic/claude-opus-4-7"
    assert before["model"] != after["model"]
    # Every other key must be byte-identical.
    for key in ("type", "family", "origin", "command", "args", "timeout"):
        assert after[key] == before[key], f"field '{key}' mutated unexpectedly"
    # Same key set — no field added or removed.
    assert set(after.keys()) == set(before.keys())


def test_model_overrides_do_not_leak_across_modes():
    """Mode A's overrides must not bleed into mode B. After running
    select_participants on mode A and then mode B on a fresh config,
    mode B's resolved models must equal the base configuration."""
    config = _base_config()
    config["modes"]["quick"]["model_overrides"] = {
        "claude": "anthropic/claude-opus-4-7",
    }
    # `review` has NO model_overrides — it should resolve to base.

    # First run resolves `quick` and pins claude on the shared dict.
    select_participants(config, "quick", current=None)
    assert (
        config["participants"]["claude"]["model"]
        == "anthropic/claude-opus-4-7"
    )

    # Now run `review` on a FRESH config (mirrors process restart /
    # fresh load_config). `review` has no override, so claude must
    # land back on the base sonnet model — proves the override is
    # mode-scoped, not a persistent edit baked into defaults.
    fresh = _base_config()
    select_participants(fresh, "review", current=None)
    assert (
        fresh["participants"]["claude"]["model"]
        == "anthropic/claude-sonnet-4-6"
    )


def test_validate_config_rejects_non_mapping_model_overrides(tmp_path):
    """A `model_overrides: [list]` is a typo — fail loudly at load."""
    config = _base_config()
    config["modes"]["review"]["model_overrides"] = ["claude", "opus"]

    with pytest.raises(ValueError, match="model_overrides must be a mapping"):
        validate_config(config)


def test_validate_config_rejects_empty_model_id():
    """`model_overrides: {claude: ""}` is a typo — empty model id
    silently breaks downstream lookups, so fail at config-load."""
    config = _base_config()
    config["modes"]["review"]["model_overrides"] = {"claude": ""}

    with pytest.raises(ValueError, match="non-empty model-id string"):
        validate_config(config)


def test_default_config_has_no_model_overrides():
    """The shipped defaults intentionally carry NO model_overrides on
    any built-in mode. This pins the v0.8 invariant: do not ship
    vendor-affinity defaults until eval-harness data supports them."""
    config = load_config(None, search=False)
    for mode_name, mode_cfg in config.get("modes", {}).items():
        assert "model_overrides" not in mode_cfg, (
            f"Built-in mode {mode_name!r} ships with model_overrides — "
            "remove before merge (see Phase D of v0.8 plan)."
        )
