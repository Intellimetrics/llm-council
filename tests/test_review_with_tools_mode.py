"""Tests for the experimental `review-with-tools` mode (Phase E — v0.8 plan).

The mode directs CLI peers (claude/codex/gemini) to use their file-read /
grep / glob tools before voting. It must:

- Be registered in DEFAULT_CONFIG and marked experimental.
- Carry a 1.8x timeout multiplier to fund tool calls.
- Route to CLI peers only (no hosted peers via mode config).
- Append the tool-use directive ONLY for CLI peers in tool families
  AND ONLY when mode == "review-with-tools".

Backward-compatibility regressions matter most: the directive must NOT
appear in `review`, `plan`, `consensus`, `mode=None`, or for hosted
families even when forcibly routed into `review-with-tools`.
"""

from __future__ import annotations

from llm_council.config import select_participants
from llm_council.context import (
    REVIEW_WITH_TOOLS_DIRECTIVE,
    apply_per_peer_directives,
)
from llm_council.defaults import DEFAULT_CONFIG


# --- mode registration --------------------------------------------------


def test_review_with_tools_mode_registered_in_defaults():
    modes = DEFAULT_CONFIG["modes"]
    assert "review-with-tools" in modes, (
        "review-with-tools mode must be present in DEFAULT_CONFIG['modes']"
    )


def test_review_with_tools_mode_is_marked_experimental():
    mode_cfg = DEFAULT_CONFIG["modes"]["review-with-tools"]
    assert mode_cfg.get("experimental") is True, (
        "review-with-tools must ship with experimental: true until the "
        "eval-harness promotion gate passes."
    )


def test_review_with_tools_timeout_multiplier_is_1_8():
    mode_cfg = DEFAULT_CONFIG["modes"]["review-with-tools"]
    assert mode_cfg.get("timeout_multiplier") == 1.8, (
        "review-with-tools needs the 1.8x multiplier to fund tool-call "
        "round-trips on top of the 240s baseline."
    )


def test_review_with_tools_routes_to_cli_peers_only(monkeypatch):
    """Default participant list (via select_participants) is CLI-only."""
    config = {
        "participants": {
            "claude": {"type": "cli", "family": "claude", "command": "claude"},
            "codex": {"type": "cli", "family": "codex", "command": "codex"},
            "gemini": {"type": "cli", "family": "gemini", "command": "gemini"},
            "antigravity": {
                "type": "cli",
                "family": "antigravity",
                "command": "agy",
            },
            "qwen_coder_plus": {"type": "openrouter", "family": "qwen"},
        },
        "modes": DEFAULT_CONFIG["modes"],
        "defaults": {},
    }
    # Case A: Antigravity fills the neutral seat.
    monkeypatch.setattr(
        "shutil.which",
        lambda name: f"/bin/{name}" if name in {"claude", "codex", "agy"} else None,
    )
    selected = select_participants(config, mode="review-with-tools", current="claude")
    assert "claude" in selected
    assert "codex" in selected
    assert "antigravity" in selected
    assert "gemini" not in selected
    assert "qwen_coder_plus" not in selected
    assert set(selected) == {"claude", "codex", "antigravity"}

    # Case B: Gemini fills the neutral seat.
    monkeypatch.setattr(
        "shutil.which",
        lambda name: f"/bin/{name}" if name in {"claude", "codex", "gemini"} else None,
    )
    selected = select_participants(config, mode="review-with-tools", current="claude")
    assert "claude" in selected
    assert "codex" in selected
    assert "gemini" in selected
    assert "antigravity" not in selected
    assert "qwen_coder_plus" not in selected
    assert set(selected) == {"claude", "codex", "gemini"}


def test_review_with_tools_does_not_add_hosted_peers():
    """Compared against `review` which adds `qwen_coder_plus`, the tools
    mode must NOT inject hosted peers — that would dilute the experiment.
    """
    mode_cfg = DEFAULT_CONFIG["modes"]["review-with-tools"]
    assert "add" not in mode_cfg or not mode_cfg.get("add"), (
        "review-with-tools must not inject hosted peers via `add`; the "
        "mode exists to test CLI tool-use specifically."
    )


# --- apply_per_peer_directives ------------------------------------------


def _base_prompt() -> str:
    return "You are a read-only participant in an LLM council.\n\nUser question:\nReview this PR.\n"


def test_directive_appended_for_claude_in_tools_mode():
    base = _base_prompt()
    result = apply_per_peer_directives(base, mode="review-with-tools", family="claude")
    assert result != base
    assert REVIEW_WITH_TOOLS_DIRECTIVE in result
    assert result.startswith(base), "Original prompt must be preserved verbatim before the directive."


def test_directive_appended_for_codex_in_tools_mode():
    base = _base_prompt()
    result = apply_per_peer_directives(base, mode="review-with-tools", family="codex")
    assert REVIEW_WITH_TOOLS_DIRECTIVE in result


def test_directive_appended_for_gemini_in_tools_mode():
    base = _base_prompt()
    result = apply_per_peer_directives(base, mode="review-with-tools", family="gemini")
    assert REVIEW_WITH_TOOLS_DIRECTIVE in result


def test_directive_NOT_appended_for_hosted_family_in_tools_mode():
    """Hosted peer routed into tools mode (via --include override) must
    NOT see the directive — the orchestrator's promise that "hosted/local
    peers do not see this instruction" has to hold defensively.
    """
    base = _base_prompt()
    for hosted_family in ("qwen", "deepseek", "glm", "kimi", "openrouter"):
        result = apply_per_peer_directives(
            base, mode="review-with-tools", family=hosted_family
        )
        assert result == base, (
            f"Hosted family {hosted_family!r} must NOT receive the tool "
            "directive even when routed into review-with-tools."
        )
        assert REVIEW_WITH_TOOLS_DIRECTIVE not in result


def test_directive_NOT_appended_for_local_ollama_family_in_tools_mode():
    """Local Ollama peers don't have tool access either; same guard."""
    base = _base_prompt()
    result = apply_per_peer_directives(
        base, mode="review-with-tools", family="local_qwen_coder"
    )
    assert result == base
    assert REVIEW_WITH_TOOLS_DIRECTIVE not in result


def test_directive_NOT_appended_in_review_mode():
    """Backward compat: `review` mode prompts must not change for any peer."""
    base = _base_prompt()
    for family in ("claude", "codex", "gemini", "qwen"):
        result = apply_per_peer_directives(base, mode="review", family=family)
        assert result == base
        assert REVIEW_WITH_TOOLS_DIRECTIVE not in result


def test_directive_NOT_appended_in_other_modes():
    """Cover plan/consensus/quick/deliberate — no behavior regression."""
    base = _base_prompt()
    for mode in ("plan", "consensus", "quick", "deliberate", "diverse", "review-cheap"):
        result = apply_per_peer_directives(base, mode=mode, family="claude")
        assert result == base, f"Mode {mode!r} must not get the directive."


def test_directive_NOT_appended_when_mode_is_None():
    """Backward compat: callers that don't pass a mode get the unchanged prompt."""
    base = _base_prompt()
    result = apply_per_peer_directives(base, mode=None, family="claude")
    assert result == base


def test_directive_NOT_appended_when_family_is_None():
    """Defensive: missing family info means we cannot decide — leave the prompt alone."""
    base = _base_prompt()
    result = apply_per_peer_directives(base, mode="review-with-tools", family=None)
    assert result == base
