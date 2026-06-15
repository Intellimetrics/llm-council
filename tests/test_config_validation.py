from pathlib import Path
import tomllib

import pytest
import yaml

from llm_council import __version__
from llm_council.config import load_config, validate_config
from llm_council.defaults import DEFAULT_CONFIG


def test_default_config_validates():
    validate_config(DEFAULT_CONFIG)


def test_package_version_matches_pyproject():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == data["project"]["version"]


def test_load_config_rejects_unknown_participant_reference(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "modes": {
                    "bad": {
                        "participants": ["missing"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown participant 'missing'"):
        load_config(path)


def test_load_config_rejects_bad_cli_args_shape(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "participants": {
                    "claude": {
                        "args": "--not-a-list",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="args must be a string list"):
        load_config(path)


def test_load_config_rejects_bad_cli_prompt_limit(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "participants": {
                    "claude": {
                        "max_prompt_chars": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_prompt_chars"):
        load_config(path)


def test_load_config_rejects_invalid_defaults_mode(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"defaults": {"mode": "missing-mode"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="defaults.mode references unknown mode"):
        load_config(path)


def test_load_config_rejects_bad_include_current(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"modes": {"quick": {"include_current": "yes"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="include_current must be a boolean"):
        load_config(path)


def test_load_config_rejects_invalid_budget_defaults(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"defaults": {"mcp_max_prompt_chars": "not-an-int"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mcp_max_prompt_chars"):
        load_config(path)

    path.write_text(
        yaml.safe_dump({"defaults": {"mcp_max_estimated_cost_usd": "nope"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mcp_max_estimated_cost_usd"):
        load_config(path)


def test_load_config_rejects_bad_defaults_independent_review(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"defaults": {"independent_review": "yes"}}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="defaults.independent_review must be a boolean"
    ):
        load_config(path)


def test_load_config_rejects_bad_mode_independent_review(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"modes": {"review": {"independent_review": 1}}}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="independent_review must be a boolean"
    ):
        load_config(path)


def test_load_config_accepts_independent_review_booleans(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "defaults": {"independent_review": True},
                "modes": {"review": {"independent_review": False}},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["defaults"]["independent_review"] is True
    assert config["modes"]["review"]["independent_review"] is False


def test_load_config_missing_explicit_path_is_clear(tmp_path: Path):
    with pytest.raises(ValueError, match="Config file does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_search_false_uses_defaults():
    config = load_config(None, search=False)
    assert "qwen_coder_flash" in config["participants"]


def test_load_config_rejects_string_fallback_chain(tmp_path: Path):
    """A bare string fallback_chain used to be character-sliced into bogus
    single-char model ids on the quota path — fail fast at load instead."""
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"participants": {"claude": {"fallback_chain": "gpt-5.4"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fallback_chain must be a string list"):
        load_config(path)


def test_load_config_accepts_valid_fallback_chain(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump(
            {"participants": {"claude": {"fallback_chain": ["a", "b"]}}}
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["participants"]["claude"]["fallback_chain"] == ["a", "b"]


def test_load_config_rejects_nonnumeric_timeout_multiplier(tmp_path: Path):
    """timeout_multiplier: "fast" used to pass load then raise an uncaught
    ValueError mid-run in _resolve_effective_timeout."""
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"modes": {"quick": {"timeout_multiplier": "fast"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timeout_multiplier must be a positive number"):
        load_config(path)


def test_load_config_rejects_nonnumeric_idle_timeout(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"participants": {"claude": {"idle_timeout": "fast"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="idle_timeout must be a positive number"):
        load_config(path)


def test_load_config_accepts_zero_timeout_per_kb_chars(tmp_path: Path):
    """0 is the documented "disable size-scaling" sentinel — must validate."""
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"participants": {"claude": {"timeout_per_kb_chars": 0}}}),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["participants"]["claude"]["timeout_per_kb_chars"] == 0


def test_load_config_rejects_negative_timeout_per_kb_chars(tmp_path: Path):
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump({"participants": {"claude": {"timeout_per_kb_chars": -5}}}),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="timeout_per_kb_chars must be a non-negative number"
    ):
        load_config(path)
