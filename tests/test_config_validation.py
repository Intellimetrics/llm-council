from pathlib import Path

import pytest
import yaml

from llm_council.config import load_config, validate_config
from llm_council.defaults import DEFAULT_CONFIG


def test_default_config_validates():
    validate_config(DEFAULT_CONFIG)


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


def test_load_config_missing_explicit_path_is_clear(tmp_path: Path):
    with pytest.raises(ValueError, match="Config file does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_search_false_uses_defaults():
    config = load_config(None, search=False)
    assert "qwen_coder_flash" in config["participants"]
