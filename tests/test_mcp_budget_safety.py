from pathlib import Path

import pytest

from llm_council import __version__
from llm_council.doctor import Check
from llm_council.mcp_server import (
    council_run_schema,
    list_modes,
    list_models,
    run_council,
    run_doctor,
)


@pytest.mark.asyncio
async def test_mcp_rejects_outside_cwd_context_even_when_allowed(
    tmp_path: Path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(project))
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside working directory"):
        await run_council(
            {
                "question": "read this",
                "working_directory": str(project),
                "context_files": [str(outside)],
                "allow_outside_cwd": True,
                "dry_run": True,
            }
        )


@pytest.mark.asyncio
async def test_mcp_rejects_working_directory_outside_project_root(
    tmp_path: Path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(project))

    with pytest.raises(ValueError, match="inside MCP project root"):
        await run_council(
            {
                "question": "read this",
                "working_directory": str(outside),
                "dry_run": True,
            }
        )


def test_mcp_schema_does_not_expose_allow_outside_cwd():
    assert "allow_outside_cwd" not in council_run_schema()["properties"]


def test_mcp_list_modes_uses_project_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
participants:
  reviewer:
    type: cli
    command: echo
modes:
  custom:
    participants:
    - reviewer
""".lstrip(),
        encoding="utf-8",
    )

    result = list_modes({"working_directory": str(tmp_path)})

    assert result["participants"] == ["reviewer"]
    assert set(result["modes"]) == {"custom"}


def test_mcp_doctor_returns_serialized_checks(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "llm_council.mcp_server.check_environment",
        lambda *args, **kwargs: [Check("cli:codex", True, "ok")],
    )

    result = run_doctor({"working_directory": str(tmp_path)})

    assert result == {
        "checks": [{"name": "cli:codex", "ok": True, "detail": "ok"}],
        "version": __version__,
    }


def test_mcp_list_models_filters_origin_and_limit(monkeypatch):
    monkeypatch.setattr(
        "llm_council.mcp_server.fetch_openrouter_models",
        lambda use_cache=True: [
            {"id": "openai/test", "name": "OpenAI Test", "origin": "US / OpenAI"},
            {"id": "qwen/test", "name": "Qwen Test", "origin": "China / Alibaba Qwen"},
            {"id": "unknown/test", "name": "Mystery Test", "origin": "Unknown"},
        ],
    )

    result = list_models({"filter": "test", "origin": "china", "limit": 1})

    assert [model["id"] for model in result["models"]] == ["qwen/test"]


@pytest.mark.asyncio
async def test_mcp_budget_rejects_large_paid_hosted_prompt(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="max_prompt_chars"):
        await run_council(
            {
                "question": "x" * 81_000,
                "working_directory": str(tmp_path),
                "participants": ["deepseek_v4_pro"],
            }
        )


@pytest.mark.asyncio
async def test_mcp_budget_does_not_inherit_global_prompt_cap(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  max_prompt_chars: 200000
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_prompt_chars"):
        await run_council(
            {
                "question": "x" * 81_000,
                "working_directory": str(tmp_path),
                "participants": ["deepseek_v4_pro"],
            }
        )


@pytest.mark.asyncio
async def test_mcp_budget_rejects_estimated_cost_when_price_is_configured(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  mcp_max_estimated_cost_usd: 0.000001
participants:
  deepseek_v4_pro:
    input_per_million: 100
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_estimated_cost_usd"):
        await run_council(
            {
                "question": "short prompt",
                "working_directory": str(tmp_path),
                "participants": ["deepseek_v4_pro"],
            }
        )


@pytest.mark.asyncio
async def test_mcp_dry_run_reports_budget_without_enforcing(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  mcp_max_estimated_cost_usd: 0.000001
participants:
  deepseek_v4_pro:
    input_per_million: 100
""".lstrip(),
        encoding="utf-8",
    )

    result = await run_council(
        {
            "question": "short prompt",
            "working_directory": str(tmp_path),
            "participants": ["deepseek_v4_pro"],
            "dry_run": True,
        }
    )

    assert result["budget"]["cost_estimate_available"] is True
    assert result["budget"]["within_budget"] is False
    assert result["budget"]["violations"][0]["limit"] == "max_estimated_cost_usd"


@pytest.mark.asyncio
async def test_mcp_budget_rejects_paid_hosted_unknown_price(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
participants:
  custom_paid:
    type: openrouter
    model: example/unknown-paid
    api_key_env: OPENROUTER_API_KEY
modes:
  custom:
    participants:
    - custom_paid
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="custom_paid"):
        await run_council(
            {
                "question": "short prompt",
                "working_directory": str(tmp_path),
                "mode": "custom",
            }
        )
