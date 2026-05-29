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

    # `config_warnings` was added to the doctor payload so MCP clients can
    # surface the same advisory the CLI prints. An empty list on a clean
    # default config is the expected baseline.
    assert result == {
        "checks": [{"name": "cli:codex", "ok": True, "detail": "ok"}],
        "version": __version__,
        "config_warnings": [],
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


def test_summarize_preflight_caps_extracts_cost_tokens_unpriced():
    """Shared run-cap reduction (CLI + MCP): prefers the retry-safety total,
    sums tokens, and flags only hosted peers with unknown price (CLI/local
    $0 rows are not 'unpriced-paid')."""
    from llm_council.budget import summarize_preflight_caps

    preflight = {
        "known_total_with_retry_safety_usd": 0.42,
        "known_total_usd": 0.30,
        "rows": [
            {"name": "claude", "type": "cli",
             "estimated_input_tokens": 100, "estimated_output_tokens": 50,
             "estimated_total_cost_usd": None},
            {"name": "hosted", "type": "openrouter",
             "estimated_input_tokens": 200, "estimated_output_tokens": 100,
             "estimated_total_cost_usd": None},
            {"name": "priced", "type": "openrouter",
             "estimated_input_tokens": 10, "estimated_output_tokens": 5,
             "estimated_total_cost_usd": 0.01},
        ],
    }
    cost, tokens, unpriced = summarize_preflight_caps(preflight)
    assert cost == 0.42  # retry-safety total preferred over known_total
    assert tokens == 100 + 50 + 200 + 100 + 10 + 5
    assert unpriced == ["hosted"]  # cli row = $0; priced row has a price


def test_summarize_preflight_caps_falls_back_to_known_total():
    from llm_council.budget import summarize_preflight_caps

    cost, tokens, unpriced = summarize_preflight_caps(
        {"known_total_usd": 0.30, "rows": []}
    )
    assert cost == 0.30
    assert tokens == 0
    assert unpriced == []


def test_mcp_budget_report_counts_cross_rank_extra_round():
    """cross_rank runs an extra ranking pass per peer; the pre-flight estimate
    must count it (~+1 round) so a hosted run that should be blocked isn't
    under-counted into passing."""
    from llm_council.budget import mcp_budget_report

    config = {
        "participants": {
            "p": {"type": "openrouter", "model": "m", "input_per_million": 1.0}
        },
        "defaults": {},
    }
    base = mcp_budget_report(
        config=config, participants=["p"], prompt_chars=1000,
        deliberate=False, max_rounds=1,
    )
    ranked = mcp_budget_report(
        config=config, participants=["p"], prompt_chars=1000,
        deliberate=False, max_rounds=1, cross_rank=True,
    )
    assert ranked["estimated_billable_prompt_chars"] == (
        2 * base["estimated_billable_prompt_chars"]
    )
    assert ranked["estimated_input_cost_usd"] > base["estimated_input_cost_usd"]


def test_mcp_budget_report_counts_synthesize_chair_call():
    """A paid-hosted synthesis chair adds one extra call to the estimate."""
    from llm_council.budget import mcp_budget_report

    config = {
        "participants": {
            "p": {"type": "openrouter", "model": "m", "input_per_million": 1.0}
        },
        "defaults": {"synthesizer": "p"},
    }
    base = mcp_budget_report(
        config=config, participants=["p"], prompt_chars=1000,
        deliberate=False, max_rounds=1,
    )
    synth = mcp_budget_report(
        config=config, participants=["p"], prompt_chars=1000,
        deliberate=False, max_rounds=1, synthesize=True,
    )
    assert synth["synthesize_billable"] is True
    assert synth["estimated_billable_prompt_chars"] == (
        base["estimated_billable_prompt_chars"] + 1000
    )
    assert synth["estimated_input_cost_usd"] > base["estimated_input_cost_usd"]


def test_mcp_budget_report_synthesize_noop_when_chair_is_free_local():
    """synthesize must NOT inflate the estimate when the chair is a free/local
    (non-paid-hosted) peer."""
    from llm_council.budget import mcp_budget_report

    config = {
        "participants": {
            "p": {"type": "openrouter", "model": "m", "input_per_million": 1.0},
            "local": {"type": "ollama", "model": "llama3"},
        },
        "defaults": {"synthesizer": "local"},
    }
    base = mcp_budget_report(
        config=config, participants=["p"], prompt_chars=1000,
        deliberate=False, max_rounds=1,
    )
    synth = mcp_budget_report(
        config=config, participants=["p"], prompt_chars=1000,
        deliberate=False, max_rounds=1, synthesize=True,
    )
    assert synth["synthesize_billable"] is False
    assert synth["estimated_billable_prompt_chars"] == (
        base["estimated_billable_prompt_chars"]
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

    budget = result["metadata"]["budget"]
    assert budget["cost_estimate_available"] is True
    assert budget["within_budget"] is False
    assert budget["violations"][0]["limit"] == "max_estimated_cost_usd"


@pytest.mark.asyncio
async def test_mcp_image_paths_outside_cwd_rejected(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(project))
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not really a png")

    with pytest.raises(ValueError, match="outside working directory"):
        await run_council(
            {
                "question": "review the screenshot",
                "working_directory": str(project),
                "image_paths": [str(outside)],
                "dry_run": True,
            }
        )


@pytest.mark.asyncio
async def test_mcp_image_paths_present_does_not_break_budget_check(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    image = tmp_path / "ui.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        + b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
        + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    # dry_run still computes the prompt+budget; image references must flow
    # through without tripping the prompt-size guard or path-resolution.
    result = await run_council(
        {
            "question": "review",
            "working_directory": str(tmp_path),
            "image_paths": [str(image.relative_to(tmp_path))],
            "dry_run": True,
        }
    )
    assert result["metadata"]["prompt_chars"] > 0


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
