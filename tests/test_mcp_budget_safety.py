from pathlib import Path

import pytest

from llm_council.mcp_server import council_run_schema, run_council


@pytest.mark.asyncio
async def test_mcp_rejects_outside_cwd_context_even_when_allowed(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
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


def test_mcp_schema_does_not_expose_allow_outside_cwd():
    assert "allow_outside_cwd" not in council_run_schema()["properties"]


@pytest.mark.asyncio
async def test_mcp_budget_rejects_large_paid_hosted_prompt(tmp_path: Path):
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
    tmp_path: Path,
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  max_estimated_cost_usd: 0.000001
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
async def test_mcp_dry_run_reports_budget_without_enforcing(tmp_path: Path):
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  max_estimated_cost_usd: 0.000001
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
