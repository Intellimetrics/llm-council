"""CLI-level result aggregation and routing precedence regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_council import cli as cli_module
from llm_council.adapters import ParticipantResult
from llm_council.cli import build_parser, cmd_run_async


def _result(name: str, label: str | None) -> ParticipantResult:
    output = (
        f"RECOMMENDATION: {label} - reviewed"
        if label is not None
        else "Review completed without a vote label."
    )
    return ParticipantResult(
        name=name,
        ok=True,
        output=output,
        error="",
        elapsed_seconds=0.1,
    )


def _config(tmp_path: Path, *, policy: str | None = None) -> dict:
    config = {
        "version": 1,
        "transcripts_dir": str(tmp_path / "runs"),
        "defaults": {
            "mode": "custom",
            "secret_scan": "off",
            "catalog_auto_refresh": False,
        },
        "participants": {
            "a": {"type": "cli", "family": "a", "command": "true"},
            "b": {"type": "cli", "family": "b", "command": "true"},
        },
        "modes": {"custom": {"participants": ["a", "b"]}},
    }
    if policy:
        config["quorum_policies"] = {"standard": {"threshold": policy}}
    return config


def _patch_run_scaffolding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: dict,
) -> list[dict]:
    transcript_calls: list[dict] = []
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_a, **_k: [])
    monkeypatch.setattr(cli_module, "maybe_print_update_nag", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_module, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(cli_module, "find_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli_module,
        "estimate_council",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("skip test estimate")),
    )
    monkeypatch.setattr(
        cli_module,
        "transcript_paths",
        lambda *_a, **_k: (tmp_path / "run.md", tmp_path / "run.json"),
    )

    def capture_transcript(*_args, **kwargs):
        transcript_calls.append(kwargs)

    monkeypatch.setattr(cli_module, "write_transcript", capture_transcript)
    return transcript_calls


@pytest.mark.asyncio
async def test_final_round_drives_metadata_webhook_and_quorum_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, policy="majority")
    transcript_calls = _patch_run_scaffolding(monkeypatch, tmp_path, config)
    results = [
        _result("a", "yes"),
        _result("b", "yes"),
        _result("a:round2", "no"),
        _result("b:round2", "no"),
    ]

    async def fake_execute(*_args, **_kwargs):
        return results, {"recommendation": "yes", "rounds": 2}

    webhook_payloads: list[dict] = []
    monkeypatch.setattr(cli_module, "execute_council", fake_execute)
    monkeypatch.setattr(
        "httpx.post",
        lambda _url, **kwargs: webhook_payloads.append(kwargs["json"]),
    )
    config["notifications"] = {"webhook_url": "https://example.invalid/hook"}
    args = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "custom", "Ship it?"]
    )

    assert await cmd_run_async(args) == 1
    metadata = transcript_calls[0]["metadata"]
    assert metadata["recommendation"] == "no"
    assert metadata["agreement_count"] == 2
    assert metadata["recommendation_counts"]["yes"] == 0
    assert metadata["recommendation_counts"]["no"] == 2
    assert "Recommendation: no" in webhook_payloads[0]["text"]
    assert "'yes': 0" in webhook_payloads[0]["text"]
    assert "'no': 2" in webhook_payloads[0]["text"]


@pytest.mark.asyncio
async def test_cli_completion_headline_counts_only_final_round_participants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _patch_run_scaffolding(monkeypatch, tmp_path, config)
    results = [
        _result("a", "yes"),
        _result("b", "yes"),
        _result("a:round2", "yes"),
        _result("b:round2", "yes"),
    ]

    async def fake_execute(*_args, **_kwargs):
        return results, {"rounds": 2, "deliberated": True}

    monkeypatch.setattr(cli_module, "execute_council", fake_execute)
    args = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "custom", "Ship it?"]
    )

    assert await cmd_run_async(args) == 0
    output = capsys.readouterr().out
    assert "2/2 participants succeeded" in output
    assert "4/4 participants succeeded" not in output


@pytest.mark.asyncio
async def test_mcp_summary_counts_final_round_but_keeps_cumulative_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_council.mcp_server as mcp_module

    config = _config(tmp_path)
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    monkeypatch.setattr(mcp_module, "load_project_env", lambda *_a, **_k: [])
    monkeypatch.setattr(mcp_module, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(mcp_module, "find_config", lambda *_a, **_k: None)
    results = [
        _result("a", "yes"),
        _result("b", "yes"),
        _result("a:round2", "yes"),
        _result("b:round2", "yes"),
    ]

    async def fake_execute(*_args, **_kwargs):
        return results, {"rounds": 2, "deliberated": True}

    monkeypatch.setattr(mcp_module, "execute_council", fake_execute)

    payload = await mcp_module.run_council(
        {
            "question": "Ship it?",
            "working_directory": str(tmp_path),
            "mode": "custom",
        }
    )

    assert "2/2 succeeded" in payload["summary_markdown"]
    assert "4/4 succeeded" not in payload["summary_markdown"]
    assert len(payload["results"]) == 4


@pytest.mark.asyncio
async def test_quorum_policy_fails_closed_without_final_vote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, policy="unanimous")
    transcript_calls = _patch_run_scaffolding(monkeypatch, tmp_path, config)

    async def fake_execute(*_args, **_kwargs):
        return [_result("a", None), _result("b", None)], {"rounds": 1}

    monkeypatch.setattr(cli_module, "execute_council", fake_execute)
    args = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "custom", "Ship it?"]
    )

    assert await cmd_run_async(args) == 1
    assert "no usable yes/no/tradeoff votes" in capsys.readouterr().err
    metadata = transcript_calls[0]["metadata"]
    assert metadata["recommendation"] == "unknown"
    assert metadata["agreement_count"] == 0
    assert metadata["total_labeled"] == 0


@pytest.mark.asyncio
async def test_tied_final_vote_is_stamped_unknown_before_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config["modes"]["custom"]["synthesize"] = True
    config["defaults"]["synthesizer"] = "a"
    transcript_calls = _patch_run_scaffolding(monkeypatch, tmp_path, config)
    execute_kwargs: dict = {}
    estimate_kwargs: dict = {}
    focus_dir = tmp_path / ".llm-council" / "review-skills" / "security"
    focus_dir.mkdir(parents=True)
    (focus_dir / "SKILL.md").write_text(
        "---\nname: security\ndescription: Security checks.\n---\nCheck authorization.",
        encoding="utf-8",
    )

    def fake_estimate(**kwargs):
        estimate_kwargs.update(kwargs)
        raise ValueError("skip test estimate")

    async def fake_execute(*_args, **kwargs):
        execute_kwargs.update(kwargs)
        return [_result("a", "yes"), _result("b", "no")], {
            "recommendation": "yes",
            "rounds": 1,
        }

    monkeypatch.setattr(cli_module, "execute_council", fake_execute)
    monkeypatch.setattr(cli_module, "estimate_council", fake_estimate)
    args = build_parser().parse_args(
        [
            "run",
            "--cwd",
            str(tmp_path),
            "--mode",
            "custom",
            "--cross-rank",
            "--focus",
            "security",
            "Ship it?",
        ]
    )

    assert await cmd_run_async(args) == 0
    metadata = transcript_calls[0]["metadata"]
    assert metadata["recommendation"] == "unknown"
    assert metadata["agreement_count"] == 0
    assert metadata["total_labeled"] == 2
    assert metadata["recommendation_tied"] is True
    assert execute_kwargs["synthesize"] is True
    assert execute_kwargs["synthesizer_name"] == "a"
    assert estimate_kwargs["synthesize"] is True
    assert estimate_kwargs["synthesizer_name"] == "a"
    assert estimate_kwargs["cross_rank"] is True
    assert "Check authorization" in estimate_kwargs["focus_directive"]
    assert estimate_kwargs["prepared_participants"] == ["a", "b"]
    assert estimate_kwargs["prepared_prompt"]
    assert "Check authorization" in estimate_kwargs["participant_prompts"]["a"]
    assert execute_kwargs["focus"][0].name == "security"


@pytest.mark.asyncio
async def test_explicit_tier_applies_after_smart_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    order: list[str] = []

    def smart(cfg: dict, _mode: str, _cwd: Path) -> None:
        order.append("smart")
        cfg["participants"]["a"]["model"] = "smart-model"

    def tier(cfg: dict, name: str) -> list[str]:
        assert name == "operator"
        assert cfg["participants"]["a"]["model"] == "smart-model"
        order.append("tier")
        cfg["participants"]["a"]["model"] = "operator-model"
        return ["a"]

    monkeypatch.setattr("llm_council.config.apply_smart_routing", smart)
    monkeypatch.setattr(cli_module, "apply_tier_override", tier)
    _patch_run_scaffolding(monkeypatch, tmp_path, config)
    args = build_parser().parse_args(
        [
            "run",
            "--cwd",
            str(tmp_path),
            "--mode",
            "custom",
            "--tier",
            "operator",
            "--dry-run",
            "Route models",
        ]
    )

    assert await cmd_run_async(args) == 0
    assert order == ["smart", "tier"]
    assert config["participants"]["a"]["model"] == "operator-model"
