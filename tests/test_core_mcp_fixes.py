from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import yaml

from llm_council.adapters import ParticipantResult
from llm_council.budget import (
    enforce_preflight_caps,
    mcp_budget_report,
    summarize_preflight_caps,
)
from llm_council.deliberation import summarize_recommendations
from llm_council.env import env_get, project_env_context, resolve_project_env
from llm_council.estimate import estimate_council
from llm_council import mcp_server


def _result(name: str, label: str) -> ParticipantResult:
    return ParticipantResult(
        name=name,
        ok=True,
        output=f"RECOMMENDATION: {label}",
        error="",
        elapsed_seconds=0.0,
    )


@pytest.mark.asyncio
async def test_project_env_context_is_request_local_and_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    (project_a / ".llm-council.env").write_text(
        "COUNCIL_SCOPE_TEST=project-a\n", encoding="utf-8"
    )
    (project_b / ".llm-council.env").write_text(
        "COUNCIL_SCOPE_TEST=project-b\n", encoding="utf-8"
    )
    monkeypatch.delenv("COUNCIL_SCOPE_TEST", raising=False)

    ready_a = asyncio.Event()
    ready_b = asyncio.Event()

    async def read_scoped(project: Path, expected: str, own, other) -> str | None:
        with project_env_context(project):
            own.set()
            await other.wait()
            await asyncio.sleep(0)
            assert env_get("COUNCIL_SCOPE_TEST") == expected
            return env_get("COUNCIL_SCOPE_TEST")

    values = await asyncio.gather(
        read_scoped(project_a, "project-a", ready_a, ready_b),
        read_scoped(project_b, "project-b", ready_b, ready_a),
    )

    assert values == ["project-a", "project-b"]
    assert "COUNCIL_SCOPE_TEST" not in os.environ


def test_resolve_project_env_matches_cli_precedence_and_interpolation(
    tmp_path: Path,
):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / ".env").write_text(
        "COUNCIL_PRECEDENCE=parent\n", encoding="utf-8"
    )
    (child / ".env").write_text(
        "COUNCIL_PRECEDENCE=child\n"
        "COUNCIL_FROM_DOTENV=${COUNCIL_PRECEDENCE}\n",
        encoding="utf-8",
    )
    (child / ".llm-council.env").write_text(
        "COUNCIL_PRECEDENCE=authoritative\n"
        "COUNCIL_FROM_COUNCIL=${COUNCIL_PRECEDENCE}\n",
        encoding="utf-8",
    )

    effective, loaded = resolve_project_env(child, base_env={})

    assert effective["COUNCIL_FROM_DOTENV"] == "child"
    assert effective["COUNCIL_PRECEDENCE"] == "authoritative"
    assert effective["COUNCIL_FROM_COUNCIL"] == "authoritative"
    assert child / ".env" in loaded
    assert child / ".llm-council.env" in loaded


def test_mcp_budget_fails_closed_when_any_paid_peer_is_unpriced():
    config = {
        "participants": {
            "priced": {
                "type": "openrouter",
                "model": "known",
                "input_per_million": 1.0,
            },
            "unknown": {
                "type": "openrouter",
                "model": "not-in-catalog-for-this-test",
            },
        },
        "defaults": {},
    }

    report = mcp_budget_report(
        config=config,
        participants=["priced", "unknown"],
        prompt_chars=1_000,
        deliberate=False,
        max_rounds=1,
    )

    assert report["within_budget"] is False
    [violation] = report["violations"]
    assert violation["limit"] == "known_paid_hosted_pricing"
    assert violation["participants"] == ["unknown"]


def test_mcp_budget_treats_local_openai_compatible_as_free():
    config = {
        "participants": {
            "local": {
                "type": "openai_compatible",
                "model": "local/qwen",
                "base_url": "http://127.0.0.1:8000/v1",
                "allow_private": True,
            }
        },
        "defaults": {},
    }

    report = mcp_budget_report(
        config=config,
        participants=["local"],
        prompt_chars=1_000,
        deliberate=False,
        max_rounds=1,
    )

    assert report["paid_hosted_participants"] == []
    assert report["within_budget"] is True


def test_mcp_budget_prices_resolved_synthesizer_alias():
    config = {
        "participants": {
            "chair": {
                "type": "openrouter",
                "model": "known",
                "input_per_million": 1.0,
            }
        },
        "defaults": {"synthesizer": "neutral_peer"},
    }

    report = mcp_budget_report(
        config=config,
        participants=[],
        prompt_chars=4_000,
        deliberate=False,
        max_rounds=1,
        synthesize=True,
        synthesizer_name="chair",
    )

    assert report["synthesize_billable"] is True
    assert report["estimated_input_cost_usd"] == 0.001


def test_recommendation_summary_requires_unique_leader():
    tied = summarize_recommendations([_result("a", "yes"), _result("b", "no")])
    assert tied.recommendation == "unknown"
    assert tied.agreement_count == 0
    assert tied.total_labeled == 2
    assert tied.tied is True

    majority = summarize_recommendations(
        [_result("a", "no"), _result("b", "no"), _result("c", "yes")]
    )
    assert majority.recommendation == "no"
    assert majority.agreement_count == 2
    assert majority.total_labeled == 3
    assert majority.tied is False


def test_deliberation_builder_honors_its_advertised_character_cap():
    from llm_council.deliberation import (
        MAX_DELIBERATION_PROMPT_CHARS,
        build_deliberation_prompt,
    )

    prompt, _truncated = build_deliberation_prompt(
        "q" * (MAX_DELIBERATION_PROMPT_CHARS * 2),
        [_result("p", "yes")],
    )

    assert len(prompt) <= MAX_DELIBERATION_PROMPT_CHARS
    assert prompt.endswith("[deliberation prompt truncated by llm-council]\n")


def test_estimate_uses_prepared_per_participant_prompts(tmp_path: Path):
    config = {
        "defaults": {},
        "participants": {
            "short": {"type": "cli", "command": "true"},
            "long": {"type": "cli", "command": "true"},
        },
        "modes": {"custom": {"participants": ["short", "long"]}},
    }

    estimate = estimate_council(
        config=config,
        cwd=tmp_path,
        question="ignored",
        mode="custom",
        current=None,
        prepared_prompt="base",
        prepared_participants=["short", "long"],
        participant_prompts={"short": "x" * 4, "long": "x" * 4_000},
        allow_network=False,
    )

    rows = {row["name"]: row for row in estimate["rows"]}
    assert rows["short"]["estimated_input_tokens"] == 1
    assert rows["long"]["estimated_input_tokens"] == 1_000


def test_estimate_separates_exact_round_one_from_deliberation_bounds_and_images(
    tmp_path: Path,
):
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
    )
    config = {
        "defaults": {},
        "participants": {
            "p": {
                "type": "openrouter",
                "model": "vendor/p",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
                "max_prompt_chars": 1_000,
                "vision": True,
            }
        },
        "modes": {"custom": {"participants": ["p"]}},
    }

    estimate = estimate_council(
        config=config,
        cwd=tmp_path,
        question="ignored",
        mode="custom",
        current=None,
        prepared_prompt="base",
        prepared_participants=["p"],
        participant_prompts={"p": "tiny"},
        deliberate=True,
        max_rounds=3,
        completion_tokens=100,
        image_paths=["pixel.png"],
        allow_network=False,
    )

    rows = {row["name"]: row for row in estimate["rows"]}
    assert rows["p"]["rounds_assumed"] == 1
    assert rows["p"]["estimated_input_tokens"] == 1 + 1_500
    assert rows["p:deliberation"]["phase"] == "deliberation"
    assert rows["p:deliberation"]["prompt_bound_chars"] == 1_000
    assert rows["p:deliberation"]["rounds_assumed"] == 2
    assert rows["p:deliberation"]["estimated_input_tokens"] == (
        250 + 1_500
    ) * 2
    assert estimate["known_total_with_retry_safety_usd"] == pytest.approx(
        estimate["known_total_usd"] * 2,
        abs=1e-6,
    )


def test_retry_safety_covers_timeout_recovery_for_synthesis(
    tmp_path: Path,
) -> None:
    config = {
        "defaults": {},
        "participants": {
            "a": {
                "type": "openrouter",
                "model": "vendor/a",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
            },
            "b": {
                "type": "openrouter",
                "model": "vendor/b",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
            },
        },
        "modes": {"custom": {"participants": ["a", "b"]}},
    }

    common = {
        "config": config,
        "cwd": tmp_path,
        "question": "review",
        "mode": "custom",
        "current": None,
        "prepared_prompt": "base",
        "prepared_participants": ["a", "b"],
        "participant_prompts": {"a": "base", "b": "base"},
        "completion_tokens": 100,
        "allow_network": False,
        "synthesize": True,
        "synthesizer_name": "b",
    }
    estimate = estimate_council(**common)

    # Synthesis turns off recommendation repair at runtime, but it still
    # inherits the adapter's default terse retry on timeout.
    assert estimate["known_total_with_retry_safety_usd"] == pytest.approx(
        estimate["known_total_usd"] * 2,
        abs=1e-6,
    )

    for cfg in config["participants"].values():
        cfg["retry_on_missing_label"] = False
        cfg["terse_retry_on_timeout"] = False
    without_outer_retries = estimate_council(**common)
    assert without_outer_retries["known_total_with_retry_safety_usd"] == (
        without_outer_retries["known_total_usd"]
    )


def test_mcp_budget_report_uses_deliberation_prompt_bounds_for_cost_and_max_call():
    config = {
        "participants": {
            "p": {
                "type": "openrouter",
                "model": "vendor/p",
                "input_per_million": 1.0,
            }
        },
        "defaults": {"mcp_max_prompt_chars": 100_000},
    }
    base = mcp_budget_report(
        config=config,
        participants=["p"],
        prompt_chars=4,
        deliberate=False,
        max_rounds=1,
        participant_prompt_chars={"p": 4},
    )
    deliberated = mcp_budget_report(
        config=config,
        participants=["p"],
        prompt_chars=4,
        deliberate=True,
        max_rounds=2,
        participant_prompt_chars={"p": 4},
        deliberation_prompt_chars={"p": 80_000},
    )

    assert deliberated["max_call_prompt_chars"] == 80_000
    assert deliberated["estimated_billable_prompt_chars"] == 80_004
    assert (
        deliberated["estimated_input_cost_usd"]
        > base["estimated_input_cost_usd"]
    )


def test_estimate_canonicalizes_legacy_mode_alias(tmp_path: Path):
    config = {
        "defaults": {},
        "participants": {"local": {"type": "cli", "command": "true"}},
        "modes": {"private-local": {"participants": ["local"]}},
    }

    estimate = estimate_council(
        config=config,
        cwd=tmp_path,
        question="review",
        mode="local-only",
        current=None,
        prepared_prompt="base",
        prepared_participants=["local"],
        allow_network=False,
    )

    assert estimate["mode"] == "private-local"


def test_estimate_counts_synthesis_cap_rows(tmp_path: Path):
    config = {
        "defaults": {"synthesizer": "b"},
        "participants": {
            "a": {
                "type": "openrouter",
                "model": "a-model",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
                "max_prompt_chars": 900,
            },
            "b": {
                "type": "openrouter",
                "model": "b-model",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
            },
        },
        "modes": {"custom": {"participants": ["a", "b"]}},
    }
    common = {
        "config": config,
        "cwd": tmp_path,
        "question": "synthesize",
        "mode": "custom",
        "current": None,
        "prepared_prompt": "base",
        "prepared_participants": ["a", "b"],
        "participant_prompts": {"a": "base", "b": "base"},
        "completion_tokens": 100,
        "allow_network": False,
    }

    base = estimate_council(**common)
    expanded = estimate_council(
        **common,
        synthesize=True,
        synthesizer_name="b",
    )

    rows = {row["name"]: row for row in expanded["rows"]}
    assert rows["b:synthesis"]["phase"] == "synthesis"
    assert rows["b:synthesis"]["prompt_bound_chars"] == 60_000
    assert expanded["paid_peer_count"] == 2

    _, base_tokens, _ = summarize_preflight_caps(base)
    expanded_cost, expanded_tokens, _ = summarize_preflight_caps(expanded)
    assert expanded_tokens > base_tokens
    assert expanded_cost > base["known_total_with_retry_safety_usd"]
    with pytest.raises(ValueError, match="exceeds --max-tokens"):
        enforce_preflight_caps(
            expanded,
            max_cost_usd=None,
            max_tokens=base_tokens,
            breakdown_hint="inspect rows",
        )


@pytest.mark.asyncio
async def test_mcp_contextual_persona_is_included_in_hard_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
participants:
  p:
    type: cli
    command: true
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    from llm_council import context as context_module

    monkeypatch.setattr(
        context_module,
        "_git_output",
        lambda _cwd, args: (
            "src/security/auth.py\n" if "--cached" in args else ""
        ),
    )
    real_estimate = estimate_council
    captured: dict[str, int | str] = {}

    def spy_estimate(**kwargs):
        full = real_estimate(**kwargs)
        base_kwargs = dict(kwargs)
        base_kwargs["participant_prompts"] = {
            name: kwargs["prepared_prompt"]
            for name in kwargs["prepared_participants"]
        }
        without_persona = real_estimate(**base_kwargs)
        _, captured["full_tokens"], _ = summarize_preflight_caps(full)
        _, captured["base_tokens"], _ = summarize_preflight_caps(without_persona)
        captured["peer_prompt"] = kwargs["participant_prompts"]["p"]
        return full

    execute_calls = 0

    async def fake_execute(*args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        return (
            [_result("p", "yes")],
            {"rounds": 1, "deliberated": False, "degraded": False},
        )

    monkeypatch.setattr(mcp_server, "estimate_council", spy_estimate)
    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)
    monkeypatch.setattr(mcp_server, "write_transcript", lambda *a, **k: None)

    await mcp_server.run_council(
        {"question": "review", "working_directory": str(tmp_path)}
    )
    assert "CONTEXTUAL ROLE ASSIGNMENT" in str(captured["peer_prompt"])
    assert int(captured["full_tokens"]) > int(captured["base_tokens"])

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        await mcp_server.run_council(
            {
                "question": "review",
                "working_directory": str(tmp_path),
                "max_tokens": int(captured["base_tokens"]),
            }
        )
    assert execute_calls == 1


def test_cli_contextual_persona_is_included_in_hard_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
participants:
  p:
    type: cli
    command: true
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    from llm_council import cli as cli_module
    from llm_council import context as context_module
    from llm_council.cli import build_parser

    monkeypatch.setattr(cli_module, "maybe_print_update_nag", lambda *_a: None)
    monkeypatch.setattr(cli_module, "build_prompt", lambda *_a, **_k: "base")
    monkeypatch.setattr(
        context_module,
        "_git_output",
        lambda _cwd, args: (
            "src/security/auth.py\n" if "--cached" in args else ""
        ),
    )
    captured: dict[str, str] = {}

    def fake_estimate(**kwargs):
        prompts = kwargs.get("participant_prompts") or {}
        peer_prompt = prompts.get("p", kwargs.get("prepared_prompt") or "base")
        captured["peer_prompt"] = peer_prompt
        return {
            "known_total_usd": 0.0,
            "known_total_with_retry_safety_usd": 0.0,
            "rows": [
                {
                    "name": "p",
                    "type": "cli",
                    "estimated_input_tokens": (len(peer_prompt) + 3) // 4,
                    "estimated_output_tokens": 0,
                    "estimated_total_cost_usd": None,
                }
            ],
        }

    monkeypatch.setattr(cli_module, "estimate_council", fake_estimate)
    args = build_parser().parse_args(
        [
            "run",
            "--cwd",
            str(tmp_path),
            "--mode",
            "custom",
            "--max-tokens",
            "1",
            "--dry-run",
            "review",
        ]
    )

    with pytest.raises(SystemExit, match="exceeds --max-tokens"):
        cli_module.cmd_run(args)
    assert "CONTEXTUAL ROLE ASSIGNMENT" in captured["peer_prompt"]


@pytest.mark.asyncio
async def test_mcp_invalid_synthesis_chair_fails_before_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  synthesizer: chair
participants:
  p:
    type: cli
    command: true
  chair:
    type: cli
    command: true
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid chair must fail before execute_council")

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)

    with pytest.raises(ValueError, match="not a configured participant"):
        await mcp_server.run_council(
            {
                "question": "review",
                "working_directory": str(tmp_path),
                "synthesize": True,
            }
        )
    assert called is False


@pytest.mark.asyncio
async def test_orchestrator_invalid_synthesis_chair_fails_before_peer_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from llm_council import orchestrator

    called = False

    async def fake_run_participants(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid chair must fail before peer launch")

    monkeypatch.setattr(orchestrator, "run_participants", fake_run_participants)

    with pytest.raises(ValueError, match="defaults.synthesizer is not configured"):
        await orchestrator.execute_council(
            participants=["p"],
            participant_cfg={"p": {"type": "cli", "command": "true"}},
            prompt="review",
            cwd=tmp_path,
            config={"defaults": {}, "modes": {}},
            synthesize=True,
        )
    assert called is False


@pytest.mark.asyncio
async def test_orchestrator_preflight_failed_chair_is_not_reinvoked_for_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from llm_council import orchestrator
    from llm_council import synthesis as synthesis_module

    launched_rosters: list[list[str]] = []
    synthesis_calls = 0

    async def fake_preflight(*args, **kwargs):
        return {"chair": "PreflightFailed: local endpoint unavailable"}

    async def fake_run_participants(names, *args, **kwargs):
        launched_rosters.append(list(names))
        return [_result(name, "yes") for name in names]

    async def fake_synthesis(*args, **kwargs):
        nonlocal synthesis_calls
        synthesis_calls += 1
        raise AssertionError("preflight-failed chair must not be invoked again")

    monkeypatch.setattr(
        orchestrator, "preflight_local_participants", fake_preflight
    )
    monkeypatch.setattr(orchestrator, "run_participants", fake_run_participants)
    monkeypatch.setattr(synthesis_module, "run_synthesis_chair", fake_synthesis)

    participant_cfg = {
        "chair": {"type": "ollama", "model": "chair-model"},
        "voter": {"type": "cli", "command": "true"},
    }
    _results, metadata = await orchestrator.execute_council(
        participants=["chair", "voter"],
        participant_cfg=participant_cfg,
        prompt="review",
        cwd=tmp_path,
        config={
            "defaults": {"synthesizer": "chair"},
            "participants": participant_cfg,
            "modes": {},
        },
        synthesize=True,
    )

    assert launched_rosters == [["voter"]]
    assert synthesis_calls == 0
    assert "failed participant preflight" in metadata["synthesis_error"]
    assert any(
        event.get("event") == "synthesis_error"
        and event.get("chair") == "chair"
        and event.get("reason") == "preflight_failed"
        for event in metadata["progress_events"]
    )


@pytest.mark.asyncio
async def test_mcp_default_cost_cap_counts_dynamic_phase_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    phase_arguments: dict[str, bool] = {"synthesize": True}
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  synthesizer: p
  mcp_max_estimated_cost_usd: 0.003
participants:
  p:
    type: openrouter
    model: vendor/p
    input_per_million: 1.0
    output_per_million: 1.0
  q:
    type: openrouter
    model: vendor/q
    input_per_million: 1.0
    output_per_million: 1.0
modes:
  custom:
    participants: [p, q]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("default MCP cost cap must fire before execution")

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)

    with pytest.raises(ValueError, match="max_estimated_cost_usd"):
        await mcp_server.run_council(
            {
                "question": "review",
                "working_directory": str(tmp_path),
                "current": "host",
                **phase_arguments,
            }
        )
    assert called is False


@pytest.mark.asyncio
async def test_mcp_tiny_base_deliberation_bound_exceeds_prompt_cap_before_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  mcp_max_prompt_chars: 1000
participants:
  p:
    type: cli
    command: true
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("deliberation prompt cap must fire before execution")

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)

    with pytest.raises(ValueError, match="max_prompt_chars"):
        await mcp_server.run_council(
            {
                "question": "tiny",
                "working_directory": str(tmp_path),
                "deliberate": True,
                "max_rounds": 2,
            }
        )
    assert called is False


@pytest.mark.asyncio
async def test_mcp_tiny_base_deliberation_bound_exceeds_token_cap_before_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  mcp_max_prompt_chars: 100000
participants:
  p:
    type: cli
    command: true
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("deliberation token cap must fire before execution")

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        await mcp_server.run_council(
            {
                "question": "tiny",
                "working_directory": str(tmp_path),
                "deliberate": True,
                "max_rounds": 2,
                "max_tokens": 2_000,
            }
        )
    assert called is False


@pytest.mark.asyncio
async def test_mcp_default_cost_cap_counts_output_and_retry_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  mcp_max_estimated_cost_usd: 0.01
participants:
  expensive_output:
    type: openrouter
    model: vendor/expensive-output
    api_key_env: OPENROUTER_API_KEY
    input_per_million: 0.001
    output_per_million: 100.0
modes:
  custom:
    participants: [expensive_output]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("whole-run MCP cost cap must fire before execution")

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)

    with pytest.raises(ValueError, match="max_estimated_cost_usd"):
        await mcp_server.run_council(
            {
                "question": "review",
                "working_directory": str(tmp_path),
                "current": "host",
            }
        )
    assert called is False


@pytest.mark.asyncio
async def test_mcp_dry_run_still_enforces_explicit_token_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
participants:
  p:
    type: cli
    command: true
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    called = False

    async def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dry-run must never execute peers")

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        await mcp_server.run_council(
            {
                "question": "review",
                "working_directory": str(tmp_path),
                "dry_run": True,
                "max_tokens": 1,
            }
        )
    assert called is False


@pytest.mark.asyncio
async def test_mcp_tied_final_vote_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
participants:
  a:
    type: cli
    command: true
  b:
    type: cli
    command: true
modes:
  custom:
    participants: [a, b]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    async def fake_execute(*args, **kwargs):
        return (
            [_result("a", "yes"), _result("b", "no")],
            {"rounds": 1, "deliberated": False, "degraded": False},
        )

    transcript_metadata: dict[str, object] = {}

    def capture_transcript(*args, **kwargs):
        transcript_metadata.update(kwargs["metadata"])

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)
    monkeypatch.setattr(mcp_server, "write_transcript", capture_transcript)

    payload = await mcp_server.run_council(
        {"question": "ship?", "working_directory": str(tmp_path)}
    )
    assert payload["recommendation"] == "unknown"
    assert payload["agreement_count"] == 0
    assert payload["total_labeled"] == 2
    assert transcript_metadata["recommendation"] == "unknown"
    assert transcript_metadata["agreement_count"] == 0
    assert transcript_metadata["total_labeled"] == 2
    assert transcript_metadata["recommendation_counts"] == {
        "yes": 1,
        "no": 1,
        "tradeoff": 0,
        "unknown": 0,
    }
    assert transcript_metadata["recommendation_tied"] is True


@pytest.mark.asyncio
async def test_mcp_explicit_tier_wins_after_smart_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  tiers:
    deep:
      p: premium-model
participants:
  p:
    type: cli
    command: true
    model: base-model
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    import llm_council.config as config_module

    def fake_smart_routing(config, mode, cwd):
        config["participants"]["p"]["model"] = "cheap-model"

    monkeypatch.setattr(config_module, "apply_smart_routing", fake_smart_routing)

    payload = await mcp_server.run_council(
        {
            "question": "review",
            "working_directory": str(tmp_path),
            "tier": "deep",
            "dry_run": True,
        }
    )
    assert payload["metadata"]["participant_models"]["p"] == "premium-model"


def test_standalone_estimates_apply_smart_routing_before_explicit_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".llm-council.yaml").write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  tiers:
    deep:
      p: premium-model
participants:
  p:
    type: openrouter
    model: base-model
    input_per_million: 1.0
    output_per_million: 1.0
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    import llm_council.config as config_module
    from llm_council import cli as cli_module
    from llm_council.cli import build_parser

    smart_calls: list[str] = []
    estimated_models: list[str] = []

    def fake_smart_routing(config, mode, cwd):
        smart_calls.append(mode)
        config["participants"]["p"]["model"] = "cheap-model"

    def fake_estimate(**kwargs):
        model = kwargs["config"]["participants"]["p"]["model"]
        estimated_models.append(model)
        return {
            "participants": ["p"],
            "captured_model": model,
            "known_total_usd": 0.0,
            "known_total_with_retry_safety_usd": 0.0,
            "rows": [],
        }

    monkeypatch.setattr(config_module, "apply_smart_routing", fake_smart_routing)
    monkeypatch.setattr(cli_module, "estimate_council", fake_estimate)
    monkeypatch.setattr(mcp_server, "estimate_council", fake_estimate)

    args = build_parser().parse_args(
        [
            "estimate",
            "--cwd",
            str(tmp_path),
            "--tier",
            "deep",
            "--json",
            "review",
        ]
    )
    assert cli_module.cmd_estimate(args) == 0

    mcp_estimate = mcp_server.estimate_run(
        {
            "question": "review",
            "working_directory": str(tmp_path),
            "tier": "deep",
        }
    )
    assert mcp_estimate["ok"] is True
    assert mcp_estimate["captured_model"] == "premium-model"
    assert smart_calls == ["custom", "custom"]
    assert estimated_models == ["premium-model", "premium-model"]


def test_mcp_config_rejects_invalid_candidate_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        """
replace_defaults: true
defaults:
  mode: custom
  max_concurrency: 2
participants:
  p:
    type: cli
    command: true
modes:
  custom:
    participants: [p]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to write invalid configuration"):
        mcp_server.run_config(
            {
                "action": "set",
                "key": "defaults.max_concurrency",
                "value": "0",
                "working_directory": str(tmp_path),
            }
        )

    assert config_path.read_text(encoding="utf-8") == before
    assert yaml.safe_load(before)["defaults"]["max_concurrency"] == 2


def test_mcp_schema_advertises_pinned_model_unverified():
    kinds = mcp_server.council_run_output_schema()["properties"]["results"][
        "items"
    ]["properties"]["error_kind"]["enum"]
    assert "pinned_model_unverified" in kinds
