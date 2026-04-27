import asyncio
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from llm_council import __version__
import llm_council.adapters as adapters_module
import llm_council.orchestrator as orchestrator_module
import llm_council.cli as cli_module
import llm_council.estimate as estimate_module
import llm_council.update_check as update_check_module
from llm_council.adapters import (
    ParticipantResult,
    _format_arg,
    clean_subprocess_env,
    redact_prompt_args,
    run_openrouter_participant,
    run_participants,
)
from llm_council.cli import build_parser, cmd_doctor, cmd_setup, cmd_transcripts, main
from llm_council.config import load_config, select_participants
from llm_council.context import build_prompt
from llm_council.context import read_context_file, read_git_diff
from llm_council.defaults import DEFAULT_CONFIG
from llm_council.deliberation import (
    build_deliberation_prompt,
    has_disagreement,
    model_comparison,
    recommendation_counts,
    recommendation_label,
)
from llm_council.doctor import _probe_ollama, _probe_openrouter, check_environment
from llm_council.doctor import Check
from llm_council.env import load_project_env
from llm_council.model_catalog import (
    _read_cache,
    _write_cache,
    infer_origin,
    normalize_openrouter_model,
    openrouter_cache_path,
)
from llm_council.mcp_server import (
    council_run_schema,
    doctor_schema,
    estimate_run,
    estimate_schema,
    last_transcript,
    last_transcript_schema,
    models_schema,
)
from llm_council.orchestrator import execute_council
from llm_council.policy import should_use_council
from llm_council.setup_wizard import mcp_config, project_config, write_setup_files
from llm_council.transcript import (
    deliberation_summary,
    final_round_results,
    latest_transcript,
    markdown_fence,
    result_to_dict,
    safe_slug,
    transcript_records,
    write_transcript,
)


def test_select_other_cli_peers_excludes_current():
    config = load_config(None)
    selected = select_participants(config, "quick", "codex")
    assert selected == ["claude", "gemini"]


def test_claude_prompt_goes_to_stdin():
    assert DEFAULT_CONFIG["participants"]["claude"]["stdin_prompt"] is True


def test_clean_subprocess_env_strips_claudecode(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    env = clean_subprocess_env()
    assert "CLAUDECODE" not in env
    assert "OPENROUTER_API_KEY" not in env


def test_prompt_arg_is_redacted_and_literal_braces_are_safe(tmp_path: Path):
    prompt = "secret prompt"
    formatted = _format_arg("{cwd}/x/{literal}", prompt=prompt, cwd=tmp_path)
    assert formatted.endswith("/x/{literal}")
    assert redact_prompt_args(["gemini", "--prompt", prompt], prompt) == [
        "gemini",
        "--prompt",
        "[prompt]",
    ]


def test_plan_mode_adds_deepseek():
    config = load_config(None)
    selected = select_participants(config, "plan", "claude")
    assert selected == ["codex", "gemini", "deepseek_v4_pro"]


def test_deliberate_mode_adds_deepseek_and_marks_expensive():
    config = load_config(None)
    selected = select_participants(config, "deliberate", "claude")
    assert selected == ["codex", "gemini", "deepseek_v4_pro"]
    assert config["modes"]["deliberate"]["deliberate"] is True


def test_explicit_participants_win():
    config = load_config(None)
    selected = select_participants(config, "quick", "codex", explicit=["glm_5_1"])
    assert selected == ["glm_5_1"]


def test_explicit_participants_respect_origin_policy_with_clear_error():
    config = load_config(None)
    with pytest.raises(ValueError, match="No participants selected"):
        select_participants(
            config,
            "quick",
            "codex",
            explicit=["deepseek_v4_pro"],
            origin_policy="us",
        )


def test_us_origin_policy_filters_non_us_additions():
    config = load_config(None)
    selected = select_participants(config, "diverse", "codex", origin_policy="us")
    assert selected == ["claude", "gemini"]


def test_origin_policy_empty_selection_is_clear():
    config = load_config(None)
    with pytest.raises(ValueError, match="No participants selected"):
        select_participants(config, "private-local", "codex", origin_policy="us")


def test_build_prompt_contains_read_only_rules(tmp_path: Path):
    prompt = build_prompt(
        "What should we do?",
        mode="quick",
        cwd=tmp_path,
        context_paths=[],
        include_diff=False,
        stdin_text=None,
    )
    assert "read-only" in prompt
    assert "Do not edit files" in prompt
    assert "What should we do?" in prompt


def test_context_file_outside_cwd_rejected_by_default(tmp_path: Path):
    outside = tmp_path.parent / "outside-context.txt"
    outside.write_text("secret")
    try:
        read_context_file(outside, cwd=tmp_path)
    except ValueError as exc:
        assert "outside working directory" in str(exc)
    else:
        raise AssertionError("expected outside context file to be rejected")


def test_context_file_outside_cwd_can_be_allowed(tmp_path: Path):
    outside = tmp_path.parent / "outside-context-allowed.txt"
    outside.write_text("ok")
    rendered = read_context_file(outside, cwd=tmp_path, allow_outside_cwd=True)
    assert "ok" in rendered


def test_missing_context_file_error_is_clear(tmp_path: Path):
    with pytest.raises(ValueError, match="Context file does not exist"):
        read_context_file("missing.txt", cwd=tmp_path)


def test_cli_missing_context_file_exits_without_traceback(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--cwd",
                str(tmp_path),
                "--dry-run",
                "--context",
                "missing.txt",
                "check",
            ]
        )
    assert str(exc.value).startswith("Context file does not exist:")


def test_cli_missing_config_file_exits_without_traceback(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "estimate",
                "--cwd",
                str(tmp_path),
                "--config",
                str(tmp_path / "missing.yaml"),
                "check",
            ]
        )
    assert str(exc.value).startswith("Config file does not exist:")


def test_estimate_cwd_without_config_uses_defaults_not_process_cwd_config(
    tmp_path: Path, capsys
):
    assert main(
        [
            "estimate",
            "--cwd",
            str(tmp_path),
            "--current",
            "claude",
            "--mode",
            "review-cheap",
            "--completion-tokens",
            "100",
            "check",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "qwen/qwen3-coder-flash" in output
    assert "qwen/qwen3-coder:free" not in output


def test_legacy_qwen_free_participant_remains_explicitly_available():
    config = load_config(None, search=False)
    selected = select_participants(
        config,
        "quick",
        "codex",
        explicit=["qwen_coder_free"],
    )
    assert selected == ["qwen_coder_free"]


def test_git_diff_includes_staged_changes(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    source = tmp_path / "file.txt"
    source.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    source.write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)

    diff = read_git_diff(tmp_path)
    assert "### Staged Changes" in diff
    assert "+new" in diff
    assert "[no diff]" not in diff


def test_git_diff_outside_repo_is_prompt_placeholder(tmp_path: Path):
    diff = read_git_diff(tmp_path)
    assert "git diff unavailable: not a git repository" in diff


def test_safe_slug():
    assert safe_slug("Hello, World!") == "hello-world"


def test_result_to_dict_includes_usage():
    result = ParticipantResult(
        name="deepseek",
        ok=True,
        output="ok",
        error="",
        elapsed_seconds=1.25,
        model="deepseek/test",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.000123,
    )
    data = result_to_dict(result)
    assert data["total_tokens"] == 15
    assert data["cost_usd"] == 0.000123


def test_run_participants_emits_progress_events(tmp_path: Path):
    events: list[dict] = []

    results = asyncio.run(
        run_participants(
            ["python"],
            {
                "python": {
                    "type": "cli",
                    "command": sys.executable,
                    "args": ["-c", "print('RECOMMENDATION: yes - ok')"],
                    "timeout": 5,
                }
            },
            "prompt",
            tmp_path,
            progress=events.append,
        )
    )

    assert results[0].ok is True
    assert events[0] == {"event": "participant_start", "participant": "python", "round": 1}
    assert events[1]["event"] == "participant_finish"
    assert events[1]["participant"] == "python"
    assert events[1]["status"] == "ok"
    assert events[1]["elapsed_seconds"] >= 0


def test_openrouter_empty_content_is_graceful(monkeypatch):
    class FakeResponse:
        def json(self):
            return {
                "model": "z-ai/glm-test",
                "choices": [
                    {
                        "message": {"content": None},
                        "finish_reason": "content_filter",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                    "cost": 0.001,
                },
            }

    async def fake_request(*_args, **_kwargs):
        return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openrouter_participant(
            "glm",
            {"type": "openrouter", "model": "z-ai/glm-test"},
            "prompt",
        )
    )

    assert result.ok is False
    assert result.output == ""
    assert result.error == "OpenRouterEmptyResponse: content_filter"
    assert result.total_tokens == 10
    assert result.cost_usd == 0.001


def test_model_comparison_and_disagreement_detection():
    results = [
        ParticipantResult("a", True, "RECOMMENDATION: yes - proceed.", "", 1),
        ParticipantResult("b", True, "RECOMMENDATION: no - defer.", "", 1, total_tokens=4),
    ]
    assert has_disagreement(results) is True
    assert recommendation_label(results[0].output) == "yes"
    assert recommendation_counts(results)["no"] == 1
    comparison = "\n".join(model_comparison(results))
    assert "a:" in comparison
    assert "4 tokens" in comparison


def test_model_comparison_prefers_recommendation_line():
    results = [
        ParticipantResult(
            "a",
            True,
            "Here is my review.\n\nRECOMMENDATION: no - wait",
            "",
            1,
        )
    ]
    comparison = "\n".join(model_comparison(results))
    assert "RECOMMENDATION: no - wait" in comparison
    assert "Here is my review" not in comparison


def test_recommendation_label_handles_common_markdown():
    assert recommendation_label("**Recommendation:** no - defer") == "no"
    assert recommendation_label("> ## Recommendation: tradeoff - depends") == "tradeoff"
    assert (
        recommendation_label(
            "```text\nRECOMMENDATION: no - sample\n```\n\n- RECOMMENDATION: yes - do it"
        )
        == "yes"
    )
    assert recommendation_label("```text\nRECOMMENDATION: no - sample\n```") == "no"


def test_disagreement_requires_labeled_positions():
    results = [
        ParticipantResult("a", True, "We should proceed.", "", 1),
        ParticipantResult("b", True, "Avoid this for now.", "", 1),
    ]
    assert recommendation_counts(results)["unknown"] == 2
    assert has_disagreement(results) is False


def test_deliberation_prompt_is_capped():
    result = ParticipantResult("a", True, "x" * 100_000, "", 1)
    prompt = build_deliberation_prompt("question" * 20_000, [result])
    assert len(prompt) < 90_000
    assert "truncated" in prompt


def test_deliberation_summary():
    assert deliberation_summary({"deliberation_status": "ran_no_labeled_disagreement"}) == (
        "ran; no labeled disagreement remained"
    )
    assert "max rounds" in deliberation_summary(
        {"deliberation_status": "ran_max_rounds_unresolved"}
    )
    assert "skipped" in deliberation_summary(
        {"deliberation_status": "skipped_no_labeled_disagreement"}
    )
    assert "max rounds is 1" in deliberation_summary(
        {"deliberation_status": "skipped_max_rounds"}
    )
    assert deliberation_summary({"deliberation_requested": False}) == "not requested"


def test_execute_council_skips_second_round_without_labeled_disagreement(
    monkeypatch, tmp_path: Path
):
    calls = 0

    async def fake_run_participants(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [
            ParticipantResult("a", True, "Proceed.", "", 1),
            ParticipantResult("b", True, "Defer.", "", 1),
        ]

    monkeypatch.setattr(orchestrator_module, "run_participants", fake_run_participants)
    results, metadata = asyncio.run(
        execute_council(["a", "b"], {}, "question", tmp_path, {}, deliberate=True)
    )
    assert calls == 1
    assert len(results) == 2
    assert metadata["deliberation_status"] == "skipped_no_labeled_disagreement"


def test_execute_council_runs_second_round_on_labeled_disagreement(
    monkeypatch, tmp_path: Path
):
    calls = 0

    async def fake_run_participants(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                ParticipantResult("a", True, "RECOMMENDATION: yes - ship", "", 1),
                ParticipantResult("b", True, "RECOMMENDATION: no - wait", "", 1),
            ]
        return [
            ParticipantResult("a", True, "RECOMMENDATION: tradeoff - guarded", "", 1),
            ParticipantResult("b", True, "RECOMMENDATION: tradeoff - guarded", "", 1),
        ]

    monkeypatch.setattr(orchestrator_module, "run_participants", fake_run_participants)
    results, metadata = asyncio.run(
        execute_council(["a", "b"], {}, "question", tmp_path, {}, deliberate=True)
    )
    assert calls == 2
    assert metadata["rounds"] == 2
    assert metadata["deliberation_status"] == "ran_no_labeled_disagreement"
    assert results[-1].name == "b:round2"


def test_execute_council_respects_max_rounds(monkeypatch, tmp_path: Path):
    calls = 0

    async def fake_run_participants(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [
            ParticipantResult("a", True, "RECOMMENDATION: yes - ship", "", 1),
            ParticipantResult("b", True, "RECOMMENDATION: no - wait", "", 1),
        ]

    monkeypatch.setattr(orchestrator_module, "run_participants", fake_run_participants)
    results, metadata = asyncio.run(
        execute_council(
            ["a", "b"], {}, "question", tmp_path, {}, deliberate=True, max_rounds=1
        )
    )
    assert calls == 1
    assert len(results) == 2
    assert metadata["deliberation_status"] == "skipped_max_rounds"


def test_execute_council_honors_max_rounds_above_two(monkeypatch, tmp_path: Path):
    calls = 0

    async def fake_run_participants(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [
            ParticipantResult("a", True, "RECOMMENDATION: yes - ship", "", 1),
            ParticipantResult("b", True, "RECOMMENDATION: no - wait", "", 1),
        ]

    monkeypatch.setattr(orchestrator_module, "run_participants", fake_run_participants)
    results, metadata = asyncio.run(
        execute_council(
            ["a", "b"], {}, "question", tmp_path, {}, deliberate=True, max_rounds=3
        )
    )
    assert calls == 3
    assert len(results) == 6
    assert metadata["rounds"] == 3
    assert metadata["deliberation_status"] == "ran_max_rounds_unresolved"
    assert results[-1].name == "b:round3"


def test_openrouter_model_normalization():
    model = normalize_openrouter_model(
        {
            "id": "deepseek/deepseek-v4-pro",
            "name": "DeepSeek V4 Pro",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000005", "completion": "0.0000015"},
        }
    )
    assert model["origin"] == "China / DeepSeek"
    assert model["input_per_million"] == 0.5
    assert model["output_per_million"] == 1.5


def test_openrouter_catalog_cache_round_trip(tmp_path: Path):
    path = tmp_path / "models.json"
    models = [{"id": "openai/gpt-test"}]
    _write_cache(path, models)
    assert _read_cache(path) == models


def test_estimate_reports_configured_openrouter_cost(tmp_path: Path, capsys):
    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "replace_defaults": True,
                "version": 1,
                "participants": {
                    "remote": {
                        "type": "openrouter",
                        "model": "provider/model",
                        "api_key_env": "OPENROUTER_API_KEY",
                        "input_per_million": 2.0,
                        "output_per_million": 4.0,
                    }
                },
                "modes": {"quick": {"participants": ["remote"]}},
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "estimate",
            "--cwd",
            str(tmp_path),
            "--config",
            str(config_path),
            "--mode",
            "quick",
            "--completion-tokens",
            "100",
            "--json",
            "Review this",
        ]
    ) == 0

    data = json.loads(capsys.readouterr().out)
    assert data["rows"][0]["name"] == "remote"
    assert data["rows"][0]["pricing_source"] == "config"
    assert data["rows"][0]["estimated_total_cost_usd"] > 0
    assert data["known_total_usd"] == data["rows"][0]["estimated_total_cost_usd"]


def test_estimate_extra_openrouter_model_uses_catalog(
    tmp_path: Path, monkeypatch, capsys
):
    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "replace_defaults": True,
                "version": 1,
                "participants": {
                    "local_cli": {
                        "type": "cli",
                        "command": "example",
                    }
                },
                "modes": {"quick": {"participants": ["local_cli"]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        estimate_module,
        "fetch_openrouter_models",
        lambda **_kwargs: [
            {
                "id": "anthropic/claude-opus-test",
                "name": "Claude Opus Test",
                "origin": "US / Anthropic",
                "context_length": 200000,
                "input_per_million": 15.0,
                "output_per_million": 75.0,
            }
        ],
    )

    assert main(
        [
            "estimate",
            "--cwd",
            str(tmp_path),
            "--config",
            str(config_path),
            "--openrouter-model",
            "anthropic/claude-opus-test",
            "--completion-tokens",
            "100",
            "Review this",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "anthropic/claude-opus-test" in output
    assert "known_total_usd:" in output
    assert "Native CLI and Ollama rows are not API-priced" in output


def test_mcp_estimate_preserves_zero_completion_tokens(tmp_path: Path, monkeypatch):
    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "replace_defaults": True,
                "version": 1,
                "participants": {
                    "remote": {
                        "type": "openrouter",
                        "model": "provider/model",
                        "api_key_env": "OPENROUTER_API_KEY",
                        "input_per_million": 2.0,
                        "output_per_million": 4.0,
                    }
                },
                "modes": {"quick": {"participants": ["remote"]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))

    result = estimate_run(
        {
            "question": "Review this",
            "working_directory": str(tmp_path),
            "completion_tokens": 0,
        }
    )

    assert result["ok"] is True
    assert result["completion_tokens_assumed_each"] == 0
    assert result["rows"][0]["estimated_output_tokens"] == 0
    assert result["rows"][0]["estimated_output_cost_usd"] == 0


def test_openrouter_cache_uses_xdg_cache_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert openrouter_cache_path() == tmp_path / "llm-council" / "openrouter-models.json"


def test_openrouter_catalog_corrupt_cache_is_ignored(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text("{not-json")
    assert _read_cache(path) is None


def test_load_project_env_does_not_override_existing(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    loaded = load_project_env(tmp_path)
    assert tmp_path / ".env" in loaded
    assert __import__("os").environ["OPENROUTER_API_KEY"] == "from-env"


def test_load_project_env_precedence(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-env-file\n")
    (tmp_path / ".env.local").write_text("OPENROUTER_API_KEY=from-local\n")
    (tmp_path / ".llm-council.env").write_text("OPENROUTER_API_KEY=from-council\n")
    load_project_env(tmp_path)
    assert __import__("os").environ["OPENROUTER_API_KEY"] == "from-council"


def test_origin_inference_for_us_model():
    assert infer_origin("openai/gpt-5.2") == "US / OpenAI"
    assert infer_origin("~anthropic/claude-opus-latest") == "US / Anthropic"


def test_mcp_run_schema_has_question():
    schema = council_run_schema()
    assert "question" in schema["required"]
    assert "include_diff" in schema["properties"]


def test_mcp_last_transcript_schema():
    schema = last_transcript_schema()
    assert "format" in schema["properties"]


def test_mcp_doctor_and_models_schema():
    assert "probe_openrouter" in doctor_schema()["properties"]
    assert "check_update" in doctor_schema()["properties"]
    assert "origin" in models_schema()["properties"]
    assert "openrouter_models" in estimate_schema()["properties"]
    assert "question" in estimate_schema()["required"]


def test_check_update_reports_available(monkeypatch, capsys):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"name": "v9.9.9"}]

    monkeypatch.setattr(
        update_check_module.httpx, "get", lambda *_args, **_kwargs: FakeResponse()
    )

    assert main(["check-update"]) == 0

    output = capsys.readouterr().out
    assert f"version: {__version__}" in output
    assert "latest: 9.9.9" in output
    assert "update_available: true" in output
    assert "uv tool install --force" in output
    assert "@v9.9.9" in output


def test_check_update_json_reports_current(monkeypatch, capsys):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"name": f"v{__version__}"}]

    monkeypatch.setattr(
        update_check_module.httpx, "get", lambda *_args, **_kwargs: FakeResponse()
    )

    assert main(["check-update", "--json"]) == 0

    data = json.loads(capsys.readouterr().out)
    assert data["current_version"] == __version__
    assert data["latest_version"] == __version__
    assert data["update_available"] is False


def test_recommendation_policy_for_architecture():
    use, mode, reason = should_use_council("architecture decision for auth")
    assert use is True
    assert mode == "plan"
    assert reason


def test_openrouter_probe_without_key_is_clear():
    check = _probe_openrouter(None)
    assert check.ok is False
    assert "not set" in check.detail


def test_ollama_probe_bad_port_is_clear():
    check = _probe_ollama("http://127.0.0.1:9")
    assert check.ok is False
    assert check.detail


def test_tri_cli_setup_omits_openrouter_modes():
    config = project_config(include_openrouter=False, include_local=False)
    assert "quick" in config["modes"]
    assert "us-only" in config["modes"]
    assert "plan" not in config["modes"]
    assert "private-local" not in config["modes"]


def test_example_config_loads_exact_modes():
    config = load_config(Path("examples/llm-council.yaml"))
    assert set(config["participants"]) == {
        "claude",
        "codex",
        "gemini",
        "deepseek_v4_pro",
        "qwen_coder_plus",
    }
    assert set(config["modes"]) == {"quick", "plan", "review"}


def test_tri_cli_setup_loaded_config_does_not_restore_defaults(tmp_path: Path):
    write_setup_files(tmp_path, include_openrouter=False, include_local=False)
    config = load_config(tmp_path / ".llm-council.yaml")
    assert set(config["participants"]) == {"claude", "codex", "gemini"}
    assert set(config["modes"]) == {"quick", "us-only"}


def test_setup_interactive_uses_preset_and_suppression_flags(
    tmp_path: Path, monkeypatch
):
    answers: list[str] = []
    monkeypatch.setattr("builtins.input", lambda _prompt: answers.pop(0) if answers else "")
    args = argparse.Namespace(
        root=str(tmp_path),
        preset="tri-cli",
        yes=False,
        force=False,
        us_only_default=True,
        no_mcp=True,
        no_instructions=True,
    )
    assert cmd_setup(args) == 0

    config = yaml.safe_load((tmp_path / ".llm-council.yaml").read_text())
    assert set(config["participants"]) == {"claude", "codex", "gemini"}
    assert config["defaults"]["origin_policy"] == "us"
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".llm-council" / "instructions").exists()


def test_setup_yes_uses_preset_and_suppression_flags(tmp_path: Path):
    args = argparse.Namespace(
        root=str(tmp_path),
        preset="tri-cli",
        yes=True,
        force=False,
        us_only_default=True,
        no_mcp=True,
        no_instructions=True,
    )
    assert cmd_setup(args) == 0

    config = yaml.safe_load((tmp_path / ".llm-council.yaml").read_text())
    assert set(config["participants"]) == {"claude", "codex", "gemini"}
    assert set(config["modes"]) == {"quick"}
    assert config["defaults"]["origin_policy"] == "us"
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".llm-council" / "instructions").exists()


def test_setup_yes_auto_selects_tri_cli_when_native_clis_exist(
    tmp_path: Path, monkeypatch, capsys
):
    def fake_which(name: str):
        return f"/usr/bin/{name}" if name in {"claude", "codex"} else None

    monkeypatch.setattr(cli_module.shutil, "which", fake_which)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    parser = build_parser()
    args = parser.parse_args(["setup", "--yes", "--root", str(tmp_path)])

    assert cmd_setup(args) == 0

    config = load_config(tmp_path / ".llm-council.yaml")
    assert set(config["participants"]) == {"claude", "codex", "gemini"}
    assert set(config["modes"]) == {"quick", "us-only"}
    assert "Auto preset selected: tri-cli" in capsys.readouterr().out


def test_setup_yes_auto_selects_openrouter_when_native_route_missing(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(cli_module.shutil, "which", lambda _name: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    parser = build_parser()
    args = parser.parse_args(["setup", "--yes", "--root", str(tmp_path)])

    assert cmd_setup(args) == 0

    config = load_config(tmp_path / ".llm-council.yaml")
    assert "claude" not in config["participants"]
    assert set(config["modes"]) >= {"quick", "plan", "review"}
    assert config["modes"]["quick"]["participants"] == [
        "deepseek_v4_flash",
        "qwen_coder_flash",
        "glm_4_7_flash",
    ]
    assert "Auto preset selected: openrouter" in capsys.readouterr().out


def test_setup_yes_auto_fails_without_usable_route(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_module.shutil, "which", lambda _name: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    parser = build_parser()
    args = parser.parse_args(["setup", "--yes", "--root", str(tmp_path)])

    with pytest.raises(SystemExit, match="could not find a usable default"):
        cmd_setup(args)


def test_setup_prints_next_steps_and_cli_warnings(tmp_path: Path, monkeypatch, capsys):
    def fake_which(name: str):
        return None if name == "claude" else f"/usr/bin/{name}"

    monkeypatch.setattr(cli_module.shutil, "which", fake_which)
    args = argparse.Namespace(
        root=str(tmp_path),
        preset="tri-cli",
        yes=True,
        force=False,
        us_only_default=False,
        no_mcp=False,
        no_instructions=False,
    )

    assert cmd_setup(args) == 0

    output = capsys.readouterr().out
    assert "Next steps:" in output
    assert "Append the full contents" in output
    assert "CLAUDE.md" in output
    assert "AGENTS.md" in output
    assert "GEMINI.md" in output
    assert "Run `llm-council doctor`" in output
    assert "Warnings:" in output
    assert "claude was not found on PATH" in output


def test_setup_writes_config_mcp_and_instructions(tmp_path: Path):
    written = write_setup_files(tmp_path)
    names = {path.relative_to(tmp_path).as_posix() for path in written}
    assert ".llm-council.yaml" in names
    assert ".mcp.json" in names
    assert ".llm-council/instructions/claude.md" in names
    assert ".llm-council/instructions/codex.md" in names
    assert ".llm-council/instructions/gemini.md" in names
    assert ".llm-council/.gitignore" in names
    assert ".gitignore" in names
    assert "llm-council" in (tmp_path / ".mcp.json").read_text()
    assert ".mcp.json" in (tmp_path / ".gitignore").read_text()
    assert "Always pass `current` as `codex`" in (
        tmp_path / ".llm-council/instructions/codex.md"
    ).read_text()


def test_mcp_config_does_not_embed_openrouter_env_reference(tmp_path: Path):
    config = mcp_config(tmp_path)
    env = config["mcpServers"]["llm-council"]["env"]
    assert env["PYTHONPATH"] == str(tmp_path.resolve())
    assert env["LLM_COUNCIL_MCP_ROOT"] == str(tmp_path.resolve())
    assert "OPENROUTER_API_KEY" not in env


def test_setup_parse_errors_are_actionable(tmp_path: Path):
    (tmp_path / ".mcp.json").write_text("{bad-json", encoding="utf-8")
    with pytest.raises(ValueError, match="rerun setup with --force"):
        write_setup_files(tmp_path)


def test_last_transcript_reads_latest_from_project_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    out_dir = tmp_path / ".llm-council" / "runs"
    md_path = out_dir / "20260425-test.md"
    json_path = out_dir / "20260425-test.json"
    write_transcript(
        md_path,
        json_path,
        question="test",
        mode="quick",
        current="codex",
        participants=[],
        prompt="prompt",
        results=[],
        transparent=True,
        metadata={"rounds": 1},
    )
    result = last_transcript({"working_directory": str(tmp_path)})
    assert result["found"] is True
    assert result["path"] == str(md_path)
    assert "test" in result["content"]
    assert "Tokens reported" in result["content"]
    assert "Model Comparison" in result["content"]
    assert latest_transcript(out_dir) == md_path
    records = transcript_records(out_dir)
    assert records[0]["question"] == "test"
    assert records[0]["total"] == 0


def test_prompt_sent_uses_longer_markdown_fence(tmp_path: Path):
    prompt = "Context:\n```text\ninside\n```"
    assert markdown_fence(prompt) == "````"
    md_path = tmp_path / "run.md"
    json_path = tmp_path / "run.json"
    write_transcript(
        md_path,
        json_path,
        question="test",
        mode="quick",
        current="codex",
        participants=[],
        prompt=prompt,
        results=[],
    )
    content = md_path.read_text(encoding="utf-8")
    assert "````text\nContext:\n```text\ninside\n```\n````" in content


def test_doctor_does_not_require_optional_ollama(monkeypatch):
    checks = [
        Check("cli:claude", True, "ok"),
        Check("cli:codex", True, "ok"),
        Check("cli:gemini", True, "ok"),
        Check("cli:ollama", False, "missing"),
        Check("python:mcp", True, "ok"),
    ]
    monkeypatch.setattr(cli_module, "check_environment", lambda *args, **kwargs: checks)
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli_module, "load_config", lambda *_args, **_kwargs: {})
    args = argparse.Namespace(
        config=None,
        json=False,
        probe_openrouter=False,
        probe_ollama=False,
    )
    assert cmd_doctor(args) == 0


def test_doctor_requires_python_mcp(monkeypatch):
    checks = [
        Check("cli:claude", True, "ok"),
        Check("cli:codex", True, "ok"),
        Check("cli:gemini", True, "ok"),
        Check("python:mcp", False, "missing"),
    ]
    monkeypatch.setattr(cli_module, "check_environment", lambda *args, **kwargs: checks)
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli_module, "load_config", lambda *_args, **_kwargs: {})
    args = argparse.Namespace(
        config=None,
        json=False,
        probe_openrouter=False,
        probe_ollama=False,
    )
    assert cmd_doctor(args) == 1


def test_doctor_openrouter_only_does_not_require_native_clis(monkeypatch):
    config = {
        "defaults": {"mode": "remote"},
        "participants": {
            "remote": {
                "type": "openrouter",
                "model": "openai/gpt-test",
                "api_key_env": "CUSTOM_OPENROUTER_KEY",
            }
        },
        "modes": {"remote": {"participants": ["remote"]}},
    }
    monkeypatch.setenv("CUSTOM_OPENROUTER_KEY", "secret")
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli_module, "load_config", lambda *_args, **_kwargs: config)
    args = argparse.Namespace(
        config=None,
        json=False,
        probe_openrouter=False,
        probe_ollama=False,
    )

    checks = check_environment(config)
    assert {check.name for check in checks} >= {"env:CUSTOM_OPENROUTER_KEY", "python:mcp"}
    assert "cli:claude" not in {check.name for check in checks}
    assert cmd_doctor(args) == 0


def test_latest_transcript_ordering_and_corrupt_records(tmp_path: Path):
    out_dir = tmp_path / ".llm-council" / "runs"
    out_dir.mkdir(parents=True)
    old_path = out_dir / "old.md"
    new_path = out_dir / "new.md"
    old_path.write_text("old", encoding="utf-8")
    new_path.write_text("new", encoding="utf-8")
    old_time = 1_700_000_000
    new_time = old_time + 10
    __import__("os").utime(old_path, (old_time, old_time))
    __import__("os").utime(new_path, (new_time, new_time))
    assert latest_transcript(out_dir) == new_path
    assert latest_transcript(tmp_path / "empty") is None

    (out_dir / "bad.json").write_text("{bad-json", encoding="utf-8")
    assert transcript_records(out_dir) == []


def test_write_transcript_uses_final_round_for_recommendations(tmp_path: Path):
    out_dir = tmp_path / ".llm-council" / "runs"
    md_path = out_dir / "multi.md"
    json_path = out_dir / "multi.json"
    results = [
        ParticipantResult("a", True, "RECOMMENDATION: yes - ship", "", 1),
        ParticipantResult("b", True, "RECOMMENDATION: no - wait", "", 1),
        ParticipantResult("a:round2", True, "RECOMMENDATION: tradeoff - guarded", "", 1),
        ParticipantResult("b:round2", True, "RECOMMENDATION: tradeoff - guarded", "", 1),
    ]
    assert [result.name for result in final_round_results(results)] == [
        "a:round2",
        "b:round2",
    ]
    write_transcript(
        md_path,
        json_path,
        question="naive unicode: naïve",
        mode="quick",
        current="codex",
        participants=["a", "b"],
        prompt="prompt",
        results=results,
        transparent=False,
        metadata={"rounds": 2, "deliberation_status": "ran_no_labeled_disagreement"},
    )
    content = md_path.read_text(encoding="utf-8")
    assert "Successful responses: 4/4 total" in content
    assert "Final-round successful responses: 2/2" in content
    assert "0 yes / 0 no / 2 tradeoff / 0 unknown" in content
    assert "Model Comparison" not in content


def test_transcripts_cli_limit_zero_and_relative_show(tmp_path: Path, capsys):
    out_dir = tmp_path / ".llm-council" / "runs"
    md_path = out_dir / "20260425-test.md"
    json_path = out_dir / "20260425-test.json"
    write_transcript(
        md_path,
        json_path,
        question="test",
        mode="quick",
        current="codex",
        participants=[],
        prompt="prompt",
        results=[],
        metadata={"rounds": 1},
    )

    parser = build_parser()
    args = parser.parse_args(
        ["transcripts", "list", "--cwd", str(tmp_path), "--limit", "0"]
    )
    assert cmd_transcripts(args) == 0
    assert capsys.readouterr().out == ""

    args = parser.parse_args(
        [
            "transcripts",
            "show",
            ".llm-council/runs/20260425-test.md",
            "--cwd",
            str(tmp_path),
        ]
    )
    assert cmd_transcripts(args) == 0
    assert "LLM Council Transcript" in capsys.readouterr().out
