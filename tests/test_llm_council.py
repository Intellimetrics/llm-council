from pathlib import Path

from llm_council.adapters import (
    ParticipantResult,
    _format_arg,
    clean_subprocess_env,
    redact_prompt_args,
)
from llm_council.config import load_config, select_participants
from llm_council.context import build_prompt
from llm_council.defaults import DEFAULT_CONFIG
from llm_council.deliberation import has_disagreement, model_comparison
from llm_council.doctor import _probe_ollama, _probe_openrouter
from llm_council.env import load_project_env
from llm_council.model_catalog import (
    _read_cache,
    _write_cache,
    infer_origin,
    normalize_openrouter_model,
)
from llm_council.mcp_server import (
    council_run_schema,
    last_transcript,
    last_transcript_schema,
)
from llm_council.policy import should_use_council
from llm_council.setup_wizard import mcp_config, project_config, write_setup_files
from llm_council.transcript import latest_transcript, result_to_dict, safe_slug, write_transcript


def test_select_other_cli_peers_excludes_current():
    config = load_config(None)
    selected = select_participants(config, "quick", "codex")
    assert selected == ["claude", "gemini"]


def test_claude_prompt_goes_to_stdin():
    assert DEFAULT_CONFIG["participants"]["claude"]["stdin_prompt"] is True


def test_clean_subprocess_env_strips_claudecode(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    env = clean_subprocess_env()
    assert "CLAUDECODE" not in env


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


def test_us_origin_policy_filters_non_us_additions():
    config = load_config(None)
    selected = select_participants(config, "diverse", "codex", origin_policy="us")
    assert selected == ["claude", "gemini"]


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


def test_model_comparison_and_disagreement_detection():
    results = [
        ParticipantResult("a", True, "Yes, proceed.", "", 1),
        ParticipantResult("b", True, "No, defer.", "", 1, total_tokens=4),
    ]
    assert has_disagreement(results) is True
    comparison = "\n".join(model_comparison(results))
    assert "a:" in comparison
    assert "4 tokens" in comparison


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


def test_mcp_run_schema_has_question():
    schema = council_run_schema()
    assert "question" in schema["required"]
    assert "include_diff" in schema["properties"]


def test_mcp_last_transcript_schema():
    schema = last_transcript_schema()
    assert "format" in schema["properties"]


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


def test_setup_writes_config_mcp_and_instructions(tmp_path: Path):
    written = write_setup_files(tmp_path)
    names = {path.relative_to(tmp_path).as_posix() for path in written}
    assert ".llm-council.yaml" in names
    assert ".mcp.json" in names
    assert ".llm-council/instructions/claude.md" in names
    assert ".llm-council/instructions/codex.md" in names
    assert ".llm-council/instructions/gemini.md" in names
    assert "llm-council" in (tmp_path / ".mcp.json").read_text()


def test_mcp_config_passes_openrouter_env_reference(tmp_path: Path):
    config = mcp_config(tmp_path)
    env = config["mcpServers"]["llm-council"]["env"]
    assert env["OPENROUTER_API_KEY"] == "${OPENROUTER_API_KEY}"


def test_last_transcript_reads_latest_from_project_cwd(tmp_path: Path):
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
