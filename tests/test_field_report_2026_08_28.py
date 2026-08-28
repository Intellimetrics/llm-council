"""Regressions for the 2026-08-28 security-council field report.

Over 48 council runs in that project the council was effectively one peer:
codex timed out in 18 (inheriting the operator's `ultra` reasoning effort
from ~/.codex/config.toml), antigravity returned an empty response in 17
(no exit code, no stderr, no retry), and three codex results were OpenAI
cyber-policy refusals recorded as `error_kind=unknown` with 542 KB of raw
stderr as the error string. Each test below pins one of the fixes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llm_council import orchestrator as orchestrator_module
from llm_council.adapters import (
    CODEX_DEFAULT_REASONING_EFFORT,
    EMPTY_RESPONSE_ERROR_PREFIX,
    ERROR_KIND_CLI_NONZERO,
    ERROR_KIND_CONTENT_REFUSED,
    ERROR_KIND_INVALID_RESPONSE,
    ERROR_KIND_QUOTA_EXHAUSTED,
    ERROR_KIND_TIMEOUT,
    ERROR_KIND_UNKNOWN,
    KNOWN_ERROR_KINDS,
    ParticipantResult,
    _bounded_stderr,
    _build_cli_command,
    _parse_cli_usage_json,
    classify_error,
    content_refusal_excerpt,
    is_content_refusal_error,
    is_empty_response_error,
    run_cli_participant,
)
from llm_council.config import (
    OLD_CODEX_MCP_STARVED_ARGS,
    load_config,
    migrate_known_cli_defaults,
)
from llm_council.defaults import DEFAULT_CONFIG
from llm_council.orchestrator import execute_council
from llm_council.setup_wizard import INSTRUCTION_TEXT, _ensure_project_gitignore
from llm_council.transcript import result_to_dict

_PATCH_TARGET = "llm_council.adapters.asyncio.create_subprocess_exec"

CODEX_REFUSAL_STDERR = (
    "OpenAI Codex v0.150.1\n--------\nworkdir: /x\nmodel: gpt-5.6-sol\n"
    "reasoning effort: ultra\n--------\nuser\nYou are a read-only participant"
    + ("\nsome trajectory line" * 3000)
    + "\n\nERROR: This content was flagged for possible cybersecurity risk. "
    "If this seems wrong, try rephrasing your request. To get authorized "
    "for security work, join the Trusted Access for Cyber program: "
    "https://chatgpt.com/cyber\ntokens used\n179,081"
)


class _Proc:
    """One-shot fake subprocess with configurable stdout / stderr / status."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, _data=None):
        return (self._stdout.encode(), self._stderr.encode())

    async def wait(self):
        return self.returncode


class _KilledProc:
    """Blocks until terminated, then hands back the stderr it had buffered."""

    def __init__(self, stderr: str):
        self.returncode: int | None = None
        self._stderr = stderr
        self._dead = asyncio.Event()

    async def communicate(self, _data=None):
        await self._dead.wait()
        return (b"", self._stderr.encode())

    def terminate(self) -> None:
        self.returncode = -15
        self._dead.set()

    def kill(self) -> None:
        self.returncode = -9
        self._dead.set()

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


def _sequence(*procs):
    """Patch context serving ``procs`` in order; returns (ctx, launches)."""
    launches: list[list[str]] = []
    queue = iter(procs)

    async def _factory(*command, **_kwargs):
        launches.append(list(command))
        return next(queue)

    return patch(_PATCH_TARGET, new=_factory), launches


def _cli_cfg(**overrides):
    cfg = {
        "type": "cli",
        "family": "claude",
        "command": "claude",
        "args": ["-p"],
        "stdin_prompt": True,
        "timeout": 30,
    }
    cfg.update(overrides)
    return cfg


def _run(cfg, tmp_path: Path, prompt: str = "question") -> ParticipantResult:
    return asyncio.run(run_cli_participant("peer", cfg, prompt, tmp_path))


# --------------------------------------------------------------------------
# LEAD 1 — codex must not inherit the operator's interactive profile
# --------------------------------------------------------------------------


def test_codex_baseline_args_include_skip_git_repo_check() -> None:
    args = DEFAULT_CONFIG["participants"]["codex"]["args"]
    assert "--skip-git-repo-check" in args
    assert "read-only" in args  # read-only sandbox survives


def test_defaults_reasoning_effort_matches_adapter_fallback() -> None:
    """`replace_defaults` configs that predate the key fall back to the
    adapter constant — the two must not drift."""
    assert (
        DEFAULT_CONFIG["participants"]["codex"]["reasoning_effort"]
        == CODEX_DEFAULT_REASONING_EFFORT
    )


def test_codex_default_command_pins_reasoning_effort() -> None:
    cfg = dict(DEFAULT_CONFIG["participants"]["codex"])
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    token = f"model_reasoning_effort={CODEX_DEFAULT_REASONING_EFFORT}"
    assert cmd.count(token) == 1
    idx = cmd.index(token)
    assert cmd[idx - 1] == "-c"
    # Option region: after exec, before the stdin positional.
    assert cmd.index("exec") < idx < cmd.index("-")


def test_codex_customized_args_still_get_reasoning_effort() -> None:
    """The field config (security-council) carried a hand-edited arg list
    without the key; injection must not depend on the exact-match
    migration, same lesson as the MCP starvation."""
    cfg = {
        "family": "codex",
        "command": "codex",
        "args": [
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "-c",
            "mcp_servers={}",
            "--cd",
            "{cwd}",
            "-",
        ],
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    assert f"model_reasoning_effort={CODEX_DEFAULT_REASONING_EFFORT}" in cmd
    assert cmd.count("mcp_servers={}") == 1


def test_codex_operator_effort_token_in_args_wins() -> None:
    cfg = {
        "family": "codex",
        "command": "codex",
        "args": ["exec", "-c", "model_reasoning_effort=low", "-"],
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    effort_tokens = [t for t in cmd if t.startswith("model_reasoning_effort")]
    assert effort_tokens == ["model_reasoning_effort=low"]


@pytest.mark.parametrize("value", [None, "inherit", "INHERIT", ""])
def test_codex_reasoning_effort_inherit_skips_injection(value) -> None:
    cfg = {
        "family": "codex",
        "command": "codex",
        "args": ["exec", "-"],
        "reasoning_effort": value,
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    assert not any(t.startswith("model_reasoning_effort") for t in cmd)


def test_codex_reasoning_effort_custom_value() -> None:
    cfg = {
        "family": "codex",
        "command": "codex",
        "args": ["exec", "-"],
        "reasoning_effort": "low",
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    assert "model_reasoning_effort=low" in cmd


def test_codex_pinned_model_keeps_effort_after_exec() -> None:
    cfg = {
        "family": "codex",
        "command": "codex",
        "model": "gpt-5.4",
        "args": ["exec", "--sandbox", "read-only", "-"],
    }
    cmd = _build_cli_command("codex", cfg, "q", Path("/tmp"))
    token = f"model_reasoning_effort={CODEX_DEFAULT_REASONING_EFFORT}"
    assert cmd.count(token) == 1
    assert cmd.index("exec") < cmd.index(token)


def test_non_codex_families_never_get_reasoning_effort() -> None:
    for family, command in (("claude", "claude"), ("antigravity", "agy")):
        cfg = {"family": family, "command": command, "args": ["exec"]}
        cmd = _build_cli_command(family, cfg, "q", Path("/tmp"))
        assert not any(t.startswith("model_reasoning_effort") for t in cmd)


def test_migration_upgrades_v22_codex_args() -> None:
    config = {
        "participants": {
            "codex": {
                "type": "cli",
                "family": "codex",
                "args": list(OLD_CODEX_MCP_STARVED_ARGS),
            }
        }
    }
    migrate_known_cli_defaults(config)
    assert (
        config["participants"]["codex"]["args"]
        == DEFAULT_CONFIG["participants"]["codex"]["args"]
    )


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("reasoning_effort", 2, "reasoning_effort"),
        ("reasoning_effort", "   ", "reasoning_effort"),
        ("retry_on_empty_response", "no", "retry_on_empty_response"),
    ],
)
def test_config_rejects_bad_new_participant_keys(
    tmp_path: Path, key, value, match
) -> None:
    path = tmp_path / ".llm-council.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "participants": {
                    "codex": {
                        "type": "cli",
                        "family": "codex",
                        "command": "codex",
                        key: value,
                    }
                },
                "modes": {"field-test": {"participants": ["codex"]}},
            }
        )
    )
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_config_accepts_inherit_and_null_reasoning_effort(tmp_path: Path) -> None:
    for value in (None, "inherit", "low"):
        path = tmp_path / ".llm-council.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "participants": {
                        "codex": {
                            "type": "cli",
                            "family": "codex",
                            "command": "codex",
                            "reasoning_effort": value,
                        }
                    },
                    # A custom mode name: `quick` would deep-merge onto the
                    # built-in strategy mode and trip "exactly one of" validation.
                    "modes": {"field-test": {"participants": ["codex"]}},
                }
            )
        )
        assert load_config(path)["participants"]["codex"]["reasoning_effort"] == value


# --------------------------------------------------------------------------
# LEAD 3 — empty responses: diagnostics + one same-prompt re-run
# --------------------------------------------------------------------------


def test_empty_response_error_carries_exit_code_and_stderr(tmp_path: Path) -> None:
    ctx, launches = _sequence(_Proc("", "agy: model returned no content\n", 0))
    with ctx:
        result = _run(_cli_cfg(retry_on_empty_response=False), tmp_path)
    assert result.ok is False
    assert len(launches) == 1
    assert result.error.startswith(EMPTY_RESPONSE_ERROR_PREFIX)
    assert is_empty_response_error(result.error)
    assert "exit 0" in result.error
    assert "agy: model returned no content" in result.error
    assert result.exit_code == 0
    assert result.stderr_tail == "agy: model returned no content"
    assert classify_error(result.error) == ERROR_KIND_INVALID_RESPONSE
    payload = result_to_dict(result)
    assert payload["exit_code"] == 0
    assert payload["stderr_tail"] == "agy: model returned no content"


def test_empty_response_with_no_stderr_says_so(tmp_path: Path) -> None:
    ctx, _ = _sequence(_Proc("", "", 0))
    with ctx:
        result = _run(_cli_cfg(retry_on_empty_response=False), tmp_path)
    assert "no stderr" in result.error
    assert result.stderr_tail is None
    assert "stderr_tail" not in result_to_dict(result)


def test_empty_response_retries_once_and_recovers(tmp_path: Path) -> None:
    ctx, launches = _sequence(_Proc("", "", 0), _Proc("RECOMMENDATION: yes - fine"))
    with ctx:
        result = _run(_cli_cfg(), tmp_path)
    assert result.ok is True, result.error
    assert len(launches) == 2
    assert launches[0] == launches[1]  # same prompt, same command
    assert result.empty_retry_attempted is True
    assert result.recovered_after_empty_retry is True
    payload = result_to_dict(result)
    assert payload["empty_retry_attempted"] is True
    assert payload["recovered_after_empty_retry"] is True
    assert "exit_code" not in payload  # success: no diagnostics noise


def test_empty_response_retry_failure_is_annotated_and_bounded(tmp_path: Path) -> None:
    ctx, launches = _sequence(_Proc("", "", 0), _Proc("", "", 0))
    with ctx:
        result = _run(_cli_cfg(), tmp_path)
    assert result.ok is False
    assert len(launches) == 2  # exactly one re-run, never a third call
    assert result.empty_retry_attempted is True
    assert result.recovered_after_empty_retry is False
    assert result.error.startswith(EMPTY_RESPONSE_ERROR_PREFIX)
    assert "One same-prompt re-run was attempted" in result.error
    assert "usage_from_json" in result.error
    assert classify_error(result.error) == ERROR_KIND_INVALID_RESPONSE


def test_empty_retry_that_returns_unlabeled_text_does_not_chain_label_repair(
    tmp_path: Path,
) -> None:
    ctx, launches = _sequence(_Proc("", "", 0), _Proc("I have thoughts but no label."))
    with ctx:
        result = _run(_cli_cfg(), tmp_path)
    assert result.ok is False
    assert len(launches) == 2  # the label-repair retry must NOT fire on top
    assert result.error.startswith("InvalidParticipantResponse: missing required")
    assert result.empty_retry_attempted is True


@pytest.mark.parametrize("overrides", [{"retries": 0}, {"retry_on_empty_response": False}])
def test_empty_retry_respects_opt_outs(tmp_path: Path, overrides) -> None:
    ctx, launches = _sequence(_Proc("", "", 0), _Proc("RECOMMENDATION: yes - x"))
    with ctx:
        result = _run(_cli_cfg(**overrides), tmp_path)
    assert result.ok is False
    assert len(launches) == 1
    assert result.empty_retry_attempted is False


def test_timeout_result_keeps_partial_stderr_tail(tmp_path: Path) -> None:
    """What a timed-out CLI wrote before the kill is the only record of
    what it was doing (codex: its tool trajectory)."""
    ctx, launches = _sequence(
        _KilledProc("reading src/app.py\nrunning grep over tests/\n")
    )
    cfg = _cli_cfg(
        family="codex",
        command="codex",
        args=["exec", "-"],
        timeout=1,
        timeout_per_kb_chars=0,
        terse_retry_on_timeout=False,
    )
    with ctx:
        result = _run(cfg, tmp_path)
    assert result.ok is False
    assert len(launches) == 1
    assert classify_error(result.error) == ERROR_KIND_TIMEOUT
    assert result.exit_code == -15
    assert result.stderr_tail is not None
    assert result.stderr_tail.endswith("running grep over tests/")
    assert result_to_dict(result)["stderr_tail"] == result.stderr_tail


# --------------------------------------------------------------------------
# LEAD 2 — the "unparsed" codex results were content-policy refusals
# --------------------------------------------------------------------------


def test_content_refused_is_a_known_error_kind() -> None:
    assert ERROR_KIND_CONTENT_REFUSED in KNOWN_ERROR_KINDS


def test_classify_error_recognizes_content_refusals() -> None:
    assert classify_error(CODEX_REFUSAL_STDERR) == ERROR_KIND_CONTENT_REFUSED
    assert classify_error("Error: content_policy_violation") == ERROR_KIND_CONTENT_REFUSED
    assert (
        classify_error("Candidate blocked: PROHIBITED_CONTENT")
        == ERROR_KIND_CONTENT_REFUSED
    )
    assert (
        classify_error("finishReason: SAFETY — response withheld")
        == ERROR_KIND_CONTENT_REFUSED
    )
    assert (
        classify_error("HTTPStatusError: 400 request blocked by safety filter")
        == ERROR_KIND_CONTENT_REFUSED
    )
    # Refusal precedes the quota scan when both phrasings appear.
    assert (
        classify_error(
            "ERROR: This content was flagged for possible cybersecurity risk\n"
            "rate_limit_exceeded"
        )
        == ERROR_KIND_CONTENT_REFUSED
    )
    # Negatives: the echoed PROMPT of a security review talks about policy
    # and blocking without being a refusal; synthesized prefixes keep their
    # own kinds.
    assert (
        classify_error("Review whether the content policy blocks uploads")
        == ERROR_KIND_UNKNOWN
    )
    assert (
        classify_error("CliExitNonZero: `codex` exited with status 1 and no stderr output")
        == ERROR_KIND_CLI_NONZERO
    )
    assert classify_error("insufficient_quota") == ERROR_KIND_QUOTA_EXHAUSTED
    assert is_content_refusal_error("") is False


def test_content_refusal_excerpt_returns_the_matched_line() -> None:
    excerpt = content_refusal_excerpt(CODEX_REFUSAL_STDERR)
    assert excerpt.startswith("ERROR: This content was flagged")
    assert "OpenAI Codex" not in excerpt  # not the banner line


def test_bounded_stderr_keeps_head_and_tail() -> None:
    bounded = _bounded_stderr(CODEX_REFUSAL_STDERR)
    assert len(bounded) < 9000
    assert bounded.startswith("OpenAI Codex v0.150.1")
    assert "chars of stderr elided" in bounded
    assert bounded.rstrip().endswith("179,081")
    assert is_content_refusal_error(bounded)
    short = "ERROR: tiny"
    assert _bounded_stderr(short) == short


def test_nonzero_exit_error_is_bounded_and_classified(tmp_path: Path) -> None:
    ctx, launches = _sequence(_Proc("", CODEX_REFUSAL_STDERR, 1))
    cfg = _cli_cfg(family="codex", command="codex", args=["exec", "-"])
    with ctx:
        result = _run(cfg, tmp_path)
    assert result.ok is False
    assert len(launches) == 1
    assert len(result.error) < 9000
    assert "chars of stderr elided" in result.error
    assert classify_error(result.error) == ERROR_KIND_CONTENT_REFUSED
    assert result.exit_code == 1
    assert result.stderr_tail is None  # already embedded in `error`
    assert result_to_dict(result)["error_kind"] == ERROR_KIND_CONTENT_REFUSED


def test_execute_council_surfaces_content_refused_peers(
    monkeypatch, tmp_path: Path
) -> None:
    async def fake_run_participants(*args, **kwargs):
        return [
            ParticipantResult("codex", False, "", CODEX_REFUSAL_STDERR, 748.5),
            ParticipantResult("claude", True, "RECOMMENDATION: yes - proceed", "", 0.8),
        ]

    monkeypatch.setattr(orchestrator_module, "run_participants", fake_run_participants)
    participant_cfg = {
        "codex": {"family": "codex", "type": "cli"},
        "claude": {"family": "claude", "type": "cli"},
    }
    _, metadata = asyncio.run(
        execute_council(["codex", "claude"], participant_cfg, "q", tmp_path, {})
    )
    refused = metadata["content_refused_peers"]
    assert len(refused) == 1
    assert refused[0]["peer"] == "codex"
    assert refused[0]["family"] == "codex"
    assert refused[0]["message"].startswith("ERROR: This content was flagged")
    assert "quota_throttled_peers" not in metadata
    events = [e for e in metadata["progress_events"] if e.get("event") == "peer_content_refused"]
    assert len(events) == 1
    assert events[0]["peer"] == "codex"
    assert events[0]["round"] == 1


def test_mcp_lifts_content_refused_peers_to_top_level(
    monkeypatch, tmp_path: Path
) -> None:
    from llm_council import mcp_server as mcp_module
    from llm_council.mcp_server import COUNCIL_RUN_OUTPUT_SCHEMA_VERSION, run_council

    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  mode: refusal-test
participants:
  local_peer:
    type: ollama
    model: llama3
    base_url: http://localhost:11434
modes:
  refusal-test:
    participants: [local_peer]
""".lstrip(),
        encoding="utf-8",
    )
    record = {
        "peer": "local_peer",
        "family": "ollama",
        "model": "llama3",
        "message": "ERROR: This content was flagged for possible cybersecurity risk.",
    }

    async def fake_execute_council(*args, **kwargs):
        return (
            [ParticipantResult("local_peer", False, "", CODEX_REFUSAL_STDERR, 1.0)],
            {
                "rounds": 1,
                "deliberated": False,
                "min_quorum": 1,
                "labeled_quorum": 0,
                "degraded": True,
                "content_refused_peers": [record],
            },
        )

    monkeypatch.setattr(mcp_module, "execute_council", fake_execute_council)
    result = asyncio.run(
        run_council({"question": "ping", "working_directory": str(tmp_path)})
    )
    assert result["schema_version"] == COUNCIL_RUN_OUTPUT_SCHEMA_VERSION == 11
    assert result["content_refused_peers"] == [record]
    assert "content_refused_peers" not in (result.get("metadata") or {})
    per_result = result["results"][0]
    assert per_result["error_kind"] == ERROR_KIND_CONTENT_REFUSED
    assert "exit_code" in per_result and "stderr_tail" in per_result


# --------------------------------------------------------------------------
# antigravity JSON output mode (the diagnostic lever for empty responses)
# --------------------------------------------------------------------------

AGY_JSON = json.dumps(
    {
        "conversation_id": "18e64562",
        "status": "SUCCESS",
        "response": "Looks fine.\n\nRECOMMENDATION: yes - bounds enforced\n",
        "duration_seconds": 36.95,
        "num_turns": 1,
        "usage": {
            "input_tokens": 18952,
            "output_tokens": 3423,
            "thinking_tokens": 3113,
            "cache_read_tokens": 28445,
            "total_tokens": 22375,
        },
    }
)


def test_parse_antigravity_usage_json_success() -> None:
    parsed = _parse_cli_usage_json("antigravity", AGY_JSON)
    assert parsed is not None
    assert parsed["text"].startswith("Looks fine.")
    assert parsed["prompt_tokens"] == 18952
    assert parsed["completion_tokens"] == 3423
    assert parsed["total_tokens"] == 22375
    assert parsed["cost_usd"] is None
    assert parsed["model"] is None
    assert parsed["status"] == "SUCCESS"


def test_parse_antigravity_usage_json_failure_status_is_a_describable_empty() -> None:
    parsed = _parse_cli_usage_json(
        "antigravity", json.dumps({"status": "ERROR", "error": "model unavailable"})
    )
    assert parsed is not None
    assert parsed["text"] == ""
    assert parsed["status"] == "ERROR"
    assert parsed["detail"] == "model unavailable"
    # Malformed / unknown shapes still fail soft.
    assert _parse_cli_usage_json("antigravity", "not json") is None
    assert _parse_cli_usage_json("antigravity", json.dumps({"foo": 1})) is None


def test_antigravity_usage_from_json_adds_output_format_flag_once() -> None:
    cfg = dict(DEFAULT_CONFIG["participants"]["antigravity"])
    cfg["usage_from_json"] = True
    cmd = _build_cli_command("antigravity", cfg, "q", Path("/tmp"))
    assert cmd.count("--output-format") == 1
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert "plan" in cmd  # read-only mode flag untouched
    pinned = dict(cfg)
    pinned["args"] = [*cfg["args"], "--output-format", "json"]
    assert _build_cli_command("antigravity", pinned, "q", Path("/tmp")).count(
        "--output-format"
    ) == 1
    off = dict(DEFAULT_CONFIG["participants"]["antigravity"])
    assert "--output-format" not in _build_cli_command("antigravity", off, "q", Path("/tmp"))


def test_antigravity_json_success_populates_usage(tmp_path: Path) -> None:
    ctx, _ = _sequence(_Proc(AGY_JSON))
    cfg = _cli_cfg(
        family="antigravity",
        command="agy",
        args=["--print", "{prompt}", "--mode", "plan"],
        stdin_prompt=False,
        usage_from_json=True,
    )
    with ctx:
        result = _run(cfg, tmp_path)
    assert result.ok is True, result.error
    assert result.total_tokens == 22375
    assert result.prompt_tokens == 18952
    assert result.model is None


def test_antigravity_json_empty_response_names_cli_status(tmp_path: Path) -> None:
    ctx, _ = _sequence(
        _Proc(json.dumps({"status": "TIMEOUT", "response": "", "usage": {}}))
    )
    cfg = _cli_cfg(
        family="antigravity",
        command="agy",
        args=["--print", "{prompt}", "--mode", "plan"],
        stdin_prompt=False,
        usage_from_json=True,
        retry_on_empty_response=False,
    )
    with ctx:
        result = _run(cfg, tmp_path)
    assert result.ok is False
    assert result.error.startswith(EMPTY_RESPONSE_ERROR_PREFIX)
    assert "CLI status TIMEOUT" in result.error


# --------------------------------------------------------------------------
# smaller items: gitignore template + prompt-phrasing guidance
# --------------------------------------------------------------------------


def test_project_gitignore_covers_cache_dir(tmp_path: Path) -> None:
    path = tmp_path / ".gitignore"
    assert _ensure_project_gitignore(path) is True
    lines = path.read_text(encoding="utf-8").splitlines()
    assert ".llm-council/cache/" in lines
    assert ".llm-council/runs/" in lines
    assert _ensure_project_gitignore(path) is False  # idempotent


def test_host_instructions_steer_security_reviews_to_verification_phrasing() -> None:
    assert "VERIFY a control" in INSTRUCTION_TEXT
    assert "find a bypass" in INSTRUCTION_TEXT
    assert "content_refused_peers" in INSTRUCTION_TEXT
