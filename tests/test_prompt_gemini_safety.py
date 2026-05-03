import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from llm_council.adapters import run_cli_participant
from llm_council.context import build_prompt
from llm_council.defaults import DEFAULT_CONFIG


def test_long_context_overflow_fails_fast_instead_of_truncating(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a" * 120_000)
    second.write_text("b" * 120_000)

    with pytest.raises(ValueError) as excinfo:
        build_prompt(
            "Should we make this change?",
            mode="review",
            cwd=tmp_path,
            context_paths=[str(first), str(second)],
            include_diff=False,
            stdin_text=None,
        )

    message = str(excinfo.value)
    assert "max_prompt_chars" in message
    assert "chunk-strategy" in message


def test_build_prompt_honors_configured_prompt_limit(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a" * 120_000)
    second.write_text("b" * 120_000)

    prompt = build_prompt(
        "Should we make this change?",
        mode="review",
        cwd=tmp_path,
        context_paths=[str(first), str(second)],
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=300_000,
    )

    assert len(prompt) > 200_000
    assert "[llm-council prompt truncated" not in prompt


def test_gemini_default_sends_large_prompt_to_stdin_not_argv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt = "x" * 120_000
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self, input=None):
            captured["stdin"] = input
            return b"RECOMMENDATION: yes - ok", b""

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured["command"] = command
        captured["stdin_pipe"] = kwargs.get("stdin")
        return FakeProcess()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    cfg = deepcopy(DEFAULT_CONFIG["participants"]["gemini"])
    result = asyncio.run(run_cli_participant("gemini", cfg, prompt, tmp_path))

    command = captured["command"]
    assert result.ok is True
    assert command == ("gemini", "--approval-mode", "plan")
    assert captured["stdin"] == prompt.encode()
    assert all(prompt not in part for part in command)
