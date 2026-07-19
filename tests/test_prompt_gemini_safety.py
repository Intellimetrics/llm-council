import asyncio
from pathlib import Path


from llm_council.adapters import run_cli_participant
from llm_council.context import build_prompt


def test_long_context_overflow_chunks_instead_of_truncating_silently(
    tmp_path: Path,
) -> None:
    """v0.8.1: context_files exceeding the cap now auto-chunk via the
    hash-aware strategy instead of either silently truncating OR
    fail-fasting. The fail-fast invariant in this test's old name was
    inverted in Fix A — the new invariant is that overflow is loudly
    surfaced via a `chunk_progress` event and never silent.
    """
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a" * 120_000)
    second.write_text("b" * 120_000)

    events: list[dict] = []
    prompt = build_prompt(
        "Should we make this change?",
        mode="review",
        cwd=tmp_path,
        context_paths=[str(first), str(second)],
        include_diff=False,
        stdin_text=None,
        chunk_progress=events.append,
    )

    # Prompt now fits under the default cap.
    assert len(prompt) <= 200_000
    # And the operator sees what happened — no silent truncation.
    chunk_events = [
        e for e in events if e.get("event") == "context_files_chunked"
    ]
    assert chunk_events, (
        "expected a context_files_chunked event; overflow must not be silent"
    )
    last = chunk_events[-1]
    assert last["strategy"] == "hash-aware"
    assert last["dropped_files"]


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


def test_gemini_family_custom_peer_sends_large_prompt_to_stdin_not_argv(
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

    cfg = {
        "type": "cli",
        "family": "gemini",
        "command": "gemini",
        "args": ["--approval-mode", "plan"],
        "timeout": 240,
        "stdin_prompt": True,
    }
    result = asyncio.run(run_cli_participant("gemini", cfg, prompt, tmp_path))

    command = captured["command"]
    assert result.ok is True
    assert command == ("gemini", "--approval-mode", "plan")
    assert captured["stdin"] == prompt.encode()
    assert all(prompt not in part for part in command)
