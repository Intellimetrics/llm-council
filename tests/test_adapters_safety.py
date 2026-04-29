import asyncio
import os
import sys
from pathlib import Path

from llm_council.adapters import (
    ParticipantResult,
    clean_subprocess_env,
    is_timeout_error,
    redact_prompt_args,
    run_cli_participant,
    run_participants,
)


def test_run_cli_participant_cleans_up_timed_out_process(tmp_path: Path):
    code = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path('child.pid').write_text(str(os.getpid())); "
        "time.sleep(60)"
    )

    result = asyncio.run(
        run_cli_participant(
            "python",
            {"type": "cli", "command": sys.executable, "args": ["-c", code], "timeout": 1},
            "prompt",
            tmp_path,
        )
    )

    assert result.ok is False
    assert result.output == ""
    assert result.command == [sys.executable, "-c", code]
    assert result.error.startswith("Timeout: `python` did not respond within 1s")
    assert "participants.python.timeout" in result.error

    pid = int((tmp_path / "child.pid").read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError(f"timed-out subprocess still exists: {pid}")


def test_run_cli_participant_skips_prompt_over_size_limit(monkeypatch, tmp_path: Path):
    async def fail_create_subprocess_exec(*_args, **_kwargs):
        raise AssertionError("subprocess should not be launched")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_create_subprocess_exec)

    result = asyncio.run(
        run_cli_participant(
            "python",
            {
                "type": "cli",
                "command": sys.executable,
                "args": ["-c", "print('ok')"],
                "max_prompt_chars": 3,
            },
            "too long",
            tmp_path,
        )
    )

    assert result.ok is False
    assert result.output == ""
    assert result.error == (
        "PromptTooLarge: participant skipped before launch; "
        "prompt has 8 chars, limit is 3"
    )
    assert result.command == [sys.executable, "-c", "print('ok')"]


def test_is_timeout_error_recognizes_new_and_legacy_messages():
    assert is_timeout_error("Timeout: `claude` did not respond within 240s ...")
    assert is_timeout_error("TimeoutError: participant exceeded 1s timeout")
    assert not is_timeout_error("OpenRouterEmptyResponse: missing message content")
    assert not is_timeout_error("")


def test_run_participants_finish_event_marks_timeout_status(tmp_path: Path):
    """A participant that times out gets status='timeout' in the progress event."""
    code = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    cfg = {
        "slowpoke": {
            "type": "cli",
            "command": sys.executable,
            "args": ["-c", code],
            "timeout": 1,
        }
    }
    events: list[dict] = []
    asyncio.run(
        run_participants(
            ["slowpoke"],
            cfg,
            "prompt",
            tmp_path,
            max_concurrency=1,
            progress=events.append,
        )
    )
    finish = [e for e in events if e.get("event") == "participant_finish"]
    assert len(finish) == 1
    assert finish[0]["status"] == "timeout"
    assert "Timeout:" in finish[0]["error"]


def test_run_participants_emits_participant_slow_before_timeout(tmp_path: Path):
    """Emit a heads-up event at the slow-warn threshold before the hard timeout."""
    code = "import time; time.sleep(0.5)"
    cfg = {
        "slow_runner": {
            "type": "cli",
            "command": sys.executable,
            "args": ["-c", code],
            "timeout": 4,
            "slow_warn_after_seconds": 0.1,
        }
    }
    events: list[dict] = []
    asyncio.run(
        run_participants(
            ["slow_runner"],
            cfg,
            "p",
            tmp_path,
            max_concurrency=1,
            progress=events.append,
        )
    )
    slow = [e for e in events if e.get("event") == "participant_slow"]
    assert len(slow) == 1
    assert slow[0]["participant"] == "slow_runner"
    assert slow[0]["timeout_seconds"] == 4
    assert slow[0]["elapsed_seconds"] == 0.1


def test_run_participants_cancels_watchdog_when_participant_start_raises(tmp_path: Path):
    """Regression: if the progress callback raises on participant_start, the
    watchdog task must still be cancelled instead of leaking into the loop."""
    code = "import time; time.sleep(0.05)"
    cfg = {
        "fast_runner": {
            "type": "cli",
            "command": sys.executable,
            "args": ["-c", code],
            "timeout": 4,
            "slow_warn_after_seconds": 0.05,
            "require_recommendation": False,
        }
    }

    seen_events: list[dict] = []

    def explosive_progress(event: dict) -> None:
        seen_events.append(event)
        if event.get("event") == "participant_start":
            raise RuntimeError("user-supplied progress callback exploded")

    async def go() -> None:
        try:
            await run_participants(
                ["fast_runner"],
                cfg,
                "p",
                tmp_path,
                max_concurrency=1,
                progress=explosive_progress,
            )
        except RuntimeError:
            pass
        # Yield once so a leaked watchdog (if any) gets a chance to fire.
        await asyncio.sleep(0.2)

    asyncio.run(go())
    # No participant_slow event should have been emitted; the watchdog was
    # cancelled by the finally block instead of leaking.
    slow = [e for e in seen_events if e.get("event") == "participant_slow"]
    assert slow == []


def test_run_participants_does_not_emit_slow_when_finishes_early(tmp_path: Path):
    """Fast-finishing participants should NOT trigger the slow event."""
    code = "print('done')"
    cfg = {
        "fast_runner": {
            "type": "cli",
            "command": sys.executable,
            "args": ["-c", code],
            "timeout": 5,
            "slow_warn_after_seconds": 2.0,
            "require_recommendation": False,
        }
    }
    events: list[dict] = []
    asyncio.run(
        run_participants(
            ["fast_runner"],
            cfg,
            "p",
            tmp_path,
            max_concurrency=1,
            progress=events.append,
        )
    )
    slow = [e for e in events if e.get("event") == "participant_slow"]
    assert slow == []


def test_clean_subprocess_env_strips_broad_secret_names(monkeypatch):
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("HOME", "/tmp/home")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/config")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("SERVICE_TOKEN", "secret")
    monkeypatch.setenv("DATABASE_PASSWORD", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("BASIC_AUTH", "secret")
    monkeypatch.setenv("CLAUDECODE", "1")

    env = clean_subprocess_env()

    for key in (
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "TMPDIR",
        "TERM",
        "LANG",
        "LC_ALL",
        "XDG_CONFIG_HOME",
    ):
        assert env[key] == os.environ[key]
    for key in (
        "OPENAI_API_KEY",
        "SERVICE_TOKEN",
        "DATABASE_PASSWORD",
        "AWS_SECRET_ACCESS_KEY",
        "BASIC_AUTH",
        "CLAUDECODE",
    ):
        assert key not in env

    env = clean_subprocess_env(["OPENAI_API_KEY"])
    assert env["OPENAI_API_KEY"] == "secret"
    assert "SERVICE_TOKEN" not in env


def test_redact_prompt_args_removes_full_prompt_and_large_fragments():
    prompt = (
        "short header\n"
        + ("sensitive prompt body " * 20)
        + "\nshort footer"
    )

    command = [
        "cli",
        f"--prompt={prompt}",
        f"--prefix={prompt[:128]}",
        f"--suffix={prompt[-128:]}",
    ]

    redacted = redact_prompt_args(command, prompt)

    assert redacted == [
        "cli",
        "--prompt=[prompt]",
        "--prefix=[prompt]",
        "--suffix=[prompt]",
    ]
    assert "sensitive prompt body" not in " ".join(redacted)
