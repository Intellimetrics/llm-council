import asyncio
import os
import sys
from pathlib import Path

from llm_council.adapters import clean_subprocess_env, redact_prompt_args, run_cli_participant


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
    assert result.error == "TimeoutError: participant exceeded 1s timeout"

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
