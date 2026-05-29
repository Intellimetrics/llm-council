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


def test_idle_read_path_does_not_deadlock_on_large_prompt(tmp_path: Path):
    """Regression for the idle-read pipe deadlock. With `idle_timeout` set
    (streamed-read path), a large stdin prompt sent to a child that emits
    stdout BEFORE consuming stdin must not deadlock. The pre-fix code wrote +
    drained + closed ALL of stdin before starting the readers, so a child that
    filled its ~64KB stdout pipe buffer blocked (no reader) while we blocked in
    drain() (child not reading) — a classic pipe deadlock that hung until the
    wall-clock cap. The interleaved write/read must complete promptly.

    The whole call is wrapped in an outer wait_for so a regression to the
    write-then-read ordering FAILS (TimeoutError) instead of hanging the suite
    forever — the old drain() ran outside the wall-clock guard.
    """
    code = (
        "import sys\n"
        "sys.stdout.write('X' * 100000)\n"   # fill stdout pipe buffer first
        "sys.stdout.flush()\n"
        "data = sys.stdin.read()\n"           # only now consume stdin
        "sys.stdout.write('\\nRECOMMENDATION: yes - read %d bytes' % len(data))\n"
        "sys.stdout.flush()\n"
    )
    big_prompt = "P" * 100000  # > 64KB so stdin drain would block under old code
    cfg = {
        "type": "cli",
        "command": sys.executable,
        "args": ["-c", code],
        "stdin_prompt": True,
        "idle_timeout": 5.0,       # triggers the streamed-read path
        "timeout": 20,             # wall-clock cap
        "max_prompt_chars": 500_000,
    }

    async def _go():
        return await asyncio.wait_for(
            run_cli_participant("slow_reader", cfg, big_prompt, tmp_path),
            timeout=35,
        )

    result = asyncio.run(_go())
    assert result.ok is True, f"expected success, got error={result.error!r}"
    assert "read 100000 bytes" in result.output
    # Completed well under the wall-clock cap — a deadlock would have burned ~20s.
    assert result.elapsed_seconds < 15


def test_run_participants_degrades_on_unguarded_setup_error(tmp_path: Path):
    """A per-peer setup crash (here a name missing from participant_cfg, which
    raises KeyError before run_one's try) must degrade that one peer to a failed
    result instead of aborting the whole round and losing the healthy peer."""
    code = "print('RECOMMENDATION: yes - ok')"
    cfg = {
        "healthy": {
            "type": "cli",
            "command": sys.executable,
            "args": ["-c", code],
            "timeout": 5,
            "require_recommendation": False,
        }
    }
    results = asyncio.run(
        run_participants(["healthy", "ghost"], cfg, "p", tmp_path, max_concurrency=2)
    )
    by_name = {r.name: r for r in results}
    assert set(by_name) == {"healthy", "ghost"}
    assert by_name["healthy"].ok is True
    assert by_name["ghost"].ok is False
    assert "KeyError" in by_name["ghost"].error


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


def test_no_default_cli_peer_auto_approves_tool_calls():
    """Read-only invariant guard. A default CLI peer must never ship a blanket
    auto-approve flag (--dangerously-skip-permissions and friends), which would
    let a misbehaving or prompt-injected peer Write/Edit files. agy in
    particular has no --approval-mode / --tools allowlist, so its read-only-ness
    is SOFT (prompt-enforced via the council read-only directive, which it
    honors) rather than flag-enforced — keeping the skip-permissions flag OFF is
    what stops a stray write from being auto-approved. (The soft guarantee is
    checked live by the opt-in tests/test_live_agy_readonly.py canary.) This
    test fails loudly if anyone re-adds such a flag to a default peer."""
    from llm_council.defaults import DEFAULT_CONFIG

    forbidden = {
        "--dangerously-skip-permissions",
        "--dangerously-bypass-approvals-and-sandbox",
        "--yolo",
        "--full-auto",
    }
    offenders = []
    for name, cfg in DEFAULT_CONFIG["participants"].items():
        if cfg.get("type") != "cli":
            continue
        args = {str(a).lower() for a in cfg.get("args", [])}
        bad = forbidden.intersection(args)
        if bad:
            offenders.append((name, sorted(bad)))
    assert not offenders, f"default CLI peers auto-approve tool calls: {offenders}"


def test_antigravity_default_is_sandboxed_without_skip_permissions():
    from llm_council.defaults import DEFAULT_CONFIG

    args = DEFAULT_CONFIG["participants"]["antigravity"]["args"]
    assert "--sandbox" in args
    assert "--dangerously-skip-permissions" not in args
    assert DEFAULT_CONFIG["participants"]["antigravity"]["read_only"] is True
