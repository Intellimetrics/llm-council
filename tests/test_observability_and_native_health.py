from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from llm_council import adapters, doctor, orchestrator
from llm_council.adapters import ParticipantResult
from llm_council.display import format_progress_message, render_summary_markdown


GEMINI_UNSUPPORTED = """Error authenticating: IneligibleTierError: client rejected
reasonCode: 'UNSUPPORTED_CLIENT'
"""


def test_gemini_unsupported_client_has_stable_error_kind() -> None:
    assert adapters.is_client_ineligible_error(GEMINI_UNSUPPORTED)
    assert (
        adapters.classify_error(GEMINI_UNSUPPORTED)
        == adapters.ERROR_KIND_CLIENT_INELIGIBLE
    )
    assert "client_ineligible" in adapters.KNOWN_ERROR_KINDS


def test_antigravity_individual_quota_has_stable_error_kind() -> None:
    error = "Error: Individual quota reached. Resets in 47h5m13s."
    assert adapters.is_quota_exhausted_error(error)
    assert adapters.classify_error(error) == adapters.ERROR_KIND_QUOTA_EXHAUSTED


def test_doctor_default_cli_check_does_not_invoke_native_model(monkeypatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda command: f"/bin/{command}")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("default doctor must not invoke a native model")

    monkeypatch.setattr(doctor.subprocess, "run", unexpected_run)
    checks = doctor.check_environment(
        {
            "participants": {
                "gemini": {
                    "type": "cli",
                    "family": "gemini",
                    "command": "gemini",
                }
            }
        }
    )

    cli_check = next(check for check in checks if check.name == "cli:gemini")
    assert cli_check.ok
    assert "authentication not probed" in cli_check.detail
    assert not any(check.name.startswith("probe:cli:") for check in checks)


def test_ollama_probe_rejects_missing_configured_model(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"models": [{"name": "qwen3-coder:30b"}]},
        ),
    )
    check = doctor._probe_ollama(
        "http://localhost:11434",
        expected_models=["qwen3-coder-next:q4_K_M"],
    )
    assert not check.ok
    assert "not installed" in check.detail


def test_opt_in_native_probe_classifies_gemini_and_offers_configured_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=GEMINI_UNSUPPORTED,
        ),
    )
    config = {
        "participants": {
            "gemini": {
                "type": "cli",
                "family": "gemini",
                "command": "gemini",
                "args": ["--approval-mode", "plan"],
                "stdin_prompt": True,
            },
            "antigravity": {
                "type": "cli",
                "family": "antigravity",
                "command": "agy",
                "stdin_prompt": True,
            },
        }
    }

    checks = doctor.check_environment(
        config,
        probe_native={"gemini"},
        probe_cwd=tmp_path,
    )
    probe = next(check for check in checks if check.name == "probe:cli:gemini")
    assert not probe.ok
    assert probe.error_kind == "client_ineligible"
    assert probe.suggested_fallback == "antigravity"
    assert "UNSUPPORTED_CLIENT" in probe.detail

    rendered = next(
        item
        for item in doctor.checks_to_dict(checks)
        if item["name"] == "probe:cli:gemini"
    )
    assert rendered["error_kind"] == "client_ineligible"
    assert rendered["suggested_fallback"] == "antigravity"


def test_cli_wall_elapsed_includes_failed_terse_retry_but_legacy_elapsed_does_not(
    monkeypatch, tmp_path: Path
) -> None:
    calls = 0

    async def fake_once(name, cfg, prompt, cwd, **_kwargs):
        nonlocal calls
        calls += 1
        # Long enough that Windows' ~15.6ms monotonic-clock granularity
        # cannot quantize two sleeps below the single-sleep threshold.
        await asyncio.sleep(0.05)
        elapsed = 3.0 if calls == 1 else 1.0
        return (
            ParticipantResult(
                name=name,
                ok=False,
                output="",
                error=f"Timeout: attempt {calls}",
                elapsed_seconds=elapsed,
                prompt_chars=len(prompt),
            ),
            {"nonzero_exit": False, "stderr": "", "exited": False},
        )

    monkeypatch.setattr(adapters, "_run_cli_once", fake_once)
    result = asyncio.run(
        adapters.run_cli_participant(
            "gemini",
            {"type": "cli", "family": "gemini", "timeout": 1},
            "question",
            tmp_path,
        )
    )

    assert calls == 2
    assert result.elapsed_seconds == 3.0
    assert result.terse_retry_attempted
    assert result.wall_elapsed_seconds is not None
    # Above one sleep (0.05) proves the wall clock spanned BOTH attempts;
    # 0.08 leaves margin for coarse-clock undermeasurement of the second.
    assert result.wall_elapsed_seconds >= 0.08


def test_participant_finish_exposes_attempt_and_wall_durations(monkeypatch, tmp_path: Path) -> None:
    async def fake_participant(*_args, **_kwargs):
        return ParticipantResult(
            "peer",
            True,
            "RECOMMENDATION: yes - ready",
            "",
            1.0,
            wall_elapsed_seconds=2.5,
        )

    monkeypatch.setattr(adapters, "run_participant", fake_participant)
    events: list[dict] = []
    asyncio.run(
        adapters.run_participants(
            ["peer"],
            {"peer": {"type": "cli", "family": "test"}},
            "question",
            tmp_path,
            progress=events.append,
        )
    )

    finish = next(event for event in events if event["event"] == "participant_finish")
    assert finish["elapsed_seconds"] == 1.0
    assert finish["wall_elapsed_seconds"] == 2.5
    assert finish["duration_seconds"] == 2.5


def test_execute_council_records_run_wall_and_timestamped_progress(
    monkeypatch, tmp_path: Path
) -> None:
    async def fake_run_participants(*_args, **_kwargs):
        # Stay above Windows' coarser scheduler/clock tick. A 10 ms sleep can
        # legitimately quantize to a zero-length monotonic interval in CI even
        # though the production timer is working.
        await asyncio.sleep(0.05)
        return [
            ParticipantResult(
                "peer",
                True,
                "RECOMMENDATION: yes - ready",
                "",
                1.25,
                wall_elapsed_seconds=1.75,
            )
        ]

    monkeypatch.setattr(orchestrator, "run_participants", fake_run_participants)
    _results, metadata = asyncio.run(
        orchestrator.execute_council(
            ["peer"],
            {"peer": {"type": "cli", "family": "test"}},
            "question",
            tmp_path,
            {"defaults": {"synthesize": False}},
            deliberate=False,
        )
    )

    assert metadata["run_wall_elapsed_seconds"] > 0.0
    assert metadata["participant_elapsed_seconds_aggregate"] == 1.25
    assert metadata["participant_wall_elapsed_seconds_aggregate"] == 1.75
    assert metadata["run_started_at"].endswith("Z")
    assert metadata["run_finished_at"].endswith("Z")
    assert all("timestamp" in event for event in metadata["progress_events"])
    assert all("run_elapsed_seconds" in event for event in metadata["progress_events"])
    finish = metadata["progress_events"][-1]
    assert finish["event"] == "council_finish"
    assert finish["duration_seconds"] == round(
        metadata["run_wall_elapsed_seconds"], 3
    )


def test_summary_distinguishes_run_wall_from_participant_aggregate() -> None:
    markdown = render_summary_markdown(
        mode="quick",
        ok_count=1,
        total=1,
        elapsed_seconds=21.0,
        wall_elapsed_seconds=10.0,
        recommendation="yes",
        per_peer_rows=[
            {
                "name": "gemini",
                "label": "yes",
                "elapsed_seconds": 5.0,
                "wall_elapsed_seconds": 8.0,
            }
        ],
        transcript_path=None,
    )

    assert "run wall=10.0s" in markdown
    assert "participant aggregate=21.0s" in markdown
    assert "| peer | label | wall time |" in markdown
    assert "| gemini | yes | 8.0s |" in markdown


def test_progress_message_shows_retry_wall_and_run_wall() -> None:
    participant = format_progress_message(
        {
            "event": "participant_finish",
            "participant": "gemini",
            "status": "error",
            "elapsed_seconds": 2.0,
            "wall_elapsed_seconds": 5.0,
        }
    )
    council = format_progress_message(
        {
            "event": "council_finish",
            "ok": 2,
            "total": 3,
            "duration_seconds": 7.5,
        }
    )
    assert participant and "wall 5.0s; attempt 2.0s" in participant
    assert council and "run wall 7.5s" in council
