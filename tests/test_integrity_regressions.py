"""Focused regressions for council integrity and transcript durability."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as RealDatetime
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council import transcript as transcript_module
from llm_council.adapters import (
    CacheContext,
    ERROR_KIND_PINNED_MODEL_UNVERIFIED,
    KNOWN_ERROR_KINDS,
    PINNED_MODEL_UNVERIFIED_PREFIX,
    ParticipantResult,
    _cache_lookup,
    classify_error,
    clean_subprocess_env,
    run_cli_participant,
)
from llm_council.cache import (
    build_payload as build_cache_payload,
    cache_path,
    compute_key,
    write_cache,
)
from llm_council.context import resolve_acceptance_contract
from llm_council.display import wants_color, wants_quiet
from llm_council.env import project_env_context
from llm_council.model_catalog import openrouter_cache_path
from llm_council.orchestrator import (
    _drop_missing_key_participants,
    execute_council,
)
from llm_council.recommend_judge import _resolve_judge_peer
from llm_council.synthesis import run_synthesis_chair
from llm_council.transcript import (
    ensure_private_transcript_dir,
    final_decision_label,
    find_transcript_by_id,
    inspect_transcript_permissions,
    transcript_dir_within_root,
    transcript_paths,
    write_transcript,
)
from llm_council.update_check import (
    NAG_OPT_OUT_ENV,
    _default_nag_cache_path,
    maybe_print_update_nag,
)
from proc_stubs import fake_proc_returning


POSIX_PERMISSION_TEST = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX mode/ownership semantics are not exposed by Windows stat",
)


def _vote(name: str, label: str) -> ParticipantResult:
    return ParticipantResult(
        name=name,
        ok=True,
        output=f"RECOMMENDATION: {label} - reason",
        error="",
        elapsed_seconds=0.1,
    )


def test_required_model_pin_fails_closed_without_served_model_telemetry(
    tmp_path: Path,
) -> None:
    cfg = {
        "type": "cli",
        "family": "claude",
        "command": "claude",
        "args": [],
        "model": "claude-fable-5",
        "usage_from_json": True,
        "require_pinned_model": True,
        "require_sections": False,
        "stdin_prompt": False,
        "timeout": 30,
        "timeout_per_kb_chars": 0,
    }
    response_without_model = json.dumps(
        {
            "result": "RECOMMENDATION: yes - looks fine",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )

    async def run() -> ParticipantResult:
        with fake_proc_returning(response_without_model):
            return await run_cli_participant("claude_fable", cfg, "prompt", tmp_path)

    result = asyncio.run(run())

    assert result.ok is False
    assert result.error.startswith(PINNED_MODEL_UNVERIFIED_PREFIX)
    assert classify_error(result.error) == ERROR_KIND_PINNED_MODEL_UNVERIFIED
    assert ERROR_KIND_PINNED_MODEL_UNVERIFIED in KNOWN_ERROR_KINDS
    assert result.model == "claude-fable-5"


def test_required_model_pin_does_not_reuse_pre_fail_closed_cache(
    tmp_path: Path,
) -> None:
    cfg = {
        "type": "cli",
        "family": "claude",
        "model": "claude-fable-5",
        "usage_from_json": True,
        "require_pinned_model": True,
    }
    prompt = "prompt"
    old_key = compute_key("claude_fable", cfg, prompt)
    old_payload = build_cache_payload(
        participant_name="claude_fable",
        prompt=prompt,
        key=old_key,
        output="RECOMMENDATION: yes - cached before telemetry was required",
        recommendation_label="yes",
        elapsed_seconds=0.1,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model="claude-fable-5",
        command=None,
    )
    write_cache(cache_path(tmp_path, "claude_fable", old_key), old_payload, 3600)

    new_key, cached = _cache_lookup(
        "claude_fable",
        cfg,
        prompt,
        CacheContext(cwd=tmp_path),
    )

    assert new_key != old_key
    assert cached is None


def test_transcript_paths_are_unique_under_concurrent_same_question_calls(
    tmp_path: Path,
) -> None:
    class FrozenDatetime:
        @classmethod
        def now(cls) -> RealDatetime:
            return RealDatetime(2026, 7, 13, 12, 34, 56, 123456)

    with patch("llm_council.transcript.datetime", FrozenDatetime):
        with ThreadPoolExecutor(max_workers=16) as pool:
            paths = list(
                pool.map(
                    lambda _: transcript_paths(tmp_path, "same question"),
                    range(128),
                )
            )

    markdown_paths = [markdown for markdown, _json in paths]
    assert len(set(markdown_paths)) == len(markdown_paths)
    assert all(
        re.fullmatch(r"20260713_123456_123456_[0-9a-f]{32}\.md", path.name)
        for path in markdown_paths
    )
    assert all("same-question" not in path.name for path in markdown_paths)
    assert all(markdown.stem == json_path.stem for markdown, json_path in paths)


def test_opaque_transcript_id_preserves_full_id_lookup(tmp_path: Path) -> None:
    _markdown, json_path = transcript_paths(
        tmp_path, "incident password sk-secret-must-never-enter-a-filename"
    )
    json_path.write_text('{"question": "opaque"}', encoding="utf-8")

    loaded = find_transcript_by_id(tmp_path, json_path.name)

    assert loaded["question"] == "opaque"
    assert Path(loaded["_path"]) == json_path
    assert "incident" not in json_path.name
    assert "secret" not in json_path.name


@POSIX_PERMISSION_TEST
def test_transcript_write_secures_absent_directory_under_umask_zero(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "absent" / "runs"
    markdown_path = runs / "run.md"
    json_path = runs / "run.json"
    previous_umask = os.umask(0)
    try:
        write_transcript(
            markdown_path,
            json_path,
            question="private?",
            mode="quick",
            current="codex",
            participants=["peer"],
            prompt="prompt",
            results=[_vote("peer", "yes")],
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(runs.stat().st_mode) == 0o700
    for path in (markdown_path, json_path, markdown_path.with_suffix(".html")):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@POSIX_PERMISSION_TEST
def test_secure_transcript_dir_tightens_precreated_permissive_directory(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o777)

    ensure_private_transcript_dir(runs)

    assert stat.S_IMODE(runs.stat().st_mode) == 0o700


@POSIX_PERMISSION_TEST
def test_transcript_writes_are_private_atomic_and_do_not_follow_symlinks(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    markdown_path = runs / "run.md"
    json_path = runs / "run.json"
    outside = tmp_path / "outside.txt"
    outside.write_text("sentinel", encoding="utf-8")
    markdown_path.symlink_to(outside)
    json_path.write_text("old", encoding="utf-8")
    json_path.chmod(0o666)

    write_transcript(
        markdown_path,
        json_path,
        question="private?",
        mode="quick",
        current="codex",
        participants=["peer"],
        prompt="prompt",
        results=[_vote("peer", "yes")],
    )

    html_path = markdown_path.with_suffix(".html")
    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert not markdown_path.is_symlink()
    assert stat.S_IMODE(runs.stat().st_mode) == 0o700
    for path in (markdown_path, json_path, html_path):
        assert path.exists()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_windows_path_fallback_writes_without_directory_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "windows-runs"
    markdown_path = runs / "run.md"
    json_path = runs / "run.json"

    monkeypatch.setattr(transcript_module, "_uses_windows_path_fallback", lambda: True)

    def unexpected_descriptor_open(*_args, **_kwargs):
        raise AssertionError("POSIX directory-descriptor path must not run")

    monkeypatch.setattr(
        transcript_module,
        "_open_owned_transcript_directory",
        unexpected_descriptor_open,
    )

    write_transcript(
        markdown_path,
        json_path,
        question="private?",
        mode="quick",
        current="codex",
        participants=["peer"],
        prompt="prompt",
        results=[_vote("peer", "yes")],
    )

    assert "private?" in markdown_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["question"] == "private?"
    assert markdown_path.with_suffix(".html").is_file()
    assert not list(runs.glob(".*.tmp"))


def test_windows_path_permission_audit_avoids_directory_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "windows-runs"
    runs.mkdir()
    artifact = runs / "20260101_010101_history.json"
    artifact.write_text("history", encoding="utf-8")
    artifact.chmod(0o666)

    monkeypatch.setattr(transcript_module, "_uses_windows_path_fallback", lambda: True)

    def unexpected_descriptor_open(*_args, **_kwargs):
        raise AssertionError("POSIX directory-descriptor path must not run")

    monkeypatch.setattr(
        transcript_module,
        "_open_owned_transcript_directory",
        unexpected_descriptor_open,
    )

    preview = inspect_transcript_permissions(runs)
    repaired = inspect_transcript_permissions(runs, repair=True)

    assert preview["eligible_files"] == 1
    assert artifact.name in (
        preview["already_private_files"] + preview["would_repair_files"]
    )
    assert repaired["eligible_files"] == 1
    assert artifact.name in (
        repaired["already_private_files"] + repaired["repaired_files"]
    )


def test_windows_path_fallback_replaces_file_link_without_following_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "windows-runs"
    runs.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("sentinel", encoding="utf-8")
    markdown_path = runs / "run.md"
    try:
        markdown_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    monkeypatch.setattr(transcript_module, "_uses_windows_path_fallback", lambda: True)
    transcript_module._atomic_write_private(markdown_path, "replacement")

    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert markdown_path.read_text(encoding="utf-8") == "replacement"
    assert not markdown_path.is_symlink()


def test_windows_path_fallback_refuses_linked_transcript_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-runs"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    monkeypatch.setattr(transcript_module, "_uses_windows_path_fallback", lambda: True)
    with pytest.raises(OSError, match="symlink or reparse-point"):
        ensure_private_transcript_dir(linked)


@POSIX_PERMISSION_TEST
def test_secure_transcript_dir_refuses_symlink_and_foreign_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o755)
    linked = tmp_path / "linked-runs"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="symlink transcript directory"):
        ensure_private_transcript_dir(linked)
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755

    owned = tmp_path / "owned-runs"
    owned.mkdir()
    owned.chmod(0o777)
    actual_uid = owned.stat().st_uid
    monkeypatch.setattr(transcript_module, "_current_euid", lambda: actual_uid + 1)
    with pytest.raises(PermissionError, match="not owned"):
        ensure_private_transcript_dir(owned)
    assert stat.S_IMODE(owned.stat().st_mode) == 0o777


@POSIX_PERMISSION_TEST
def test_historical_permission_repair_is_narrow_and_does_not_follow_links(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o777)
    artifacts = [
        runs / "20260101_010101_legacy.md",
        runs / "20260101_010101_legacy.json",
        runs / "20260101_010101_legacy.html",
    ]
    for artifact in artifacts:
        artifact.write_text("history", encoding="utf-8")
        artifact.chmod(0o666)
    unrelated = runs / "notes.md"
    unrelated.write_text("not a transcript", encoding="utf-8")
    unrelated.chmod(0o666)
    outside = tmp_path / "outside.txt"
    outside.write_text("sentinel", encoding="utf-8")
    outside.chmod(0o666)
    linked_name = "20260101_010101_link.md"
    (runs / linked_name).symlink_to(outside)
    hardlink_target = tmp_path / "hardlink-target.txt"
    hardlink_target.write_text("shared inode", encoding="utf-8")
    hardlink_target.chmod(0o666)
    hardlink_name = "20260101_010101_hardlink.json"
    os.link(hardlink_target, runs / hardlink_name)

    preview = inspect_transcript_permissions(runs)
    report = inspect_transcript_permissions(runs, repair=True)

    expected_names = sorted(path.name for path in artifacts)
    assert preview["would_repair_files"] == expected_names
    assert preview["skipped_symlinks"] == [linked_name]
    assert preview["skipped_hardlinks"] == [hardlink_name]
    assert report["directory_repaired"] is True
    assert report["repaired_files"] == expected_names
    assert report["skipped_symlinks"] == [linked_name]
    assert report["skipped_hardlinks"] == [hardlink_name]
    assert stat.S_IMODE(runs.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in artifacts)
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o666
    assert stat.S_IMODE(outside.stat().st_mode) == 0o666
    assert stat.S_IMODE(hardlink_target.stat().st_mode) == 0o666


@POSIX_PERMISSION_TEST
def test_transcript_dir_within_root_rejects_escape_and_existing_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    cwd = root / "project"
    cwd.mkdir(parents=True)
    inside = transcript_dir_within_root(
        cwd, {"transcripts_dir": ".llm-council/runs"}, root=root
    )
    assert inside == (cwd / ".llm-council/runs").resolve()

    with pytest.raises(ValueError, match="inside MCP project root"):
        transcript_dir_within_root(cwd, {"transcripts_dir": "../../outside"}, root=root)

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = cwd / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside MCP project root"):
        transcript_dir_within_root(cwd, {"transcripts_dir": "linked"}, root=root)


def test_dashboard_decision_uses_final_peer_votes_and_reports_ties(
    tmp_path: Path,
) -> None:
    markdown_path = tmp_path / "tie.md"
    json_path = tmp_path / "tie.json"
    tied = [_vote("a", "yes"), _vote("b", "no")]

    write_transcript(
        markdown_path,
        json_path,
        question="ship?",
        mode="quick",
        current=None,
        participants=["a", "b"],
        prompt="ship?",
        results=tied,
        metadata={
            "recommendation": "yes",
            "synthesis": {
                "ok": True,
                "decision_label": "yes",
                "output": "### Decision\nYes. Ship it.",
            },
        },
    )

    dashboard = markdown_path.with_suffix(".html").read_text(encoding="utf-8")
    assert final_decision_label(tied) == "unknown"
    assert "Decision: UNKNOWN" in dashboard
    assert "Decision: YES" not in dashboard


def test_dashboard_decision_uses_latest_round_only(tmp_path: Path) -> None:
    markdown_path = tmp_path / "latest.md"
    results = [
        _vote("a", "yes"),
        _vote("b", "yes"),
        _vote("a:round2", "no"),
        _vote("b:round2", "no"),
    ]

    write_transcript(
        markdown_path,
        tmp_path / "latest.json",
        question="ship?",
        mode="deliberate",
        current=None,
        participants=["a", "b"],
        prompt="ship?",
        results=results,
        metadata={"recommendation": "yes"},
    )

    dashboard = markdown_path.with_suffix(".html").read_text(encoding="utf-8")
    assert final_decision_label(results) == "no"
    assert "Decision: NO" in dashboard


def test_synthesis_chair_parses_its_required_decision_section(tmp_path: Path) -> None:
    async def fake_run_participant(*args, **kwargs) -> ParticipantResult:
        return ParticipantResult(
            name="chair",
            ok=True,
            output=(
                "### Decision\nTradeoff. Ship after the migration test.\n\n"
                "### Consensus blockers\n- none"
            ),
            error="",
            elapsed_seconds=0.1,
        )

    with patch(
        "llm_council.synthesis.run_participant",
        side_effect=fake_run_participant,
    ):
        payload = asyncio.run(
            run_synthesis_chair(
                question="ship?",
                results=[_vote("peer", "yes")],
                convergence=None,
                participant_cfg={"chair": {"type": "cli"}},
                cwd=tmp_path,
                chair_name="chair",
            )
        )

    assert payload["decision_label"] == "tradeoff"


def test_preflight_failed_peer_is_not_reintroduced_during_deliberation(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    async def fake_preflight(*args, **kwargs) -> dict[str, str]:
        return {"dead": "PreflightFailed: local endpoint unavailable"}

    async def fake_run_participants(selected, *args, **kwargs):
        calls.append(list(selected))
        if len(calls) == 1:
            return [_vote("a", "yes"), _vote("b", "no")]
        return [_vote(name, "yes") for name in selected]

    participant_cfg = {
        name: {"type": "cli", "command": "true"} for name in ("a", "b", "dead")
    }
    with (
        patch(
            "llm_council.orchestrator.preflight_local_participants",
            side_effect=fake_preflight,
        ),
        patch(
            "llm_council.orchestrator.run_participants",
            side_effect=fake_run_participants,
        ),
    ):
        _results, metadata = asyncio.run(
            execute_council(
                participants=["a", "b", "dead"],
                participant_cfg=participant_cfg,
                prompt="question",
                cwd=tmp_path,
                config={},
                deliberate=True,
                max_rounds=2,
            )
        )

    assert calls == [["a", "b"], ["a", "b"]]
    skip_event = next(
        event
        for event in metadata["progress_events"]
        if event.get("event") == "deliberation_skip_participants"
    )
    assert skip_event["skipped"] == ["dead"]


def test_long_literal_acceptance_contract_is_not_statted_as_a_path(
    tmp_path: Path,
) -> None:
    contract = "accept when " + ("x" * 10_000)
    assert resolve_acceptance_contract(contract, cwd=tmp_path) == contract


def test_request_local_environment_reaches_all_runtime_consumers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    key_name = "COUNCIL_REQUEST_ONLY_KEY"
    for name in (
        key_name,
        "NO_COLOR",
        "LLM_COUNCIL_QUIET",
        "XDG_CACHE_HOME",
        NAG_OPT_OUT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    request_cache = tmp_path / "request-cache"
    (tmp_path / ".llm-council.env").write_text(
        (
            f"{key_name}=request-secret\n"
            "NO_COLOR=1\n"
            "LLM_COUNCIL_QUIET=1\n"
            f"XDG_CACHE_HOME={request_cache}\n"
            f"{NAG_OPT_OUT_ENV}=1\n"
        ),
        encoding="utf-8",
    )
    hosted_cfg = {
        "hosted": {
            "type": "openrouter",
            "model": "vendor/model",
            "api_key_env": key_name,
        }
    }
    judge_config = {
        "defaults": {"recommend_judge": "hosted"},
        "participants": hosted_cfg,
    }

    class Tty:
        @staticmethod
        def isatty() -> bool:
            return True

    def fail_update_check(_version: str):
        raise AssertionError("opt-out must skip update checker")

    with project_env_context(tmp_path):
        child_env = clean_subprocess_env([key_name], strict=True)
        active, dropped = _drop_missing_key_participants(["hosted"], hosted_cfg)
        resolved_judge = _resolve_judge_peer(judge_config)
        color_enabled = wants_color(Tty())
        quiet_enabled = wants_quiet()
        catalog_cache = openrouter_cache_path()
        update_cache = _default_nag_cache_path()
        update_check_skipped = not maybe_print_update_nag(
            "0.0.0",
            checker=fail_update_check,
        )

    assert child_env[key_name] == "request-secret"
    assert active == ["hosted"]
    assert dropped == []
    assert resolved_judge is not None
    assert resolved_judge["api_key"] == "request-secret"
    assert color_enabled is False
    assert quiet_enabled is True
    assert catalog_cache == request_cache / "llm-council" / "openrouter-models.json"
    assert update_cache == request_cache / "llm-council" / "update-check.json"
    assert update_check_skipped is True
    # Context cleanup is just as important as propagation: no request secret
    # may leak into later work on the same long-lived host process.
    active_after, dropped_after = _drop_missing_key_participants(["hosted"], hosted_cfg)
    assert active_after == []
    assert dropped_after[0]["peer"] == "hosted"


def test_request_local_key_reaches_hosted_adapter_and_doctor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import llm_council.adapters as adapters_module
    import llm_council.doctor as doctor_module

    key_name = "COUNCIL_HOSTED_REQUEST_KEY"
    monkeypatch.delenv(key_name, raising=False)
    (tmp_path / ".llm-council.env").write_text(
        f"{key_name}=hosted-request-secret\n",
        encoding="utf-8",
    )
    observed_headers: dict[str, str] = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "model": "vendor/model",
                "choices": [
                    {
                        "message": {
                            "content": "RECOMMENDATION: yes - request key resolved"
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def request(self, method, url, **kwargs):
            observed_headers.update(kwargs.get("headers") or {})
            return FakeResponse()

    cfg = {
        "type": "openrouter",
        "model": "vendor/model",
        "api_key_env": key_name,
        "require_sections": False,
        "retries": 0,
    }
    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        doctor_module,
        "_check_openrouter_catalog_age",
        lambda _config: doctor_module.Check("catalog:openrouter", True, "fresh"),
    )

    with project_env_context(tmp_path):
        hosted_result = asyncio.run(
            adapters_module._run_openai_compatible_inner(
                "hosted",
                cfg,
                "prompt",
            )
        )
        doctor_checks = doctor_module.check_environment(
            {"participants": {"hosted": cfg}}
        )

    assert hosted_result.ok is True
    assert observed_headers["Authorization"] == "Bearer hosted-request-secret"
    env_check = next(
        check for check in doctor_checks if check.name == f"env:{key_name}"
    )
    assert env_check.ok is True
