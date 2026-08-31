"""Integration tests: OKF enrichment inside build_prompt + surfaces."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import llm_council.okf_context as okf
from llm_council.context import build_prompt
from llm_council.okf_context import OKF_SECTION_HEADER, OkfSettings


def _git_repo_with_diff(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "mod.py").write_text(
        "\n".join(f"line{i}" for i in range(1, 31)) + "\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "mod.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    lines = (tmp_path / "mod.py").read_text(encoding="utf-8").splitlines()
    lines[11] = "line12-changed"
    (tmp_path / "mod.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def _stub_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "stub-bundle"
    (bundle / "functions" / "mod").mkdir(parents=True)
    (bundle / "index.md").write_text(
        '---\nsource_revision: "cafe1234"\n---\n', encoding="utf-8"
    )
    (bundle / "functions" / "mod" / "touched_fn.md").write_text(
        "---\n"
        "type: Python Function\n"
        "title: touched_fn\n"
        "resource: mod.py#L1-L50\n"
        "relationships:\n"
        "  called_by:\n"
        "  - target: functions/other/consumer\n"
        "---\n\n# Signature\n\n`def touched_fn() -> None`\n",
        encoding="utf-8",
    )
    return bundle


def _patch_bundle(monkeypatch: pytest.MonkeyPatch, bundle: Path) -> None:
    monkeypatch.setattr(
        okf,
        "generate_ephemeral_bundle",
        lambda cwd, out_dir, *, binary, timeout: (bundle, None, None),
    )
    monkeypatch.setattr(okf, "_head_revision", lambda cwd: "cafe1234")


def _build(cwd: Path, **kwargs) -> str:
    defaults = dict(
        mode="quick",
        cwd=cwd,
        context_paths=[],
        include_diff=True,
        stdin_text=None,
    )
    defaults.update(kwargs)
    return build_prompt("review this change", **defaults)


def test_prompt_byte_identical_when_disabled(tmp_path: Path):
    repo = _git_repo_with_diff(tmp_path)
    baseline = _build(repo)
    omitted = _build(repo)  # kwargs omitted entirely
    explicit_none = _build(repo, okf_settings=None)
    disabled = _build(repo, okf_settings=OkfSettings(enabled=False))
    assert baseline == omitted == explicit_none == disabled
    assert OKF_SECTION_HEADER not in baseline


def test_prompt_byte_identical_on_okf_failure(tmp_path: Path):
    repo = _git_repo_with_diff(tmp_path)
    baseline = _build(repo)
    statuses: list[dict] = []
    enabled_but_failing = _build(
        repo,
        okf_settings=OkfSettings(
            enabled=True, binary="definitely-not-a-real-binary-xyz"
        ),
        okf_status=statuses.append,
    )
    assert enabled_but_failing == baseline
    assert statuses and statuses[-1]["status"] == "binary_missing"


def test_okf_section_inserted_after_diff_before_context_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = _git_repo_with_diff(tmp_path)
    (repo / "notes.txt").write_text("attached context\n", encoding="utf-8")
    _patch_bundle(monkeypatch, _stub_bundle(tmp_path))

    statuses: list[dict] = []
    prompt = _build(
        repo,
        context_paths=["notes.txt"],
        okf_settings=OkfSettings(enabled=True),
        okf_status=statuses.append,
    )
    assert statuses[-1]["status"] == "attached"
    assert statuses[-1]["concepts"] == 1
    diff_pos = prompt.index("## Git Diff")
    okf_pos = prompt.index(OKF_SECTION_HEADER)
    file_pos = prompt.index("## File: notes.txt")
    assert diff_pos < okf_pos < file_pos
    assert "functions/other/consumer" in prompt
    assert "`def touched_fn() -> None`" in prompt


def test_okf_no_diff_status_when_diff_not_requested(tmp_path: Path):
    repo = _git_repo_with_diff(tmp_path)
    statuses: list[dict] = []
    prompt = _build(
        repo,
        include_diff=False,
        okf_settings=OkfSettings(enabled=True),
        okf_status=statuses.append,
    )
    assert statuses == [{"status": "no_diff"}]
    assert OKF_SECTION_HEADER not in prompt


def test_okf_never_causes_overflow_and_skips_when_no_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = _git_repo_with_diff(tmp_path)
    _patch_bundle(monkeypatch, _stub_bundle(tmp_path))

    # Cap sized so the base prompt fits with less headroom than the floor:
    # the excerpt is skipped and the prompt must be byte-identical to a
    # disabled run at the same cap.
    baseline = _build(repo)
    tight_cap = len(baseline) + 100  # < OKF_EXCERPT_FLOOR_CHARS of headroom
    statuses: list[dict] = []
    prompt = _build(
        repo,
        max_prompt_chars=tight_cap,
        okf_settings=OkfSettings(enabled=True),
        okf_status=statuses.append,
    )
    assert prompt == _build(repo, max_prompt_chars=tight_cap)
    assert statuses[-1]["status"] == "excerpt_over_budget"

    # Roomy cap: the enriched prompt still respects the cap.
    roomy_cap = len(baseline) + 5_000
    enriched = _build(
        repo,
        max_prompt_chars=roomy_cap,
        okf_settings=OkfSettings(enabled=True),
    )
    assert OKF_SECTION_HEADER in enriched
    assert len(enriched) <= roomy_cap


def test_okf_overflowing_base_prompt_chunks_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = _git_repo_with_diff(tmp_path)
    # Blow the diff up beyond the cap so the chunking path fires.
    (repo / "mod.py").write_text(
        "\n".join(f"line{i}-changed" for i in range(1, 3_000)) + "\n",
        encoding="utf-8",
    )
    _patch_bundle(monkeypatch, _stub_bundle(tmp_path))

    kwargs = dict(max_prompt_chars=8_000, chunk_strategy="hash-aware")
    baseline_events: list[dict] = []
    baseline = _build(repo, chunk_progress=baseline_events.append, **kwargs)
    statuses: list[dict] = []
    enriched_events: list[dict] = []
    enriched = _build(
        repo,
        okf_settings=OkfSettings(enabled=True),
        okf_status=statuses.append,
        chunk_progress=enriched_events.append,
        **kwargs,
    )
    assert enriched == baseline
    assert len(enriched) <= 8_000
    assert statuses[-1]["status"] == "excerpt_over_budget"
    assert [e["event"] for e in enriched_events] == [
        e["event"] for e in baseline_events
    ]


def test_estimate_parity_with_okf_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from llm_council.defaults import DEFAULT_CONFIG
    from llm_council.estimate import estimate_council

    repo = _git_repo_with_diff(tmp_path)
    _patch_bundle(monkeypatch, _stub_bundle(tmp_path))

    kwargs = dict(
        config=DEFAULT_CONFIG,
        cwd=repo,
        question="review this change",
        mode="quick",
        current="claude",
        include_diff=True,
        allow_network=False,
    )
    est_plain = estimate_council(**kwargs)
    est_okf = estimate_council(okf_context=True, **kwargs)
    # The excerpt is prompt-affecting, so the estimate must grow with it.
    assert est_okf["prompt_chars"] > est_plain["prompt_chars"]

    # Exact parity with a directly-built enriched prompt.
    from llm_council.okf_context import resolve_okf_settings

    direct = build_prompt(
        "review this change",
        mode="quick",
        cwd=repo,
        context_paths=[],
        include_diff=True,
        stdin_text=None,
        okf_settings=resolve_okf_settings(DEFAULT_CONFIG, "quick", True),
    )
    assert est_okf["prompt_chars"] == len(direct)
    assert OKF_SECTION_HEADER in direct


async def test_mcp_run_metadata_okf_context_and_no_toplevel_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from llm_council import mcp_server
    from llm_council.adapters import ParticipantResult
    from llm_council.mcp_server import COUNCIL_RUN_OUTPUT_SCHEMA_VERSION

    repo = _git_repo_with_diff(tmp_path)
    _patch_bundle(monkeypatch, _stub_bundle(tmp_path))
    (repo / ".llm-council.yaml").write_text(
        "replace_defaults: true\n"
        "defaults:\n  mode: custom\n"
        "participants:\n  a:\n    type: cli\n    command: true\n"
        "modes:\n  custom:\n    participants: [a]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(repo))

    async def fake_execute(*args, **kwargs):
        return (
            [
                ParticipantResult(
                    name="a", ok=True, output="RECOMMENDATION: yes",
                    error="", elapsed_seconds=0.0,
                )
            ],
            {"rounds": 1, "deliberated": False, "degraded": False},
        )

    transcript_metadata: dict[str, object] = {}

    def capture_transcript(*args, **kwargs):
        transcript_metadata.update(kwargs["metadata"])

    monkeypatch.setattr(mcp_server, "execute_council", fake_execute)
    monkeypatch.setattr(mcp_server, "write_transcript", capture_transcript)

    payload = await mcp_server.run_council(
        {
            "question": "ship?",
            "working_directory": str(repo),
            "include_diff": True,
            "okf_context": True,
        }
    )
    record = transcript_metadata["okf_context"]
    assert record["status"] == "attached"
    assert record["concepts"] == 1
    # Metadata-only surfacing: no top-level key, schema unchanged.
    assert "okf_context" not in payload
    assert payload["metadata"]["okf_context"]["status"] == "attached"
    assert COUNCIL_RUN_OUTPUT_SCHEMA_VERSION == 11

    # Disabled (arg omitted): no metadata key at all.
    transcript_metadata.clear()
    await mcp_server.run_council(
        {
            "question": "ship?",
            "working_directory": str(repo),
            "include_diff": True,
        }
    )
    assert "okf_context" not in transcript_metadata


def test_transcript_header_okf_bullet(tmp_path: Path):
    from llm_council.adapters import ParticipantResult
    from llm_council.transcript import write_transcript

    def _render(metadata: dict) -> str:
        md = tmp_path / "t.md"
        write_transcript(
            md,
            tmp_path / "t.json",
            question="q",
            mode="quick",
            current="claude",
            participants=["a"],
            prompt="p",
            results=[
                ParticipantResult(
                    name="a", ok=True, output="RECOMMENDATION: yes",
                    error="", elapsed_seconds=0.0,
                )
            ],
            metadata=metadata,
        )
        return md.read_text(encoding="utf-8")

    attached = _render(
        {
            "okf_context": {
                "status": "attached",
                "concepts": 3,
                "chars": 2100,
                "source_revision": "abc1234",
                "source": "ephemeral",
                "stale": False,
            }
        }
    )
    assert "- OKF blast radius: 3 concept(s), 2100 chars" in attached
    assert "`abc1234`" in attached

    stale = _render(
        {
            "okf_context": {
                "status": "stale_attached",
                "concepts": 1,
                "chars": 900,
                "source_revision": "old999",
                "source": "existing",
                "stale": True,
            }
        }
    )
    assert "STALE bundle, line locators approximate" in stale

    failed = _render({"okf_context": {"status": "binary_missing"}})
    assert "⚠️ OKF blast radius requested but not attached: `binary_missing`" in failed

    absent = _render({})
    assert "OKF blast radius" not in absent


def test_okf_status_callback_failure_is_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A raising status callback (e.g. stderr write failure in the CLI
    printer) must not abort the run — the callback sits outside
    build_okf_section's own fail-soft wrapper."""
    repo = _git_repo_with_diff(tmp_path)
    _patch_bundle(monkeypatch, _stub_bundle(tmp_path))

    def _explode(event: dict) -> None:
        raise OSError("stderr gone")

    prompt = _build(
        repo,
        okf_settings=OkfSettings(enabled=True),
        okf_status=_explode,
    )
    assert OKF_SECTION_HEADER in prompt
