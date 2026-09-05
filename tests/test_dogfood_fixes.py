"""Regression cases reproduced during the live September dogfood run."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

from llm_council import adapters, citations, mcp_server, okf_context, query
from llm_council.blocking import run_process, run_blocking
from llm_council.stats import aggregate


def _alive(pid):
    if os.name == "posix" and Path(f"/proc/{pid}/stat").exists():
        # Orphaned grandchildren can await init's reap after being killed.
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_successful_generator_stops_background_child(tmp_path):
    marker = tmp_path / "pids.json"
    code = (
        "import os,json,subprocess,sys; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid]))"
    )
    try:
        result = run_process([sys.executable, "-c", code], timeout=5,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert result.returncode == 0
        parent, child = json.loads(marker.read_text())
        for _ in range(100):
            if not _alive(child):
                break
            time.sleep(.01)
        assert not _alive(parent)
        assert not _alive(child)
    finally:
        if marker.exists():
            for pid in json.loads(marker.read_text()):
                if _alive(pid):
                    os.kill(pid, 9)


@pytest.mark.parametrize("escape", ["bundle", "index", "direct_index"])
def test_generated_bundle_must_stay_in_output_root(tmp_path, monkeypatch, escape):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "index.md").write_text("---\nsource_revision: fixture\n---\n")
    if escape == "bundle":
        (output / "knowledge").symlink_to(outside, target_is_directory=True)
    else:
        directory = output if escape == "direct_index" else output / "knowledge"
        directory.mkdir(exist_ok=True)
        (directory / "index.md").symlink_to(outside / "index.md")
    monkeypatch.setattr(okf_context.shutil, "which", lambda command: command)
    monkeypatch.setattr(okf_context, "run_process", lambda *a, **k: subprocess.CompletedProcess([], 0))
    bundle, failure, _ = okf_context.generate_ephemeral_bundle(tmp_path, output, binary="stub", timeout=5)
    assert bundle is None
    assert failure == "generate_failed"


def test_verified_evidence_has_its_own_stat_bucket():
    row = {"name": "peer", "ok": True, "output": "RECOMMENDATION: yes",
           "evidence": [{"tag": "verified", "verified": True, "text": "code.py:1"}]}
    result = aggregate([{"mtime": 1, "data": {"results": [row]}}])
    tags = result["participants"][0]["evidence_tag_distribution"]
    assert tags["verified"] == 1
    assert tags["untagged"] == 0


def _peer(code):
    return {"type": "cli", "family": "custom", "command": sys.executable,
            "args": ["-c", code], "timeout": 60, "retries": 0, "stdin_prompt": True}


@pytest.mark.parametrize("concurrency", [1, 2])
async def test_request_deadline_preserves_completed_votes(tmp_path, monkeypatch, concurrency):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    cfg = {"defaults": {"max_concurrency": concurrency}, "participants": {
        "fast": _peer("print('RECOMMENDATION: yes')"),
        "slow": _peer("import time; time.sleep(30)"),
        "queued": _peer("from pathlib import Path; Path('queued-started').touch(); print('RECOMMENDATION: yes')"),
    }}
    (tmp_path / ".llm-council.yaml").write_text(yaml.safe_dump(cfg))
    started = time.monotonic()
    result = await mcp_server.run_council({
        "working_directory": str(tmp_path), "current": "codex", "mode": "quick",
        "participants": ["fast", "slow", "queued"], "min_quorum": 1,
        "request_timeout_seconds": 2, "question": "Check the fixture", "deliberate": False,
    })
    assert time.monotonic() - started < 3
    assert result["metadata"]["partial"] is True
    assert result["recommendation"] == "yes"
    peers = {p["name"]: p for p in result["results"]}
    assert peers["fast"]["ok"]
    assert peers["slow"]["error_kind"] == "timeout"
    assert not peers["slow"]["terse_retry_attempted"]
    if concurrency == 1:
        assert not (tmp_path / "queued-started").exists()
        assert peers["queued"]["error_kind"] == "timeout"
    saved = json.loads(Path(result["json"]).read_text())
    assert saved["metadata"]["partial"] is True
    assert saved["results"][0]["ok"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
async def test_cancel_native_peer_stops_its_tool_children(tmp_path):
    marker = tmp_path / "pids.json"
    code = (
        "import os,json,subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(30)"
    )
    task = asyncio.create_task(adapters.run_participant("slow", _peer(code), "q", tmp_path))
    try:
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert all(not _alive(pid) for pid in json.loads(marker.read_text()))
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_citation_reads_are_bounded_and_do_not_block_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(citations, "MAX_CITATION_FILE_CHARS", 8)
    monkeypatch.setattr(citations, "MAX_CITATION_RUN_CHARS", 10)
    (tmp_path / "large").write_text("a\n" * 10000)
    (tmp_path / "other").write_text("b\n" * 10000)
    verifier = citations.CitationVerifier(tmp_path)
    original = Path.open

    class SlowReader:
        def __init__(self, handle): self.handle = handle
        def __enter__(self): return self
        def __exit__(self, *args): self.handle.close()
        def read(self, count):
            time.sleep(.02)
            assert count <= 8
            return self.handle.read(count)

    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: SlowReader(original(self, *a, **kw)))
    task = asyncio.create_task(run_blocking(verifier.verify, citations.VerifiedRef("large", 1, 4)))
    await asyncio.sleep(.005)
    assert not task.done()
    assert await task
    assert not verifier.verify(citations.VerifiedRef("large", 5, 5))
    assert verifier.verify(citations.VerifiedRef("other", 1, 1))
    assert not verifier.verify(citations.VerifiedRef("other", 2, 2))


def test_search_limits_reads_and_keeps_newest_window(tmp_path, monkeypatch):
    monkeypatch.setattr(query, "_MAX_INDEX_RECORDS", 2)
    monkeypatch.setattr(query, "_MAX_TRANSCRIPT_BYTES", 1024)
    for n in range(5):
        path = tmp_path / f"run{n}.json"
        path.write_text(json.dumps({"question": "auth migration", "results": []}))
        os.utime(path, (100 + n, 100 + n))
    (tmp_path / "huge.json").write_bytes(b"x" * 2048)
    reads = []
    original = query._read_search_record
    def read(path):
        reads.append(path.name)
        return original(path)
    monkeypatch.setattr(query, "_read_search_record", read)
    scope = {}
    matches = query.search_similar("auth", runs_dir=tmp_path, diagnostics=scope)
    query.search_similar("migration", runs_dir=tmp_path)
    assert set(reads) == {"run3.json", "run4.json"}
    assert len(reads) == 2
    assert len(matches) == 2
    assert scope["limited"]
    assert scope["skipped_oversize_files"] == 1


def test_search_bounds_total_cold_read_work(tmp_path, monkeypatch):
    body = json.dumps({"question": "auth", "results": []})
    for n in range(4):
        (tmp_path / f"run{n}.json").write_text(body)
    monkeypatch.setattr(query, "_MAX_QUERY_READ_BYTES", len(body))
    scope = {}
    assert len(query.search_similar("auth", runs_dir=tmp_path, diagnostics=scope)) == 1
    assert scope["deferred_files"] == 3
    assert scope["limited"]
