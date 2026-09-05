"""Behavioral regressions from the September repository review; no live LLMs."""

import asyncio
import json
import os
from pathlib import Path
import sys
import threading
import tracemalloc

import pytest

from llm_council import adapters, okf_context, query
from llm_council.blocking import check_cancelled, run_blocking
from llm_council.citations import CitationVerifier, VerifiedRef
from llm_council.config import resolve_config_data
from llm_council.findings import build_matrix_from_results


DIFF = "diff --git a/code.py b/code.py\n--- a/code.py\n+++ b/code.py\n@@ -1 +1 @@\n-old\n+new\n"
CONCEPT = "---\nresource: code.py#L1-L2\n---\n# Signature\n`external marker`\n"


def _bundle(root):
    (root / "functions").mkdir(parents=True)
    (root / "index.md").write_text("---\nsource_revision: old\n---\n")
    (root / "functions" / "code.md").write_text(CONCEPT)
    return root


@pytest.mark.parametrize("link_kind", ["bundle", "index", "directory", "concept"])
def test_okf_rejects_external_symlinks(tmp_path, link_kind):
    project = tmp_path / "project"
    project.mkdir()
    outside = _bundle(tmp_path / "outside")
    bundle = project / "knowledge"
    if link_kind == "bundle":
        bundle.symlink_to(outside, target_is_directory=True)
    else:
        _bundle(bundle)
        target = {"index": bundle / "index.md", "directory": bundle / "functions",
                  "concept": bundle / "functions" / "code.md"}[link_kind]
        if target.is_dir():
            (target / "code.md").unlink()
            target.rmdir()
        else:
            target.unlink()
        target.symlink_to(outside / target.relative_to(bundle), target_is_directory=link_kind == "directory")
    section, _ = okf_context.build_okf_section(
        project, DIFF, okf_context.OkfSettings(enabled=True, binary="no-such-okf-review"), headroom=12000
    )
    assert section is None


def test_okf_reads_prefix_before_allocating_large_file(tmp_path):
    bundle = _bundle(tmp_path / "knowledge")
    with (bundle / "functions" / "code.md").open("a") as handle:
        handle.write("x" * (8 * 1024 * 1024))
    tracemalloc.start()
    try:
        concepts = okf_context.load_concepts(bundle)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(concepts) == 1
    assert peak < 2 * 1024 * 1024


def test_okf_limits_directory_walk_before_materialization(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "knowledge")
    for n in range(20):
        (bundle / "functions" / f"empty{n}").mkdir()
    monkeypatch.setattr(okf_context, "MAX_BUNDLE_ENTRIES_SCANNED", 5)
    with pytest.raises(ValueError, match="scan limit"):
        okf_context.load_concepts(bundle)


def test_okf_fallback_at_head_is_still_stale_for_working_tree(tmp_path, monkeypatch):
    _bundle(tmp_path / "knowledge")
    monkeypatch.setattr(okf_context, "_head_revision", lambda cwd: "old")
    section, status = okf_context.build_okf_section(
        tmp_path, DIFF, okf_context.OkfSettings(enabled=True, binary="no-such-okf-review"),
        headroom=12000,
    )
    assert section is not None
    assert status["source_revision"] == "old"
    assert status["status"] == "stale_attached"


async def test_repeated_cancellation_waits_for_worker_cleanup():
    started, cleaning, release, finished = (threading.Event() for _ in range(4))

    def work():
        started.set()
        try:
            while not release.wait(.01):
                check_cancelled()
        finally:
            cleaning.set()
            release.wait(3)
            finished.set()

    task = asyncio.create_task(run_blocking(work))
    try:
        async with asyncio.timeout(3):
            while not started.is_set():
                await asyncio.sleep(.01)
            task.cancel()
            while not cleaning.is_set():
                await asyncio.sleep(.01)
            task.cancel()
            await asyncio.sleep(.01)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
    finally:
        release.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.parametrize("mode", ["quick", "review-with-tools"])
async def test_cli_rereads_source_after_edit(tmp_path, mode):
    source = tmp_path / "code.txt"
    source.write_text("yes")
    cfg = {"type": "cli", "family": "custom", "command": sys.executable,
           "args": ["-c", "from pathlib import Path; print('RECOMMENDATION: '+Path('code.txt').read_text())"],
           "timeout": 5, "retries": 0}
    ctx = adapters.CacheContext(cwd=tmp_path)
    first = await adapters.run_cli_participant("reader", cfg, "Review code.txt", tmp_path, cache_ctx=ctx, mode=mode)
    source.write_text("no")
    second = await adapters.run_cli_participant("reader", cfg, "Review code.txt", tmp_path, cache_ctx=ctx, mode=mode)
    assert first.output.strip() == "RECOMMENDATION: yes"
    assert second.output.strip() == "RECOMMENDATION: no"
    assert not second.from_cache
    assert not (tmp_path / ".llm-council" / "cache").exists()


@pytest.mark.skipif(os.name != "posix", reason="executable Python fixture")
async def test_okf_cancellation_keeps_loop_responsive_and_reaps_process(tmp_path, monkeypatch):
    marker = tmp_path / "started.json"
    binary = tmp_path / "okf-stub"
    binary.write_text(
        f"#!{sys.executable}\nimport json, os, sys, time\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),sys.argv[sys.argv.index('-o')+1]]))\n"
        "time.sleep(30)\n"
    )
    binary.chmod(0o755)
    monkeypatch.setattr(okf_context, "_head_revision", lambda cwd: None)
    task = asyncio.create_task(run_blocking(
        okf_context.build_okf_section, tmp_path, DIFF,
        okf_context.OkfSettings(enabled=True, binary=str(binary)), headroom=12000,
    ))
    try:
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(.01)
        # Reaching here while the generator sleeps proves the loop is free.
        pid, output_dir = json.loads(marker.read_text())
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert not Path(output_dir).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _write_run(path, votes):
    path.write_text(json.dumps({"question": "review auth migration", "results": [
        {"name": str(i), "ok": True, "output": "RECOMMENDATION: " + label}
        for i, label in enumerate(votes)
    ]}))


@pytest.mark.parametrize("votes,expected", [
    (["yes", "no", "no"], "no"), (["yes", "no"], None),
    (["no", "tradeoff"], "leaning-no"), (["tradeoff", "yes"], "leaning-yes"),
])
def test_search_reports_council_decision(tmp_path, votes, expected):
    _write_run(tmp_path / "20260905_100000_run.json", votes)
    assert query.search_similar("auth migration", runs_dir=tmp_path)[0].recommendation_label == expected


def test_search_index_reads_only_changed_files_and_observes_pruning(tmp_path, monkeypatch):
    path = tmp_path / "20260905_100000_run.json"
    _write_run(path, ["yes"])
    reads = []
    original = Path.open
    def read(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read)
    query.search_similar("auth", runs_dir=tmp_path)
    query.search_similar("migration", runs_dir=tmp_path)
    assert reads == [path]
    _write_run(path, ["no"])
    assert query.search_similar("auth", runs_dir=tmp_path)[0].recommendation_label == "no"
    # The write itself also opens the file.
    assert reads == [path, path, path]
    path.unlink()
    assert query.search_similar("auth", runs_dir=tmp_path) == []


def test_citation_line_counts_shared_but_invalidated_on_edit(tmp_path, monkeypatch):
    source = tmp_path / "code.py"
    source.write_text("one\ntwo\n")
    verifier = CitationVerifier(tmp_path)
    reads = []
    original = Path.open
    def open_file(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    assert verifier.verify(VerifiedRef("code.py", 1, 1))
    assert verifier.verify(VerifiedRef("code.py", 2, 2))
    assert reads == [source]
    source.write_text("one\n")
    assert not verifier.verify(VerifiedRef("code.py", 2, 2))


def test_findings_need_positive_verification_even_without_evidence_section(tmp_path):
    peers = [adapters.ParticipantResult(
        name=name, ok=True, error="", elapsed_seconds=0,
        output="RECOMMENDATION: no\nFINDINGS:\n- id: F1\n  severity: blocker\n"
               "  claim: missing check\n  evidence: [VERIFIED:ghost.py:1-2]\n",
    ) for name in ("claude", "codex")]
    matrix = build_matrix_from_results(peers, verifier=CitationVerifier(tmp_path))
    assert not matrix.clusters
    assert len(matrix.single_peer_concerns) == 2
    assert all(p.evidence_verification_failures == ["ghost.py:1-2"] for p in peers)
    # A VERIFIED token without either a filesystem check or a receipt is insufficient.
    assert not build_matrix_from_results(peers).clusters


@pytest.mark.parametrize("served", ["claude-fable-5", "claude-fable-5-10", "claude-fable-5-1-mini"])
def test_pin_rejects_related_but_different_models(served):
    assert not adapters._model_pin_satisfied("claude-fable-5-1", served)


@pytest.mark.parametrize("served", ["claude-fable-5-1", "claude-fable-5-1-20260901"])
def test_pin_accepts_exact_model_and_dated_snapshot(served):
    assert adapters._model_pin_satisfied("claude-fable-5-1", served)


def test_migrate_shipped_fallbacks_preserves_custom_chains_and_pins():
    config = resolve_config_data({"participants": {"codex": {
        "model": "gpt-5.4", "fallback_chain": ["gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini"],
    }}})
    assert config["participants"]["codex"]["model"] == "gpt-5.4"
    assert config["participants"]["codex"]["fallback_chain"] == ["gpt-5.6-terra", "gpt-5.6-luna"]
    config = resolve_config_data({"participants": {"codex": {"fallback_chain": ["my-model"]}}})
    assert config["participants"]["codex"]["fallback_chain"] == ["my-model"]


def test_cached_receipts_do_not_inflate_spend_or_retry_recoveries():
    from llm_council.stats import aggregate
    original = {"name": "hosted", "ok": True, "output": "RECOMMENDATION: yes",
                "elapsed_seconds": 10, "total_tokens": 1000, "cost_usd": .1,
                "terse_retry_attempted": True, "recovered_after_timeout": True}
    records = [{"mtime": 1, "data": {"results": [original]}},
               {"mtime": 2, "data": {"results": [{**original, "from_cache": True}]}}]
    peer = aggregate(records)["participants"][0]
    assert peer["runs"] == 2
    assert peer["cache_hits"] == 1
    assert peer["cost_total"] == .1
    assert peer["tokens_total"] == 1000
    assert peer["timeout_recoveries"] == 1
    assert peer["terse_retry_attempts"] == 1
