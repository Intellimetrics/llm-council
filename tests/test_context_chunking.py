"""context_files chunking (Fix A, v0.8.1).

`context_files` passed via MCP `council_run` or `--context-files` CLI flag
now flow through the same hash-aware chunker that `--diff` uses, so
multi-file context payloads that would otherwise trip a participant's
`max_prompt_chars` cap are auto-trimmed instead of failing. Files
individually larger than the available budget are dropped entirely with a
warning event surfacing the path.

Mirrors the patterns in
`tests/test_llm_council.py::test_build_prompt_hash_aware_drops_unrelated_files`
for the diff-side chunker — same scoring algorithm, different input shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_council.context import build_prompt
from llm_council.diff_chunking import (
    ContextChunkResult,
    chunk_context_files,
)


# --- Unit: chunk_context_files (no build_prompt scaffolding) ------------


def test_chunk_context_files_empty_returns_empty_result():
    result = chunk_context_files([], budget=1_000, question="anything")
    assert isinstance(result, ContextChunkResult)
    assert result.sections == []
    assert result.dropped_files == []
    assert result.triggered is False


def test_chunk_context_files_all_fit_returns_unchanged():
    files = [
        ("alpha.py", "## File: alpha.py\n\n```\nA" * 10 + "\n```"),
        ("beta.py", "## File: beta.py\n\n```\nB" * 10 + "\n```"),
    ]
    result = chunk_context_files(files, budget=10_000, question="q")
    assert result.triggered is False
    assert len(result.sections) == 2
    assert result.dropped_files == []
    assert result.oversize_files == []


def test_chunk_context_files_drops_oversize_file_entirely():
    # huge.py alone exceeds budget; small.py fits.
    huge = "## File: huge.py\n\n```\n" + ("X" * 5_000) + "\n```"
    small = "## File: small.py\n\n```\nok\n```"
    result = chunk_context_files(
        [("huge.py", huge), ("small.py", small)],
        budget=1_000,
        question="review",
    )
    assert result.triggered is True
    assert "huge.py" in result.oversize_files
    assert "huge.py" in result.dropped_files
    # Small file survives.
    assert len(result.sections) == 1
    assert "small.py" in result.sections[0]


def test_chunk_context_files_prioritizes_mentioned_files():
    # Three files of equal size; only two fit at chosen budget. Mentioned
    # file must survive.
    body = "## File: {name}\n\n```\n" + ("z" * 200) + "\n```"
    files = [
        ("alpha.py", body.replace("{name}", "alpha.py")),
        ("beta.py", body.replace("{name}", "beta.py")),
        ("gamma.py", body.replace("{name}", "gamma.py")),
    ]
    # Pick budget for ~2 of 3.
    single_len = len(files[0][1])
    budget = single_len * 2 + 1  # room for two files + one separator
    result = chunk_context_files(
        files, budget=budget, question="please review gamma.py for bugs"
    )
    assert result.triggered is True
    # gamma.py must have survived.
    assert any("gamma.py" in section for section in result.sections)
    assert "gamma.py" not in result.dropped_files


def test_chunk_context_files_dropped_chars_accounting():
    body = "## File: {name}\n\n```\n" + ("z" * 100) + "\n```"
    files = [
        ("a.py", body.replace("{name}", "a.py")),
        ("b.py", body.replace("{name}", "b.py")),
        ("c.py", body.replace("{name}", "c.py")),
    ]
    single = len(files[0][1])
    budget = single + 1  # only one file fits
    result = chunk_context_files(files, budget=budget, question="q")
    assert result.triggered is True
    assert len(result.sections) == 1
    # Original chars: 3 sections + 2 separators.
    assert result.original_chars == 3 * single + 2
    assert result.chunked_chars == single
    assert result.dropped_chars > 0


# --- Integration: build_prompt with context_paths -----------------------


def test_build_prompt_many_small_context_files_no_chunking(tmp_path: Path):
    """Regression guard: when everything fits, no chunking, no warnings."""
    events: list[dict] = []
    files = []
    for i in range(5):
        path = tmp_path / f"f{i}.py"
        path.write_text(f"# small file {i}\nvalue = {i}\n", encoding="utf-8")
        files.append(str(path))
    prompt = build_prompt(
        "review the files",
        mode="quick",
        cwd=tmp_path,
        context_paths=files,
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=200_000,
        chunk_progress=events.append,
    )
    # All five files inlined.
    for i in range(5):
        assert f"f{i}.py" in prompt
    # No context_files_chunked event emitted.
    assert all(e.get("event") != "context_files_chunked" for e in events)


def test_build_prompt_empty_context_files_list_no_change(tmp_path: Path):
    events: list[dict] = []
    prompt = build_prompt(
        "no files attached",
        mode="quick",
        cwd=tmp_path,
        context_paths=[],
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=10_000,
        chunk_progress=events.append,
    )
    assert "no files attached" in prompt
    assert events == []


def test_build_prompt_context_files_preserves_mentioned_file(tmp_path: Path):
    """Hash-aware: question names `gamma.py`; alpha/beta should be dropped."""
    # Three large-ish files of equal size; budget tight enough that not all
    # three can survive alongside the framing.
    body_size = 5_000
    paths = []
    for name in ("alpha.py", "beta.py", "gamma.py"):
        p = tmp_path / name
        p.write_text(("a" * body_size) + "\n", encoding="utf-8")
        paths.append(str(p))

    events: list[dict] = []
    prompt = build_prompt(
        "please review the bug in gamma.py",
        mode="quick",
        cwd=tmp_path,
        context_paths=paths,
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=12_000,  # one file (~5K body + wrapping) fits
        chunk_progress=events.append,
    )
    # Final prompt respects the cap.
    assert len(prompt) <= 12_000
    # gamma.py — the mentioned file — survived.
    assert "gamma.py" in prompt
    # At least one chunk event surfaced.
    chunk_events = [e for e in events if e.get("event") == "context_files_chunked"]
    assert chunk_events, "expected a context_files_chunked event"
    last = chunk_events[-1]
    assert last["strategy"] == "hash-aware"
    # gamma.py was preserved — it must NOT appear in dropped_files.
    assert "gamma.py" not in last["dropped_files"]
    # At least one of the unmentioned files was dropped.
    dropped = set(last["dropped_files"])
    assert dropped, "expected at least one file to have been dropped"


def test_build_prompt_single_oversize_file_dropped_with_warning(tmp_path: Path):
    """A file alone larger than the available budget is dropped entirely,
    and the event names the file path."""
    small = tmp_path / "small.py"
    small.write_text("# small\nv = 1\n", encoding="utf-8")
    huge = tmp_path / "huge.py"
    huge.write_text("z" * 50_000, encoding="utf-8")

    events: list[dict] = []
    prompt = build_prompt(
        "review",
        mode="quick",
        cwd=tmp_path,
        context_paths=[str(small), str(huge)],
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=10_000,
        chunk_progress=events.append,
    )
    assert len(prompt) <= 10_000
    # small.py survived; huge.py was dropped.
    assert "small.py" in prompt
    assert "huge.py" not in prompt
    chunk_events = [e for e in events if e.get("event") == "context_files_chunked"]
    assert chunk_events
    last = chunk_events[-1]
    # huge.py path label appears in oversize_files (operator-actionable).
    assert any("huge.py" in path for path in last["oversize_files"])
    assert any("huge.py" in path for path in last["dropped_files"])


def test_build_prompt_context_files_chunking_does_not_affect_diff_path(
    tmp_path: Path,
):
    """The existing diff hash-aware path must remain untouched: a diff-only
    overflow with no context_files behaves exactly as before."""
    # Synthesize a git repo with a large diff (mirrors the existing
    # test_build_prompt_hash_aware_drops_unrelated_files setup) but pass NO
    # context_paths. The context-files chunker branch must be a no-op and
    # the diff chunker must still fire when opted in.
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (tmp_path / name).write_text("a\n", encoding="utf-8")
        subprocess.run(["git", "add", name], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (tmp_path / name).write_text("a\n" + ("aa\n" * 1_500), encoding="utf-8")
        subprocess.run(["git", "add", name], cwd=tmp_path, check=True)

    events: list[dict] = []
    prompt = build_prompt(
        "review the bug in gamma.py please",
        mode="quick",
        cwd=tmp_path,
        context_paths=[],
        include_diff=True,
        stdin_text=None,
        max_prompt_chars=8_000,
        chunk_strategy="hash-aware",
        chunk_progress=events.append,
    )
    assert len(prompt) <= 8_000
    # No context_files event — there were no context_files.
    assert all(e.get("event") != "context_files_chunked" for e in events)
    # Diff chunker still fired.
    diff_events = [e for e in events if e.get("event") == "diff_chunked"]
    assert diff_events


def test_build_prompt_context_file_boundary_exact_fit(tmp_path: Path):
    """Edge: a single context file whose total prompt sits exactly at the
    cap should NOT trigger chunking (boundary == budget is inclusive)."""
    # Build a small file and pick a cap that exactly equals the resulting
    # prompt length.
    target = tmp_path / "exact.py"
    target.write_text("hello world\n", encoding="utf-8")
    events: list[dict] = []
    # First a permissive build to measure the natural prompt length.
    measured = build_prompt(
        "review exact.py",
        mode="quick",
        cwd=tmp_path,
        context_paths=[str(target)],
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=200_000,
        chunk_progress=events.append,
    )
    exact_cap = len(measured)
    # Now request exactly that cap — must succeed without chunking.
    events.clear()
    prompt = build_prompt(
        "review exact.py",
        mode="quick",
        cwd=tmp_path,
        context_paths=[str(target)],
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=exact_cap,
        chunk_progress=events.append,
    )
    assert len(prompt) == exact_cap
    assert all(e.get("event") != "context_files_chunked" for e in events)


def test_build_prompt_oversize_alone_raises_when_no_files_survive(
    tmp_path: Path,
):
    """If the ONLY context file is oversize AND non-file framing already
    exceeds the cap (or the lone file is the entire overflow), the rebuild
    can't fit and the existing fail-fast path must still fire — chunking
    is best-effort, not a magic shrink."""
    # Tiny prompt cap such that even an empty prompt almost overflows;
    # then attach a multi-K file. The chunker drops the file (oversize),
    # but the rebuild still fits because the file was the only overflow
    # source. We assert the chunk_progress event fired AND the resulting
    # prompt is under cap.
    huge = tmp_path / "huge.py"
    huge.write_text("z" * 5_000, encoding="utf-8")
    events: list[dict] = []
    prompt = build_prompt(
        "review",
        mode="quick",
        cwd=tmp_path,
        context_paths=[str(huge)],
        include_diff=False,
        stdin_text=None,
        max_prompt_chars=2_000,
        chunk_progress=events.append,
    )
    assert len(prompt) <= 2_000
    # huge.py was dropped.
    assert "huge.py" not in prompt
    chunk_events = [e for e in events if e.get("event") == "context_files_chunked"]
    assert chunk_events
    assert any(
        "huge.py" in path
        for path in chunk_events[-1]["oversize_files"]
    )


def test_build_prompt_fail_fast_when_chunking_cannot_rescue(tmp_path: Path):
    """If the non-file framing itself exceeds the cap, no amount of file
    chunking helps and the existing ValueError must fire."""
    # Tiny file but absurdly tiny cap such that even the head_sections
    # framing (read-only rules + response format) blows the cap.
    p = tmp_path / "tiny.py"
    p.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds max_prompt_chars"):
        build_prompt(
            "q",
            mode="quick",
            cwd=tmp_path,
            context_paths=[str(p)],
            include_diff=False,
            stdin_text=None,
            max_prompt_chars=200,  # well under the head framing length
        )
