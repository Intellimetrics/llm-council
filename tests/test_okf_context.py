"""Unit tests for llm_council.okf_context (pure + process functions)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from llm_council.okf_context import (
    DEFAULT_OKF_BINARY,
    DEFAULT_OKF_GENERATE_TIMEOUT_SECONDS,
    DEFAULT_OKF_MAX_EXCERPT_CHARS,
    OKF_EXCERPT_FLOOR_CHARS,
    OKF_SECTION_HEADER,
    Concept,
    OkfSettings,
    build_okf_section,
    find_bundle,
    generate_ephemeral_bundle,
    load_concepts,
    match_concepts,
    render_okf_section,
    resolve_okf_settings,
    touched_ranges_from_diff,
)


ENABLED = OkfSettings(enabled=True)


# ---------------------------------------------------------------------------
# touched_ranges_from_diff


def test_touched_ranges_edge_case_matrix():
    diff = "\n".join(
        [
            # Modified file with two hunks (one pure-deletion `+c,0` hunk).
            "diff --git a/pkg/mod.py b/pkg/mod.py",
            "--- a/pkg/mod.py",
            "+++ b/pkg/mod.py",
            "@@ -10,3 +12,4 @@ def f():",
            "@@ -40,2 +44,0 @@ def g():",
            # New file.
            "diff --git a/pkg/new.py b/pkg/new.py",
            "--- /dev/null",
            "+++ b/pkg/new.py",
            "@@ -0,0 +1,5 @@",
            # Deleted file: old path gets the whole-file sentinel.
            "diff --git a/pkg/gone.py b/pkg/gone.py",
            "--- a/pkg/gone.py",
            "+++ /dev/null",
            # Pure rename, no hunks: contributes nothing.
            "diff --git a/pkg/old_name.py b/pkg/new_name.py",
            "rename from pkg/old_name.py",
            "rename to pkg/new_name.py",
            # Rename with edits: ranges land on the new path.
            "diff --git a/pkg/before.py b/pkg/after.py",
            "--- a/pkg/before.py",
            "+++ b/pkg/after.py",
            "@@ -1,2 +3 @@",
            # Mode-only change: no hunks, nothing recorded.
            "diff --git a/scripts/run.sh b/scripts/run.sh",
            "old mode 100644",
            "new mode 100755",
            # Binary file.
            "diff --git a/img/logo.png b/img/logo.png",
            "Binary files a/img/logo.png and b/img/logo.png differ",
            # Quoted path with a space.
            'diff --git "a/docs/has space.md" "b/docs/has space.md"',
            '--- "a/docs/has space.md"',
            '+++ "b/docs/has space.md"',
            "@@ -1 +2,2 @@",
            r"\ No newline at end of file",
        ]
    )
    touched = touched_ranges_from_diff(diff)
    assert touched["pkg/mod.py"] == [(12, 15), (44, 44)]
    assert touched["pkg/new.py"] == [(1, 5)]
    assert touched["pkg/gone.py"] == [(1, 10**9)]
    assert "pkg/old_name.py" not in touched
    assert "pkg/new_name.py" not in touched
    assert touched["pkg/after.py"] == [(3, 3)]
    assert "scripts/run.sh" not in touched
    assert "img/logo.png" not in touched
    assert touched["docs/has space.md"] == [(2, 3)]


def test_touched_ranges_merges_staged_and_unstaged_duplicates():
    half = "\n".join(
        [
            "diff --git a/x.py b/x.py",
            "--- a/x.py",
            "+++ b/x.py",
            "@@ -1,2 +{start},2 @@",
        ]
    )
    raw = half.format(start=1) + "\n\n" + half.format(start=9)
    touched = touched_ranges_from_diff(raw)
    assert touched["x.py"] == [(1, 2), (9, 10)]


def test_touched_ranges_empty_and_prose_only():
    assert touched_ranges_from_diff("") == {}
    assert touched_ranges_from_diff("[git diff output truncated after 42 bytes]") == {}


# ---------------------------------------------------------------------------
# bundle fixtures


def _write_concept(
    bundle: Path,
    concept_id: str,
    *,
    resource: str,
    signature: str | None = None,
    calls: list[str] | None = None,
    called_by: list[str] | None = None,
    extra_frontmatter: str = "",
) -> None:
    path = bundle / (concept_id + ".md")
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_lines = []
    if calls or called_by:
        rel_lines.append("relationships:")
        if calls:
            rel_lines.append("  calls:")
            for target in calls:
                rel_lines.append(f"  - target: {target}")
                rel_lines.append("    confidence: exact")
        if called_by:
            rel_lines.append("  called_by:")
            for target in called_by:
                rel_lines.append(f"  - target: {target}")
    body = "\n# Signature\n\n" + (f"`{signature}`\n" if signature else "")
    path.write_text(
        "---\n"
        "type: Python Function\n"
        f"title: {concept_id.rsplit('/', 1)[-1]}\n"
        f"resource: {resource}\n"
        + ("\n".join(rel_lines) + "\n" if rel_lines else "")
        + extra_frontmatter
        + "---\n" + body,
        encoding="utf-8",
    )


def _mini_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "functions").mkdir(parents=True)
    (bundle / "index.md").write_text(
        "---\nokf_version: \"0.2\"\nsource_revision: \"cafe1234\"\n---\n# KB\n",
        encoding="utf-8",
    )
    _write_concept(
        bundle,
        "functions/pkg/mod/target_fn",
        resource="pkg/mod.py#L10-L20",
        signature="def target_fn(x: int) -> str",
        calls=["functions/pkg/util/helper"],
        called_by=["functions/pkg/api/handler", "functions/pkg/cli/main"],
    )
    _write_concept(
        bundle,
        "functions/pkg/util/helper",
        resource="pkg/util.py#L5-L9",
        signature="def helper() -> None",
    )
    _write_concept(
        bundle,
        "functions/pkg/api/handler",
        resource="pkg/api.py#L30-L60",
        signature="def handler(req) -> Response",
    )
    # No signature body, unmatched by diffs (different file).
    _write_concept(
        bundle,
        "functions/pkg/cli/main",
        resource="pkg/cli.py#L1-L99",
    )
    return bundle


# ---------------------------------------------------------------------------
# load_concepts


def test_load_concepts_parses_frontmatter_and_skips_malformed(tmp_path: Path):
    bundle = _mini_bundle(tmp_path)
    # Malformed: no frontmatter at all.
    (bundle / "functions" / "plain.md").write_text("just prose\n", encoding="utf-8")
    # Malformed: unparseable YAML.
    (bundle / "functions" / "bad.md").write_text(
        "---\n: [unbalanced\n---\nbody\n", encoding="utf-8"
    )
    # Missing usable resource (no line range).
    _write_concept(bundle, "functions/pkg/norange", resource="pkg/mod.py")
    # Numeric confidence below 0.5 is dropped; string confidence kept.
    _write_concept(
        bundle,
        "functions/pkg/lowconf",
        resource="pkg/low.py#L1-L2",
    )
    (bundle / "functions" / "pkg" / "lowconf.md").write_text(
        "---\n"
        "type: Python Function\n"
        "title: lowconf\n"
        "resource: pkg/low.py#L1-L2\n"
        "relationships:\n"
        "  calls:\n"
        "  - target: functions/pkg/dropme\n"
        "    confidence: 0.2\n"
        "  - target: functions/pkg/keepme\n"
        "    confidence: exact\n"
        "---\n",
        encoding="utf-8",
    )

    concepts = load_concepts(bundle)
    assert "functions/pkg/mod/target_fn" in concepts
    assert "functions/plain" not in concepts
    assert "functions/bad" not in concepts
    assert "functions/pkg/norange" not in concepts
    target = concepts["functions/pkg/mod/target_fn"]
    assert target.path == "pkg/mod.py"
    assert (target.start, target.end) == (10, 20)
    assert target.signature == "def target_fn(x: int) -> str"
    assert target.called_by == [
        "functions/pkg/api/handler",
        "functions/pkg/cli/main",
    ]
    assert concepts["functions/pkg/lowconf"].calls == ["functions/pkg/keepme"]
    # index.md files are never treated as concepts.
    assert all(not cid.endswith("/index") for cid in concepts)


# ---------------------------------------------------------------------------
# match_concepts


def test_match_concepts_range_intersection_and_sorting(tmp_path: Path):
    concepts = load_concepts(_mini_bundle(tmp_path))
    touched = {
        "pkg/mod.py": [(15, 16)],  # inside target_fn
        "pkg/api.py": [(1, 29)],  # just above handler — no match
        "pkg/util.py": [(9, 40)],  # overlaps helper's last line
    }
    matched = match_concepts(concepts, touched)
    assert [c.concept_id for c in matched] == [
        "functions/pkg/mod/target_fn",
        "functions/pkg/util/helper",
    ]
    # Deterministic: same result twice.
    assert [c.concept_id for c in match_concepts(concepts, touched)] == [
        c.concept_id for c in matched
    ]


def test_match_concepts_deleted_file_sentinel_matches_everything(tmp_path: Path):
    concepts = load_concepts(_mini_bundle(tmp_path))
    matched = match_concepts(concepts, {"pkg/cli.py": [(1, 10**9)]})
    assert [c.concept_id for c in matched] == ["functions/pkg/cli/main"]


# ---------------------------------------------------------------------------
# render_okf_section


def test_render_okf_section_deterministic_and_content(tmp_path: Path):
    concepts = load_concepts(_mini_bundle(tmp_path))
    matched = match_concepts(concepts, {"pkg/mod.py": [(10, 20)]})
    text1, count1 = render_okf_section(
        matched,
        concepts,
        budget=10_000,
        source_revision="cafe1234",
        source_kind="ephemeral working-tree generation",
        stale=False,
    )
    text2, count2 = render_okf_section(
        matched,
        concepts,
        budget=10_000,
        source_revision="cafe1234",
        source_kind="ephemeral working-tree generation",
        stale=False,
    )
    assert text1 == text2 and count1 == count2 == 1
    assert text1.startswith(OKF_SECTION_HEADER)
    assert "cafe1234" in text1
    assert "`def target_fn(x: int) -> str`" in text1
    assert "pkg/mod.py#L10-L20" in text1
    # Neighbor with a signature renders it; neighbor without renders id+locator.
    assert "`def handler(req) -> Response` — pkg/api.py#L30-L60" in text1
    assert "functions/pkg/cli/main — pkg/cli.py#L1-L99" in text1
    # Neighbors sorted by id: api/handler's line before cli/main's line.
    assert text1.index("def handler(req)") < text1.index("functions/pkg/cli/main")


def test_render_okf_section_budget_truncation(tmp_path: Path):
    concepts = load_concepts(_mini_bundle(tmp_path))
    matched = match_concepts(
        concepts,
        {"pkg/mod.py": [(10, 20)], "pkg/util.py": [(5, 9)], "pkg/api.py": [(30, 60)]},
    )
    assert len(matched) == 3
    full, _ = render_okf_section(
        matched, concepts, budget=100_000,
        source_revision=None, source_kind="ephemeral working-tree generation",
        stale=False,
    )
    tight_budget = len(full) - 10
    text, count = render_okf_section(
        matched, concepts, budget=tight_budget,
        source_revision=None, source_kind="ephemeral working-tree generation",
        stale=False,
    )
    assert text is not None and count < 3
    assert len(text) <= tight_budget
    assert f"[okf excerpt truncated: showing {count} of 3 touched concepts]" in text
    # Budget too small for even the header + one block -> None.
    none_text, none_count = render_okf_section(
        matched, concepts, budget=50,
        source_revision=None, source_kind="ephemeral working-tree generation",
        stale=False,
    )
    assert none_text is None and none_count == 0


def test_render_okf_section_stale_note():
    concept = Concept(
        concept_id="functions/x/f", path="x.py", start=1, end=2, signature="def f()"
    )
    text, _ = render_okf_section(
        [concept], {concept.concept_id: concept}, budget=5_000,
        source_revision="old123", source_kind="pre-existing bundle", stale=True,
    )
    assert text is not None
    assert "predates the current HEAD" in text


# ---------------------------------------------------------------------------
# resolve_okf_settings


def test_resolve_okf_settings_precedence_and_defaults():
    config = {
        "defaults": {"okf_context": True, "okf_max_excerpt_chars": 5_000},
        "modes": {"review": {"okf_context": False}, "quick": {}},
    }
    # Mode-explicit false overrides defaults true (None-aware, not `or`).
    assert resolve_okf_settings(config, "review", None).enabled is False
    # Mode silent -> defaults.
    assert resolve_okf_settings(config, "quick", None).enabled is True
    # Override wins over both.
    assert resolve_okf_settings(config, "review", True).enabled is True
    assert resolve_okf_settings(config, "quick", False).enabled is False
    settings = resolve_okf_settings(config, "quick", None)
    assert settings.max_excerpt_chars == 5_000
    assert settings.binary == DEFAULT_OKF_BINARY
    assert settings.generate_timeout_seconds == DEFAULT_OKF_GENERATE_TIMEOUT_SECONDS
    # Feature absent everywhere -> off, library defaults.
    bare = resolve_okf_settings({}, "quick", None)
    assert bare.enabled is False
    assert bare.max_excerpt_chars == DEFAULT_OKF_MAX_EXCERPT_CHARS


# ---------------------------------------------------------------------------
# generate_ephemeral_bundle / find_bundle


def test_generate_ephemeral_bundle_binary_missing_fail_soft(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    bundle_dir, status, detail = generate_ephemeral_bundle(
        tmp_path, out, binary="definitely-not-a-real-binary-xyz", timeout=5
    )
    assert bundle_dir is None
    assert status == "binary_missing"
    assert "definitely-not-a-real-binary-xyz" in (detail or "")


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stub")
def test_generate_ephemeral_bundle_stub_success_and_no_project_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "src.py").write_text("x = 1\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "okf-stub"
    # Writes index.md into the -o dir; asserts --no-cache was passed by
    # refusing to run without it.
    stub.write_text(
        "#!/bin/sh\n"
        'case "$*" in *--no-cache*) ;; *) echo "missing --no-cache" >&2; exit 3;; esac\n'
        'out=""\n'
        'prev=""\n'
        'for arg in "$@"; do if [ "$prev" = "-o" ]; then out="$arg"; fi; prev="$arg"; done\n'
        'mkdir -p "$out/functions"\n'
        'printf -- "---\\nsource_revision: \\"beef\\"\\n---\\n" > "$out/index.md"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    before = sorted(p.name for p in project.iterdir())

    out = tmp_path / "out"
    out.mkdir()
    bundle_dir, status, detail = generate_ephemeral_bundle(
        project, out, binary=str(stub), timeout=10
    )
    assert status is None and detail is None
    assert bundle_dir is not None and (bundle_dir / "index.md").is_file()
    # Nothing written into the project (no .okf-cache.json, no knowledge/).
    assert sorted(p.name for p in project.iterdir()) == before


def test_find_bundle_okf_toml_output_key_and_staleness(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    # No bundle anywhere.
    assert find_bundle(tmp_path) is None

    # okf.toml pointing at a custom output dir wins over ./knowledge.
    custom = tmp_path / "kb"
    custom.mkdir()
    (custom / "index.md").write_text(
        f'---\nsource_revision: "{head}"\n---\n', encoding="utf-8"
    )
    (tmp_path / "okf.toml").write_text('output = "kb"\n', encoding="utf-8")
    found = find_bundle(tmp_path)
    assert found is not None
    bundle_root, revision = found
    assert bundle_root == custom.resolve()
    assert revision == head

    # Fallback location without okf.toml.
    (tmp_path / "okf.toml").unlink()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "index.md").write_text(
        '---\nsource_revision: "0ld"\n---\n', encoding="utf-8"
    )
    found = find_bundle(tmp_path)
    assert found is not None
    assert found[0] == knowledge
    assert found[1] == "0ld"


# ---------------------------------------------------------------------------
# build_okf_section status matrix


_DIFF = "\n".join(
    [
        "diff --git a/pkg/mod.py b/pkg/mod.py",
        "--- a/pkg/mod.py",
        "+++ b/pkg/mod.py",
        "@@ -10,2 +12,3 @@",
    ]
)


def test_build_okf_section_no_diff(tmp_path: Path):
    text, status = build_okf_section(tmp_path, "  \n", ENABLED, headroom=50_000)
    assert text is None and status["status"] == "no_diff"


def test_build_okf_section_excerpt_over_budget(tmp_path: Path):
    text, status = build_okf_section(
        tmp_path, _DIFF, ENABLED, headroom=OKF_EXCERPT_FLOOR_CHARS - 1
    )
    assert text is None and status["status"] == "excerpt_over_budget"


def test_build_okf_section_binary_missing_without_fallback(tmp_path: Path):
    settings = OkfSettings(enabled=True, binary="definitely-not-a-real-binary-xyz")
    text, status = build_okf_section(tmp_path, _DIFF, settings, headroom=50_000)
    assert text is None
    assert status["status"] == "binary_missing"


def test_build_okf_section_stale_fallback_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import llm_council.okf_context as okf

    bundle = _mini_bundle(tmp_path)
    monkeypatch.setattr(
        okf, "generate_ephemeral_bundle",
        lambda cwd, out_dir, *, binary, timeout: (None, "generate_failed", "boom"),
    )
    monkeypatch.setattr(okf, "find_bundle", lambda cwd: (bundle, "cafe1234"))
    monkeypatch.setattr(okf, "_head_revision", lambda cwd: "feed5678")
    text, status = build_okf_section(tmp_path, _DIFF, ENABLED, headroom=50_000)
    assert text is not None
    assert status["status"] == "stale_attached"
    assert status["stale"] is True
    assert status["source"] == "existing"
    assert "predates the current HEAD" in text


def test_build_okf_section_no_matched_concepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import llm_council.okf_context as okf

    bundle = _mini_bundle(tmp_path)
    monkeypatch.setattr(
        okf, "generate_ephemeral_bundle",
        lambda cwd, out_dir, *, binary, timeout: (bundle, None, None),
    )
    monkeypatch.setattr(okf, "_head_revision", lambda cwd: "cafe1234")
    unrelated = _DIFF.replace("pkg/mod.py", "elsewhere/other.py")
    text, status = build_okf_section(tmp_path, unrelated, ENABLED, headroom=50_000)
    assert text is None and status["status"] == "no_matched_concepts"


def test_build_okf_section_attached_ephemeral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import llm_council.okf_context as okf

    bundle = _mini_bundle(tmp_path)
    monkeypatch.setattr(
        okf, "generate_ephemeral_bundle",
        lambda cwd, out_dir, *, binary, timeout: (bundle, None, None),
    )
    monkeypatch.setattr(okf, "_head_revision", lambda cwd: "cafe1234")
    text, status = build_okf_section(tmp_path, _DIFF, ENABLED, headroom=50_000)
    assert text is not None and text.startswith(OKF_SECTION_HEADER)
    assert status["status"] == "attached"
    assert status["source"] == "ephemeral"
    assert status["stale"] is False
    assert status["concepts"] == 1
    assert status["chars"] == len(text)


def test_build_okf_section_internal_error_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import llm_council.okf_context as okf

    def _boom(raw_diff):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(okf, "touched_ranges_from_diff", _boom)
    text, status = build_okf_section(tmp_path, _DIFF, ENABLED, headroom=50_000)
    assert text is None
    assert status["status"] == "internal_error"
    assert "kaboom" in status["detail"]


def test_resolve_okf_settings_clamps_generate_timeout():
    from llm_council.okf_context import OKF_GENERATE_TIMEOUT_CEILING_SECONDS

    config = {"defaults": {"okf_context": True, "okf_generate_timeout_seconds": 600}}
    settings = resolve_okf_settings(config, "quick", None)
    assert settings.generate_timeout_seconds == OKF_GENERATE_TIMEOUT_CEILING_SECONDS


def test_find_bundle_rejects_okf_toml_output_outside_cwd(tmp_path: Path):
    # A hostile okf.toml in a reviewed repo must not aim the concept walk
    # outside cwd.
    outside = tmp_path / "outside-bundle"
    outside.mkdir()
    (outside / "index.md").write_text("---\n---\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "okf.toml").write_text('output = "../outside-bundle"\n', encoding="utf-8")
    assert find_bundle(project) is None

    (project / "okf.toml").write_text(
        f'output = "{outside}"\n', encoding="utf-8"
    )
    assert find_bundle(project) is None


def test_build_okf_section_stale_when_fallback_revision_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import llm_council.okf_context as okf

    bundle = _mini_bundle(tmp_path)
    monkeypatch.setattr(
        okf, "generate_ephemeral_bundle",
        lambda cwd, out_dir, *, binary, timeout: (None, "generate_failed", "boom"),
    )
    monkeypatch.setattr(okf, "find_bundle", lambda cwd: (bundle, None))
    monkeypatch.setattr(okf, "_head_revision", lambda cwd: None)
    text, status = build_okf_section(tmp_path, _DIFF, ENABLED, headroom=50_000)
    assert text is not None
    # Unknown bundle vintage must not read as fresh.
    assert status["status"] == "stale_attached"
    assert status["stale"] is True


def test_build_okf_section_no_tempdir_leak_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ctrl-C during generation must propagate (not be swallowed by the
    fail-soft Exception catch) AND leave no llm-council-okf-* tempdir
    behind — the TemporaryDirectory context manager cleans on
    BaseException too."""
    import tempfile as _tempfile

    import llm_council.okf_context as okf

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(okf, "run_process", _interrupt)
    monkeypatch.setattr(okf.shutil, "which", lambda binary: "/usr/bin/true")

    def _leftovers() -> set[str]:
        return {
            p.name
            for p in Path(_tempfile.gettempdir()).glob("llm-council-okf-*")
        }

    before = _leftovers()
    with pytest.raises(KeyboardInterrupt):
        build_okf_section(tmp_path, _DIFF, ENABLED, headroom=50_000)
    assert _leftovers() == before
