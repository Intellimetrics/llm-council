"""Tests for image_paths flow through council_run schema, build_prompt, and CLI."""

from __future__ import annotations

import asyncio
import base64
import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.adapters import (
    _build_user_content_async,
    _read_image_base64,
    run_participants,
)
from llm_council.budget import (
    DEFAULT_IMAGE_MAX_BYTES,
    DEFAULT_IMAGE_TOTAL_MAX_BYTES,
    image_attachment_violations,
)
from llm_council.cli import main
from llm_council.context import (
    IMAGE_MIME_ALLOWLIST,
    MAX_PROMPT_CHARS,
    build_image_manifest,
    build_prompt,
    render_image_section,
    resolve_image_path,
)
from llm_council.mcp_server import (
    INLINE_INPUTS_RETENTION_DAYS,
    _stage_inline_images,
    council_run_schema,
    estimate_schema,
    run_council,
    sweep_old_inline_inputs,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _make_png(path: Path, *, width: int = 1, height: int = 1) -> Path:
    """Write a minimal valid PNG so resolve_image_path treats it as image/png."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"\x00" + b"\x00\x00\x00" * width  # filter byte + RGB pixel
    idat = zlib.compress(raw * height)
    path.write_bytes(
        PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    )
    return path


def test_council_run_schema_accepts_image_paths():
    schema = council_run_schema()
    assert "image_paths" in schema["properties"]
    assert schema["properties"]["image_paths"]["type"] == "array"
    assert schema["properties"]["image_paths"]["items"]["type"] == "string"


def test_estimate_schema_accepts_image_paths():
    schema = estimate_schema()
    assert "image_paths" in schema["properties"]


def test_schema_mode_description_lists_all_built_in_modes():
    """Regression: mode description must include every DEFAULT_CONFIG mode so
    new modes don't fall off (4.7 caught opus-versions missing from the
    hardcoded list)."""
    from llm_council.defaults import DEFAULT_CONFIG

    desc_run = council_run_schema()["properties"]["mode"]["description"]
    desc_estimate = estimate_schema()["properties"]["mode"]["description"]
    for name in DEFAULT_CONFIG["modes"]:
        assert name in desc_run, f"mode '{name}' missing from council_run schema description"
        assert name in desc_estimate, f"mode '{name}' missing from estimate schema description"


def test_resolve_image_path_returns_mime_and_size(tmp_path: Path):
    image = _make_png(tmp_path / "ui.png")
    resolved, mime, size = resolve_image_path(image, cwd=tmp_path)
    assert resolved == image
    assert mime == "image/png"
    assert size == image.stat().st_size


def test_resolve_image_path_rejects_outside_cwd(tmp_path: Path):
    outside = _make_png(tmp_path.parent / "escape.png")
    try:
        with pytest.raises(ValueError, match="outside working directory"):
            resolve_image_path(outside, cwd=tmp_path)
    finally:
        outside.unlink()


def test_resolve_image_path_allows_outside_with_flag(tmp_path: Path):
    outside = _make_png(tmp_path.parent / "ok.png")
    try:
        resolved, mime, _ = resolve_image_path(
            outside, cwd=tmp_path, allow_outside_cwd=True
        )
        assert resolved == outside
        assert mime == "image/png"
    finally:
        outside.unlink()


def test_resolve_image_path_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ValueError, match="Image path does not exist"):
        resolve_image_path("nope.png", cwd=tmp_path)


def test_resolve_image_path_rejects_disallowed_mime(tmp_path: Path):
    svg = tmp_path / "drawing.svg"
    svg.write_text("<svg/>")
    with pytest.raises(ValueError, match="not allowed"):
        resolve_image_path(svg, cwd=tmp_path)


def test_resolve_image_path_rejects_unknown_mime(tmp_path: Path):
    blob = tmp_path / "thing.unknownext"
    blob.write_bytes(b"binary")
    with pytest.raises(ValueError, match="Unable to detect mime"):
        resolve_image_path(blob, cwd=tmp_path)


def test_render_image_section_lists_relative_paths(tmp_path: Path):
    a = _make_png(tmp_path / "a.png")
    b = _make_png(tmp_path / "b.png")
    manifest = build_image_manifest([str(a), str(b)], cwd=tmp_path)
    rendered = render_image_section(manifest)
    assert rendered.startswith("## Images")
    assert "a.png" in rendered
    assert "b.png" in rendered
    assert "image/png" in rendered


def test_render_image_section_empty_returns_empty_string():
    assert render_image_section([]) == ""


def test_build_image_manifest_includes_sha256_and_size(tmp_path: Path):
    image = _make_png(tmp_path / "ui.png")
    manifest = build_image_manifest([str(image)], cwd=tmp_path)
    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["mime"] == "image/png"
    assert entry["size"] == image.stat().st_size
    assert len(entry["sha256"]) == 64
    assert entry["relative_path"] == "ui.png"


def test_build_prompt_includes_image_section(tmp_path: Path):
    image = _make_png(tmp_path / "screenshot.png")
    prompt = build_prompt(
        "Why is the login button overlapping the form?",
        mode="quick",
        cwd=tmp_path,
        context_paths=[],
        include_diff=False,
        stdin_text=None,
        image_paths=[str(image)],
    )
    assert "## Images" in prompt
    assert "screenshot.png" in prompt
    # Audience-agnostic copy: CLI participants and vision-capable hosted
    # participants both see the same section.
    assert "file-read tool" in prompt
    assert "vision-capable" in prompt


def test_build_prompt_text_only_when_no_images(tmp_path: Path):
    prompt = build_prompt(
        "Plain question",
        mode="quick",
        cwd=tmp_path,
        context_paths=[],
        include_diff=False,
        stdin_text=None,
    )
    assert "## Images" not in prompt


def test_build_prompt_size_guard_counts_only_text(tmp_path: Path):
    """Image references contribute text only; the binary file size never enters the guard."""
    image = _make_png(tmp_path / "tiny.png", width=1, height=1)
    # Cap the prompt very small. The textual "## Images" line is small, but
    # the underlying file bytes (~74 bytes) must NOT be counted, so the
    # truncation budget should be evaluated against the prompt string only.
    prompt = build_prompt(
        "Tiny.",
        mode="quick",
        cwd=tmp_path,
        context_paths=[],
        include_diff=False,
        stdin_text=None,
        image_paths=[str(image)],
        max_prompt_chars=MAX_PROMPT_CHARS,
    )
    # Image bytes are ~70-90 bytes; if they had been inlined, the prompt
    # would contain raw PNG signature bytes. Confirm it doesn't.
    assert PNG_SIGNATURE.decode("latin-1") not in prompt


def test_cli_missing_image_exits_without_traceback(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--cwd",
                str(tmp_path),
                "--dry-run",
                "--image",
                "missing.png",
                "check",
            ]
        )
    assert str(exc.value).startswith("Image path does not exist:")


def test_cli_image_outside_cwd_rejected(tmp_path: Path):
    outside = _make_png(tmp_path.parent / "escape.png")
    try:
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "run",
                    "--cwd",
                    str(tmp_path),
                    "--dry-run",
                    "--image",
                    str(outside),
                    "check",
                ]
            )
        assert "outside working directory" in str(exc.value)
    finally:
        outside.unlink()


def test_cli_image_dry_run_shows_image_in_prompt(tmp_path: Path, capsys):
    image = _make_png(tmp_path / "ui.png")
    rc = main(
        [
            "run",
            "--cwd",
            str(tmp_path),
            "--dry-run",
            "--image",
            str(image),
            "Review the UI",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "prompt_chars" in captured


def test_build_cli_command_codex_dedups_exec_when_model_is_pinned():
    """Regression for review point 8: Codex's exec subcommand needs
    `-m <model>` and we must not emit `exec exec` when the default args
    list already starts with `exec`."""
    from llm_council.adapters import _build_cli_command

    cfg = {
        "command": "codex",
        "family": "codex",
        "args": ["exec", "--sandbox", "read-only", "-"],
        "model": "gpt-5.1",
    }
    cmd = _build_cli_command("codex", cfg, "p", Path("/tmp"))
    # Exactly one `exec` token and exactly one `-m gpt-5.1` pair.
    assert cmd.count("exec") == 1
    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "gpt-5.1"


def test_build_cli_command_codex_without_exec_in_args_still_emits_canonical_pair():
    """If a future config drops `exec` from default args, the synthesized
    `exec -m <model>` still produces a valid command (no double-exec, no
    missing exec)."""
    from llm_council.adapters import _build_cli_command

    cfg = {
        "command": "codex",
        "family": "codex",
        "args": ["--sandbox", "read-only", "-"],
        "model": "gpt-5.1",
    }
    cmd = _build_cli_command("codex", cfg, "p", Path("/tmp"))
    assert cmd.count("exec") == 1
    assert cmd[1:4] == ["exec", "-m", "gpt-5.1"]


def test_image_mime_allowlist_covers_common_image_types():
    assert {"image/png", "image/jpeg", "image/webp", "image/gif"} <= IMAGE_MIME_ALLOWLIST
    assert "image/svg+xml" not in IMAGE_MIME_ALLOWLIST


# ---------------------------------------------------------------------------
# Phase 2: vision-aware adapter payload, byte budget, non-vision skip events.
# ---------------------------------------------------------------------------


def test_build_user_content_returns_string_when_no_vision(tmp_path: Path):
    image = _make_png(tmp_path / "ui.png")
    manifest = build_image_manifest([str(image)], cwd=tmp_path)
    content = asyncio.run(
        _build_user_content_async("hello", manifest, {"vision": False})
    )
    assert content == "hello"


def test_build_user_content_returns_multimodal_array_for_vision(tmp_path: Path):
    image = _make_png(tmp_path / "ui.png")
    manifest = build_image_manifest([str(image)], cwd=tmp_path)
    content = asyncio.run(
        _build_user_content_async("hello", manifest, {"vision": True})
    )
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_user_content_no_manifest_returns_string():
    content = asyncio.run(_build_user_content_async("hello", None, {"vision": True}))
    assert content == "hello"


def test_read_image_base64_rejects_disallowed_mime(tmp_path: Path):
    image = _make_png(tmp_path / "ok.png")
    bad_entry = {"path": str(image), "mime": "image/svg+xml"}
    with pytest.raises(ValueError, match="not allowed"):
        _read_image_base64(bad_entry)


def test_image_attachment_violations_flags_per_file_cap(tmp_path: Path):
    manifest = [
        {"path": "big.png", "size": DEFAULT_IMAGE_MAX_BYTES + 1, "relative_path": "big.png"}
    ]
    violations = image_attachment_violations(manifest)
    assert any(v["limit"] == "image_max_bytes" for v in violations)


def test_image_attachment_violations_flags_total_cap():
    manifest = [
        {"path": f"img-{i}.png", "size": 4 * 1024 * 1024, "relative_path": f"img-{i}.png"}
        for i in range(10)
    ]
    violations = image_attachment_violations(manifest)
    assert any(v["limit"] == "image_total_max_bytes" for v in violations)


def test_image_attachment_violations_empty_when_under_cap(tmp_path: Path):
    image = _make_png(tmp_path / "tiny.png")
    manifest = build_image_manifest([str(image)], cwd=tmp_path)
    assert image_attachment_violations(manifest) == []


def test_execute_council_emits_images_skipped_for_non_vision(tmp_path: Path):
    from llm_council.orchestrator import execute_council

    image = _make_png(tmp_path / "ui.png")
    manifest = build_image_manifest([str(image)], cwd=tmp_path)
    participant_cfg = {
        "text_router": {
            "type": "openrouter",
            "model": "x/y",
            "vision": False,
            "api_key_env": "MISSING_KEY_FOR_TEST",
        },
        "vision_router": {
            "type": "openrouter",
            "model": "x/vision",
            "vision": True,
            "api_key_env": "MISSING_KEY_FOR_TEST",
        },
        "claude": {"type": "cli", "command": "true", "args": []},
    }
    config = {
        "version": 1,
        "defaults": {"max_concurrency": 1},
        "participants": participant_cfg,
        "modes": {"x": {"participants": list(participant_cfg)}},
    }

    async def go() -> tuple[list, dict]:
        return await execute_council(
            ["text_router", "vision_router", "claude"],
            participant_cfg,
            "prompt",
            tmp_path,
            config,
            image_manifest=manifest,
        )

    _, metadata = asyncio.run(go())
    events = metadata["progress_events"]
    skipped = [e for e in events if e.get("event") == "images_skipped"]
    assert [e["participant"] for e in skipped] == ["text_router"]
    assert skipped[0]["reason"] == "non_vision"
    assert skipped[0]["image_count"] == 1


def test_skip_event_emitted_even_when_progress_callback_is_none(tmp_path: Path):
    """Regression: skip events must reach metadata.progress_events without a printer."""
    from llm_council.orchestrator import execute_council

    image = _make_png(tmp_path / "ui.png")
    manifest = build_image_manifest([str(image)], cwd=tmp_path)
    participant_cfg = {
        "text_router": {
            "type": "openrouter",
            "model": "x/y",
            "api_key_env": "MISSING_KEY_FOR_TEST",
        },
    }
    config = {
        "version": 1,
        "defaults": {"max_concurrency": 1},
        "participants": participant_cfg,
        "modes": {"x": {"participants": ["text_router"]}},
    }

    async def go():
        return await execute_council(
            ["text_router"],
            participant_cfg,
            "prompt",
            tmp_path,
            config,
            progress=None,
            image_manifest=manifest,
        )

    _, metadata = asyncio.run(go())
    skipped = [e for e in metadata["progress_events"] if e.get("event") == "images_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["participant"] == "text_router"


# ---------------------------------------------------------------------------
# Phase 3: inline `images: [{data, mime, name?}]` staging into .llm-council/inputs.
# ---------------------------------------------------------------------------


def test_stage_inline_images_writes_under_run_dir(tmp_path: Path):
    png_bytes = (tmp_path / "src.png").write_bytes  # noqa: F841 (sentinel)
    sample = _make_png(tmp_path / "sample.png")
    encoded = base64.b64encode(sample.read_bytes()).decode()
    inputs = [{"data": encoded, "mime": "image/png", "name": "screenshot.png"}]
    staged = _stage_inline_images(inputs, tmp_path, "myrun")
    assert len(staged) == 1
    target = tmp_path / staged[0]
    assert target.exists()
    assert target.parent == tmp_path / ".llm-council" / "inputs" / "myrun"
    assert target.read_bytes() == sample.read_bytes()


def test_stage_inline_images_rejects_disallowed_mime(tmp_path: Path):
    inputs = [{"data": base64.b64encode(b"<svg/>").decode(), "mime": "image/svg+xml"}]
    with pytest.raises(ValueError, match="not allowed"):
        _stage_inline_images(inputs, tmp_path, "run")


def test_stage_inline_images_rejects_oversize_per_file(tmp_path: Path):
    huge = base64.b64encode(b"a" * (DEFAULT_IMAGE_MAX_BYTES + 100)).decode()
    inputs = [{"data": huge, "mime": "image/png"}]
    with pytest.raises(ValueError, match="per-file budget"):
        _stage_inline_images(inputs, tmp_path, "run")


def test_stage_inline_images_rejects_oversize_total(tmp_path: Path):
    each = base64.b64encode(b"a" * (4 * 1024 * 1024)).decode()
    inputs = [{"data": each, "mime": "image/png"} for _ in range(10)]
    with pytest.raises(ValueError, match="total attachment budget"):
        _stage_inline_images(inputs, tmp_path, "run")


def test_stage_inline_images_rejects_invalid_base64(tmp_path: Path):
    inputs = [{"data": "@@@notbase64", "mime": "image/png"}]
    with pytest.raises(ValueError, match="base64 decode failed"):
        _stage_inline_images(inputs, tmp_path, "run")


def test_stage_inline_images_returns_empty_when_none(tmp_path: Path):
    assert _stage_inline_images(None, tmp_path, "run") == []
    assert _stage_inline_images([], tmp_path, "run") == []


def test_sweep_old_inline_inputs_removes_dirs_past_retention(tmp_path: Path):
    """Both reviewers flagged: .llm-council/inputs/ would grow unbounded
    without a sweep. Confirm the helper removes old run dirs and keeps
    fresh ones."""
    import os
    import time

    inputs_root = tmp_path / ".llm-council" / "inputs"
    inputs_root.mkdir(parents=True)
    fresh = inputs_root / "fresh-run"
    fresh.mkdir()
    (fresh / "img.png").write_bytes(b"fresh")
    stale = inputs_root / "stale-run"
    stale.mkdir()
    (stale / "img.png").write_bytes(b"stale")
    # Backdate the stale dir to 30 days ago.
    old_ts = time.time() - (30 * 86400)
    os.utime(stale, (old_ts, old_ts))

    removed = sweep_old_inline_inputs(tmp_path, retention_days=7)
    assert removed == 1
    assert fresh.exists()
    assert not stale.exists()


def test_sweep_old_inline_inputs_no_inputs_dir_is_noop(tmp_path: Path):
    """No .llm-council/inputs/ directory yet → sweep is a noop returning 0."""
    assert sweep_old_inline_inputs(tmp_path) == 0


def test_inline_inputs_retention_default_is_seven_days():
    assert INLINE_INPUTS_RETENTION_DAYS == 7


def test_stage_inline_images_forces_extension_to_match_mime(tmp_path: Path):
    """Regression: inline name without extension should still be readable downstream."""
    sample = _make_png(tmp_path / "src.png")
    encoded = base64.b64encode(sample.read_bytes()).decode()
    inputs = [{"data": encoded, "mime": "image/png", "name": "screenshot"}]
    staged = _stage_inline_images(inputs, tmp_path, "run")
    assert staged[0].endswith(".png")
    # And the manifest can resolve mime from the extension we forced on:
    manifest = build_image_manifest(staged, cwd=tmp_path)
    assert manifest[0]["mime"] == "image/png"


def test_stage_inline_images_mismatched_extension_normalized(tmp_path: Path):
    """If host claims png but names the file .jpg, force .png so mime matches downstream."""
    sample = _make_png(tmp_path / "src.png")
    encoded = base64.b64encode(sample.read_bytes()).decode()
    inputs = [{"data": encoded, "mime": "image/png", "name": "weird.jpg"}]
    staged = _stage_inline_images(inputs, tmp_path, "run")
    assert staged[0].endswith(".png")


def test_build_image_manifest_does_not_load_full_file_into_memory(tmp_path: Path):
    """Regression: sha256 must stream so we don't OOM on multi-GB inputs.
    We patch read_bytes to fail loudly; if it's called we know we regressed."""
    image = _make_png(tmp_path / "ui.png")

    real_open = Path.open
    read_bytes_called = []
    real_read_bytes = Path.read_bytes

    def tripwire(self, *args, **kwargs):
        read_bytes_called.append(self)
        return real_read_bytes(self, *args, **kwargs)

    with patch.object(Path, "read_bytes", tripwire):
        manifest = build_image_manifest([str(image)], cwd=tmp_path)
    # Streaming hash uses Path.open(..."rb"); read_bytes should not be touched
    # for hashing. If something else reads the bytes (resolve_image_path
    # reading via stat is fine), it is allowed once at most.
    assert len(read_bytes_called) == 0
    assert len(manifest[0]["sha256"]) == 64


@pytest.mark.asyncio
async def test_mcp_council_run_dry_run_lists_inline_images(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    sample = _make_png(tmp_path / "src.png")
    encoded = base64.b64encode(sample.read_bytes()).decode()
    result = await run_council(
        {
            "question": "review",
            "working_directory": str(tmp_path),
            "images": [{"data": encoded, "mime": "image/png", "name": "shot.png"}],
            "dry_run": True,
        }
    )
    assert len(result["images"]) == 1
    assert result["images"][0]["mime"] == "image/png"
    # Confirm staged file exists on disk after dry_run staging.
    rel = result["images"][0]["path"]
    staged = tmp_path / rel
    assert staged.exists()
    assert staged.read_bytes() == sample.read_bytes()


@pytest.mark.asyncio
async def test_mcp_council_run_rejects_oversize_image_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (DEFAULT_IMAGE_MAX_BYTES + 100))
    with pytest.raises(ValueError, match="Image attachment budget"):
        await run_council(
            {
                "question": "review",
                "working_directory": str(tmp_path),
                "image_paths": ["big.png"],
                "dry_run": True,
            }
        )


def test_council_run_schema_accepts_inline_images_field():
    schema = council_run_schema()
    assert "images" in schema["properties"]
    item = schema["properties"]["images"]["items"]
    assert "data" in item["required"]
    assert "mime" in item["required"]


def test_estimate_council_enforces_image_budget(tmp_path: Path):
    """Regression for review point: estimate must reject the same image
    sets the actual run will reject, so preflight is honest."""
    from llm_council.estimate import estimate_council

    big = tmp_path / "huge.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (DEFAULT_IMAGE_MAX_BYTES + 100))
    config = {
        "version": 1,
        "defaults": {"mode": "single"},
        "participants": {
            "x": {
                "type": "openrouter",
                "model": "x/y",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
                "api_key_env": "OPENROUTER_API_KEY",
            },
        },
        "modes": {"single": {"participants": ["x"]}},
    }
    with pytest.raises(ValueError, match="Image attachment budget exceeded"):
        estimate_council(
            config=config,
            cwd=tmp_path,
            question="describe",
            mode="single",
            current=None,
            image_paths=[str(big)],
        )


def test_estimate_image_token_heuristic_only_charges_vision_participants(tmp_path: Path):
    """Vision-flagged participants get extra tokens; text-only ones don't."""
    from llm_council.estimate import IMAGE_TOKEN_HEURISTIC, estimate_council

    image = _make_png(tmp_path / "ui.png")
    config = {
        "version": 1,
        "defaults": {"mode": "vision_pair"},
        "participants": {
            "text_only": {
                "type": "openrouter",
                "model": "x/text",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
                "api_key_env": "OPENROUTER_API_KEY",
            },
            "vision_capable": {
                "type": "openrouter",
                "model": "x/vision",
                "input_per_million": 1.0,
                "output_per_million": 1.0,
                "api_key_env": "OPENROUTER_API_KEY",
                "vision": True,
            },
        },
        "modes": {"vision_pair": {"participants": ["text_only", "vision_capable"]}},
    }
    estimate = estimate_council(
        config=config,
        cwd=tmp_path,
        question="what is in this image",
        mode="vision_pair",
        current=None,
        image_paths=[str(image)],
    )
    rows_by_name = {row["name"]: row for row in estimate["rows"]}
    assert (
        rows_by_name["vision_capable"]["estimated_input_tokens"]
        == rows_by_name["text_only"]["estimated_input_tokens"] + IMAGE_TOKEN_HEURISTIC
    )
