"""MCP server wrapper for llm-council."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import re
from pathlib import Path
from typing import Any

from llm_council import __version__
from llm_council.budget import (
    DEFAULT_IMAGE_MAX_BYTES,
    DEFAULT_IMAGE_TOTAL_MAX_BYTES,
    enforce_mcp_budget,
    image_attachment_violations,
    mcp_budget_report,
)
from llm_council.context import IMAGE_MIME_ALLOWLIST
from llm_council.defaults import DEFAULT_CONFIG
from llm_council.config import (
    detect_current_agent,
    find_config,
    load_config,
    select_participants,
)
from llm_council.context import MAX_PROMPT_CHARS, build_image_manifest, build_prompt
from llm_council.doctor import check_environment, checks_to_dict
from llm_council.env import load_project_env
from llm_council.estimate import estimate_council
from llm_council.model_catalog import fetch_openrouter_models
from llm_council.orchestrator import execute_council
from llm_council.policy import should_use_council
from llm_council.stats import compute_stats
from llm_council.transcript import latest_transcript, transcript_paths, write_transcript
from llm_council.update_check import check_for_update


def _mode_description() -> str:
    names = ", ".join(sorted(DEFAULT_CONFIG["modes"]))
    return f"Council mode. Built-in choices: {names}."


def council_run_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "mode": {
                "type": "string",
                "description": _mode_description(),
            },
            "current": {"type": "string", "enum": ["claude", "codex", "gemini"]},
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit participants. Overrides mode routing.",
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra participants to add to mode routing.",
            },
            "context_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to include as read-only context.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Repo-relative paths to image files (PNG/JPEG/WebP/GIF) the host has staged for council review. CLI participants Read them with their own tools.",
            },
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Base64-encoded image bytes."},
                        "mime": {
                            "type": "string",
                            "enum": sorted(IMAGE_MIME_ALLOWLIST),
                        },
                        "name": {"type": "string"},
                    },
                    "required": ["data", "mime"],
                    "additionalProperties": False,
                },
                "description": "Inline base64 images. llm-council writes them under .llm-council/inputs/<run-id>/ before participants run. Use only when the host cannot stage to disk; image_paths is preferred.",
            },
            "include_diff": {"type": "boolean", "default": False},
            "working_directory": {"type": "string"},
            "dry_run": {"type": "boolean", "default": False},
            "transparent": {"type": "boolean", "default": False},
            "deliberate": {
                "type": "boolean",
                "default": False,
                "description": "Run an expensive second round if first-round responses disagree.",
            },
            "max_rounds": {"type": "integer", "minimum": 1, "maximum": 3},
            "origin_policy": {
                "type": "string",
                "enum": ["any", "us"],
                "description": "Set to 'us' to allow only US-origin participants.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }


def last_transcript_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "working_directory": {"type": "string"},
            "format": {
                "type": "string",
                "enum": ["markdown", "json"],
                "default": "markdown",
            },
        },
        "additionalProperties": False,
    }


def recommend_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "failed_attempts": {"type": "integer", "minimum": 0, "default": 0},
            "files_touched": {"type": "integer", "minimum": 0, "default": 0},
            "risk": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "default": "medium",
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }


def estimate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "mode": {
                "type": "string",
                "description": _mode_description(),
            },
            "current": {"type": "string", "enum": ["claude", "codex", "gemini"]},
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit participants. Overrides mode routing.",
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra participants to add to mode routing.",
            },
            "context_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to include as read-only context.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Repo-relative paths to image files (PNG/JPEG/WebP/GIF). Counted against prompt-size guard as text references only.",
            },
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "mime": {"type": "string", "enum": sorted(IMAGE_MIME_ALLOWLIST)},
                        "name": {"type": "string"},
                    },
                    "required": ["data", "mime"],
                    "additionalProperties": False,
                },
                "description": "Inline base64 images. Estimate stages them to .llm-council/inputs/<run-id>/ before computing prompt size.",
            },
            "include_diff": {"type": "boolean", "default": False},
            "working_directory": {"type": "string"},
            "deliberate": {"type": "boolean", "default": False},
            "max_rounds": {"type": "integer", "minimum": 1, "maximum": 3},
            "completion_tokens": {
                "type": "integer",
                "minimum": 0,
                "default": 1500,
                "description": "Assumed output tokens per participant per round.",
            },
            "openrouter_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra OpenRouter model IDs to price without editing config.",
            },
            "origin_policy": {
                "type": "string",
                "enum": ["any", "us"],
                "description": "Set to 'us' to allow only US-origin participants.",
            },
            "no_cache": {"type": "boolean", "default": False},
        },
        "required": ["question"],
        "additionalProperties": False,
    }


def doctor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "working_directory": {"type": "string"},
            "probe_openrouter": {"type": "boolean", "default": False},
            "probe_ollama": {"type": "boolean", "default": False},
            "check_update": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
    }


def stats_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "working_directory": {"type": "string"},
            "since_days": {
                "type": "integer",
                "minimum": 1,
                "description": "Only consider transcripts within the last N days.",
            },
            "participant": {
                "type": "string",
                "description": "Filter the per-participant view to one peer.",
            },
        },
        "additionalProperties": False,
    }


def models_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "filter": {"type": "string"},
            "origin": {"type": "string", "enum": ["us", "china", "unknown"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "no_cache": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
    }


async def run_council(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    question = arguments["question"]
    mode = arguments.get("mode") or config.get("defaults", {}).get("mode", "quick")
    current = arguments.get("current") or detect_current_agent()
    participants = select_participants(
        config,
        mode,
        current,
        explicit=arguments.get("participants"),
        include=arguments.get("include"),
        origin_policy=arguments.get("origin_policy"),
    )
    transcripts_root = Path(config.get("transcripts_dir", ".llm-council/runs"))
    if not transcripts_root.is_absolute():
        transcripts_root = cwd / transcripts_root
    md_path, json_path = transcript_paths(transcripts_root, question)
    sweep_old_inline_inputs(cwd)
    inline_staged = _stage_inline_images(arguments.get("images"), cwd, md_path.stem)
    image_path_inputs = list(arguments.get("image_paths") or []) + inline_staged
    image_manifest = (
        build_image_manifest(image_path_inputs, cwd=cwd, allow_outside_cwd=False)
        if image_path_inputs
        else []
    )
    image_violations = image_attachment_violations(image_manifest)
    if image_violations:
        raise ValueError(
            "Image attachment budget exceeded: "
            + ", ".join(
                f"{item['limit']} {item.get('actual')} > {item.get('maximum')}"
                for item in image_violations
            )
        )
    prompt = build_prompt(
        question,
        mode=mode,
        cwd=cwd,
        context_paths=arguments.get("context_files") or [],
        include_diff=bool(arguments.get("include_diff")),
        stdin_text=None,
        allow_outside_cwd=False,
        max_prompt_chars=config.get("defaults", {}).get("max_prompt_chars")
        or MAX_PROMPT_CHARS,
        image_manifest=image_manifest or None,
    )

    mode_cfg = config.get("modes", {}).get(mode, {})
    transparent = bool(
        arguments.get("transparent") or config.get("defaults", {}).get("transparent")
    )
    deliberate = bool(arguments.get("deliberate") or mode_cfg.get("deliberate"))
    max_rounds = int(
        arguments.get("max_rounds")
        or mode_cfg.get("max_rounds")
        or config.get("defaults", {}).get("max_deliberation_rounds")
        or 2
    )
    budget = mcp_budget_report(
        config=config,
        participants=participants,
        prompt_chars=len(prompt),
        deliberate=deliberate,
        max_rounds=max_rounds,
    )

    if arguments.get("dry_run"):
        return {
            "mode": mode,
            "current": current,
            "participants": participants,
            "prompt_chars": len(prompt),
            "deliberate": deliberate,
            "max_rounds": max_rounds,
            "budget": budget,
            "images": [_public_image_entry(entry, cwd) for entry in image_manifest],
        }

    enforce_mcp_budget(budget)
    cfg = config.get("participants", {})
    results, metadata = await execute_council(
        participants,
        cfg,
        prompt,
        cwd,
        config,
        deliberate=deliberate,
        max_rounds=max_rounds,
        image_manifest=image_manifest or None,
    )
    if image_manifest:
        metadata["images"] = [
            _public_image_entry(entry, cwd) for entry in image_manifest
        ]
    write_transcript(
        md_path,
        json_path,
        question=question,
        mode=mode,
        current=current,
        participants=participants,
        prompt=prompt,
        results=results,
        transparent=transparent,
        metadata=metadata,
    )
    return {
        "mode": mode,
        "current": current,
        "participants": participants,
        "metadata": metadata,
        "transcript": str(md_path),
        "json": str(json_path),
        "results": [
            {
                "name": result.name,
                "ok": result.ok,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "error": result.error,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "cost_usd": result.cost_usd,
            }
            for result in results
        ],
    }


def last_transcript(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    path = latest_transcript(out_dir, suffix=".json" if arguments.get("format") == "json" else ".md")
    if path is None:
        return {"found": False, "path": None, "content": ""}
    return {"found": True, "path": str(path), "content": path.read_text(encoding="utf-8")}


def run_doctor(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    result: dict[str, Any] = {
        "checks": checks_to_dict(
            check_environment(
                config,
                probe_openrouter=bool(arguments.get("probe_openrouter")),
                probe_ollama=bool(arguments.get("probe_ollama")),
            )
        )
    }
    result["version"] = __version__
    if arguments.get("check_update"):
        result["update"] = check_for_update(__version__).to_dict()
    return result


def estimate_run(arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        cwd = _resolve_working_directory(arguments)
        load_project_env(cwd)
        config = load_config(find_config(cwd), search=False)
        mode = arguments.get("mode") or config.get("defaults", {}).get("mode", "quick")
        current = arguments.get("current") or detect_current_agent()
        completion_tokens = (
            1500
            if arguments.get("completion_tokens") is None
            else int(arguments["completion_tokens"])
        )
        from datetime import datetime

        estimate_slug = (
            "estimate-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        sweep_old_inline_inputs(cwd)
        inline_staged = _stage_inline_images(
            arguments.get("images"), cwd, estimate_slug
        )
        image_path_inputs = list(arguments.get("image_paths") or []) + inline_staged
        estimate = estimate_council(
            config=config,
            cwd=cwd,
            question=arguments["question"],
            mode=mode,
            current=current,
            explicit=arguments.get("participants"),
            include=arguments.get("include"),
            origin_policy=arguments.get("origin_policy"),
            context_paths=arguments.get("context_files") or [],
            include_diff=bool(arguments.get("include_diff")),
            stdin_text=None,
            allow_outside_cwd=False,
            deliberate=bool(arguments.get("deliberate")),
            max_rounds=arguments.get("max_rounds"),
            completion_tokens=completion_tokens,
            openrouter_models=arguments.get("openrouter_models") or [],
            use_cache=not bool(arguments.get("no_cache")),
            image_paths=image_path_inputs or None,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, **estimate}


def list_models(arguments: dict[str, Any]) -> dict[str, Any]:
    models = fetch_openrouter_models(use_cache=not bool(arguments.get("no_cache")))
    if arguments.get("filter"):
        needle = str(arguments["filter"]).lower()
        models = [
            model
            for model in models
            if needle in model["id"].lower() or needle in model["name"].lower()
        ]
    if arguments.get("origin"):
        prefix = {"us": "US /", "china": "China /", "unknown": "Unknown"}[
            arguments["origin"]
        ]
        models = [model for model in models if str(model["origin"]).startswith(prefix)]
    limit = int(arguments.get("limit") or 40)
    return {"models": models[:limit]}


def run_stats(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    since_days = arguments.get("since_days")
    if since_days is not None:
        since_days = int(since_days)
        if since_days <= 0:
            raise ValueError("since_days must be a positive integer")
    participant = arguments.get("participant")
    if participant is not None and not isinstance(participant, str):
        raise ValueError("participant must be a string")
    return compute_stats(
        out_dir,
        participant=participant or None,
        since_days=since_days,
    )


def list_modes(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    return {
        "modes": config.get("modes", {}),
        "participants": list(config.get("participants", {}).keys()),
    }


async def _serve() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except Exception as exc:  # pragma: no cover - depends on environment install
        raise SystemExit(
            "The 'mcp' Python package is required for MCP server mode. "
            "Install project requirements first."
        ) from exc

    app = Server("llm-council")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="council_run",
                description="Run a read-only multi-agent council.",
                inputSchema=council_run_schema(),
            ),
            Tool(
                name="council_recommend",
                description="Recommend whether a task should go to council and which mode to use.",
                inputSchema=recommend_schema(),
            ),
            Tool(
                name="council_estimate",
                description="Estimate prompt size and OpenRouter costs before running council.",
                inputSchema=estimate_schema(),
            ),
            Tool(
                name="council_list_modes",
                description="List configured council modes and participants.",
                inputSchema={
                    "type": "object",
                    "properties": {"working_directory": {"type": "string"}},
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="council_last_transcript",
                description="Read the latest council transcript from the project.",
                inputSchema=last_transcript_schema(),
            ),
            Tool(
                name="council_doctor",
                description="Diagnose local CLI, OpenRouter, Ollama, and MCP readiness.",
                inputSchema=doctor_schema(),
            ),
            Tool(
                name="council_models",
                description="List cached OpenRouter models with optional filter/origin.",
                inputSchema=models_schema(),
            ),
            Tool(
                name="council_stats",
                description=(
                    "Aggregate per-participant metrics across recorded "
                    "transcripts: run count, success rate, recommendation "
                    "label distribution, tokens, cost, and last-used time."
                ),
                inputSchema=stats_schema(),
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "council_run":
            result = await run_council(arguments)
        elif name == "council_recommend":
            use, mode, reason = should_use_council(
                arguments["task"],
                failed_attempts=int(arguments.get("failed_attempts") or 0),
                files_touched=int(arguments.get("files_touched") or 0),
                risk=arguments.get("risk") or "medium",
            )
            result = {"use_council": use, "mode": mode, "reason": reason}
        elif name == "council_estimate":
            result = estimate_run(arguments)
        elif name == "council_list_modes":
            result = list_modes(arguments)
        elif name == "council_last_transcript":
            result = last_transcript(arguments)
        elif name == "council_doctor":
            result = run_doctor(arguments)
        elif name == "council_models":
            result = list_models(arguments)
        elif name == "council_stats":
            result = run_stats(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run llm-council MCP server")
    parser.parse_args(argv or [])
    asyncio.run(_serve())
    return 0


_INLINE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
INLINE_INPUTS_RETENTION_DAYS = 7


def sweep_old_inline_inputs(
    cwd: Path, *, retention_days: int = INLINE_INPUTS_RETENTION_DAYS
) -> int:
    """Best-effort cleanup of staged inline-image directories older than
    `retention_days`. Returns the number of directories removed.

    Called opportunistically before staging new inputs, so a long-running
    project doesn't accumulate gigabytes of screenshot artifacts. Failures
    are swallowed: cleanup must never block a council run.
    """

    import shutil
    import time

    inputs_root = cwd / ".llm-council" / "inputs"
    if not inputs_root.is_dir():
        return 0
    cutoff = time.time() - max(0, retention_days) * 86400
    removed = 0
    try:
        candidates = list(inputs_root.iterdir())
    except OSError:
        return 0
    for entry in candidates:
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        try:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except OSError:
            continue
    return removed


def _stage_inline_images(
    images: list[dict[str, Any]] | None,
    cwd: Path,
    run_slug: str,
) -> list[str]:
    if not images:
        return []
    if not isinstance(images, list):
        raise ValueError("images must be an array of {data, mime, name?} entries")
    inputs_root = cwd / ".llm-council" / "inputs" / run_slug
    inputs_root.mkdir(parents=True, exist_ok=True)
    staged_relative: list[str] = []
    total_bytes = 0
    for index, entry in enumerate(images):
        if not isinstance(entry, dict):
            raise ValueError("images entry must be an object")
        mime = entry.get("mime")
        if mime not in IMAGE_MIME_ALLOWLIST:
            raise ValueError(
                f"Inline image #{index} mime '{mime}' is not allowed. "
                f"Allowed: {', '.join(sorted(IMAGE_MIME_ALLOWLIST))}."
            )
        data = entry.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError(f"Inline image #{index} missing base64 'data'")
        # Cheap pre-decode size guard: base64 expands ~4/3.
        approx_bytes = (len(data) * 3) // 4
        if approx_bytes > DEFAULT_IMAGE_MAX_BYTES:
            raise ValueError(
                f"Inline image #{index} exceeds per-file budget before decode "
                f"(~{approx_bytes} > {DEFAULT_IMAGE_MAX_BYTES})"
            )
        if total_bytes + approx_bytes > DEFAULT_IMAGE_TOTAL_MAX_BYTES:
            raise ValueError(
                "Inline images exceed total attachment budget before decode "
                f"(~{total_bytes + approx_bytes} > {DEFAULT_IMAGE_TOTAL_MAX_BYTES})"
            )
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Inline image #{index} base64 decode failed: {exc}") from exc
        total_bytes += len(decoded)
        if len(decoded) > DEFAULT_IMAGE_MAX_BYTES:
            raise ValueError(
                f"Inline image #{index} exceeds per-file budget after decode "
                f"({len(decoded)} > {DEFAULT_IMAGE_MAX_BYTES})"
            )
        if total_bytes > DEFAULT_IMAGE_TOTAL_MAX_BYTES:
            raise ValueError(
                "Inline images exceed total attachment budget "
                f"({total_bytes} > {DEFAULT_IMAGE_TOTAL_MAX_BYTES})"
            )
        ext = _MIME_TO_EXT.get(mime, "")
        raw_name = entry.get("name") or f"img-{index:02d}{ext}"
        safe_name = _INLINE_NAME_RE.sub("-", str(raw_name)).strip("-") or f"img-{index:02d}{ext}"
        # Force the extension to match the declared mime so downstream
        # mimetypes.guess_type matches what the host claimed.
        if Path(safe_name).suffix.lower() != ext:
            safe_name = Path(safe_name).stem + ext
        target = inputs_root / safe_name
        # Avoid path collisions if the host reuses names.
        suffix = 0
        while target.exists():
            suffix += 1
            target = inputs_root / f"{Path(safe_name).stem}-{suffix}{ext}"
        target.write_bytes(decoded)
        staged_relative.append(str(target.resolve().relative_to(cwd.resolve())))
    return staged_relative


def _public_image_entry(entry: dict[str, Any], cwd: Path) -> dict[str, Any]:
    return {
        "path": entry.get("relative_path") or entry.get("path"),
        "mime": entry.get("mime"),
        "size": entry.get("size"),
        "sha256": entry.get("sha256"),
    }


def _resolve_working_directory(arguments: dict[str, Any]) -> Path:
    root = Path(os.environ.get("LLM_COUNCIL_MCP_ROOT") or ".").resolve()
    cwd = Path(arguments.get("working_directory") or root).resolve()
    if not cwd.exists():
        raise ValueError(f"working_directory does not exist: {cwd}")
    if not cwd.is_dir():
        raise ValueError(f"working_directory is not a directory: {cwd}")
    try:
        cwd.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"working_directory must be inside MCP project root: {root}"
        ) from exc
    return cwd


if __name__ == "__main__":
    raise SystemExit(main())
