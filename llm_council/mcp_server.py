"""MCP server wrapper for llm-council."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from llm_council.budget import enforce_mcp_budget, mcp_budget_report
from llm_council.config import (
    detect_current_agent,
    find_config,
    load_config,
    select_participants,
)
from llm_council.context import MAX_PROMPT_CHARS, build_prompt
from llm_council.doctor import check_environment, checks_to_dict
from llm_council.env import load_project_env
from llm_council.model_catalog import fetch_openrouter_models
from llm_council.orchestrator import execute_council
from llm_council.policy import should_use_council
from llm_council.transcript import latest_transcript, transcript_paths, write_transcript


def council_run_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "mode": {
                "type": "string",
                "description": "Council mode such as quick, plan, review, diverse, review-cheap, private-local.",
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


def doctor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "working_directory": {"type": "string"},
            "probe_openrouter": {"type": "boolean", "default": False},
            "probe_ollama": {"type": "boolean", "default": False},
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
    config = load_config(find_config(cwd))
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
    )
    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    md_path, json_path = transcript_paths(out_dir, question)
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
    config = load_config(find_config(cwd))
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
    config = load_config(find_config(cwd))
    return {
        "checks": checks_to_dict(
            check_environment(
                config,
                probe_openrouter=bool(arguments.get("probe_openrouter")),
                probe_ollama=bool(arguments.get("probe_ollama")),
            )
        )
    }


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


def list_modes(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd)
    config = load_config(find_config(cwd))
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
        elif name == "council_list_modes":
            result = list_modes(arguments)
        elif name == "council_last_transcript":
            result = last_transcript(arguments)
        elif name == "council_doctor":
            result = run_doctor(arguments)
        elif name == "council_models":
            result = list_models(arguments)
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
