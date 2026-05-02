"""Command line interface for llm-council."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from llm_council import __version__
from llm_council.config import (
    detect_current_agent,
    find_config,
    load_config,
    parse_csv,
    select_participants,
)
from llm_council.budget import image_attachment_violations
from llm_council.context import MAX_PROMPT_CHARS, build_image_manifest, build_prompt
from llm_council.doctor import check_environment, checks_to_dict
from llm_council.env import load_project_env
from llm_council.estimate import estimate_council
from llm_council.model_catalog import fetch_openrouter_models
from llm_council.orchestrator import execute_council
from llm_council.policy import should_use_council
from llm_council.setup_wizard import write_setup_files
from llm_council.stats import compute_stats, format_stats_text
from llm_council.transcript import (
    find_transcript_by_id,
    format_prior_council_context,
    latest_transcript,
    normalize_run_id,
    transcript_paths,
    transcript_records,
    write_transcript,
)
from llm_council.update_check import check_for_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-council",
        description="Lightweight read-only multi-agent council for coding projects.",
    )
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a council prompt")
    run.add_argument("question", nargs="*", help="Question or prompt")
    run.add_argument("--config", help="Path to config YAML")
    run.add_argument("--mode", default=None, help="Council mode")
    run.add_argument("--current", choices=["claude", "codex", "gemini"])
    run.add_argument("--participants", help="Comma-separated explicit participants")
    run.add_argument("--include", help="Comma-separated extra participants")
    run.add_argument(
        "--origin-policy",
        choices=["any", "us"],
        help="Filter participants by model/lab origin",
    )
    run.add_argument("--context", action="append", default=[], help="Context file")
    run.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image file path (PNG/JPEG/WebP/GIF) for council to inspect; repeatable",
    )
    run.add_argument(
        "--allow-outside-cwd",
        action="store_true",
        help="Allow --context and --image files outside the working directory",
    )
    run.add_argument("--diff", action="store_true", help="Include git diff")
    run.add_argument("--stdin", action="store_true", help="Append stdin as context")
    run.add_argument("--cwd", default=".", help="Working directory")
    run.add_argument("--dry-run", action="store_true", help="Print plan without calls")
    run.add_argument("--json", action="store_true", help="Print JSON result summary")
    run.add_argument(
        "--transparent",
        action="store_true",
        help="Print usage/cost and per-model comparison when available",
    )
    run.add_argument(
        "--deliberate",
        action="store_true",
        help="Run an expensive second round if first-round responses disagree",
    )
    run.add_argument("--max-rounds", type=int, help="Maximum deliberation rounds")
    run.add_argument(
        "--min-quorum",
        type=int,
        default=None,
        help=(
            "Minimum label-producing peers in the final round before the "
            "result is considered trustworthy. Default: 2 when 2+ peers "
            "are configured, else equal to the peer count. Setting higher "
            "than the configured peer count will always report the council "
            "as degraded."
        ),
    )
    run.add_argument(
        "--continue",
        dest="continue_id",
        default=None,
        help=(
            "Run id (timestamp prefix or filename) of a prior council "
            "transcript whose summary should be prepended to the new prompt."
        ),
    )
    run.add_argument(
        "--cache",
        dest="cache_mode",
        choices=["on", "off", "refresh"],
        default="on",
        help=(
            "Per-participant on-disk result cache keyed on prompt+config. "
            "`on` reads and writes (default). `off` skips both. `refresh` "
            "ignores the read but still writes."
        ),
    )
    run.add_argument(
        "--chunk-strategy",
        dest="chunk_strategy",
        choices=["fail", "head", "tail", "hash-aware"],
        default="fail",
        help=(
            "How to handle a diff that pushes the prompt over max_prompt_chars. "
            "Default `fail` preserves fail-fast behavior. `head`/`tail` keep "
            "the first/last bytes that fit. `hash-aware` drops lower-relevance "
            "files (per-file `diff --git` blocks) until the prompt fits."
        ),
    )

    sub.add_parser("list", help="List participants and modes")
    init = sub.add_parser("init", help="Write an example project config")
    init.add_argument("--path", default=".llm-council.yaml")

    setup = sub.add_parser("setup", help="Walk through or write project setup")
    setup.add_argument("--root", default=".", help="Project root")
    setup.add_argument(
        "--preset",
        choices=[
            "auto",
            "tri-cli",
            "openrouter",
            "tri-cli-openrouter",
            "local-private",
            "all",
        ],
        default="auto",
        help=(
            "Setup scope: auto detects local CLIs/OpenRouter, "
            "tri-cli for Claude/Codex/Gemini only, openrouter for hosted-only, "
            "tri-cli-openrouter for native CLIs plus hosted models, "
            "local-private for native CLIs plus Ollama, all for every preset"
        ),
    )
    setup.add_argument("--yes", action="store_true", help="Non-interactive defaults")
    setup.add_argument(
        "--plan",
        action="store_true",
        help="Print detected setup routes without writing files",
    )
    setup.add_argument("--force", action="store_true", help="Overwrite existing files")
    setup.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a preset even when required CLIs or API keys are not detected",
    )
    setup.add_argument(
        "--us-only-default",
        action="store_true",
        help="Default generated config to US-origin participants only",
    )
    setup.add_argument("--no-mcp", action="store_true", help="Do not write .mcp.json")
    setup.add_argument(
        "--no-instructions", action="store_true", help="Do not write instruction snippets"
    )

    doctor = sub.add_parser("doctor", help="Check local council environment")
    doctor.add_argument("--config", help="Path to config YAML")
    doctor.add_argument("--json", action="store_true", help="Print JSON")
    doctor.add_argument(
        "--probe-openrouter",
        action="store_true",
        help="Validate OPENROUTER_API_KEY with OpenRouter auth endpoint",
    )
    doctor.add_argument(
        "--probe-ollama",
        action="store_true",
        help="Validate Ollama is serving its tags endpoint",
    )
    doctor.add_argument(
        "--check-update",
        action="store_true",
        help="Check public GitHub version and print update guidance",
    )

    update = sub.add_parser("check-update", help="Check whether llm-council is current")
    update.add_argument("--json", action="store_true", help="Print JSON")

    recommend = sub.add_parser("recommend", help="Recommend whether to use council")
    recommend.add_argument("task", nargs="*", help="Task description")
    recommend.add_argument("--failed-attempts", type=int, default=0)
    recommend.add_argument("--files-touched", type=int, default=0)
    recommend.add_argument(
        "--risk", choices=["low", "medium", "high"], default="medium"
    )
    recommend.add_argument("--json", action="store_true", help="Print JSON")

    estimate = sub.add_parser(
        "estimate", help="Estimate prompt size and OpenRouter costs before a run"
    )
    estimate.add_argument("question", nargs="*", help="Question or prompt")
    estimate.add_argument("--config", help="Path to config YAML")
    estimate.add_argument("--mode", default=None, help="Council mode")
    estimate.add_argument("--current", choices=["claude", "codex", "gemini"])
    estimate.add_argument("--participants", help="Comma-separated explicit participants")
    estimate.add_argument("--include", help="Comma-separated extra participants")
    estimate.add_argument(
        "--origin-policy",
        choices=["any", "us"],
        help="Filter participants by model/lab origin",
    )
    estimate.add_argument("--context", action="append", default=[], help="Context file")
    estimate.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image file path (PNG/JPEG/WebP/GIF) for council to inspect; repeatable",
    )
    estimate.add_argument(
        "--allow-outside-cwd",
        action="store_true",
        help="Allow --context and --image files outside the working directory",
    )
    estimate.add_argument("--diff", action="store_true", help="Include git diff")
    estimate.add_argument("--stdin", action="store_true", help="Append stdin as context")
    estimate.add_argument("--cwd", default=".", help="Working directory")
    estimate.add_argument(
        "--deliberate",
        action="store_true",
        help="Estimate an opt-in deliberation run",
    )
    estimate.add_argument("--max-rounds", type=int, help="Maximum deliberation rounds")
    estimate.add_argument(
        "--completion-tokens",
        type=int,
        default=1500,
        help="Assumed output tokens per participant per round",
    )
    estimate.add_argument(
        "--openrouter-model",
        action="append",
        default=[],
        help="Extra OpenRouter model ID to price without editing config",
    )
    estimate.add_argument("--no-cache", action="store_true", help="Bypass model cache")
    estimate.add_argument("--json", action="store_true", help="Print JSON")

    last = sub.add_parser("last", help="Print the latest council transcript path/content")
    last.add_argument("--cwd", default=".", help="Working directory")
    last.add_argument("--json-file", action="store_true", help="Use JSON transcript")
    last.add_argument("--path-only", action="store_true", help="Only print path")

    transcripts = sub.add_parser("transcripts", help="Inspect council transcripts")
    transcripts_sub = transcripts.add_subparsers(dest="transcripts_command")
    transcripts_list = transcripts_sub.add_parser("list", help="List recent transcripts")
    transcripts_list.add_argument("--cwd", default=".", help="Working directory")
    transcripts_list.add_argument("--limit", type=int, default=10)
    transcripts_list.add_argument("--json", action="store_true", help="Print JSON")
    transcripts_show = transcripts_sub.add_parser("show", help="Show a transcript")
    transcripts_show.add_argument("path", nargs="?", help="Transcript path; defaults to latest")
    transcripts_show.add_argument("--cwd", default=".", help="Working directory")
    transcripts_show.add_argument("--json-file", action="store_true", help="Show JSON")
    transcripts_summary = transcripts_sub.add_parser(
        "summary", help="Summarize transcript totals"
    )
    transcripts_summary.add_argument("--cwd", default=".", help="Working directory")

    stats = sub.add_parser(
        "stats", help="Aggregate per-participant metrics over recorded transcripts"
    )
    stats.add_argument("--cwd", default=".", help="Working directory")
    stats.add_argument(
        "--since",
        type=int,
        default=None,
        help="Only consider transcripts within the last N days",
    )
    stats.add_argument(
        "--participant",
        default=None,
        help="Filter the per-participant table to one peer",
    )
    stats.add_argument("--json", action="store_true", help="Print JSON")

    sub.add_parser("mcp-server", help="Run llm-council MCP server over stdio")

    models = sub.add_parser("models", help="Inspect live model catalogs")
    models_sub = models.add_subparsers(dest="models_command")
    openrouter = models_sub.add_parser(
        "openrouter", help="Fetch and print OpenRouter models"
    )
    openrouter.add_argument("--filter", help="Case-insensitive filter over id/name")
    openrouter.add_argument(
        "--origin",
        choices=["us", "china", "unknown"],
        help="Filter by inferred country of origin",
    )
    openrouter.add_argument("--limit", type=int, default=40)
    openrouter.add_argument("--no-cache", action="store_true", help="Bypass disk cache")
    openrouter.add_argument("--json", action="store_true", help="Print JSON")

    return parser


def _question_from_args(parts: list[str]) -> str:
    question = " ".join(parts).strip()
    if not question:
        raise SystemExit("question is required")
    return question


def cmd_list(args: argparse.Namespace) -> int:
    load_project_env(Path.cwd())
    config = load_config(getattr(args, "config", None))
    print("Participants:")
    for name, cfg in config.get("participants", {}).items():
        model = cfg.get("model") or "cli default"
        print(f"  {name:20} {cfg.get('type'):10} {model}")
    print("\nModes:")
    for name, cfg in config.get("modes", {}).items():
        print(f"  {name:20} {cfg.get('description', '')}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing config: {target}")
    sample = Path(__file__).resolve().parent.parent / "examples" / "llm-council.yaml"
    if sample.exists():
        target.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        target.write_text("version: 1\n", encoding="utf-8")
    print(f"Wrote {target}")
    return 0


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def cmd_setup(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    load_project_env(root)
    if getattr(args, "plan", False):
        _print_setup_plan(root)
        return 0
    preset = args.preset
    if preset == "auto" and not args.yes:
        preset = _prompt_setup_preset(root)
    elif preset == "auto":
        preset = _auto_setup_preset()
        print(f"Auto preset selected: {preset}")

    _guard_setup_preset(preset, args)

    include_native = preset != "openrouter"
    include_openrouter = preset in {"openrouter", "tri-cli-openrouter", "all"}
    include_local = preset in {"local-private", "all"}

    if not args.yes:
        print("LLM Council setup")
        print(f"Project root: {root}")
        print("\nDetected CLIs:")
        for name in ("claude", "codex", "gemini", "ollama"):
            print(f"  {name:8} {shutil.which(name) or 'not found'}")
        include_openrouter = _confirm(
            "Include OpenRouter participant presets?", include_openrouter
        )
        include_local = _confirm("Include local/Ollama participant preset?", include_local)
        us_only_default = _confirm(
            "Default to US-origin participants only?", args.us_only_default
        )
        write_mcp = False if args.no_mcp else _confirm(
            "Write/update .mcp.json for project MCP?", True
        )
        write_instructions = False if args.no_instructions else _confirm(
            "Write keyword instruction snippets?", True
        )
    else:
        write_mcp = not args.no_mcp
        write_instructions = not args.no_instructions
        us_only_default = args.us_only_default

    try:
        written = write_setup_files(
            root,
            include_native=include_native,
            include_openrouter=include_openrouter,
            include_local=include_local,
            us_only_default=us_only_default,
            write_mcp=write_mcp,
            write_instructions=write_instructions,
            force=args.force,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if written:
        print("Wrote:")
        for path in written:
            print(f"  {path}")
    else:
        print("No files written; existing setup left in place.")
    _print_setup_next_steps(
        root,
        include_native=include_native,
        write_mcp=write_mcp,
        write_instructions=write_instructions,
        include_openrouter=include_openrouter,
        include_local=include_local,
    )
    return 0


def _auto_setup_preset() -> str:
    native_names = ("claude", "codex", "gemini")
    native_count = sum(1 for name in native_names if shutil.which(name))
    if native_count >= 2:
        return "tri-cli"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    found = ", ".join(name for name in native_names if shutil.which(name)) or "none"
    raise SystemExit(
        "Auto setup could not find a usable default council route. "
        f"Found native CLIs: {found}. "
        "Install at least two of claude/codex/gemini, or set OPENROUTER_API_KEY "
        "in your shell, .env, .env.local, or .llm-council.env and rerun setup. "
        "Advanced users who intentionally want to stage an incomplete config "
        "can choose an explicit preset with --allow-incomplete."
    )


def _detect_setup_routes() -> dict[str, object]:
    native_names = ("claude", "codex", "gemini")
    native_paths = {name: shutil.which(name) for name in native_names}
    native_count = sum(1 for path in native_paths.values() if path)
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    ollama_path = shutil.which("ollama")
    return {
        "native_paths": native_paths,
        "native_count": native_count,
        "has_openrouter": has_openrouter,
        "ollama_path": ollama_path,
        "auto": _auto_setup_preset_or_none(),
    }


def _auto_setup_preset_or_none() -> str | None:
    try:
        return _auto_setup_preset()
    except SystemExit:
        return None


def _preset_status(preset: str, routes: dict[str, object]) -> tuple[str, str]:
    native_count = int(routes["native_count"])
    has_openrouter = bool(routes["has_openrouter"])
    has_ollama = bool(routes["ollama_path"])
    if preset == "auto":
        selected = routes["auto"]
        if selected:
            return "recommended", f"would select `{selected}`"
        return "blocked", "needs at least two native CLIs or OPENROUTER_API_KEY"
    if preset == "tri-cli":
        if native_count >= 2:
            return "available", "uses installed Claude/Codex/Gemini CLI accounts"
        return "blocked", "needs at least two of claude/codex/gemini"
    if preset == "openrouter":
        if has_openrouter:
            return "available", "uses hosted OpenRouter reviewers"
        return "blocked", "needs OPENROUTER_API_KEY"
    if preset == "tri-cli-openrouter":
        if native_count >= 2 and has_openrouter:
            return "available", "uses native CLI peers plus hosted reviewers"
        missing = []
        if native_count < 2:
            missing.append("two native CLIs")
        if not has_openrouter:
            missing.append("OPENROUTER_API_KEY")
        return "blocked", "needs " + " and ".join(missing)
    if preset == "local-private":
        if native_count >= 2 and has_ollama:
            return "available", "uses native CLI peers plus local Ollama"
        missing = []
        if native_count < 2:
            missing.append("two native CLIs")
        if not has_ollama:
            missing.append("ollama")
        return "blocked", "needs " + " and ".join(missing)
    if preset == "all":
        if native_count >= 2 and has_openrouter and has_ollama:
            return "available", "writes native, hosted, and local presets"
        missing = []
        if native_count < 2:
            missing.append("two native CLIs")
        if not has_openrouter:
            missing.append("OPENROUTER_API_KEY")
        if not has_ollama:
            missing.append("ollama")
        return "blocked", "needs " + " and ".join(missing)
    return "unknown", ""


def _guard_setup_preset(preset: str, args: argparse.Namespace) -> None:
    if getattr(args, "allow_incomplete", False):
        return
    status, detail = _preset_status(preset, _detect_setup_routes())
    if status != "blocked":
        return
    message = (
        f"Preset `{preset}` is not usable in this environment: {detail}. "
        "Run `llm-council setup --plan` to see available presets."
    )
    if getattr(args, "yes", False):
        raise SystemExit(
            message
            + " To write this config anyway, rerun with `--allow-incomplete`."
        )
    if not _confirm(message + " Write it anyway?", default=False):
        raise SystemExit("Setup cancelled.")


def _print_setup_plan(root: Path) -> None:
    routes = _detect_setup_routes()
    native_paths = routes["native_paths"]
    assert isinstance(native_paths, dict)
    print("LLM Council setup plan")
    print(f"Project root: {root}")
    print()
    print("Detected:")
    for name in ("claude", "codex", "gemini"):
        print(f"  {name:8} {native_paths.get(name) or 'not found'}")
    print(f"  openrouter {'OPENROUTER_API_KEY set' if routes['has_openrouter'] else 'OPENROUTER_API_KEY not set'}")
    print(f"  ollama   {routes['ollama_path'] or 'not found'}")
    print()
    print("Preset choices:")
    for preset in (
        "auto",
        "tri-cli",
        "openrouter",
        "tri-cli-openrouter",
        "local-private",
        "all",
    ):
        status, detail = _preset_status(preset, routes)
        print(f"  {preset:19} {status:11} {detail}")
    print()
    print("Agent installers: show this plan to the user and ask which preset to write.")
    print("Do not choose a blocked preset unless the user is deliberately preparing config for later.")
    print("Then run: llm-council setup --yes --preset <chosen-preset>")


def _prompt_setup_preset(root: Path) -> str:
    _print_setup_plan(root)
    default = _auto_setup_preset_or_none() or "openrouter"
    valid = {
        "auto",
        "tri-cli",
        "openrouter",
        "tri-cli-openrouter",
        "local-private",
        "all",
    }
    answer = input(f"Choose setup preset [{default}]: ").strip()
    if not answer:
        answer = default
    if answer not in valid:
        raise SystemExit(
            f"Unknown preset '{answer}'. Choose one of: {', '.join(sorted(valid))}."
        )
    if answer == "auto":
        selected = _auto_setup_preset()
        print(f"Auto preset selected: {selected}")
        return selected
    return answer


def _print_setup_next_steps(
    root: Path,
    *,
    include_native: bool,
    write_mcp: bool,
    write_instructions: bool,
    include_openrouter: bool,
    include_local: bool,
) -> None:
    print()
    print("Next steps:")
    if write_instructions:
        print(
            "  1. For each CLI you use, append the full contents of "
            f"{root / '.llm-council/instructions/claude.md'} to CLAUDE.md."
        )
        print(
            "     Append the full contents of "
            f"{root / '.llm-council/instructions/codex.md'} to AGENTS.md."
        )
        print(
            "     Append the full contents of "
            f"{root / '.llm-council/instructions/gemini.md'} to GEMINI.md."
        )
    else:
        print("  1. Add council instructions to CLAUDE.md, AGENTS.md, and GEMINI.md.")
    if write_mcp:
        print("  2. Restart the CLI session(s) you use so MCP reloads.")
        print(
            "     `.mcp.json` contains local absolute paths; setup adds it to "
            ".gitignore unless it is already ignored."
        )
        print(
            "     If `.mcp.json` was already committed, use "
            "`git rm --cached .mcp.json` after confirming it should stay local."
        )
    else:
        print("  2. Add the llm-council MCP server to your MCP config, then restart CLIs.")
    print("  3. Run `llm-council doctor` from the project root.")
    if include_openrouter:
        print(
            "  4. Run `llm-council estimate --mode review-cheap \"Review this\"` "
            "before paid hosted calls."
        )

    warnings: list[str] = []
    if include_native:
        for name in ("claude", "codex", "gemini"):
            if shutil.which(name) is None:
                warnings.append(f"{name} was not found on PATH; native CLI modes need it.")
    if include_openrouter and not os.environ.get("OPENROUTER_API_KEY"):
        warnings.append(
            "OPENROUTER_API_KEY is not exported; hosted OpenRouter modes need it."
        )
    if include_local and shutil.which("ollama") is None:
        warnings.append("ollama was not found on PATH; private-local mode needs it.")
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def cmd_doctor(args: argparse.Namespace) -> int:
    env_start = Path(args.config).expanduser() if args.config else Path.cwd()
    load_project_env(env_start)
    config = load_config(args.config)
    checks = check_environment(
        config,
        probe_openrouter=args.probe_openrouter,
        probe_ollama=args.probe_ollama,
    )
    check_update = bool(getattr(args, "check_update", False))
    if args.json:
        if check_update:
            result = {"checks": checks_to_dict(checks)}
            result["update"] = check_for_update(__version__).to_dict()
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(checks_to_dict(checks), indent=2))
    else:
        for check in checks:
            status = "ok" if check.ok else "missing"
            print(f"{status:8} {check.name:24} {check.detail}")
        if check_update:
            _print_update_status(check_for_update(__version__))
    required_names = {"python:mcp"}
    default_mode = config.get("defaults", {}).get("mode", "quick")
    try:
        default_participants = select_participants(config, default_mode, current=None)
    except ValueError:
        default_participants = []
    for name in default_participants:
        participant = config.get("participants", {}).get(name, {})
        if participant.get("type") == "cli":
            required_names.add(f"cli:{name}")
        elif participant.get("type") == "openrouter":
            api_key_env = participant.get("api_key_env") or "OPENROUTER_API_KEY"
            required_names.add(f"env:{api_key_env}")
        elif participant.get("type") == "ollama":
            required_names.add("cli:ollama")
    required = [check for check in checks if check.name in required_names]
    if args.probe_openrouter:
        required.extend(
            check for check in checks if check.name == "probe:openrouter"
        )
    if args.probe_ollama:
        required.extend(check for check in checks if check.name == "probe:ollama")
    return 0 if all(check.ok for check in required) else 1


def cmd_check_update(args: argparse.Namespace) -> int:
    status = check_for_update(__version__)
    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        _print_update_status(status)
    return 0 if status.error is None else 1


def _print_update_status(status) -> None:
    print(f"version: {status.current_version}")
    if status.error:
        print(f"update_check: unavailable ({status.error})")
        print(f"update_command: {status.install_command}")
        return
    print(f"latest: {status.latest_version}")
    if status.update_available:
        print("update_available: true")
        print(f"update_command: {status.install_command}")
    else:
        print("update_available: false")


def cmd_recommend(args: argparse.Namespace) -> int:
    task = _question_from_args(args.task)
    use, mode, reason = should_use_council(
        task,
        failed_attempts=args.failed_attempts,
        files_touched=args.files_touched,
        risk=args.risk,
    )
    result = {"use_council": use, "mode": mode, "reason": reason}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"use_council: {str(use).lower()}")
        print(f"mode: {mode}")
        print(f"reason: {reason}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    question = _question_from_args(args.question)
    load_project_env(cwd)
    try:
        config = load_config(args.config or find_config(cwd), search=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    mode = args.mode or config.get("defaults", {}).get("mode", "quick")
    current = args.current or detect_current_agent()
    stdin_text = sys.stdin.read() if args.stdin else None
    try:
        estimate = estimate_council(
            config=config,
            cwd=cwd,
            question=question,
            mode=mode,
            current=current,
            explicit=parse_csv(args.participants),
            include=parse_csv(args.include),
            origin_policy=args.origin_policy,
            context_paths=args.context,
            include_diff=args.diff,
            stdin_text=stdin_text,
            allow_outside_cwd=args.allow_outside_cwd,
            deliberate=args.deliberate,
            max_rounds=args.max_rounds,
            completion_tokens=args.completion_tokens,
            openrouter_models=args.openrouter_model,
            use_cache=not args.no_cache,
            image_paths=args.image or None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(estimate, indent=2))
    else:
        _print_estimate(estimate)
    return 0


def _print_estimate(estimate: dict) -> None:
    participants = list(estimate.get("participants") or [])
    extras = [
        f"openrouter:{model}"
        for model in estimate.get("extra_openrouter_models") or []
    ]
    print(f"mode: {estimate['mode']}")
    print(f"current: {estimate.get('current') or 'unknown'}")
    print("participants: " + ", ".join(participants + extras))
    print(f"prompt_chars: {estimate['prompt_chars']}")
    print(f"estimated_prompt_tokens: {estimate['estimated_prompt_tokens']}")
    print(f"budgeted_rounds: {estimate['budgeted_rounds']}")
    print(
        "completion_tokens_assumed_each: "
        f"{estimate['completion_tokens_assumed_each']}"
    )
    print()
    print(
        f"{'participant/model':44} {'type':10} {'in/1M':>9} "
        f"{'out/1M':>9} {'input':>10} {'output':>10} {'total':>10}"
    )
    for row in estimate["rows"]:
        label = row["name"] if row["model"] == "cli default" else row["model"]
        print(
            f"{label[:44]:44} "
            f"{row['type'][:10]:10} "
            f"{_fmt_cost(row['input_per_million']):>9} "
            f"{_fmt_cost(row['output_per_million']):>9} "
            f"{_fmt_usd(row['estimated_input_cost_usd']):>10} "
            f"{_fmt_usd(row['estimated_output_cost_usd']):>10} "
            f"{_fmt_usd(row['estimated_total_cost_usd']):>10}"
        )
    print()
    print(f"known_total_usd: {_fmt_usd(estimate['known_total_usd'])}")
    if estimate.get("unknown_cost_rows"):
        print("unknown_cost_rows: " + ", ".join(estimate["unknown_cost_rows"]))
    if estimate.get("notes"):
        print("notes:")
        for note in estimate["notes"]:
            print(f"  - {note}")


def cmd_last(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    path = latest_transcript(out_dir, suffix=".json" if args.json_file else ".md")
    if path is None:
        raise SystemExit(f"No council transcripts found in {out_dir}")
    if args.path_only:
        print(path)
    else:
        print(path.read_text(encoding="utf-8"))
    return 0


def _transcript_dir(cwd: Path, config: dict) -> Path:
    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    return out_dir if out_dir.is_absolute() else cwd / out_dir


def cmd_stats(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    out_dir = _transcript_dir(cwd, config)
    if args.since is not None and args.since <= 0:
        raise SystemExit("--since must be a positive integer")
    stats = compute_stats(
        out_dir,
        participant=args.participant,
        since_days=args.since,
    )
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_stats_text(stats))
    return 0


def cmd_transcripts(args: argparse.Namespace) -> int:
    if not args.transcripts_command:
        raise SystemExit("transcripts subcommand is required")
    cwd = Path(args.cwd).resolve()
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    out_dir = _transcript_dir(cwd, config)

    if args.transcripts_command == "list":
        records = transcript_records(out_dir)
        records = records[-args.limit :] if args.limit > 0 else []
        if args.json:
            print(json.dumps(records, indent=2))
        else:
            for record in records:
                print(
                    f"{record['ok']}/{record['total']} "
                    f"${record['cost_usd']:.6f} "
                    f"{record['mode'] or '-':10} "
                    f"{record['question'][:80]} "
                    f"({record['markdown']})"
                )
        return 0

    if args.transcripts_command == "show":
        if args.path:
            path = Path(args.path)
            if not path.is_absolute():
                path = cwd / path
        else:
            path = latest_transcript(out_dir, suffix=".json" if args.json_file else ".md")
            if path is None:
                raise SystemExit(f"No council transcripts found in {out_dir}")
        print(path.read_text(encoding="utf-8"))
        return 0

    if args.transcripts_command == "summary":
        records = transcript_records(out_dir)
        runs = len(records)
        tokens = sum(record["tokens"] for record in records)
        cost = sum(record["cost_usd"] for record in records)
        successes = sum(record["ok"] for record in records)
        participants = sum(record["total"] for record in records)
        print(f"runs: {runs}")
        print(f"participant_successes: {successes}/{participants}")
        print(f"tokens: {tokens}")
        print(f"cost_usd: ${cost:.6f}")
        return 0

    raise SystemExit(f"Unknown transcripts subcommand: {args.transcripts_command}")


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    return f"${value:.3f}"


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    if value < 0.001:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _print_progress_event(event: dict) -> None:
    kind = event.get("event")
    participant = event.get("participant")
    round_label = f"round {event.get('round')}" if event.get("round") else "round ?"
    if kind == "participant_start":
        print(f"- {participant}: starting {round_label}", flush=True)
        return
    if kind == "participant_slow":
        elapsed = float(event.get("elapsed_seconds") or 0)
        timeout = float(event.get("timeout_seconds") or 0)
        print(
            f"- {participant}: still running after {elapsed:.1f}s "
            f"(hard timeout at {timeout:.0f}s)",
            flush=True,
        )
        return
    if kind == "participant_finish":
        status = event.get("status") or ("ok" if event.get("ok") else "error")
        details = [f"{float(event.get('elapsed_seconds') or 0):.1f}s"]
        if event.get("total_tokens") is not None:
            details.append(f"{event['total_tokens']} tokens")
        if event.get("cost_usd") is not None:
            details.append(f"${float(event['cost_usd']):.6f}")
        if event.get("from_cache"):
            details.append("cached")
        print(f"- {participant}: {status} {round_label} ({'; '.join(details)})", flush=True)
        if event.get("error"):
            print(f"  {event['error']}", flush=True)
        return
    if kind == "deliberation_skip_participants":
        skipped = ", ".join(event.get("skipped") or [])
        print(
            f"Deliberation: skipping {skipped} from round {event.get('round')} "
            f"({event.get('reason')})",
            flush=True,
        )
        return
    if kind == "deliberation_pending":
        print(f"Deliberation: disagreement detected; starting round {event.get('round')}", flush=True)
        return
    if kind == "deliberation_round_start":
        print(f"Deliberation: running round {event.get('round')}", flush=True)
        return
    if kind == "deliberation_skip":
        print(f"Deliberation: skipped ({event.get('reason')})", flush=True)
        return
    if kind == "deliberation_finish":
        print(f"Deliberation: {event.get('status')} after {event.get('rounds')} rounds", flush=True)
        return
    if kind == "degraded_consensus":
        labeled = event.get("labeled_quorum")
        threshold = event.get("min_quorum")
        print(
            f"Quorum: {labeled} of {threshold} required peers labeled — DEGRADED",
            flush=True,
        )
        return
    if kind == "images_skipped":
        print(
            f"- {participant}: image attachments skipped ({event.get('reason')}; "
            f"{event.get('image_count')} image(s) referenced as text only)",
            flush=True,
        )


def cmd_models(args: argparse.Namespace) -> int:
    if args.models_command != "openrouter":
        raise SystemExit("models subcommand is required")
    models = fetch_openrouter_models(use_cache=not args.no_cache)
    if args.filter:
        needle = args.filter.lower()
        models = [
            model
            for model in models
            if needle in model["id"].lower() or needle in model["name"].lower()
        ]
    if args.origin:
        prefix = {"us": "US /", "china": "China /", "unknown": "Unknown"}[args.origin]
        models = [model for model in models if str(model["origin"]).startswith(prefix)]
    models = models[: max(args.limit, 0)]
    if args.json:
        print(json.dumps(models, indent=2))
        return 0

    print(f"{'model':44} {'origin':24} {'ctx':>9} {'in/1M':>9} {'out/1M':>9}")
    for model in models:
        print(
            f"{model['id'][:44]:44} "
            f"{model['origin'][:24]:24} "
            f"{str(model['context_length'] or 'n/a'):>9} "
            f"{_fmt_cost(model['input_per_million']):>9} "
            f"{_fmt_cost(model['output_per_million']):>9}"
        )
    return 0


async def cmd_run_async(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    question = _question_from_args(args.question)
    load_project_env(cwd)
    try:
        config = load_config(args.config or find_config(cwd), search=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    mode = args.mode or config.get("defaults", {}).get("mode", "quick")
    current = args.current or detect_current_agent()
    explicit = parse_csv(args.participants)
    include = parse_csv(args.include)
    try:
        participants = select_participants(
            config,
            mode,
            current,
            explicit=explicit,
            include=include,
            origin_policy=args.origin_policy,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    stdin_text = sys.stdin.read() if args.stdin else None
    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    parent_run_id: str | None = None
    prior_context: str | None = None
    continue_id = getattr(args, "continue_id", None)
    if continue_id:
        try:
            normalize_run_id(continue_id)
            prior_transcript = find_transcript_by_id(out_dir, continue_id)
            prior_path = prior_transcript.get("_path")
            parent_run_id = (
                Path(str(prior_path)).stem
                if prior_path
                else normalize_run_id(continue_id)
            )
            prior_context = format_prior_council_context(
                prior_transcript, run_id=parent_run_id
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    try:
        image_manifest = (
            build_image_manifest(
                args.image, cwd=cwd, allow_outside_cwd=args.allow_outside_cwd
            )
            if args.image
            else []
        )
        if image_manifest:
            violations = image_attachment_violations(image_manifest)
            if violations:
                raise ValueError(
                    "Image attachment budget exceeded: "
                    + ", ".join(
                        f"{v['limit']} {v.get('actual')} > {v.get('maximum')}"
                        for v in violations
                    )
                )
        chunk_events: list[dict] = []

        def _record_chunk_event(event: dict) -> None:
            chunk_events.append(event)
            dropped_files = event.get("dropped_files") or []
            file_note = (
                f"; dropped files: {', '.join(dropped_files)}" if dropped_files else ""
            )
            print(
                f"warning: diff chunking applied (strategy={event.get('strategy')}, "
                f"original={event.get('original_chars')} chars, "
                f"chunked={event.get('chunked_chars')} chars, "
                f"dropped={event.get('dropped_chars')} chars{file_note})",
                file=sys.stderr,
                flush=True,
            )

        prompt = build_prompt(
            question,
            mode=mode,
            cwd=cwd,
            context_paths=args.context,
            include_diff=args.diff,
            stdin_text=stdin_text,
            allow_outside_cwd=args.allow_outside_cwd,
            max_prompt_chars=config.get("defaults", {}).get("max_prompt_chars")
            or MAX_PROMPT_CHARS,
            image_manifest=image_manifest or None,
            prior_context=prior_context,
            chunk_strategy=getattr(args, "chunk_strategy", "fail"),
            chunk_progress=_record_chunk_event,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        print(f"mode: {mode}")
        print(f"current: {current or 'unknown'}")
        print("participants: " + ", ".join(participants))
        print(f"prompt_chars: {len(prompt)}")
        return 0

    mode_cfg = config.get("modes", {}).get(mode, {})
    transparent = bool(args.transparent or config.get("defaults", {}).get("transparent"))
    deliberate = bool(args.deliberate or mode_cfg.get("deliberate"))
    max_rounds = int(
        args.max_rounds
        or mode_cfg.get("max_rounds")
        or config.get("defaults", {}).get("max_deliberation_rounds")
        or 2
    )
    min_quorum_value: int | None
    if args.min_quorum is not None:
        min_quorum_value = int(args.min_quorum)
    elif mode_cfg.get("min_quorum") is not None:
        min_quorum_value = int(mode_cfg["min_quorum"])
    else:
        min_quorum_value = None
    participant_cfg = config.get("participants", {})
    if not args.json:
        print(
            f"Council starting: mode={mode}, current={current or 'unknown'}, "
            f"participants={', '.join(participants)}, prompt_chars={len(prompt)}",
            flush=True,
        )
        if deliberate:
            print(f"Deliberation enabled: max_rounds={max_rounds}", flush=True)
    results, metadata = await execute_council(
        participants,
        participant_cfg,
        prompt,
        cwd,
        config,
        deliberate=deliberate,
        max_rounds=max_rounds,
        progress=None if args.json else _print_progress_event,
        image_manifest=image_manifest or None,
        min_quorum=min_quorum_value,
        mode=mode,
        cache_mode=getattr(args, "cache_mode", "on"),
    )
    if image_manifest:
        metadata["images"] = [
            {
                "path": entry.get("relative_path") or entry.get("path"),
                "mime": entry.get("mime"),
                "size": entry.get("size"),
                "sha256": entry.get("sha256"),
            }
            for entry in image_manifest
        ]
    if chunk_events:
        latest = chunk_events[-1]
        metadata["diff_chunking"] = {
            "strategy": latest.get("strategy"),
            "original_chars": latest.get("original_chars"),
            "chunked_chars": latest.get("chunked_chars"),
            "dropped_chars": latest.get("dropped_chars"),
            "dropped_files": list(latest.get("dropped_files") or []),
        }
        progress_events = metadata.setdefault("progress_events", [])
        if isinstance(progress_events, list):
            progress_events.append(latest)

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
        parent_run_id=parent_run_id,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "transcript": str(md_path),
                    "json": str(json_path),
                    "metadata": metadata,
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
                },
                indent=2,
            )
        )
    else:
        from llm_council.adapters import is_timeout_error

        ok = sum(1 for result in results if result.ok)
        timed_out = [result for result in results if is_timeout_error(result.error)]
        print(f"Council complete: {ok}/{len(results)} participants succeeded")
        if metadata.get("deliberated"):
            print("Deliberation: second round ran after disagreement detection")
        for result in results:
            if result.ok:
                status = "ok"
            elif is_timeout_error(result.error):
                status = "timeout"
            else:
                status = "error"
            print(f"- {result.name}: {status} ({result.elapsed_seconds:.1f}s)")
            if transparent:
                details = []
                if result.total_tokens is not None:
                    details.append(f"{result.total_tokens} tokens")
                if result.cost_usd is not None:
                    details.append(f"${result.cost_usd:.6f}")
                if details:
                    print(f"  {'; '.join(details)}")
            if not result.ok:
                print(f"  {result.error}")
        if timed_out:
            names = ", ".join(
                sorted({r.name.split(":round")[0] for r in timed_out})
            )
            print(
                f"Note: {names} timed out. Increase `participants.<name>.timeout` "
                "in `.llm-council.yaml` for slower models, or shorten the prompt."
            )
        print(f"Transcript: {md_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return asyncio.run(cmd_run_async(args))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "setup":
        return cmd_setup(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "check-update":
        return cmd_check_update(args)
    if args.command == "recommend":
        return cmd_recommend(args)
    if args.command == "estimate":
        return cmd_estimate(args)
    if args.command == "last":
        return cmd_last(args)
    if args.command == "transcripts":
        return cmd_transcripts(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "models":
        return cmd_models(args)
    if args.command == "mcp-server":
        from llm_council.mcp_server import main as mcp_main

        return mcp_main()
    parser.print_help()
    return 2
