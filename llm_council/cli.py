"""Command line interface for llm-council."""

from __future__ import annotations

import argparse
import asyncio
import json
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
from llm_council.context import MAX_PROMPT_CHARS, build_prompt
from llm_council.doctor import check_environment, checks_to_dict
from llm_council.env import load_project_env
from llm_council.model_catalog import fetch_openrouter_models
from llm_council.orchestrator import execute_council
from llm_council.policy import should_use_council
from llm_council.setup_wizard import write_setup_files
from llm_council.transcript import (
    latest_transcript,
    transcript_paths,
    transcript_records,
    write_transcript,
)


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
        "--allow-outside-cwd",
        action="store_true",
        help="Allow --context files outside the working directory",
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

    sub.add_parser("list", help="List participants and modes")
    init = sub.add_parser("init", help="Write an example project config")
    init.add_argument("--path", default=".llm-council.yaml")

    setup = sub.add_parser("setup", help="Walk through or write project setup")
    setup.add_argument("--root", default=".", help="Project root")
    setup.add_argument(
        "--preset",
        choices=["tri-cli", "tri-cli-openrouter", "local-private", "all"],
        default="all",
    )
    setup.add_argument("--yes", action="store_true", help="Non-interactive defaults")
    setup.add_argument("--force", action="store_true", help="Overwrite existing files")
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

    recommend = sub.add_parser("recommend", help="Recommend whether to use council")
    recommend.add_argument("task", nargs="*", help="Task description")
    recommend.add_argument("--failed-attempts", type=int, default=0)
    recommend.add_argument("--files-touched", type=int, default=0)
    recommend.add_argument(
        "--risk", choices=["low", "medium", "high"], default="medium"
    )
    recommend.add_argument("--json", action="store_true", help="Print JSON")

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
    include_openrouter = args.preset in {"tri-cli-openrouter", "all"}
    include_local = args.preset in {"local-private", "all"}

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
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    env_start = Path(args.config).expanduser() if args.config else Path.cwd()
    load_project_env(env_start)
    config = load_config(args.config)
    checks = check_environment(
        config,
        probe_openrouter=args.probe_openrouter,
        probe_ollama=args.probe_ollama,
    )
    if args.json:
        print(json.dumps(checks_to_dict(checks), indent=2))
    else:
        for check in checks:
            status = "ok" if check.ok else "missing"
            print(f"{status:8} {check.name:24} {check.detail}")
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


def cmd_last(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    load_project_env(cwd)
    config = load_config(find_config(cwd))
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


def cmd_transcripts(args: argparse.Namespace) -> int:
    if not args.transcripts_command:
        raise SystemExit("transcripts subcommand is required")
    cwd = Path(args.cwd).resolve()
    load_project_env(cwd)
    config = load_config(find_config(cwd))
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


def _print_progress_event(event: dict) -> None:
    kind = event.get("event")
    participant = event.get("participant")
    round_label = f"round {event.get('round')}" if event.get("round") else "round ?"
    if kind == "participant_start":
        print(f"- {participant}: starting {round_label}", flush=True)
        return
    if kind == "participant_finish":
        status = event.get("status") or ("ok" if event.get("ok") else "error")
        details = [f"{float(event.get('elapsed_seconds') or 0):.1f}s"]
        if event.get("total_tokens") is not None:
            details.append(f"{event['total_tokens']} tokens")
        if event.get("cost_usd") is not None:
            details.append(f"${float(event['cost_usd']):.6f}")
        print(f"- {participant}: {status} {round_label} ({'; '.join(details)})", flush=True)
        if event.get("error"):
            print(f"  {event['error']}", flush=True)
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
    config = load_config(args.config or find_config(cwd))
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
    try:
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
        ok = sum(1 for result in results if result.ok)
        print(f"Council complete: {ok}/{len(results)} participants succeeded")
        if metadata.get("deliberated"):
            print("Deliberation: second round ran after disagreement detection")
        for result in results:
            status = "ok" if result.ok else "error"
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
    if args.command == "recommend":
        return cmd_recommend(args)
    if args.command == "last":
        return cmd_last(args)
    if args.command == "transcripts":
        return cmd_transcripts(args)
    if args.command == "models":
        return cmd_models(args)
    if args.command == "mcp-server":
        from llm_council.mcp_server import main as mcp_main

        return mcp_main()
    parser.print_help()
    return 2
