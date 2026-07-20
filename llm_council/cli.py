"""Command line interface for llm-council."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from llm_council import __version__
from llm_council.config import (
    apply_tier_override,
    canonical_mode_name,
    config_warnings,
    detect_current_agent,
    find_config,
    load_config,
    parse_csv,
    participant_api_key_env,
    resolve_config_data,
    select_participants,
)
from llm_council.adapters import classify_error
from llm_council import budget
from llm_council.budget import image_attachment_violations
from llm_council import display
from llm_council.context import (
    MAX_PROMPT_CHARS,
    apply_per_peer_directives,
    build_image_manifest,
    build_prompt,
)
from llm_council.doctor import (
    Check,
    check_environment,
    checks_to_dict,
    probe_local_openai,
)
from llm_council.env import load_project_env
from llm_council.estimate import (
    CLI_DEFAULT_MODEL_LABEL,
    deliberation_prompt_char_bounds,
    estimate_council,
)
from llm_council.model_catalog import (
    fetch_openrouter_models,
    refresh_openrouter_cache,
)
from llm_council.orchestrator import execute_council
from llm_council.policy import should_use_council
from llm_council.setup_wizard import write_setup_files
from llm_council.stats import compute_stats, format_stats_text
from llm_council.transcript import (
    continuation_depth_limit_error,
    find_transcript_by_id,
    final_round_results,
    format_prior_council_context,
    latest_transcript,
    inspect_transcript_permissions,
    normalize_run_id,
    transcript_dir,
    transcript_paths,
    transcript_records,
    write_transcript,
)
from llm_council.update_check import (
    check_for_update,
    hydrate_nag_cache_from_status,
    maybe_print_update_nag,
)


_SETUP_PRESETS = (
    "auto",
    "tri-cli",
    "openrouter",
    "tri-cli-openrouter",
    "private-local",
    "all",
)
_SETUP_PRESET_ALIASES = {"local-private": "private-local"}


def _setup_preset_arg(value: str) -> str:
    canonical = _SETUP_PRESET_ALIASES.get(value, value)
    if canonical not in _SETUP_PRESETS:
        raise argparse.ArgumentTypeError(
            f"unknown preset {value!r}; choose one of: "
            + ", ".join(_SETUP_PRESETS)
        )
    return canonical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-council",
        description="Lightweight read-only multi-agent council for coding projects.",
    )
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a council prompt")
    run.add_argument("question", nargs="*", help="Question or prompt")
    run.add_argument(
        "--question",
        dest="question_flag",
        default=None,
        help=(
            "Question text. Alias for the positional argument; useful when "
            "the question contains characters that make positional parsing "
            "awkward, or when matching the MCP `council_run` arg name."
        ),
    )
    run.add_argument("--config", help="Path to config YAML")
    run.add_argument("--mode", default=None, help="Council mode")
    run.add_argument("--current", choices=["claude", "codex", "antigravity"])
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
    run.add_argument(
        "--synthesize",
        action="store_true",
        help=(
            "After peers respond, invoke the configured synthesis chair "
            "(defaults.synthesizer) to produce a decision memo. Chair "
            "output is metadata; the headline recommendation still comes "
            "from peer votes. Costs one extra participant call."
        ),
    )
    run.add_argument(
        "--no-require-sections",
        dest="require_sections",
        action="store_false",
        default=None,
        help=(
            "Disable section-coverage validation. By default, when the prompt "
            "contains `PART N — TITLE (REQUIRED)` markers, peers must reference "
            "each in their responses or fail with error_kind=incomplete_response."
        ),
    )
    run.add_argument(
        "--strict-evidence",
        dest="strict_evidence",
        action="store_true",
        default=None,
        help=(
            "Require every EVIDENCE bullet to carry one of "
            "[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE] tags. "
            "Untagged entries trigger one repair retry then "
            "error_kind=untagged_evidence. Off by default in v0.7."
        ),
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
        "--allow-privacy-downgrade",
        action="store_true",
        help=(
            "Explicitly allow a private-local continuation to send prior "
            "question/summary context to non-local participants. Refused by "
            "default."
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
        "--stance",
        action="append",
        default=[],
        metavar="PEER=for|against|neutral",
        help=(
            "Override or extend stance assignment for one peer. Repeatable: "
            "`--stance claude=for --stance codex=against`. Adds an "
            "ethical-override clause to the peer's prompt to attack "
            "groupthink. Useful with `--mode consensus` or any mode where "
            "you want to inject roles without forking the mode config."
        ),
    )
    run.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help=(
            "Hard ceiling on the council's pre-flight estimated cost in USD. "
            "If the estimate exceeds this, the run is refused before any "
            "subprocess or HTTP call. Free/local participants count as $0; "
            "hosted or non-local participants with unknown pricing are refused "
            "when this cap is set."
        ),
    )
    run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Hard ceiling on estimated prompt+completion tokens across all "
            "participants and budgeted rounds. Refuses the run if exceeded."
        ),
    )
    run.add_argument(
        "--cost-warn-usd",
        type=float,
        default=None,
        help=(
            "Soft, advisory-only cost-warning threshold in USD. When the "
            "pre-flight estimate is at or above this value the run still "
            "proceeds, but a non-fatal warning is printed to stderr and "
            "stamped into the transcript metadata. Never blocks — use "
            "--max-cost-usd for a hard ceiling. Overrides defaults.cost_warn_usd."
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
    run.add_argument(
        "--tier",
        default=None,
        help=(
            "Swap participant models per `defaults.tiers.<name>` in "
            ".llm-council.yaml (e.g. `--tier deep` for top-end thinking "
            "models, `--tier fast` for budget models). Pin the tier->model "
            "map yourself; missing peers in the map keep their default "
            "model so a tier can swap a subset."
        ),
    )
    run.add_argument(
        "--independent-review",
        dest="independent_review",
        action="store_true",
        default=False,
        help=(
            "On a --continue run, suppress the prior council's per-peer "
            "labels/rationales so this round forms its verdict independently "
            "(advisory; prior_context is simply not injected). No effect "
            "without --continue or when no prior context was produced."
        ),
    )
    run.add_argument(
        "--open",
        action="store_true",
        default=False,
        help="Automatically open the HTML transcript in the default browser at the end of the run.",
    )

    sub.add_parser("list", help="List participants and modes")
    init = sub.add_parser("init", help="Write an example project config")
    init.add_argument("--path", default=".llm-council.yaml")

    setup = sub.add_parser("setup", help="Walk through or write project setup")
    setup.add_argument("--root", default=".", help="Project root")
    setup.add_argument(
        "--preset",
        type=_setup_preset_arg,
        metavar="{" + ",".join(_SETUP_PRESETS) + "}",
        default="auto",
        help=(
            "Setup scope: auto detects local CLIs/OpenRouter, "
            "tri-cli for Claude/Codex plus a Gemini-family CLI, openrouter for hosted-only, "
            "tri-cli-openrouter for native CLIs plus hosted models, "
            "private-local for offline Ollama-only review, all for every participant route"
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
    setup.add_argument(
        "--probe-local",
        action="store_true",
        help=(
            "Scan well-known local OpenAI-compatible ports during interactive "
            "setup and offer to scaffold participant blocks for any reachable "
            "endpoint (vLLM, sglang, LM Studio, llama.cpp --api, MLX). "
            "Ignored under --yes (probing requires interactive prompts for "
            "origin and family)."
        ),
    )

    hook = sub.add_parser("install-hook", help="Install LLM Council git hooks")
    hook.add_argument("--root", default=".", help="Project root (defaults to current dir)")
    hook.add_argument(
        "--hook-type",
        choices=["pre-commit", "pre-push"],
        default="pre-commit",
        help="The git hook type to install (default: pre-commit)",
    )
    hook.add_argument(
        "--mode",
        default="consensus",
        help="The mode to run the hook in (default: consensus)",
    )
    hook.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing hook instead of refusing to overwrite it",
    )

    cfg_parser = sub.add_parser("config", help="Get or set configuration values in .llm-council.yaml")
    cfg_sub = cfg_parser.add_subparsers(dest="config_command")
    
    cfg_get = cfg_sub.add_parser("get", help="Get a configuration value")
    cfg_get.add_argument("key", help="The dot-notation key to retrieve (e.g., defaults.auto_open_browser)")
    cfg_get.add_argument("--cwd", default=".", help="Working directory")
    
    cfg_set = cfg_sub.add_parser("set", help="Set a configuration value")
    cfg_set.add_argument("key", help="The dot-notation key to set (e.g., defaults.auto_open_browser)")
    cfg_set.add_argument("value", help="The value to set (automatically parsed as boolean, int, float, or string)")
    cfg_set.add_argument("--cwd", default=".", help="Working directory")

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
        "--probe-native",
        action="store_true",
        help="Invoke configured native CLIs with a bounded readiness probe",
    )
    doctor.add_argument(
        "--repair-transcript-permissions",
        action="store_true",
        help="Tighten owned transcript directories/files to 0700/0600",
    )
    doctor.add_argument(
        "--probe-local-openai",
        nargs="?",
        const="",
        default=None,
        metavar="BASE_URL",
        help=(
            "Probe local OpenAI-compatible inference servers (vLLM, LM "
            "Studio, llama.cpp --api, sglang, TGI, MLX, Ollama /v1). With "
            "no value, scans well-known ports on 127.0.0.1 (8000, 1234, "
            "8080, 11434, 5000). With a URL, probes that endpoint only."
        ),
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
    estimate.add_argument("--current", choices=["claude", "codex", "antigravity"])
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
    estimate.add_argument(
        "--chunk-strategy",
        choices=["fail", "head", "tail", "hash-aware"],
        default="fail",
        help=(
            "How to fit an oversized git diff into the prompt budget; matches "
            "the run command."
        ),
    )
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
    estimate.add_argument(
        "--tier",
        default=None,
        help=(
            "Swap participant models per `defaults.tiers.<name>` in "
            ".llm-council.yaml before estimating, so the per-peer cost "
            "reflects the tier you'd actually run."
        ),
    )
    estimate.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help=(
            "Hard ceiling on the council's pre-flight estimated cost in USD. "
            "The breakdown is still printed, but the command exits non-zero "
            "if the estimate exceeds this. Free/local participants count as "
            "$0; hosted or non-local participants with unknown pricing are "
            "refused when this cap is set."
        ),
    )
    estimate.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Hard ceiling on estimated prompt+completion tokens across all "
            "participants and budgeted rounds. Exits non-zero if exceeded."
        ),
    )
    estimate.add_argument("--json", action="store_true", help="Print JSON")

    last = sub.add_parser("last", help="Print the latest council transcript path/content")
    last.add_argument("--cwd", default=".", help="Working directory")
    last.add_argument("--json-file", action="store_true", help="Use JSON transcript")
    last.add_argument("--html-file", action="store_true", help="Use HTML transcript")
    last.add_argument("--path-only", action="store_true", help="Only print path")
    last.add_argument("--open", action="store_true", help="Open transcript in browser")

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
    transcripts_show.add_argument("--html-file", action="store_true", help="Show HTML")
    transcripts_show.add_argument("--open", action="store_true", help="Open in browser")
    transcripts_summary = transcripts_sub.add_parser(
        "summary", help="Summarize transcript totals"
    )
    transcripts_summary.add_argument("--cwd", default=".", help="Working directory")
    transcripts_summary.add_argument(
        "--since",
        type=_parse_since_arg,
        default=None,
        help=(
            "Only summarize transcripts within the last N days (e.g. `7`) "
            "or since an absolute ISO date (e.g. `2026-04-01`)"
        ),
    )
    transcripts_prune = transcripts_sub.add_parser(
        "prune",
        help="Delete old transcripts (paired md+json) by count or age",
    )
    transcripts_prune.add_argument("--cwd", default=".", help="Working directory")
    transcripts_prune.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help=(
            "Keep the N most recent transcripts; older are pruned. Combined "
            "with --keep-since the UNION is retained (a transcript survives if "
            "it matches EITHER rule)."
        ),
    )
    transcripts_prune.add_argument(
        "--keep-since",
        type=_parse_keep_since_arg,
        default=None,
        help=(
            "Keep transcripts newer than the cutoff (integer days back "
            "or ISO date YYYY-MM-DD; ISO dates snap to midnight UTC); older "
            "are pruned. Combined with --keep-last the UNION is retained (a "
            "transcript survives if it matches EITHER rule)."
        ),
    )
    transcripts_prune.add_argument(
        "--delete",
        dest="apply",
        action="store_true",
        help="Delete transcripts matching the retention policy; default is a dry run",
    )
    transcripts_prune.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    transcripts_prune.add_argument("--json", action="store_true", help="Print JSON")

    stats = sub.add_parser(
        "stats", help="Aggregate per-participant metrics over recorded transcripts"
    )
    stats.add_argument("--cwd", default=".", help="Working directory")
    stats.add_argument(
        "--since",
        type=_parse_since_arg,
        default=None,
        help=(
            "Only consider transcripts within the last N days (e.g. `7`) "
            "or since an absolute ISO date (e.g. `2026-04-01`)"
        ),
    )
    stats.add_argument(
        "--participant",
        default=None,
        help="Filter participant metrics to one participant",
    )
    stats.add_argument("--json", action="store_true", help="Print JSON")
    stats.add_argument(
        "--peer",
        dest="participant",
        help=argparse.SUPPRESS,
    )

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

    refresh = models_sub.add_parser(
        "refresh",
        help="Force-fetch the OpenRouter catalog and overwrite the local cache",
    )
    refresh.add_argument("--json", action="store_true", help="Print JSON summary")

    return parser


def _parse_stance_args(values: list[str]) -> dict[str, str]:
    """Parse repeated `--stance peer=for|against|neutral` flags.

    Empty input returns an empty dict so callers can compose with the mode's
    stance map. Validation of stance values is left to render_stance_section,
    which already raises on unknown stances.
    """
    from llm_council.defaults import VALID_STANCES

    parsed: dict[str, str] = {}
    for item in values:
        text = (item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(
                f"--stance must be of the form peer=for|against|neutral, got '{item}'"
            )
        peer, _, stance = text.partition("=")
        peer = peer.strip()
        stance = stance.strip().lower()
        if not peer:
            raise SystemExit(f"--stance peer name is empty in '{item}'")
        if stance not in VALID_STANCES:
            raise SystemExit(
                f"--stance value '{stance}' for peer '{peer}' must be one of "
                f"{', '.join(VALID_STANCES)}"
            )
        parsed[peer] = stance
    return parsed


def _parse_since_arg(raw: str) -> int:
    """Accept either an integer (days back) or an ISO date (YYYY-MM-DD).

    Returns the equivalent `since_days` integer for compute_stats. ISO dates
    must be in the past; future dates are rejected via the cmd_stats
    `args.since <= 0` check below.
    """
    raw = (raw or "").strip()
    if not raw:
        raise argparse.ArgumentTypeError("--since cannot be empty")
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        cutoff = date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--since must be an integer (days back) or ISO date "
            f"(YYYY-MM-DD): got '{raw}' ({exc})"
        ) from exc
    today = datetime.now(timezone.utc).date()
    return (today - cutoff).days


def _parse_keep_since_arg(raw: str) -> float:
    """Same surface as `_parse_since_arg` but returns a precise epoch cutoff.

    For prune the ``rolling now - N*86400`` semantics that ``stats --since``
    uses would silently discard files mtime'd earlier in the day on the
    cutoff date. Snap the ISO-date case to midnight UTC of that date so the
    cutoff means "at the start of the day you named", and keep the integer
    case as a precise N-day rolling window.
    """
    raw = (raw or "").strip()
    if not raw:
        raise argparse.ArgumentTypeError("--keep-since cannot be empty")
    now = datetime.now(timezone.utc)
    try:
        days = int(raw)
    except ValueError:
        days = None
    if days is not None:
        if days < 0:
            raise argparse.ArgumentTypeError(
                "--keep-since must be a non-negative integer (days back)"
            )
        return (now - timedelta(days=days)).timestamp()
    try:
        cutoff_date = date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--keep-since must be an integer (days back) or ISO date "
            f"(YYYY-MM-DD): got '{raw}' ({exc})"
        ) from exc
    cutoff_dt = datetime(
        cutoff_date.year,
        cutoff_date.month,
        cutoff_date.day,
        tzinfo=timezone.utc,
    )
    if cutoff_dt > now:
        raise argparse.ArgumentTypeError(
            f"--keep-since date {raw} is in the future"
        )
    return cutoff_dt.timestamp()


def _question_from_args(parts: list[str], flag_value: str | None = None) -> str:
    """Resolve the question text from positional parts or the --question alias.

    Positional and --question are mutually exclusive; passing both is rejected
    so a future contributor doesn't accidentally pass conflicting strings and
    silently ship the wrong one.
    """
    positional = " ".join(parts).strip()
    flag = (flag_value or "").strip()
    if positional and flag:
        raise SystemExit(
            "question may be passed as a positional argument OR via "
            "--question, not both"
        )
    question = positional or flag
    if not question:
        raise SystemExit("question is required")
    return question


def _emit_config_warnings(config: dict) -> None:
    """Print non-fatal config advisories to stderr, prefixed for grep-ability.

    Called by every command that loads a project config. Today this surfaces
    near-miss origin typos (see `config.config_warnings`); other advisory
    classes can be added there without touching every command handler.
    """
    for warning in config_warnings(config):
        print(f"llm-council warning: {warning}", file=sys.stderr)


def cmd_list(args: argparse.Namespace) -> int:
    load_project_env(Path.cwd())
    config = load_config(getattr(args, "config", None))
    _emit_config_warnings(config)
    print("Participants:")
    for name, cfg in config.get("participants", {}).items():
        model = cfg.get("model") or "cli default (unreported)"
        print(f"  {name:20} {cfg.get('type'):10} {model}")
    print("\nModes:")
    modes = config.get("modes", {})
    for name, cfg in modes.items():
        if name in {"local-only", "local-private"} and "private-local" in modes:
            continue
        flag = " [EXPERIMENTAL]" if cfg.get("experimental") else ""
        description = str(cfg.get("description", ""))
        if flag and description.upper().startswith("EXPERIMENTAL"):
            description = description.partition("—")[2].strip() or description
        print(f"  {name:20}{flag} {description}")
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


def cmd_install_hook(args: argparse.Namespace) -> int:
    import shlex
    import stat
    import subprocess
    import tempfile

    root = Path(args.root).resolve()
    try:
        resolved = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "--git-path",
                f"hooks/{args.hook_type}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(
            f"Error: {root} is not a usable git worktree: {detail.strip()}",
            file=sys.stderr,
        )
        return 1

    hook_path = Path(resolved.stdout.strip())
    hook_file = hook_path if hook_path.is_absolute() else root / hook_path

    load_project_env(root)
    try:
        config = load_config(find_config(root), search=False)
    except (OSError, ValueError) as exc:
        print(f"Error: Could not load council config: {exc}", file=sys.stderr)
        return 1
    known_modes = config.get("modes", {})
    hook_mode = canonical_mode_name(config, args.mode)
    if hook_mode not in known_modes:
        choices = ", ".join(sorted(known_modes))
        print(
            f"Error: Unknown council mode '{args.mode}'. Known modes: {choices}",
            file=sys.stderr,
        )
        return 1
    if (hook_file.exists() or hook_file.is_symlink()) and not getattr(
        args, "force", False
    ):
        print(
            f"Error: Refusing to overwrite existing hook: {hook_file}. "
            "Rerun with --force to replace it.",
            file=sys.stderr,
        )
        return 1

    quoted_mode = shlex.quote(hook_mode)
    quoted_question = shlex.quote(f"{args.hook_type} validation.")
    script_content = f"""#!/bin/sh
# Git hook installed by llm-council
echo "Running LLM Council {args.hook_type} audit..."
exec llm-council run --diff --mode {quoted_mode} {quoted_question}
"""

    temp_path: Path | None = None
    try:
        hook_file.parent.mkdir(parents=True, exist_ok=True)
        if getattr(args, "force", False):
            # Populate a sibling file and atomically replace the hook path so
            # --force never follows an attacker-controlled symlink target.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=hook_file.parent,
                prefix=f".{hook_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(script_content)
                temp_path = Path(handle.name)
            os.chmod(temp_path, 0o755)
            os.replace(temp_path, hook_file)
            temp_path = None
        else:
            try:
                fd = os.open(
                    hook_file,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o755,
                )
            except FileExistsError:
                print(
                    f"Error: Refusing to overwrite existing hook: {hook_file}. "
                    "Rerun with --force to replace it.",
                    file=sys.stderr,
                )
                return 1
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(script_content)
            st = os.stat(hook_file, follow_symlinks=False)
            try:
                os.chmod(
                    hook_file,
                    st.st_mode | stat.S_IEXEC,
                    follow_symlinks=False,
                )
            except NotImplementedError:
                # Windows exposes the keyword but cannot honor no-follow
                # chmod.  Its chmod implementation does not manage executable
                # bits anyway, so keep the safely-created hook rather than
                # retrying with a symlink-following call.  POSIX failures stay
                # fatal because the executable bit is meaningful there.
                if os.name != "nt":
                    raise
        print(f"Successfully installed LLM Council {args.hook_type} hook to {hook_file}")
        return 0
    except Exception as e:
        print(f"Error: Failed to write git hook: {e}", file=sys.stderr)
        return 1
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _parse_config_value(value_str: str) -> Any:
    # Boolean
    if value_str.lower() in ("true", "yes", "on"):
        return True
    if value_str.lower() in ("false", "no", "off"):
        return False
    # None/null
    if value_str.lower() in ("none", "null"):
        return None
    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    # List (e.g., if it starts with [ and ends with ])
    if value_str.startswith("[") and value_str.endswith("]"):
        import ast
        try:
            return ast.literal_eval(value_str)
        except Exception:
            pass
    # Fallback to string
    return value_str


def _get_nested_val(d: dict, key_path: str) -> Any:
    parts = key_path.split(".")
    curr = d
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr


def _set_nested_val(d: dict, key_path: str, val: Any) -> None:
    parts = key_path.split(".")
    curr = d
    for part in parts[:-1]:
        if part not in curr or not isinstance(curr[part], dict):
            curr[part] = {}
        curr = curr[part]
    curr[parts[-1]] = val


def cmd_config(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    cfg_file = find_config(cwd)
    if not cfg_file:
        raise SystemExit("Configuration file not found. Run setup first.")
    cfg_path = Path(cfg_file)
        
    import yaml
    try:
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise SystemExit(f"Failed to read configuration: {e}")
        
    if args.config_command == "get":
        val = _get_nested_val(config, args.key)
        if val is None:
            print("")
        else:
            print(val)
        return 0
        
    elif args.config_command == "set":
        parsed_val = _parse_config_value(args.value)
        _set_nested_val(config, args.key, parsed_val)

        try:
            resolve_config_data(config)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Refusing to write invalid configuration: {exc}"
            ) from exc

        import tempfile

        temp_path: Path | None = None
        try:
            payload = yaml.safe_dump(config, sort_keys=False)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cfg_path.parent,
                prefix=f".{cfg_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
            temp_path.chmod(cfg_path.stat().st_mode)
            os.replace(temp_path, cfg_path)
            print(f"Successfully set {args.key} to {parsed_val}")
        except Exception as e:
            raise SystemExit(f"Failed to write configuration: {e}")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return 0
        
    else:
        raise SystemExit("Subcommand required: get, set")


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _pick_origin_interactive(default: str | None = None) -> str:
    """Prompt the user to pick an origin from `KNOWN_ORIGIN_STRINGS` or
    enter free text. Returns the chosen string verbatim — typo detection
    via `config_warnings` will catch near-misses on canonical strings."""
    from llm_council.defaults import KNOWN_ORIGIN_STRINGS

    options = list(KNOWN_ORIGIN_STRINGS)
    print("\n  Pick an origin (reflects the *model*, not the network location):")
    for index, value in enumerate(options, start=1):
        marker = " (default)" if value == default else ""
        print(f"    {index}) {value}{marker}")
    print(f"    {len(options) + 1}) Other (free text)")
    while True:
        prompt = "  Choice"
        if default and default in options:
            prompt = f"{prompt} [{options.index(default) + 1}]"
        answer = input(f"{prompt}: ").strip()
        if not answer and default and default in options:
            return default
        if answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            if choice == len(options) + 1:
                free = input("  Custom origin string: ").strip()
                if free:
                    return free
                print("  Origin must be non-empty.")
                continue
        print(f"  Invalid choice. Enter a number 1-{len(options) + 1}.")


def _derive_default_family(model_id: str) -> str:
    """Heuristic: pick a reasonable default `family` from a model id.

    `Qwen/Qwen3.6-27B` → `qwen`
    `meta-llama/Llama-3.3-70B-Instruct` → `llama`
    `lmstudio-community/Meta-Llama-3.3-70B-…` → `llama`
    `deepseek-ai/DeepSeek-V4` → `deepseek`

    Best-effort substring match against well-known family keywords; falls
    back to the model id's first segment when nothing matches. The user
    can always override the default at the wizard prompt.

    TODO: when a new prominent family lands (granite, command-r,
    nemotron, exaone, …) extend this list. The fallback path keeps the
    wizard usable for unknown families — a missing keyword surfaces as
    a default like `local_acme_some_model`, which the user can rename
    inline.
    """
    lower = model_id.lower()
    for keyword in (
        "qwen", "deepseek", "llama", "mistral", "gemma",
        "phi", "claude", "gemini", "kimi", "glm",
    ):
        if keyword in lower:
            return keyword
    if "/" in model_id:
        first, _ = model_id.split("/", 1)
        return re.sub(r"[^a-z0-9]+", "_", first.lower()) or "local"
    return "local"


def _derive_default_participant_name(model_id: str) -> str:
    """Turn `Qwen/Qwen3.6-27B` into `local_qwen_qwen3_6_27b`."""
    family = _derive_default_family(model_id)
    last = model_id.rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "_", last.lower()).strip("_") or "model"
    return f"local_{family}_{slug}"[:64]


def _probe_and_collect_local_participants(
    *, probe_url: str | None = None
) -> dict[str, dict]:
    """Probe local OpenAI-compatible endpoints, prompt the user to scaffold
    each as a participant. Returns a {name: cfg} dict (possibly empty).

    Caller is responsible for ensuring this only runs in interactive mode.
    """
    from llm_council.doctor import discover_local_openai

    print("\nProbing local OpenAI-compatible endpoints…")
    probes = discover_local_openai(probe_url)
    extras: dict[str, dict] = {}
    for probe in probes:
        if not probe.ok:
            print(f"  - {probe.label}: {probe.detail}")
            continue
        served_models = list(probe.models)
        print(f"\n  Found: {probe.detail}")
        print(f"  Endpoint: {probe.base_url}")
        if not _confirm("  Scaffold a council participant for this endpoint?", True):
            continue
        if served_models:
            if len(served_models) == 1:
                model_id = served_models[0]
                print(f"  Model: {model_id}")
            else:
                print("  Available models:")
                for i, mid in enumerate(served_models, 1):
                    print(f"    {i}) {mid}")
                while True:
                    raw = input(
                        f"  Pick a model [1-{len(served_models)}]: "
                    ).strip()
                    if raw.isdigit() and 1 <= int(raw) <= len(served_models):
                        model_id = served_models[int(raw) - 1]
                        break
                    print("  Invalid choice.")
        else:
            model_id = input(
                "  Endpoint did not list models — enter the model id manually: "
            ).strip()
            if not model_id:
                print("  Skipping (no model id).")
                continue
        base_url = probe.base_url
        default_name = _derive_default_participant_name(model_id)
        name_input = input(f"  Participant name [{default_name}]: ").strip()
        name = name_input or default_name
        default_family = _derive_default_family(model_id)
        family_input = input(f"  Family [{default_family}]: ").strip()
        family = family_input or default_family
        origin = _pick_origin_interactive()
        extras[name] = {
            "type": "openai_compatible",
            "family": family,
            "origin": origin,
            "base_url": base_url,
            "model": model_id,
            "api_key_env": "LOCAL_OPENAI_API_KEY",
            "allow_private": True,
            "timeout": 360,
        }
        print(f"  Will write participant {name!r}.")
    if extras:
        print(
            "\n  Note: export LOCAL_OPENAI_API_KEY=dummy (or a real key) "
            "before running the council. The openai_compatible adapter "
            "requires a non-empty Authorization header even for "
            "unauthenticated local servers."
        )
    return extras


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

    include_native = preset not in {"openrouter", "private-local"}
    include_openrouter = preset in {"openrouter", "tri-cli-openrouter", "all"}
    include_local = preset in {"private-local", "all"}

    if not args.yes:
        print("LLM Council setup")
        print(f"Project root: {root}")
        print("\nDetected CLIs:")
        for name in ("claude", "codex", "gemini", "antigravity", "ollama"):
            cmd = "agy" if name == "antigravity" else name
            print(f"  {name:12} {shutil.which(cmd) or 'not found'}")
        include_openrouter = _confirm(
            "Include hosted OpenRouter participants?", include_openrouter
        )
        include_local = _confirm("Include local Ollama participants?", include_local)
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

    extra_local_participants: dict[str, dict] = {}
    if getattr(args, "probe_local", False):
        if args.yes:
            print(
                "llm-council: --probe-local requires interactive prompts "
                "(origin/family/model). Ignored under --yes.",
                file=sys.stderr,
            )
        else:
            extra_local_participants = _probe_and_collect_local_participants()

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
            extra_local_participants=extra_local_participants or None,
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


def _make_which_cache():
    """Per-setup-invocation memoizer for `shutil.which` lookups.

    Setup route detection probes the same native binaries from several
    helpers; without memoization each binary is re-resolved against PATH
    multiple times per `setup` invocation. The cache lives only as long
    as the caller holds it, so probe results never leak across invocations.
    """
    _cache: dict[str, str | None] = {}

    def _which(binary: str) -> str | None:
        if binary not in _cache:
            _cache[binary] = shutil.which(binary)
        return _cache[binary]

    return _which


def _auto_setup_preset(which=None) -> str:
    if which is None:
        which = _make_which_cache()
    has_claude = bool(which("claude"))
    has_codex = bool(which("codex"))
    has_neutral = bool(which("agy"))
    if has_neutral and (has_claude or has_codex):
        return "tri-cli"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"

    found_list = []
    if has_claude:
        found_list.append("claude")
    if has_codex:
        found_list.append("codex")
    if which("agy"):
        found_list.append("antigravity")
    found = ", ".join(found_list) or "none"

    raise SystemExit(
        "Auto setup could not find a usable default council route. "
        f"Found native CLIs: {found}. "
        "Install Antigravity plus at least one of Claude or Codex, or set "
        "OPENROUTER_API_KEY in your shell, .env, .env.local, or .llm-council.env "
        "and rerun setup. Advanced users who intentionally want to stage an "
        "incomplete config can choose an explicit preset with --allow-incomplete."
    )


def _detect_setup_routes() -> dict[str, object]:
    which = _make_which_cache()
    # `gemini` stays in native_paths purely for informational display (a
    # legacy/enterprise binary may still be on PATH) but no longer feeds
    # route gating below: standalone Gemini CLI was retired for individual
    # accounts in 2026-06, so only Antigravity satisfies the Gemini-family
    # seat for auto-detection purposes.
    native_names = ("claude", "codex", "gemini", "antigravity")
    native_paths = {name: which("agy" if name == "antigravity" else name) for name in native_names}
    has_claude = bool(native_paths.get("claude"))
    has_codex = bool(native_paths.get("codex"))
    has_neutral = bool(native_paths.get("antigravity"))
    native_count = sum([has_claude, has_codex, has_neutral])
    native_usable = has_neutral and (has_claude or has_codex)
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    ollama_path = which("ollama")
    return {
        "native_paths": native_paths,
        "native_count": native_count,
        "native_usable": native_usable,
        "has_openrouter": has_openrouter,
        "ollama_path": ollama_path,
        "auto": _auto_setup_preset_or_none(which),
    }


def _auto_setup_preset_or_none(which=None) -> str | None:
    try:
        return _auto_setup_preset(which)
    except SystemExit:
        return None


def _preset_status(preset: str, routes: dict[str, object]) -> tuple[str, str]:
    preset = _SETUP_PRESET_ALIASES.get(preset, preset)
    native_usable = bool(routes["native_usable"])
    has_openrouter = bool(routes["has_openrouter"])
    has_ollama = bool(routes["ollama_path"])
    if preset == "auto":
        selected = routes["auto"]
        if selected:
            return "recommended", f"would select `{selected}`"
        return (
            "blocked",
            "needs Antigravity plus Claude/Codex, or OPENROUTER_API_KEY",
        )
    if preset == "tri-cli":
        if native_usable:
            return "available", "uses installed native CLI participants"
        return "blocked", "needs Antigravity plus at least one of Claude or Codex"
    if preset == "openrouter":
        if has_openrouter:
            return "available", "uses hosted OpenRouter participants"
        return "blocked", "needs OPENROUTER_API_KEY"
    if preset == "tri-cli-openrouter":
        if native_usable and has_openrouter:
            return "available", "uses native CLI and hosted OpenRouter participants"
        missing = []
        if not native_usable:
            missing.append("a usable native route")
        if not has_openrouter:
            missing.append("OPENROUTER_API_KEY")
        return "blocked", "needs " + " and ".join(missing)
    if preset == "private-local":
        if has_ollama:
            return "available", "uses local Ollama only (default mode: private-local)"
        return "blocked", "needs ollama"
    if preset == "all":
        if native_usable and has_openrouter and has_ollama:
            return "available", "writes native, hosted, and local participant routes"
        missing = []
        if not native_usable:
            missing.append("a usable native route")
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
    for name in ("claude", "codex", "gemini", "antigravity"):
        print(f"  {name:12} {native_paths.get(name) or 'not found'}")
    print(f"  openrouter {'OPENROUTER_API_KEY set' if routes['has_openrouter'] else 'OPENROUTER_API_KEY not set'}")
    print(f"  ollama   {routes['ollama_path'] or 'not found'}")
    print()
    print("Preset choices:")
    for preset in _SETUP_PRESETS:
        status, detail = _preset_status(preset, routes)
        print(f"  {preset:19} {status:11} {detail}")
    print()
    print("Agent installers: show this plan to the user and ask which preset to write.")
    print("Do not choose a blocked preset unless the user is deliberately preparing config for later.")
    print("Then run: llm-council setup --yes --preset <chosen-preset>")


def _prompt_setup_preset(root: Path) -> str:
    _print_setup_plan(root)
    default = _auto_setup_preset_or_none() or "openrouter"
    valid = set(_SETUP_PRESETS)
    answer = input(f"Choose setup preset [{default}]: ").strip()
    if not answer:
        answer = default
    answer = _SETUP_PRESET_ALIASES.get(answer, answer)
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
            f"{root / '.llm-council/instructions/antigravity.md'} to GEMINI.md."
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
            "  4. Run `llm-council estimate --mode review \"Review this\"` "
            "before paid hosted calls."
        )

    warnings: list[str] = []
    if include_native:
        has_primary = bool(shutil.which("claude") or shutil.which("codex"))
        if not has_primary:
            warnings.append(
                "neither claude nor codex was found on PATH; native CLI "
                "modes need at least one."
            )
        has_agy = bool(shutil.which("agy"))
        if not has_agy:
            warnings.append(
                "antigravity (agy) was not found on PATH; native CLI modes "
                "need a Gemini-family participant."
            )
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
    _emit_config_warnings(config)
    checks = check_environment(
        config,
        probe_openrouter=args.probe_openrouter,
        probe_ollama=args.probe_ollama,
        probe_native=bool(getattr(args, "probe_native", False)),
        probe_cwd=env_start.resolve() if env_start.is_dir() else env_start.parent.resolve(),
    )
    if getattr(args, "repair_transcript_permissions", False):
        repair_root = transcript_dir(
            env_start.resolve() if env_start.is_dir() else env_start.parent.resolve(),
            config,
        )
        if repair_root.is_dir():
            report = inspect_transcript_permissions(repair_root, repair=True)
            repaired = len(report["repaired_files"])
            skipped = sum(
                len(value)
                for key, value in report.items()
                if key.startswith("skipped_") and isinstance(value, list)
            )
            checks.append(Check(
                name="transcripts:permissions",
                ok=skipped == 0,
                detail=(
                    f"repaired {repaired} file(s); directory_repaired="
                    f"{report['directory_repaired']}; skipped={skipped}; "
                    f"directory={repair_root}"
                ),
            ))
        else:
            checks.append(Check(
                name="transcripts:permissions",
                ok=True,
                detail=f"no transcript directory yet: {repair_root}",
            ))
    probe_local = getattr(args, "probe_local_openai", None)
    if probe_local is not None:
        # `--probe-local-openai` (no arg) → scan defaults; with a URL → probe
        # that single endpoint. The empty-string sentinel comes from
        # argparse's `const=""` for the bare flag.
        explicit_url = probe_local or None
        checks.extend(probe_local_openai(explicit_url))
    default_mode = config.get("defaults", {}).get("mode", "quick")
    try:
        default_participants = select_participants(config, default_mode, current=None)
    except ValueError as exc:
        default_participants = []
        checks.append(
            Check(
                name="route:default-mode",
                ok=False,
                detail=f"mode '{default_mode}' is not runnable: {exc}",
            )
        )
    else:
        checks.append(
            Check(
                name="route:default-mode",
                ok=True,
                detail=(
                    f"mode '{default_mode}' selects "
                    + ", ".join(default_participants)
                ),
            )
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
    required_names = {"python:mcp", "route:default-mode"}
    for name in default_participants:
        participant = config.get("participants", {}).get(name, {})
        if participant.get("type") == "cli":
            required_names.add(f"cli:{name}")
        elif api_key_env := participant_api_key_env(participant):
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
    # Only gate on an explicit URL probe — the bare port-scan is discovery
    # and "no local servers running" is not an error condition.
    if probe_local:
        required.extend(
            check
            for check in checks
            if check.name.startswith("probe:local-openai")
        )
    return 0 if all(check.ok for check in required) else 1


def cmd_check_update(args: argparse.Namespace) -> int:
    status = check_for_update(__version__)
    # An explicit check is at least as authoritative as the passive 24h
    # nag refresh, so it should hydrate the same cache. Without this, a
    # user who manually runs `check-update` would still see the old nag
    # message on next `run` until the cache organically expires.
    hydrate_nag_cache_from_status(status)
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
    _emit_config_warnings(config)

    mode = canonical_mode_name(
        config, args.mode or config.get("defaults", {}).get("mode", "quick")
    )
    tier = getattr(args, "tier", None)
    if tier:
        try:
            apply_tier_override(config, tier)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    current = args.current or detect_current_agent()
    stdin_text = sys.stdin.read() if args.stdin else None
    chunk_events: list[dict[str, Any]] = []
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
            chunk_strategy=getattr(args, "chunk_strategy", "fail"),
            chunk_progress=chunk_events.append,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if chunk_events:
        estimate["chunk_events"] = chunk_events

    if args.json:
        print(json.dumps(estimate, indent=2))
    else:
        _print_estimate(estimate)

    # Budget gates mirror `cmd_run`'s logic so the same cap can be enforced
    # pre-flight from a wrapper or CI without re-running the cost math.
    # The breakdown is printed unconditionally above so the caller still
    # sees per-peer costs along with the non-zero exit.
    max_cost_usd = getattr(args, "max_cost_usd", None)
    max_tokens = getattr(args, "max_tokens", None)
    try:
        budget.enforce_preflight_caps(
            estimate,
            max_cost_usd=max_cost_usd,
            max_tokens=max_tokens,
            breakdown_hint=(
                "drop expensive peers, raise the cap, or see the per-peer "
                "breakdown above. To exclude the repair-retry margin, set "
                "retry_on_missing_label: false on individual participants."
            ),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
        label = row["name"] if row["model"] == CLI_DEFAULT_MODEL_LABEL else row["model"]
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
    out_dir = transcript_dir(cwd, config)
    
    if getattr(args, "html_file", False) and getattr(args, "json_file", False):
        raise SystemExit("Error: --html-file and --json-file are mutually exclusive.")
        
    suffix = ".md"
    if getattr(args, "html_file", False):
        suffix = ".html"
    elif getattr(args, "json_file", False):
        suffix = ".json"
    elif getattr(args, "open", False):
        # By default, --open implies --html-file
        suffix = ".html"
        
    path = latest_transcript(out_dir, suffix=suffix)
    if path is None:
        if latest_transcript(out_dir, suffix=".md") is not None:
            raise SystemExit(f"No transcripts with suffix '{suffix}' found in {out_dir}.")
        raise SystemExit(f"No council transcripts found in {out_dir}")
        
    if getattr(args, "open", False):
        if not path.is_file():
            raise SystemExit(f"Failed to read transcript {path}: file does not exist")
        import webbrowser
        print(f"Opening transcript: {path}")
        if not webbrowser.open(path.as_uri()):
            raise SystemExit(f"Failed to open browser for transcript {path}. You can view the path directly.")
        return 0
        
    if args.path_only:
        print(path)
    else:
        print(path.read_text(encoding="utf-8"))
    return 0


def _transcript_dir(cwd: Path, config: dict) -> Path:
    return transcript_dir(cwd, config)


def cmd_stats(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    load_project_env(cwd)
    config = load_config(find_config(cwd), search=False)
    out_dir = _transcript_dir(cwd, config)
    if args.since is not None and args.since <= 0:
        raise SystemExit("--since must be a positive integer or ISO date in the past")
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
        if getattr(args, "html_file", False) and getattr(args, "json_file", False):
            raise SystemExit("Error: --html-file and --json-file are mutually exclusive.")

        suffix = ".md"
        if getattr(args, "html_file", False):
            suffix = ".html"
        elif getattr(args, "json_file", False):
            suffix = ".json"
        elif getattr(args, "open", False) and not args.path:
            # By default, --open without an explicit path implies --html-file
            suffix = ".html"

        if args.path:
            path = Path(args.path)
            if not path.is_absolute():
                path = cwd / path
            if getattr(args, "open", False) and not getattr(args, "html_file", False) and not getattr(args, "json_file", False):
                if path.suffix == ".md" and path.with_suffix(".html").is_file():
                    path = path.with_suffix(".html")
            else:
                if getattr(args, "html_file", False) or getattr(args, "json_file", False):
                    path = path.with_suffix(suffix)
        else:
            path = latest_transcript(out_dir, suffix=suffix)
            if path is None:
                if latest_transcript(out_dir, suffix=".md") is not None:
                    raise SystemExit(f"No transcripts with suffix '{suffix}' found in {out_dir}.")
                raise SystemExit(f"No council transcripts found in {out_dir}")

        if not path.is_file():
            raise SystemExit(f"Failed to read transcript {path}: file does not exist")

        if getattr(args, "open", False):
            import webbrowser
            print(f"Opening transcript: {path}")
            if not webbrowser.open(path.as_uri()):
                raise SystemExit(f"Failed to open browser for transcript {path}. You can view the path directly.")
            return 0

        try:
            print(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SystemExit(f"Failed to read transcript {path}: {exc}") from exc
        return 0

    if args.transcripts_command == "summary":
        records = transcript_records(out_dir)
        since_days = getattr(args, "since", None)
        if since_days is not None:
            if since_days <= 0:
                raise SystemExit(
                    "--since must be a positive integer or ISO date in the past"
                )
            cutoff = time.time() - since_days * 86400
            records = [r for r in records if r.get("mtime", 0) >= cutoff]
        runs = len(records)
        tokens = sum(record["tokens"] for record in records)
        cost = sum(record["cost_usd"] for record in records)
        successes = sum(record["ok"] for record in records)
        participants = sum(record["total"] for record in records)
        print(f"runs: {runs}")
        if since_days is not None:
            print(f"since: last {since_days}d")
        print(f"participant_successes: {successes}/{participants}")
        print(f"tokens: {tokens}")
        print(f"cost_usd: ${cost:.6f}")
        return 0

    if args.transcripts_command == "prune":
        return _cmd_transcripts_prune(args, out_dir)

    raise SystemExit(f"Unknown transcripts subcommand: {args.transcripts_command}")


def _cmd_transcripts_prune(args: argparse.Namespace, out_dir: Path) -> int:
    if args.keep_last is None and args.keep_since is None:
        raise SystemExit(
            "transcripts prune requires --keep-last N and/or --keep-since "
            "<days_or_iso_date>; refusing to act with no retention policy"
        )
    if args.keep_last is not None and args.keep_last < 0:
        raise SystemExit("--keep-last must be non-negative")
    # args.keep_since is now a precise epoch cutoff (float) when supplied.

    if not out_dir.exists():
        message = f"No transcripts directory at {out_dir}"
        if args.json:
            print(json.dumps({"pruned": [], "kept": [], "message": message}))
        else:
            print(message)
        return 0

    json_paths = sorted(out_dir.glob("*.json"))
    indexed: list[tuple[Path, float]] = [(p, p.stat().st_mtime) for p in json_paths]
    indexed.sort(key=lambda item: item[1], reverse=True)

    keep_set: set[Path] = set()
    if args.keep_last is not None:
        for path, _mtime in indexed[: args.keep_last]:
            keep_set.add(path)
    if args.keep_since is not None:
        cutoff = float(args.keep_since)
        for path, mtime in indexed:
            if mtime >= cutoff:
                keep_set.add(path)

    pruned: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for path, mtime in indexed:
        sibling_md = path.with_suffix(".md")
        record = {
            "json": str(path),
            "markdown": str(sibling_md) if sibling_md.exists() else None,
            "mtime": mtime,
        }
        if path in keep_set:
            kept.append(record)
            continue
        pruned.append(record)
        if args.apply:
            try:
                path.unlink(missing_ok=True)
                if sibling_md.exists():
                    sibling_md.unlink()
            except OSError as exc:
                raise SystemExit(f"failed to remove {path}: {exc}") from exc

    if args.json:
        print(
            json.dumps(
                {
                    "pruned": pruned,
                    "kept_count": len(kept),
                    "applied": args.apply,
                },
                indent=2,
            )
        )
    else:
        verb = "removed" if args.apply else "would remove"
        print(f"transcripts {verb}: {len(pruned)} (kept {len(kept)})")
        for record in pruned:
            print(f"  - {record['json']}")
        if not args.apply and pruned:
            print("re-run with --delete to delete these transcripts")
    return 0


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    return f"${value:.3f}"


def _fmt_usd(value: float | None) -> str:
    return display.format_usd(value)


def _make_progress_printer(ordered_peers: list[str] | tuple[str, ...] | None = None):
    """Build a `_print_progress_event`-shaped callback with a peer roster
    closure for per-peer color accents.

    Closure exists so `execute_council`'s sync `progress` callback contract
    stays single-argument while still giving us deterministic per-peer
    color rotation. Falls back to the bold-cyan gutter for peers that
    aren't in the roster (e.g., custom CLIs registered late, future peer
    types).
    """
    peers = list(ordered_peers or [])

    def _accent_for(name: str | None) -> str | None:
        if not name or not peers:
            return None
        return display.peer_accent(name, peers)

    def _emit(event: dict) -> None:
        kind = event.get("event")
        participant = event.get("participant")
        round_label = f"round {event.get('round')}" if event.get("round") else "round ?"
        # `wants_quiet()` suppresses color (layout stays); NO_COLOR / non-TTY
        # is already handled by `wants_color`.
        color = display.wants_color(sys.stderr) and not display.wants_quiet()
        peer_color = _accent_for(participant) if color else None
        if kind == "participant_start":
            print(
                display.format_gutter(
                    participant or "peer",
                    f"start {round_label}",
                    color=color,
                    token_color=peer_color,
                ),
                flush=True,
            )
            return
        if kind == "participant_slow":
            elapsed = float(event.get("elapsed_seconds") or 0)
            timeout = float(event.get("timeout_seconds") or 0)
            slow = display.colorize_status("slow", color=color)
            print(
                display.format_gutter(
                    participant or "peer",
                    f"{slow} after {elapsed:.1f}s (hard timeout at {timeout:.0f}s)",
                    color=color,
                    token_color=peer_color,
                ),
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
            colored_status = display.colorize_status(status, color=color)
            print(
                display.format_gutter(
                    participant or "peer",
                    f"{colored_status} {round_label} ({'; '.join(details)})",
                    color=color,
                    token_color=peer_color,
                ),
                flush=True,
            )
            if event.get("error"):
                print(
                    display.format_gutter("", event["error"], color=color),
                    flush=True,
                )
            return
        if kind == "deliberation_skip_participants":
            skipped = ", ".join(event.get("skipped") or [])
            print(
                display.format_gutter(
                    display.VERB_DELIBERATING,
                    f"skipping {skipped} from round {event.get('round')} "
                    f"({event.get('reason')})",
                    color=color,
                ),
                flush=True,
            )
            return
        if kind == "deliberation_pending":
            print(
                display.format_gutter(
                    display.VERB_DELIBERATING,
                    f"disagreement detected; starting round {event.get('round')}",
                    color=color,
                ),
                flush=True,
            )
            return
        if kind == "deliberation_round_start":
            print(
                display.format_gutter(
                    display.VERB_ROUND,
                    f"{event.get('round')} (deliberation)",
                    color=color,
                ),
                flush=True,
            )
            return
        if kind == "deliberation_skip":
            print(
                display.format_gutter(
                    display.VERB_DELIBERATING,
                    f"skipped ({event.get('reason')})",
                    color=color,
                ),
                flush=True,
            )
            return
        if kind == "deliberation_finish":
            status = event.get("status") or "done"
            colored_status = display.colorize_status(status, color=color)
            print(
                display.format_gutter(
                    display.VERB_DELIBERATING,
                    f"{colored_status} after {event.get('rounds')} rounds",
                    color=color,
                ),
                flush=True,
            )
            return
        if kind == "degraded_consensus":
            labeled = event.get("labeled_quorum")
            threshold = event.get("min_quorum")
            degraded = display.colorize_status("DEGRADED", color=color)
            print(
                display.format_gutter(
                    "Quorum",
                    f"{labeled} of {threshold} required peers labeled — {degraded}",
                    color=color,
                ),
                flush=True,
            )
            return
        if kind == "images_skipped":
            print(
                display.format_gutter(
                    participant or "peer",
                    f"image attachments skipped ({event.get('reason')}; "
                    f"{event.get('image_count')} image(s) referenced as text only)",
                    color=color,
                    token_color=peer_color,
                ),
                flush=True,
            )

    return _emit


# Back-compat: the previous module-level callback kept for tests/imports
# that call it directly. Built without a peer roster, so the per-peer
# accent rotation degrades to the default gutter.
_print_progress_event = _make_progress_printer()


def cmd_models(args: argparse.Namespace) -> int:
    if args.models_command == "refresh":
        return _cmd_models_refresh(args)
    if args.models_command != "openrouter":
        raise SystemExit("models subcommand is required (openrouter|refresh)")
    try:
        models = fetch_openrouter_models(use_cache=not args.no_cache)
    except Exception as exc:
        message = f"openrouter catalog fetch failed: {type(exc).__name__}: {exc}"
        if args.json:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            print(message, file=sys.stderr)
        return 1
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


def _cmd_models_refresh(args: argparse.Namespace) -> int:
    try:
        summary = refresh_openrouter_cache()
    except Exception as exc:
        message = f"openrouter catalog refresh failed: {type(exc).__name__}: {exc}"
        if args.json:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            print(message, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"ok": True, **summary}, indent=2))
    else:
        print(
            f"refreshed openrouter catalog: {summary['model_count']} models -> "
            f"{summary['cache_path']}"
        )
    return 0


async def cmd_run_async(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    question = _question_from_args(
        args.question, flag_value=getattr(args, "question_flag", None)
    )
    load_project_env(cwd)
    # Cached daily nag — skips network on cache hit, opt-out via
    # LLM_COUNCIL_NO_UPDATE_CHECK=1, never raises. mcp_server.py does
    # not call this so the stdio transport stays clean.
    maybe_print_update_nag(__version__)
    try:
        config = load_config(args.config or find_config(cwd), search=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _emit_config_warnings(config)

    # Resolve continuation metadata before routing. A private-local parent
    # keeps its privacy boundary when --mode is omitted; an explicit move to
    # hosted peers is refused after participant selection unless the caller
    # acknowledges it with --allow-privacy-downgrade.
    out_dir = transcript_dir(cwd, config)
    parent_run_id: str | None = None
    prior_context: str | None = None
    prior_transcript: dict[str, Any] | None = None
    continue_id = getattr(args, "continue_id", None)
    if continue_id:
        try:
            normalize_run_id(continue_id)
            depth_error = continuation_depth_limit_error(config, out_dir, continue_id)
            if depth_error:
                raise SystemExit(depth_error)
            prior_transcript = find_transcript_by_id(out_dir, continue_id)
            prior_path = prior_transcript.get("_path")
            parent_run_id = (
                Path(str(prior_path)).stem
                if prior_path
                else normalize_run_id(continue_id)
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    inherited_mode: str | None = None
    if args.mode is None and prior_transcript is not None:
        from llm_council.privacy import transcript_was_private_local

        if transcript_was_private_local(prior_transcript):
            inherited_mode = str(prior_transcript.get("mode") or "private-local")
    mode = canonical_mode_name(
        config,
        args.mode
        or inherited_mode
        or config.get("defaults", {}).get("mode", "quick"),
    )
    tier = getattr(args, "tier", None)
    if tier:
        try:
            swapped = apply_tier_override(config, tier)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not args.json and swapped:
            print(
                f"Tier '{tier}' applied: swapped models for {', '.join(swapped)}",
                flush=True,
            )
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

    if prior_transcript is not None:
        from llm_council.privacy import privacy_downgrade_error

        downgrade_error = privacy_downgrade_error(
            prior_transcript,
            participants=participants,
            participant_cfg=config.get("participants", {}) or {},
            allow_privacy_downgrade=bool(
                getattr(args, "allow_privacy_downgrade", False)
            ),
        )
        if downgrade_error:
            raise SystemExit(downgrade_error)
        prior_context = format_prior_council_context(
            prior_transcript, run_id=parent_run_id
        )
    stdin_text = sys.stdin.read() if args.stdin else None
    # Independent-review isolation (advisory). Resolution order:
    # CLI flag > per-mode independent_review > defaults.independent_review.
    # Default OFF. When ON and a prior_context WOULD have been injected from a
    # continuation, drop it so this round forms its verdict without anchoring
    # on prior verdicts. Suppression is recorded post-run as a metadata flag.
    _independent_mode_cfg = config.get("modes", {}).get(mode, {})
    if not isinstance(_independent_mode_cfg, dict):
        _independent_mode_cfg = {}
    # None-aware precedence (NOT an `or` chain): a higher-priority layer's
    # explicit `false` must override a lower layer's `true` — e.g. a mode that
    # opts out of a globally-defaulted-on independent_review. The CLI flag is
    # store_true, so it can only force ON; when unset it defers to the mode
    # value (if set, including explicit false), then the global default.
    # (codex WU5 review.)
    if getattr(args, "independent_review", False):
        independent_review = True
    else:
        _iv = _independent_mode_cfg.get("independent_review")
        if _iv is None:
            _iv = config.get("defaults", {}).get("independent_review")
        independent_review = bool(_iv)
    prior_context_suppressed = False
    if independent_review and prior_context:
        prior_context = None
        prior_context_suppressed = True
        print(
            "note: --independent-review active; prior council context "
            "suppressed for this run.",
            file=sys.stderr,
            flush=True,
        )
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

        participant_cfg = config.get("participants", {})
        if not isinstance(participant_cfg, dict):
            participant_cfg = {}
        mode_cfg = config.get("modes", {}).get(mode, {})
        if not isinstance(mode_cfg, dict):
            mode_cfg = {}
        mode_stances = mode_cfg.get("stances")
        cli_stance_overrides = _parse_stance_args(getattr(args, "stance", []) or [])
        if cli_stance_overrides:
            base = dict(mode_stances) if isinstance(mode_stances, dict) else {}
            base.update(cli_stance_overrides)
            mode_stances = base
        from llm_council.config import balance_stances
        mode_stances = balance_stances(participants, mode_stances)
        default_max = (
            config.get("defaults", {}).get("max_prompt_chars") or MAX_PROMPT_CHARS
        )
        # Chunk against the tightest budget any selected peer enforces, not
        # the global default — adapters re-check per-participant before
        # launch, so a prompt sized to the global default would still be
        # rejected by stricter peers.
        peer_caps = [
            int(participant_cfg.get(name, {}).get("max_prompt_chars"))
            for name in participants
            if isinstance(participant_cfg.get(name), dict)
            and participant_cfg.get(name, {}).get("max_prompt_chars")
        ]
        effective_max = min([int(default_max), *peer_caps]) if peer_caps else int(default_max)
        safe_context = bool(
            (config.get("modes", {}) or {}).get(mode, {}).get("safe_context")
        )
        prompt = build_prompt(
            question,
            mode=mode,
            cwd=cwd,
            context_paths=args.context,
            include_diff=args.diff,
            stdin_text=stdin_text,
            allow_outside_cwd=args.allow_outside_cwd,
            max_prompt_chars=effective_max,
            image_manifest=image_manifest or None,
            stances=mode_stances if isinstance(mode_stances, dict) else None,
            participants=participant_cfg or None,
            prior_context=prior_context,
            safe_context=safe_context,
            chunk_strategy=getattr(args, "chunk_strategy", "fail"),
            chunk_progress=_record_chunk_event,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    from llm_council.safety import apply_secret_scan_policy, redact_secrets

    defaults_cfg = config.get("defaults", {}) or {}
    scan_policy = str(defaults_cfg.get("secret_scan") or "warn").lower()
    scan_allowlist = str(
        defaults_cfg.get("secret_scan_allowlist")
        or ".llm-council-secrets-allow"
    )
    try:
        scan_result = apply_secret_scan_policy(
            prompt,
            policy=scan_policy,
            cwd=cwd,
            allowlist_filename=scan_allowlist,
        )
    except ValueError as exc:
        # block-mode hit; surface and halt before any participant runs.
        raise SystemExit(str(exc)) from exc
    # redact policy: swap in the masked prompt before it reaches peers OR the
    # transcript. Pop it so the (large) redacted prompt isn't duplicated into
    # metadata.secret_scan below.
    _redacted = scan_result.pop("redacted_prompt", None)
    persisted_question = question
    if _redacted is not None:
        prompt = _redacted
        # `question` is persisted separately and is also reused by
        # synthesis prompts. Redact that copy too so the policy protects every
        # artifact and every later participant call, not only round one.
        persisted_question, _question_findings = redact_secrets(
            question,
            cwd=cwd,
            allowlist_filename=scan_allowlist,
        )
    if scan_result.get("detected_count"):
        kinds_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(scan_result["kinds"].items())
        )
        print(
            display.format_gutter(
                "warn",
                f"secret_scan: {scan_result['detected_count']} likely "
                f"credential(s) detected in prompt ({kinds_summary}). "
                f"Allowlist: ./{scan_allowlist}. Policy: {scan_policy}.",
                color=display.wants_color(sys.stdout),
            ),
            flush=True,
        )

    mode_cfg = config.get("modes", {}).get(mode, {})
    transparent = bool(args.transparent or config.get("defaults", {}).get("transparent"))
    deliberate = bool(args.deliberate or mode_cfg.get("deliberate"))
    synthesize = bool(
        getattr(args, "synthesize", False)
        or mode_cfg.get("synthesize")
        or config.get("defaults", {}).get("synthesize")
    )
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

    tool_call_voting = bool(mode_cfg.get("tool_call_voting"))
    participant_prompts: dict[str, str] = {}
    for name in participants:
        peer_cfg = participant_cfg.get(name) or {}
        assigned_stance = (
            mode_stances.get(name)
            if isinstance(mode_stances, dict)
            else peer_cfg.get("stance")
        )
        participant_prompts[name] = apply_per_peer_directives(
            prompt,
            mode=mode,
            family=peer_cfg.get("family"),
            tool_call_voting=tool_call_voting,
            stance=assigned_stance,
        )

    deliberation_prompt_bounds = (
        deliberation_prompt_char_bounds(
            participants=participants,
            participant_cfg=participant_cfg,
            mode=mode,
            tool_call_voting=tool_call_voting,
            stances=mode_stances if isinstance(mode_stances, dict) else None,
            effective_prompt_cap=config.get("defaults", {}).get(
                "max_prompt_chars"
            ),
        )
        if deliberate and max_rounds > 1
        else {}
    )

    synthesizer_name: str | None = None
    if synthesize:
        from llm_council.synthesis import select_synthesizer

        active_participant_cfg = {
            name: participant_cfg[name]
            for name in participants
            if name in participant_cfg
        }
        try:
            synthesizer_name = select_synthesizer(
                config,
                active_participant_cfg,
                stances=mode_stances if isinstance(mode_stances, dict) else None,
                current=current,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    # Budget gates run BEFORE the dry-run early-return so users can validate
    # their cap is configured correctly without spending real-call dollars.
    max_cost_usd = getattr(args, "max_cost_usd", None)
    max_tokens = getattr(args, "max_tokens", None)
    # M6 soft cost-warning threshold. None-aware precedence (NOT an `or`
    # chain, so an explicit 0 at either layer is respected): CLI flag >
    # defaults.cost_warn_usd.
    cost_warn_usd = getattr(args, "cost_warn_usd", None)
    if cost_warn_usd is None:
        cost_warn_usd = config.get("defaults", {}).get("cost_warn_usd")
    # Compute the pre-flight estimate once and reuse it for: the hard
    # budget gate (when a cap is set), the M6 soft warning, and the L7
    # compact metadata echo. allow_network=False keeps it cheap (cached
    # catalog only) so the always-on L7 echo adds no meaningful latency.
    preflight: dict[str, Any] | None = None
    cost_warning_payload: dict[str, Any] | None = None
    cost_estimate_block: dict[str, Any] | None = None
    try:
        preflight = estimate_council(
            config=config,
            cwd=cwd,
            question=persisted_question,
            mode=mode,
            current=current,
            explicit=parse_csv(args.participants),
            include=parse_csv(args.include),
            origin_policy=args.origin_policy,
            context_paths=args.context,
            include_diff=args.diff,
            stdin_text=stdin_text,
            allow_outside_cwd=args.allow_outside_cwd,
            deliberate=deliberate,
            max_rounds=max_rounds,
            use_cache=True,
            allow_network=False,
            image_paths=args.image,
            prepared_prompt=prompt,
            prepared_participants=participants,
            participant_prompts=participant_prompts,
            synthesize=synthesize,
            synthesizer_name=synthesizer_name,
            deliberation_prompt_chars=deliberation_prompt_bounds,
        )
    except (OSError, ValueError) as exc:
        # The hard caps MUST fail closed if the estimate can't be computed;
        # the soft warning / echo are best-effort and degrade silently.
        if max_cost_usd is not None or max_tokens is not None:
            raise SystemExit(
                f"failed to compute pre-flight estimate: {exc}"
            ) from exc
    if preflight is not None:
        from llm_council.estimate import compact_cost_estimate

        cost_estimate_block = compact_cost_estimate(preflight)
        if max_cost_usd is not None or max_tokens is not None:
            # enforce_preflight_caps uses the retry-safety total so a worst-case
            # repair/timeout recovery can't silently push spend past the cap,
            # and flags
            # hosted peers with unknown catalog price (which would otherwise slip
            # past a $-cap). Shared with cmd_estimate + the MCP run pipeline.
            try:
                budget.enforce_preflight_caps(
                    preflight,
                    max_cost_usd=max_cost_usd,
                    max_tokens=max_tokens,
                    breakdown_hint=(
                        "drop expensive peers, raise the cap, or run `llm-council "
                        "estimate ...` for a per-peer breakdown. To exclude the "
                        "outer-retry margin, set retries: 0 on individual "
                        "participants."
                    ),
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        # M6: soft warning off the SAME reduction the hard gate uses, so soft
        # and hard numbers can never drift. Only fires when the run proceeds
        # (the hard gate above would have refused first if both tripped).
        if cost_warn_usd is not None:
            soft_cost_total, _soft_tokens, _soft_unpriced = (
                budget.summarize_preflight_caps(preflight)
            )
            if soft_cost_total >= float(cost_warn_usd):
                cost_warning_payload = {
                    "estimated_usd": soft_cost_total,
                    "threshold_usd": float(cost_warn_usd),
                }
                if not args.json:
                    print(
                        f"warning: estimated ${soft_cost_total:.6f} exceeds "
                        f"cost_warn_usd ${float(cost_warn_usd):.6f}; proceeding.",
                        file=sys.stderr,
                        flush=True,
                    )

    if args.dry_run:
        # Surface the resolved per-peer model so callers can verify a tier
        # override (or any participant-level model: setting) actually landed
        # without having to run the council for real.
        participant_models = {
            name: (participant_cfg.get(name, {}) or {}).get("model")
            for name in participants
        }
        if args.json:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "mode": mode,
                        "current": current,
                        "participants": participants,
                        "participant_models": participant_models,
                        "prompt_chars": len(prompt),
                    },
                    indent=2,
                )
            )
        else:
            print(f"mode: {mode}")
            print(f"current: {current or 'unknown'}")
            print("participants: " + ", ".join(participants))
            models_line = ", ".join(
                f"{name}={participant_models[name] or 'default'}"
                for name in participants
            )
            print(f"models: {models_line}")
            print(f"prompt_chars: {len(prompt)}")
        return 0
    if not args.json:
        color = display.wants_color(sys.stderr)
        print(
            display.format_gutter(
                display.VERB_CONVENING,
                f"llm-council starting: mode={mode}, current={current or 'unknown'}, "
                f"participants={', '.join(participants)}, prompt_chars={len(prompt)}",
                color=color,
            ),
            flush=True,
        )
        if deliberate:
            print(
                display.format_gutter(
                    display.VERB_DELIBERATING,
                    f"enabled: max_rounds={max_rounds}",
                    color=color,
                ),
                flush=True,
            )
    # CLI flags override the config-file defaults for per-call toggles
    # the validator reads off `cfg`. None means "no flag given — keep
    # whatever the config has." Propagating into config["defaults"] lets
    # execute_council push them into each peer's cfg consistently.
    _cli_require_sections = getattr(args, "require_sections", None)
    _cli_strict_evidence = getattr(args, "strict_evidence", None)
    if _cli_require_sections is not None:
        config.setdefault("defaults", {})["require_sections"] = bool(_cli_require_sections)
    if _cli_strict_evidence is not None:
        config.setdefault("defaults", {})["strict_evidence"] = bool(_cli_strict_evidence)
    try:
        results, metadata = await execute_council(
            participants,
            participant_cfg,
            prompt,
            cwd,
            config,
            deliberate=deliberate,
            max_rounds=max_rounds,
            progress=None if args.json else _make_progress_printer(participants),
            image_manifest=image_manifest or None,
            min_quorum=min_quorum_value,
            mode=mode,
            cache_mode=getattr(args, "cache_mode", "on"),
            stances=mode_stances if isinstance(mode_stances, dict) else None,
            synthesize=synthesize,
            synthesizer_name=synthesizer_name,
            current=current,
            question=persisted_question,
            deliberation_prompt_cap=config.get("defaults", {}).get(
                "max_prompt_chars"
            ),
        )
    except ValueError as exc:
        # The nested-council refusal is an operator-facing guard, not a bug;
        # present it as a clean CLI error instead of a traceback.
        if str(exc).startswith("NestedCouncilRefused:"):
            raise SystemExit(str(exc)) from exc
        raise
    from llm_council.deliberation import summarize_recommendations

    final_results = final_round_results(results)
    recommendation_summary = summarize_recommendations(final_results)
    metadata["recommendation"] = recommendation_summary.recommendation
    metadata["agreement_count"] = recommendation_summary.agreement_count
    metadata["total_labeled"] = recommendation_summary.total_labeled
    metadata["recommendation_counts"] = dict(recommendation_summary.counts)
    metadata["recommendation_tied"] = recommendation_summary.tied
    # Record the secret-scan result in metadata for transcript-based
    # audit tooling. The stderr warning above is for the live terminal;
    # transcripts need their own copy because audit pipelines don't read
    # stderr. Keep parity with mcp_server.run_council, which stamps the
    # same field.
    if scan_result.get("detected_count") or scan_policy != "off":
        metadata["secret_scan"] = scan_result
    # Record the pre-run independent-review suppression (only when it actually
    # occurred — see resolution above). Surfaced as a metadata flag rather than
    # a mid-run progress event because the decision precedes execute_council.
    if prior_context_suppressed:
        metadata["prior_context_suppressed_for_independence"] = True
    # Surface any non-configuration synthesis failure reported after the
    # early chair validation (for example, a chair call that fails at runtime).
    if metadata.get("synthesis_error"):
        print(
            display.format_gutter(
                "warn",
                f"synthesis skipped: {metadata['synthesis_error']}",
                color=display.wants_color(sys.stdout),
            ),
            flush=True,
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
    # L7: compact cost-estimate echo so a caller who skipped `estimate`
    # still sees the cost signal in the transcript metadata.
    if cost_estimate_block is not None:
        metadata["cost_estimate"] = cost_estimate_block
    # M6: non-fatal soft cost-warning (omitted when not triggered).
    if cost_warning_payload is not None:
        metadata["cost_warning"] = cost_warning_payload

    md_path, json_path = transcript_paths(out_dir, persisted_question)
    write_transcript(
        md_path,
        json_path,
        question=persisted_question,
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
                            "wall_elapsed_seconds": (
                                round(result.wall_elapsed_seconds, 3)
                                if result.wall_elapsed_seconds is not None
                                else None
                            ),
                            "error": result.error,
                            "error_kind": classify_error(result.error),
                            "model": result.model,
                            "total_tokens": result.total_tokens,
                            "cost_usd": result.cost_usd,
                            "from_cache": result.from_cache,
                            "cache_hit_seconds": result.cache_hit_seconds,
                            "recovered_after_launch_retry": result.recovered_after_launch_retry,
                            "repair_retry_recovered": result.repair_retry_recovered,
                            "recovered_after_timeout": result.recovered_after_timeout,
                            "prompt_chars": result.prompt_chars,
                            "stance": result.stance,
                            "effort": result.effort,
                            "confidence": result.confidence,
                            "risk": result.risk,
                            "blockers": list(result.blockers),
                            "evidence": list(result.evidence),
                            "tests_to_run": list(result.tests_to_run),
                            "assumptions": list(result.assumptions),
                        }
                        for result in results
                    ],
                },
                indent=2,
            )
        )
    else:
        from llm_council.adapters import is_timeout_error

        ok = sum(1 for result in final_results if result.ok)
        timed_out = [result for result in results if is_timeout_error(result.error)]
        color = display.wants_color(sys.stdout)
        unicode_safe = display.wants_unicode_rule(sys.stdout)
        complete_word = display.colorize_status("complete", color=color)
        print(
            display.format_gutter(
                display.VERB_CONCLUDED,
                f"llm-council {complete_word}: {ok}/{len(final_results)} participants succeeded",
                color=color,
            )
        )
        if metadata.get("deliberated"):
            print(
                display.format_gutter(
                    display.VERB_DELIBERATING,
                    "second round ran after disagreement detection",
                    color=color,
                )
            )
        for result in results:
            if result.ok:
                status = "ok"
            elif is_timeout_error(result.error):
                status = "timeout"
            else:
                status = "error"
            colored_status = display.colorize_status(status, color=color)
            print(
                display.format_gutter(
                    result.name,
                    f"{colored_status} ({result.elapsed_seconds:.1f}s)",
                    color=color,
                )
            )
            if transparent:
                details = []
                if result.total_tokens is not None:
                    details.append(f"{result.total_tokens} tokens")
                if result.cost_usd is not None:
                    details.append(f"${result.cost_usd:.6f}")
                if details:
                    print(
                        display.format_gutter(
                            "", "; ".join(details), color=color
                        )
                    )
            if not result.ok:
                print(display.format_gutter("", result.error, color=color))
        if timed_out:
            names = ", ".join(
                sorted({r.name.split(":round")[0] for r in timed_out})
            )
            print(
                display.format_gutter(
                    "Note",
                    f"{names} timed out. Increase "
                    f"`participants.<name>.timeout` in `.llm-council.yaml` "
                    "for slower models, or shorten the prompt.",
                    color=color,
                )
            )
        # Horizontal rule → transcript path. The rule's gutter token is
        # blank; the rule itself is the content. Above the path so the
        # reader's eye lands on the path last (council-recommended).
        print(
            display.format_gutter(
                "",
                display.horizontal_rule(
                    unicode_safe=unicode_safe, color=color
                ),
                color=color,
            )
        )
        print(display.format_gutter("Transcript", str(md_path), color=color))

    # Notification webhooks
    notify_cfg = config.get("notifications")
    if isinstance(notify_cfg, dict) and notify_cfg.get("webhook_url"):
        webhook_url = notify_cfg.get("webhook_url")
        counts = recommendation_summary.counts
        summary_text = (
            f"LLM Council Run Finished!\n"
            f"Question: {question[:150] if question else 'None'}...\n"
            f"Recommendation: {metadata.get('recommendation', 'unknown')}\n"
            f"Votes: {counts}\n"
            f"Transcript: {md_path}\n"
        )
        import httpx
        try:
            httpx.post(webhook_url, json={"text": summary_text}, timeout=5.0)
        except Exception as e:
            print(f"Warning: Failed to send notification: {e}", file=sys.stderr)

    # Quorum policy enforcement
    policies = config.get("quorum_policies")
    if isinstance(policies, dict):
        counts = recommendation_summary.counts
        total_votes = counts["yes"] + counts["no"] + counts["tradeoff"]
        
        active_policy = policies.get("standard")
        
        # Scan files in git diff and --context to find if they match any key
        changed_files = []
        if args.diff:
            from llm_council.context import _git_output
            git_staged_files = _git_output(cwd, ["diff", "--cached", "--name-only"])
            git_unstaged_files = _git_output(cwd, ["diff", "--name-only"])
            for f_list in (git_staged_files, git_unstaged_files):
                if f_list:
                    changed_files.extend(f_list.splitlines())
        for c_file in args.context:
            changed_files.append(c_file)
            
        # Match file patterns
        for pattern, pol in policies.items():
            if pattern == "standard":
                continue
            for f in changed_files:
                if pattern in f:
                    active_policy = pol
                    break
                    
        if isinstance(active_policy, dict):
            threshold = active_policy.get("threshold", "majority")
            if total_votes == 0:
                print(
                    "\nQUORUM ERROR: Policy failed closed because the final "
                    "round produced no usable yes/no/tradeoff votes.",
                    file=sys.stderr,
                )
                return 1
            if threshold == "unanimous":
                if counts["no"] > 0 or counts["tradeoff"] > 0:
                    print(
                        f"\nQUORUM ERROR: Policy '{threshold}' failed. "
                        f"Found {counts['no']} 'no' votes and {counts['tradeoff']} 'tradeoff' votes.",
                        file=sys.stderr
                    )
                    return 1
            elif threshold == "majority":
                if total_votes > 0 and counts["yes"] <= total_votes / 2:
                    print(
                        f"\nQUORUM ERROR: Policy '{threshold}' failed. "
                        f"Only {counts['yes']}/{total_votes} voted 'yes'.",
                        file=sys.stderr
                    )
                    return 1

    # Auto-open HTML transcript in browser if configured or requested
    auto_open = False
    if getattr(args, "open", False):
        auto_open = True
    else:
        defaults_cfg = config.get("defaults", {})
        if isinstance(defaults_cfg, dict) and defaults_cfg.get("auto_open_browser"):
            auto_open = True
            
    if auto_open:
        html_path = md_path.with_suffix(".html")
        if html_path.is_file():
            import webbrowser
            if not getattr(args, "json", False):
                print(f"[Auto-Open] Opening transcript: {html_path}")
            if not webbrowser.open(html_path.resolve().as_uri()):
                print(f"Warning: Failed to auto-open browser for transcript {html_path}", file=sys.stderr)

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
    if args.command == "install-hook":
        return cmd_install_hook(args)
    if args.command == "config":
        return cmd_config(args)
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
