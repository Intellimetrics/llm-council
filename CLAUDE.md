# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`llm-council` is a Python 3.11+ MCP server and CLI that lets one coding agent
ask a "council" of other LLMs (Claude Code, Codex CLI, Gemini CLI, OpenRouter
hosted models, local Ollama) for read-only second opinions. It is published as
the `llm-council` console script and as the `llm-council` MCP server.

## Common commands

```bash
# Install editable with dev deps
python -m pip install -e ".[dev]"

# Run the full test suite
pytest -q

# Run a single test file or test
pytest tests/test_config_validation.py -q
pytest tests/test_llm_council.py::test_specific_name -q

# Local CLI invocation without install
python -m llm_council --help

# Setup wizard (writes .llm-council.yaml, .mcp.json, instruction snippets)
llm-council setup --plan                       # detect routes, do not write
llm-council setup --yes --preset tri-cli       # write without prompting

# Diagnostics
llm-council doctor [--probe-openrouter] [--probe-ollama] [--check-update]

# Direct council run (skips MCP)
llm-council run --current codex --mode review --diff "Review this change"

# Run as MCP server over stdio (what `.mcp.json` invokes)
llm-council mcp-server
```

CI (`.github/workflows/test.yml`) runs `pytest -q` on Python 3.11 and 3.12.
There is no separate lint/format step configured.

## Architecture

The codebase is small and single-package (`llm_council/`). Two surfaces share
the same core:

1. `cli.py` (`main` -> `llm-council` script)
2. `mcp_server.py` (`llm-council mcp-server`, exposing `council_run`,
   `council_estimate`, `council_recommend`, `council_doctor`,
   `council_list_modes`, `council_last_transcript`, `council_models`,
   `council_stats`)

Both flow through the same pipeline:

```
load_project_env -> load_config (defaults + project YAML)
                 -> select_participants (mode + current CLI + origin filter)
                 -> build_prompt (question + optional context files / diff / stdin)
                 -> execute_council (orchestrator)
                    -> run_participants (adapters: cli | openrouter | ollama)
                    -> optional deliberation round on disagreement
                 -> write_transcript (markdown + json under transcripts_dir)
```

Key modules:

- `defaults.py` — built-in `DEFAULT_CONFIG`. Project YAML is deep-merged on top
  via `config.deep_merge`. The set of legal participant types (`cli`,
  `openrouter`, `openai_compatible`, `ollama`) and built-in modes (`quick`,
  `peer-only`, `plan`, `review`, `review-cheap`, `diverse`, `private-local`,
  `local-only`, `us-only`, `deliberate`, `consensus`, plus the temporary
  `opus-versions`) live here.
- `config.py` — config discovery (`find_config` walks up from cwd looking for
  `.llm-council.yaml` etc.), validation, the `other_cli_peers` strategy used by
  most modes, `detect_current_agent` (parent-process walk on `/proc`), and
  `migrate_known_cli_defaults` which silently rewrites previously generated
  unsafe Claude/Codex args at load time.
- `adapters.py` — three execution paths. CLI participants run via
  `asyncio.create_subprocess_exec` with `{prompt}`/`{cwd}` template
  substitution and prompt-on-stdin by default; OpenRouter uses `httpx`;
  Ollama hits a local `/api/chat`. Successful CLI output without a
  `RECOMMENDATION: yes|no|tradeoff` label is treated as failure
  (`_response_validation_error`).
- `orchestrator.py` — runs round 1, then optional deliberation rounds (helpers
  live in `deliberation.py`). Emits `progress_events` consumed both by the
  CLI's stream output and by the MCP tool's `metadata.progress_events` field.
- `policy.py` — `should_use_council` heuristic callers use to decide whether
  invoking the council is worth the cost for a given request.
- `update_check.py` — backs `llm-council doctor --check-update` and the
  startup version nag.
- `context.py` — builds the user-facing prompt and enforces `MAX_PROMPT_CHARS`.
- `setup_wizard.py` — writes `.llm-council.yaml`, `.mcp.json`, and the
  per-CLI instruction snippets in `.llm-council/instructions/`. Setup is
  guarded by `_preset_status` in `cli.py`; presets whose required CLIs/keys
  are missing are blocked unless `--allow-incomplete` is passed.
- `budget.py` / `estimate.py` / `model_catalog.py` — token/cost estimation
  and OpenRouter model catalog fetch (cached on disk).
- `transcript.py` — paired markdown + JSON transcripts under
  `.llm-council/runs/`. `latest_transcript` and `transcript_records` back the
  `last` and `transcripts` subcommands.

## Invariants worth preserving

- **Read-only by default.** Council participants must not edit files. CLI
  adapters pass flags like `--permission-mode default` (Claude),
  `--sandbox read-only` (Codex), `--approval-mode plan` (Gemini). Don't
  remove these from `defaults.py` without an explicit reason.
- **`RECOMMENDATION:` label.** CLI output is rejected if it lacks the label;
  prompts in `context.py` ask for it. Adapter and prompt changes must keep
  these in sync.
- **Config migration is silent.** `migrate_known_cli_defaults` rewrites old
  `OLD_CLAUDE_PLAN_ARGS` / `OLD_CODEX_APPROVAL_ARGS` and back-fills
  `peer-only` mode and `include_current` for built-in `other_cli_peers`
  modes. When changing baseline args in defaults, update the migration
  constants too.
- **Prompt-size guard.** `max_prompt_chars` is enforced both globally and
  per-participant before any subprocess launches; preserve this so oversized
  prompts fail fast rather than after a long hosted/CLI timeout.
- **`.mcp.json` stays local.** Setup adds it to `.gitignore`. It contains
  absolute paths and must not be committed.
- **Version bumps.** `__version__` in `llm_council/__init__.py` and the
  `version` in `pyproject.toml` and the README badge are kept in sync, with
  a matching `CHANGELOG.md` entry. Releases are tagged `vX.Y.Z`.

## Failure taxonomy

`adapters.classify_error(error)` maps any non-empty result.error to a stable
machine-readable kind (also surfaced as `error_kind` in transcripts and
`--json` stdout). Add new kinds explicitly here when introducing a new
failure path; do not let strings drift.

| `error_kind`         | When                                                                                |
|----------------------|-------------------------------------------------------------------------------------|
| `timeout`            | Participant exceeded its `timeout`. Prefix: `Timeout:` or `TimeoutError:`           |
| `context_overflow`   | Estimated tokens exceed `max_context_tokens`. Prefix: `ContextOverflowExcluded:`    |
| `prompt_too_large`   | Prompt skipped before launch (per-participant `max_prompt_chars`)                    |
| `invalid_response`   | CLI/HTTP succeeded but lacked `RECOMMENDATION:` label after one repair retry         |
| `downstream_error`   | httpx / hosted-API failures (HTTPStatusError, ConnectError, ReadTimeout, etc.)       |
| `cli_nonzero_exit`   | CLI participant exited with a nonzero status and empty stderr. Prefix: `CliExitNonZero:` |
| `preflight_failed`   | Local participant's `base_url` was unreachable at run start. Prefix: `PreflightFailed:` |
| `unknown`            | Non-empty error that did not match any known prefix — file a dogfood note            |

## Custom CLI participant: minimal template

When defining a one-off CLI participant (in `.llm-council.yaml` or a temp
config) the deep-merge from `defaults.py` only fills keys that exist on a
built-in baseline. For an entirely new family, you generally need:

```yaml
participants:
  my_cli:
    type: cli              # required: routes through the CLI adapter
    family: my_cli         # required when a participant doesn't share a baseline
    origin: us             # `us` | `china` | `unknown` — origin filtering
    command: my-cli        # binary on PATH (or absolute path)
    args: ["--flag"]       # optional; uses {prompt}/{cwd} template substitution
    model: my-model        # optional model identifier
    timeout: 240           # seconds before the participant is killed
    max_prompt_chars: 120000  # per-peer prompt cap (chunking targets this)
    read_only: true        # advisory marker; the read-only invariant is
                           # actually enforced by the per-CLI args baked
                           # into defaults.py (e.g. --permission-mode default
                           # for Claude, --sandbox read-only for Codex), so
                           # custom CLIs need to pass equivalent flags via
                           # `args` for the invariant to hold
    stdin_prompt: true     # whether the prompt is delivered via stdin (default)
                           # vs. {prompt} arg substitution
```

Forget `family` and the participant works but config validation may flag it
as orphaned. Forget `stdin_prompt: true` and an unsubstituted-`{prompt}`
arg gets shipped as literal text. The read-only invariant lives in the
host CLI's own permission flags (passed via `args`), not in the
`read_only:` key — that key is documentation-only today.

## Continuation chain depth

`continuation_id` (CLI `--continue`) prepends a summary of the prior
transcript. Each link summarizes only its immediate parent (not the full
history), so depth growth is linear, not exponential. Still, the default
`max_continuation_depth` of 5 caps how many parents can chain before the
run is refused — set `defaults.max_continuation_depth: <N>` in
`.llm-council.yaml` to override.

## Run-level budget caps

`--max-cost-usd` and `--max-tokens` (CLI) / `max_cost_usd`,
`max_tokens` (MCP `council_run`) gate the run on the **pre-flight
estimate** before any subprocess or HTTP call is made. The estimate sums
known `cost_usd` per participant from the OpenRouter catalog; free/local
peers count as $0 and unknown-cost peers (catalog miss) cannot be
enforced — those raise no error but are visible in the estimate. Use
`llm-council estimate ...` for a per-peer breakdown when a cap fails.
