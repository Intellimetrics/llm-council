# LLM Council

LLM Council is a lightweight, read-only-first multi-agent tool for coding projects.

Baseline assumption: the user usually works inside one of three premium CLIs:

- Claude Code
- Codex CLI
- Gemini CLI

By default, `llm-council` asks the other two CLIs for read-only opinions. OpenRouter and local models are explicit participants, not fallbacks.

## Quick Start

From this checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
llm-council run --dry-run --current codex "Should we add an MCP wrapper?"
```

Project setup:

```bash
llm-council setup --yes
llm-council doctor
llm-council doctor --probe-openrouter --probe-ollama
```

The setup command writes:

- `.llm-council.yaml`
- `.mcp.json`
- `.llm-council/instructions/claude.md`
- `.llm-council/instructions/codex.md`
- `.llm-council/instructions/gemini.md`

The generated `.mcp.json` exposes the `llm-council` MCP server.

Secrets can live in a local `.env`, `.env.local`, or `.llm-council.env` file.
These files are loaded from the project directory and parent directories without
overriding already-exported environment variables.

Live call:

```bash
llm-council run --current codex "Review this architecture decision"
```

Turn on transparency for terminal usage/cost details and a transcript comparison:

```bash
llm-council run --transparent --current codex --mode review "Compare the risks"
```

Run expensive opt-in deliberation. This runs a second round only when first-round
responses appear to disagree:

```bash
llm-council run --mode deliberate --transparent "Decide between these designs"
llm-council run --deliberate --max-rounds 2 "Decide between these designs"
```

Include a file and git diff:

```bash
llm-council run --current claude --mode review --context server.py --diff "Find risks in this change"
```

Ask whether a task is worth taking to council:

```bash
llm-council recommend --risk high "Refactor auth and database session handling"
```

Read the latest transcript:

```bash
llm-council last --path-only
llm-council last
llm-council transcripts list
llm-council transcripts summary
```

Use explicit participants:

```bash
llm-council run --participants deepseek_v4_pro,qwen_coder_plus "Compare these two designs"
```

US-origin only:

```bash
llm-council run --mode us-only "Review this plan"
llm-council run --mode diverse --origin-policy us "Review this plan"
```

For users who want this as the default:

```bash
llm-council setup --yes --us-only-default
```

## Current-Agent Routing

`quick`, `plan`, and `review` use `strategy: other_cli_peers`.

If current is `codex`, default native participants are:

- `claude`
- `gemini`

If current is unknown, all three native CLIs are used.

You can set current explicitly:

```bash
LLM_COUNCIL_CURRENT=gemini llm-council run "Question"
```

or:

```bash
llm-council run --current gemini "Question"
```

## Model Routing

Native CLI participants do not specify models by default. This lets paid/max CLI accounts use their own best current defaults.

Use explicit models only when you want exact routing:

```yaml
participants:
  claude:
    model: claude-opus-4-7
  gemini:
    model: gemini-3.1-pro-preview
  codex:
    model: gpt-5.4
```

OpenRouter participants must specify models:

```yaml
participants:
  deepseek_v4_pro:
    type: openrouter
    model: deepseek/deepseek-v4-pro
    api_key_env: OPENROUTER_API_KEY
```

Refresh the live OpenRouter catalog when deciding what to add:

```bash
llm-council models openrouter --filter deepseek
llm-council models openrouter --origin china --limit 20
llm-council models openrouter --origin us --limit 20
llm-council models openrouter --no-cache --filter qwen
```

Prices are normalized to dollars per million input/output tokens. Origin labels
are inferred from provider IDs, so treat unknown or edge cases as a starting
point rather than a compliance database. OpenRouter model listings are cached
under `~/.cache/llm-council` for one hour by default.

## Safety

The built-in CLI defaults are read-only:

- Claude: `--permission-mode plan` with read/search/list tools only
- Codex: `exec --sandbox read-only --ask-for-approval never --ephemeral`
- Gemini: `--approval-mode plan`

The prompt also tells participants not to edit files or run write operations.

## MCP Tools

The MCP server exposes:

- `council_run`: run a read-only council
- `council_recommend`: decide whether a task is worth taking to council
- `council_list_modes`: list configured modes and participants
- `council_last_transcript`: read the most recent local transcript
- `council_doctor`: diagnose CLI, OpenRouter, Ollama, and MCP readiness
- `council_models`: list cached OpenRouter models with filter/origin options

`council_run` accepts `transparent`, `deliberate`, and `max_rounds` for the same
usage/cost and opt-in deliberation behavior as the CLI. Context files are
restricted to `working_directory` by default; pass `allow_outside_cwd` only when
external files are intentionally part of the prompt.

Run it manually:

```bash
python -m llm_council mcp-server
```

or:

```bash
python -m llm_council.mcp_server
```

## Keyword Layer

Setup writes instruction snippets for each CLI. The intended user language is:

```text
go to council
go to council with deepseek
go to council with qwen and glm
go to council on the diff
go to council cheap
go to council private
ask council if this plan is sound
```

Agents should use `council_recommend` when the task looks risky, broad, ambiguous,
or has already failed multiple times. Council should stay manual or suggested,
not automatic for every task.

## Threat Model

Council participants receive the prompt, selected context files, optional git
diffs, and stdin context you provide. Native CLI participants run as local
subprocesses in read-only/plan modes. OpenRouter participants send the prompt to
OpenRouter and the selected upstream model. Ollama participants send the prompt
to your configured local Ollama server.

Do not include secrets in context files or diffs. By default, context files must
be inside the working directory; use `--allow-outside-cwd` only when you intend
to share external files. Subprocess CLI participants do not inherit common API
key environment variables by default.

## When Not To Use Council

Skip council for formatting, exact user-directed edits, obvious syntax fixes,
single-file changes you already understand, or tasks where extra model calls
would add cost without reducing risk.

## Transcripts

Each run writes:

- `.llm-council/runs/<timestamp>_<slug>.md`
- `.llm-council/runs/<timestamp>_<slug>.json`

These are intentionally local run artifacts.

