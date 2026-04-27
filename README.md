# LLM Council

Get a second opinion from other LLMs without leaving your coding CLI.

LLM Council lets Claude Code, Codex CLI, Gemini CLI, OpenRouter models, and
local Ollama models review the same question, plan, diff, or architecture
choice. It is read-only by default: council participants are asked to analyze
and advise, not edit your files.

Use it when one model is not enough: architecture tradeoffs, risky refactors,
security-sensitive changes, release gates, stubborn bugs, or any point where
you want another model to challenge the plan before you act.

## What It Does

From your terminal:

```bash
llm-council run --current codex --mode review --diff "Review this change before I ship it"
```

Or from an MCP-enabled coding agent:

```text
Use llm-council to review this plan with the other models before editing files.
```

A run shows what the council is doing, then writes a transcript:

```text
Council starting: mode=quick, current=codex, participants=claude, gemini
- claude: starting round 1
- gemini: starting round 1
- gemini: ok round 1
- claude: ok round 1
Council complete: 2/2 participants succeeded
Transcript: .llm-council/runs/20260427_090457_review.md
```

## Choose Your Setup

| Your situation | Best preset | What it uses |
| --- | --- | --- |
| You have one coding CLI and want more opinions | `openrouter` | Hosted API models through OpenRouter |
| You have Claude Code, Codex CLI, and/or Gemini CLI installed | `auto` or `tri-cli` | Your existing CLI accounts |
| You want native CLIs plus hosted frontier or budget reviewers | `tri-cli-openrouter` | CLI accounts plus OpenRouter |
| You want local/private review | `local-private` | Ollama on your machine |

`setup --yes` defaults to `--preset auto`. Auto setup writes a config only when
it finds a usable council route: at least two native CLIs, or
`OPENROUTER_API_KEY` for hosted reviewers. If you only have one CLI installed,
set up OpenRouter or choose a preset explicitly.

## Quick Start

Install the command, then run setup from the project you want to use council in:

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/your/project
llm-council setup --yes
llm-council doctor
```

If you do not use `uv`, use `pipx`:

```bash
pipx install --force git+https://github.com/Intellimetrics/llm-council.git
```

Do not use `uvx` for project installation. `setup` writes the MCP command into
the target project's `.mcp.json`, so the project needs a stable installed
`llm-council` executable.

Common setup choices:

```bash
llm-council setup --yes --preset auto
llm-council setup --yes --preset openrouter
llm-council setup --yes --preset tri-cli
llm-council setup --yes --preset tri-cli-openrouter
llm-council setup --yes --preset local-private
```

Setup writes:

- `.llm-council.yaml`
- `.mcp.json`
- `.llm-council/instructions/claude.md`
- `.llm-council/instructions/codex.md`
- `.llm-council/instructions/gemini.md`

Restart your coding CLI after setup so `.mcp.json` and project instructions
reload.

## Let Your Coding Agent Install It

Paste one of these prompts into the coding CLI you are using inside the target
project.

### Claude Code

```text
Install LLM Council into this project. Follow these steps exactly and stop if any step fails.

1. Run `command -v uv`. If it exists, run `uv tool install --force git+https://github.com/Intellimetrics/llm-council.git`.
2. If `uv` is not installed, run `command -v pipx`. If it exists, run `pipx install --force git+https://github.com/Intellimetrics/llm-council.git`.
3. Do not use `uvx`.
4. From the project root, run `llm-council setup --yes`.
5. If setup says there is no usable route, ask me whether to set `OPENROUTER_API_KEY` or install another native CLI. Do not guess.
6. Append the full contents of `.llm-council/instructions/claude.md` to `CLAUDE.md` in the project root. Create `CLAUDE.md` if needed. Do not overwrite existing content.
7. Run `llm-council doctor` and show me the output.
8. Tell me to restart this Claude Code session so MCP and project instructions reload.
```

### Codex CLI

```text
Install LLM Council into this project. Follow these steps exactly and stop if any step fails.

1. Run `command -v uv`. If it exists, run `uv tool install --force git+https://github.com/Intellimetrics/llm-council.git`.
2. If `uv` is not installed, run `command -v pipx`. If it exists, run `pipx install --force git+https://github.com/Intellimetrics/llm-council.git`.
3. Do not use `uvx`.
4. From the project root, run `llm-council setup --yes`.
5. If setup says there is no usable route, ask me whether to set `OPENROUTER_API_KEY` or install another native CLI. Do not guess.
6. Append the full contents of `.llm-council/instructions/codex.md` to `AGENTS.md` in the project root. Create `AGENTS.md` if needed. Do not overwrite existing content.
7. Run `llm-council doctor` and show me the output.
8. Tell me to restart this Codex session so MCP and project instructions reload.
```

### Gemini CLI

```text
Install LLM Council into this project. Follow these steps exactly and stop if any step fails.

1. Run `command -v uv`. If it exists, run `uv tool install --force git+https://github.com/Intellimetrics/llm-council.git`.
2. If `uv` is not installed, run `command -v pipx`. If it exists, run `pipx install --force git+https://github.com/Intellimetrics/llm-council.git`.
3. Do not use `uvx`.
4. From the project root, run `llm-council setup --yes`.
5. If setup says there is no usable route, ask me whether to set `OPENROUTER_API_KEY` or install another native CLI. Do not guess.
6. Append the full contents of `.llm-council/instructions/gemini.md` to `GEMINI.md` in the project root. Create `GEMINI.md` if needed. Do not overwrite existing content.
7. Run `llm-council doctor` and show me the output.
8. Tell me to restart this Gemini session so MCP and project instructions reload.
```

### Verify An Agent Install

```text
Verify the LLM Council install in this project:
- `.mcp.json` exists and has an `llm-council` MCP server entry.
- `.llm-council.yaml` exists.
- The generated instruction snippet for this CLI exists under `.llm-council/instructions/`.
- The snippet was appended to the right project instruction file: `CLAUDE.md`, `AGENTS.md`, or `GEMINI.md`.
- `llm-council doctor` passes or explains exactly what is missing.
- `llm-council --version` prints a version.
```

## Wire Your CLI Manually

`setup` creates instruction snippets, but it does not overwrite your project
instruction files. Append the right snippet yourself if you are not using the
agent prompts above:

- Claude Code: append `.llm-council/instructions/claude.md` to `CLAUDE.md`.
- Codex CLI: append `.llm-council/instructions/codex.md` to `AGENTS.md`.
- Gemini CLI: append `.llm-council/instructions/gemini.md` to `GEMINI.md`.

Restart the relevant CLI session after editing `.mcp.json` or instruction
files.

## First Runs

Preview the route without calling participants:

```bash
llm-council run --dry-run --current codex "Should we split this service?"
```

Ask for a review:

```bash
llm-council run --current codex --mode review "Review this architecture plan"
```

Include the current git diff:

```bash
llm-council run --current claude --mode review --diff "Find risks in this change"
```

Show usage, cost when available, and a model-by-model comparison:

```bash
llm-council run --transparent --current codex --mode review "Compare the risks"
```

Ask whether a task is worth taking to council:

```bash
llm-council recommend --risk high "Refactor auth and database session handling"
```

Read transcripts:

```bash
llm-council last
llm-council transcripts list
llm-council transcripts summary
```

## Costs In 30 Seconds

- Native CLI participants use your installed Claude Code, Codex CLI, or Gemini
  CLI account. Billing and limits are whatever those tools use.
- OpenRouter participants are hosted API calls billed by token. Use
  `llm-council estimate` before spending money.
- Ollama participants run locally. They use your machine, not a hosted API.

Estimate first:

```bash
llm-council estimate --mode review-cheap "Review this plan"
llm-council estimate --mode review-cheap --completion-tokens 2500 "Review this diff"
```

Use cheap hosted modes for broad feedback. Escalate to frontier hosted models
for release gates, security-sensitive decisions, architecture tradeoffs, or
bugs where cheaper reviewers are not enough.

## MCP Tools

The generated `.mcp.json` exposes an MCP (Model Context Protocol) server named
`llm-council`. Coding agents can call:

- `council_run`: ask the configured council a question.
- `council_estimate`: estimate prompt size and hosted model cost before a run.
- `council_recommend`: ask whether council is worth using for a task.
- `council_doctor`: check local setup, version, and optional update status.
- `council_list_modes`: inspect configured modes and participants.
- `council_last_transcript`: fetch the latest transcript path or content.
- `council_models`: inspect configured or hosted model choices.

MCP calls return participant progress in `metadata.progress_events` when the
tool call completes.

## Manual MCP Install

Prefer `llm-council setup --yes`. If you must configure MCP by hand, resolve
the executable path dynamically.

Autonomous agents configuring this MCP server must run `command -v llm-council`
and must not copy the placeholder path below literally.

```json
{
  "mcpServers": {
    "llm-council": {
      "type": "stdio",
      "command": "/absolute/path/from/command-v/llm-council",
      "args": ["mcp-server"],
      "env": {
        "LLM_COUNCIL_MCP_ROOT": "/absolute/path/to/your/project"
      }
    }
  }
}
```

If the project already has `.mcp.json`, add only the `llm-council` entry under
the existing `mcpServers` object.

## Safety

- Council participants are instructed to be read-only.
- The MCP server is scoped to the configured project root.
- Generated MCP config does not embed API keys.
- Secrets can live in `.env`, `.env.local`, or `.llm-council.env`.
- Prompt-size guards skip oversized participants before long timeouts where
  possible.

## Versioning And Updates

```bash
llm-council --version
llm-council check-update
llm-council doctor --check-update
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
```

Releases are recorded in [CHANGELOG](CHANGELOG.md) and tagged as `vX.Y.Z`.

## More

- [Operator reference](docs/llm-council.md): config, participants, costs, MCP
  details, and custom modes.
- [Model catalog notes](docs/model-catalog-2026-04-25.md): model selection
  context from the initial release.
- [Changelog](CHANGELOG.md): release history.
