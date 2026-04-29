# LLM Council

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-ready-2f855a)](docs/llm-council.md)
[![Read Only](https://img.shields.io/badge/default-read--only-6b7280)](#safety)
[![Version](https://img.shields.io/badge/version-0.3.1-111827)](CHANGELOG.md)

Give your coding agent a council of other models.

LLM Council is an MCP server for Claude Code, Codex CLI, Gemini CLI, and other
MCP-capable coding agents. Once installed, you do not normally run
`llm-council` by hand. You keep working in your usual agent and say things
like:

```text
Review this migration plan and use council before editing files.
```

```text
This auth refactor feels risky. Take it to council and compare the tradeoffs.
```

```text
Use council on the current diff. I want Claude, Codex, and Gemini to challenge the plan.
```

Your current agent asks the configured council for read-only review, shows what
the models said, and keeps a local transcript for auditability.

```text
Council starting: mode=quick, current=codex, participants=claude, codex, gemini
- claude: starting round 1
- codex: starting round 1
- gemini: starting round 1
- codex: ok round 1
- gemini: ok round 1
- claude: ok round 1
Council complete: 3/3 participants succeeded
Transcript: .llm-council/runs/20260427_090457_review.md
```

## Why Use It

Single-agent coding is fast, but it is also easy for one model to overfit its
own plan. LLM Council gives your agent a lightweight way to ask independent
reviewers for:

- architecture pushback before a big implementation
- security and data-handling review before touching sensitive paths
- second opinions on stubborn bugs
- release-gate review of a diff
- model diversity when one agent keeps looping
- local/private review through Ollama when hosted calls are not appropriate

Council participants are advisory and read-only by default. They are asked to
analyze, not edit.

## Install The Way Users Actually Will

Open the project where you want council installed, then paste this into your
active coding agent:

```text
Install LLM Council into this project from https://github.com/Intellimetrics/llm-council.

Use the agent-first install path:
1. Check for `uv` with `command -v uv`. If present, run:
   `uv tool install --force git+https://github.com/Intellimetrics/llm-council.git`
2. If `uv` is not installed, check for `pipx` with `command -v pipx`. If present, run:
   `pipx install --force git+https://github.com/Intellimetrics/llm-council.git`
3. Do not use `uvx`; this must be a stable project install.
4. From this project root, run `llm-council setup --plan`.
5. Show me the detected routes and ask which preset I want: `auto`, `tri-cli`, `openrouter`, `tri-cli-openrouter`, `local-private`, or `all`. Do not choose silently unless I explicitly say to use the recommendation.
6. Run `llm-council setup --yes --preset <my-choice>`.
7. If setup reports no usable council route, stop and ask me whether to set `OPENROUTER_API_KEY` or install another native CLI.
8. After setup, read the generated snippet for this CLI from `.llm-council/instructions/`, then append that file's full contents to the correct project instruction file without overwriting existing content:
   - Claude Code: `.llm-council/instructions/claude.md` -> `CLAUDE.md`
   - Codex CLI: `.llm-council/instructions/codex.md` -> `AGENTS.md`
   - Gemini CLI: `.llm-council/instructions/gemini.md` -> `GEMINI.md`
9. Confirm the destination file now contains the LLM Council routing rules.
10. Run `llm-council doctor` and show me the result.
11. Tell me to restart this CLI session so MCP and project instructions reload.
```

That is the primary install path. It avoids the common mistakes agents make:
using `uvx`, copying placeholder paths into `.mcp.json`, overwriting existing
project instructions, silently accepting the wrong preset, skipping the
instruction-file append step, or declaring success before `doctor` passes.

## After Install

Restart your coding agent, then talk naturally:

```text
Use council to review this plan before implementing it.
```

```text
Ask council whether this database migration is safe. Include the current diff.
```

```text
Take this bug to council. I want independent theories before we change code.
```

```text
Use cheap council first, then tell me whether this is worth a frontier review.
```

Generated project instructions teach your agent the routing rules:

- `go to council`, `ask council`, or `use council` calls `council_run`
- the active CLI passes its identity, so transcripts show which host will
  synthesize and act
- `quick` asks Claude, Codex, and Gemini as explicit read-only participants
- `peer-only` excludes the current host subprocess when you only want outside
  perspectives
- `on the diff` includes the current git diff
- `cheap` uses budget hosted reviewers
- `private` or `local` uses the local Ollama route
- council feedback is advisory unless you explicitly ask the agent to act

## Pick Your Council

| If you have... | Choose... | Best for... |
| --- | --- | --- |
| One coding CLI | `openrouter` | adding outside model opinions with one API key |
| Two or more of Claude Code, Codex CLI, Gemini CLI | `auto` | using accounts you already have |
| Native CLIs plus hosted models | `tri-cli-openrouter` | stronger diversity and frontier escalation |
| Local models through Ollama | `local-private` | private/offline review |

Agent installs should run `llm-council setup --plan` first and ask you which
preset to write. `llm-council setup --yes` uses `auto` only when you explicitly
accept the default. Auto writes a config only when it finds a usable route: at
least two native CLIs, or `OPENROUTER_API_KEY` for hosted reviewers. If you only
have one CLI account, OpenRouter is usually the easiest way to add additional
reviewers.

Setup stops before writing presets whose required CLI tools or API keys are
missing. In interactive mode, it asks for confirmation first. Advanced users can
add `--allow-incomplete` when they deliberately want to write config before
installing the missing tools.

## What Setup Creates

```text
.llm-council.yaml                  shared project council config
.mcp.json                          local MCP command and project path
.llm-council/instructions/*.md      snippets to append to agent instructions
.llm-council/runs/                  local transcripts
```

`.mcp.json` contains absolute paths for one machine. Setup adds it to
`.gitignore`. Commit `.llm-council.yaml` only if your team wants shared council
modes; each developer should run setup locally.

If `.mcp.json` was already committed:

```bash
git rm --cached .mcp.json
```

## Costs And Data Boundaries

Council can call different kinds of participants:

- Native CLI participants use your installed Claude Code, Codex CLI, or Gemini
  CLI account. Billing and limits are controlled by those tools.
- OpenRouter participants are hosted API calls billed by token. Run an estimate
  before expensive reviews.
- Ollama participants run locally on your machine.

Ask your agent:

```text
Estimate the council cost for reviewing the current diff before running it.
```

Or run directly:

```bash
llm-council estimate --mode review-cheap --diff "Review this change"
```

Do not use council for classified, CUI, regulated, customer, production,
credential, or `DEPLOY_MODE=secret` content unless every configured participant
is approved for that data. US-origin model/company origin is not the same as
GovCloud, FedRAMP, or enterprise data-handling approval.

## Manual Terminal Use

Most users will interact through their coding agent. The terminal command is
still useful for setup, diagnostics, transcripts, and occasional direct runs.

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/project
llm-council setup --plan
llm-council setup --yes --preset <chosen-preset>
llm-council doctor
llm-council check-update
```

Advanced staging example:

```bash
llm-council setup --yes --preset openrouter --allow-incomplete
```

Direct review:

```bash
llm-council run --current codex --mode review --diff "Review this change"
```

Transcript tools:

```bash
llm-council last
llm-council transcripts list
llm-council transcripts summary
```

## MCP Tools

The generated `.mcp.json` exposes an MCP server named `llm-council`.

| Tool | Purpose |
| --- | --- |
| `council_run` | ask the configured council a question |
| `council_estimate` | estimate prompt size and hosted cost before a run |
| `council_recommend` | ask whether council is worth using for a task |
| `council_doctor` | check setup, version, and optional update status |
| `council_list_modes` | inspect configured modes and participants |
| `council_last_transcript` | fetch the latest transcript path or content |
| `council_models` | inspect configured or hosted model choices |

MCP calls return participant progress in `metadata.progress_events` when the
tool call completes.

## Safety

- Council participants are instructed to be read-only.
- The MCP server is scoped to the configured project root.
- Generated MCP config does not embed API keys.
- Secrets can live in `.env`, `.env.local`, or `.llm-council.env`.
- Prompt-size guards skip oversized participants before long timeouts where
  possible.
- Hosted model calls are explicit through your config and provider keys.

## Update

```bash
llm-council --version
llm-council check-update
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
```

Releases are tagged as `vX.Y.Z` and recorded in [CHANGELOG](CHANGELOG.md).

## More

- [Operator reference](docs/llm-council.md): config, participants, costs, MCP
  details, and custom modes.
- [Model catalog notes](docs/model-catalog-2026-04-25.md): model selection
  context from the initial release.
- [Changelog](CHANGELOG.md): release history.
