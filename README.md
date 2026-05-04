# LLM Council

[![Tests](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-ready-2f855a)](docs/llm-council.md)
[![Read-only](https://img.shields.io/badge/default-read--only-6b7280)](#read-only-means-read-only)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Intellimetrics/llm-council?style=flat&color=yellow)](https://github.com/Intellimetrics/llm-council/stargazers)
[![Version](https://img.shields.io/badge/version-0.4.2-111827)](CHANGELOG.md)

Your coding agent is confident.

That is useful right up until it confidently ships a bad migration, hides the real bug behind a nicer-looking patch, or refactors the one file nobody should have touched.

**LLM Council** gives that agent a way to ask other models for read-only second opinions before it does something expensive.

It is a Python 3.11+ MCP server and CLI. It works with the tools developers already use: Claude Code, Codex CLI, Gemini CLI, hosted models through OpenRouter, and local models through Ollama.

The console command is `llm-council`. The MCP server name is also `llm-council`.

MIT licensed. Current version: `0.4.0`.

GitHub: <https://github.com/Intellimetrics/llm-council>

## Why this exists

A coding agent can move fast enough to be dangerous.

That is the point. You want the speed. You do not want a single model's blind spot to become your production incident.

LLM Council is for the moment when you pause and think:

> This looks plausible, but I want another model to try to break it.

Ask council. The peers read the same prompt or diff. They answer independently. They cannot edit your files. Each answer must end with one of three plain labels:

```text
RECOMMENDATION: yes
RECOMMENDATION: no
RECOMMENDATION: tradeoff
```

That label matters. It means your coding agent does not have to guess whether a long answer was approval, rejection, or "this depends." If a peer forgets the label, LLM Council treats that answer as failed instead of quietly pretending it was fine.

## Install

Use `uv tool install` for a real install:

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
```

Or use `pipx`:

```bash
pipx install --force git+https://github.com/Intellimetrics/llm-council.git
```

Do not use `uvx` for the install. `uvx` is good for a one-shot trial, but it does not give your project a stable tool install.

Now move into the project where your coding agent works:

```bash
cd /path/to/your/project
llm-council setup --plan
llm-council setup --yes --preset auto
llm-council doctor
```

Then restart your coding agent so it reloads MCP config and project instructions.

## The thirty-second version

Once installed, talk to your coding agent like this:

```text
Ask council to review the current diff before we ship it.
```

Or:

```text
Take this bug to council. I want independent theories, not another patch yet.
```

Or:

```text
Use private-local council. This code cannot leave my machine.
```

LLM Council is not another chat UI. It is a way for your existing agent to slow down at the right moment and ask peers to look for the thing it missed.

## What gets asked

You can ask a direct question:

```bash
llm-council run --mode quick "Why is this test flaky?"
```

You can ask about the current diff:

```bash
llm-council run --mode review --diff "Is this safe to merge?"
```

You can make the council argue harder:

```bash
llm-council run --mode consensus --diff "Should we ship this migration?"
```

And you can ask for the cost first:

```bash
llm-council estimate --mode consensus --diff "Should we ship this migration?"
```

Most day-to-day use should happen through MCP, from your coding agent. The CLI is there for setup, diagnosis, transcripts, and the occasional manual run.

## Pick a setup

`llm-council setup --plan` looks at what is already available on your machine and tells you which routes can work.

Then choose one preset:

| Preset | Use it when |
|---|---|
| `auto` | You want LLM Council to use the best working mix it can find. |
| `tri-cli` | You have Claude Code, Codex CLI, and Gemini CLI installed. |
| `openrouter` | You want hosted peers through one OpenRouter key. |
| `tri-cli-openrouter` | You want native CLIs plus hosted fallback or variety. |
| `local-private` | You want local Ollama peers only. |
| `all` | You want every available route configured. |

The setup flow is intentionally boring:

```bash
llm-council setup --plan
llm-council setup --yes --preset <name>
llm-council doctor
```

If setup cannot find enough working peers, add another native CLI, configure OpenRouter, or set up Ollama.

## Peers

LLM Council can ask three kinds of peers.

Native CLI peers use the tools you may already have installed:

- Claude Code
- Codex CLI
- Gemini CLI

Hosted peers go through OpenRouter.

Local peers go through Ollama.

That lets you choose the shape of the review. You can keep everything local, use accounts you already pay for, or add hosted models when you want a broader spread.

## Read-only means read-only

Council peers are reviewers, not co-authors.

Every native CLI peer is launched under that host's read-only or approval-only flags. The council process asks for opinions; it does not hand peers permission to edit your project.

LLM Council also rejects peer replies that do not include a `RECOMMENDATION: yes|no|tradeoff` line. A vague essay is not enough. Your agent gets an answer it can act on, or it gets a failure.

This is the whole bargain: let other models inspect the work without giving them the keys.

## Modes

Modes are named ways to assemble the council.

| Mode | What it is for |
|---|---|
| `quick` | A fast second opinion. Good default when your agent seems stuck. |
| `peer-only` | Outside voices only; excludes the current host CLI. |
| `plan` | Architecture and approach questions before code changes. |
| `review` | Diff review before merge or release. |
| `review-cheap` | Budget hosted review when you want a first pass before spending more. |
| `diverse` | A wider spread across Claude, Codex, Gemini, and OpenRouter. |
| `private-local` | Ollama only. Use when the prompt must stay on your machine. |
| `us-only` | Filters to US-origin participants. |
| `deliberate` | Forces a second round even if the first answers mostly agree. |
| `consensus` | Makes peers argue from different sides, then gives them a chance to revise. |

You can define your own modes in `.llm-council.yaml`. The full schema is in the [operator reference](docs/llm-council.md).

## Consensus

`consensus` is the mode to use when "looks fine" is not enough.

It gives peers different jobs: one looks for the case to proceed, one looks for the case against, and one tries to stay neutral. If they disagree, LLM Council runs another round where they can respond to the strongest opposing points.

That is useful for questions like:

```text
Is this migration safe to run during business hours?
```

```text
Does this auth change fix the bug, or does it just move the hole?
```

```text
Is this refactor actually equivalent?
```

A peer is never forced to defend something unsafe just because it was assigned a side. If the honest answer is "do not ship this," it can say so.

If the peers still disagree at the end, the transcript says that clearly. No fake unanimity.

## Costs, data boundaries, safety

LLM Council may call native CLIs, hosted models, or local models depending on your setup.

Native CLI peers use your installed Claude Code, Codex CLI, or Gemini CLI accounts. Their billing and rate limits belong to those tools.

OpenRouter peers are hosted API calls and may cost money by token.

Ollama peers run locally.

Use cost caps when you want a hard stop before anything runs:

```bash
llm-council run --mode consensus --diff \
  --max-cost-usd 0.50 \
  --max-tokens 200000 \
  "Is this migration safe to ship?"
```

Use an estimate when you just want to see the shape of the run:

```bash
llm-council estimate --mode consensus --diff \
  "Is this migration safe to ship?"
```

Secrets are not written into `.mcp.json`. Put API keys in `.env`, `.env.local`, or `.llm-council.env`.

Oversized prompts are refused before peers launch. Hosted peers without known pricing do not get to sneak past a cost cap.

> [!CAUTION]
> Do not use council for classified, CUI, regulated, customer, production,
> credential, or `DEPLOY_MODE=secret` content unless every configured
> participant is approved for that data. US-origin model/company origin is
> not the same as GovCloud, FedRAMP, or enterprise data-handling approval.

## MCP tools

Setup exposes an MCP server named `llm-council`.

It provides these tools:

| Tool | What your agent uses it for |
|---|---|
| `council_run` | Ask the council a question. |
| `council_estimate` | Estimate size and cost before asking. |
| `council_recommend` | Ask whether council is worth using for the current task. |
| `council_doctor` | Check whether setup is healthy. |
| `council_list_modes` | See configured modes and participants. |
| `council_last_transcript` | Fetch the latest transcript path or content. |
| `council_models` | Inspect configured or hosted model choices. |
| `council_stats` | Summarize past transcript usage. |

The important one is `council_run`. Your agent sends the prompt, mode, and optional diff context. LLM Council returns the peer answers, recommendation labels, and transcript path.

## Agent-driven install

The easiest way to install LLM Council into a project is to ask your coding agent to do it.

Paste this into Claude Code, Codex CLI, or Gemini CLI from the project root:

```text
Install LLM Council into this project from
https://github.com/Intellimetrics/llm-council.

Use the stable install path:
1. Check for `uv` with `command -v uv`. If present, run:
   `uv tool install --force git+https://github.com/Intellimetrics/llm-council.git`
2. If `uv` is not installed, check for `pipx`. If present, run:
   `pipx install --force git+https://github.com/Intellimetrics/llm-council.git`
3. Do not use `uvx`; that is only for a one-shot trial.
4. From this project root, run `llm-council setup --plan`.
5. Show me the detected routes and ask which preset I want:
   `auto`, `tri-cli`, `openrouter`, `tri-cli-openrouter`,
   `local-private`, or `all`.
6. Run `llm-council setup --yes --preset <my-choice>`.
7. If setup reports no usable council route, stop and ask me whether to
   set `OPENROUTER_API_KEY`, install another native CLI, or configure Ollama.
8. After setup, append the matching snippet from `.llm-council/instructions/`
   to the project instruction file without overwriting it:
   - Claude Code: `.llm-council/instructions/claude.md` -> `CLAUDE.md`
   - Codex CLI: `.llm-council/instructions/codex.md` -> `AGENTS.md`
   - Gemini CLI: `.llm-council/instructions/gemini.md` -> `GEMINI.md`
9. Confirm the destination file now contains the LLM Council routing rules.
10. Run `llm-council doctor` and show me the result.
11. Tell me to restart this CLI session so MCP and project instructions reload.
```

That last restart matters. Without it, your agent may not see the new MCP server or the new routing instructions.

## Manual use

These commands are useful when you are setting up, debugging, or reading old runs:

```bash
llm-council setup --plan
llm-council doctor
llm-council doctor --probe-openrouter
llm-council doctor --probe-ollama
llm-council doctor --check-update
```

Run council directly:

```bash
llm-council run --mode quick "What is the likely cause of this failing test?"
llm-council run --mode review --diff "Review the current diff."
llm-council run --mode consensus --diff "Should this ship?"
```

Inspect transcripts:

```bash
llm-council last
llm-council transcripts list
llm-council transcripts summary
llm-council transcripts prune --keep-since 2026-04-01 --apply
```

Refresh hosted model data:

```bash
llm-council models refresh
```

More options, including conversation continuation and prompt chunking, are in the [operator reference](docs/llm-council.md).

## Try it once with uvx

Use `uvx` only when you want to try the CLI without installing it:

```bash
uvx --from git+https://github.com/Intellimetrics/llm-council.git llm-council \
  run --mode quick "Explain the tradeoff in this design."
```

That does not install the tool for your project. It also does not give your coding agent MCP access.

For real use, install with `uv tool install` or `pipx`.

## Smithery

This repo includes `smithery.yaml` for registering `llm-council mcp-server` as a stdio MCP server.

Install through the Smithery marketplace UI if that is how you manage MCP servers. The manifest exposes configuration for:

- `OPENROUTER_API_KEY`
- `OLLAMA_HOST`
- `LLM_COUNCIL_MCP_ROOT`

Native CLI peers still need to be installed on the host.

## Update

Check your version:

```bash
llm-council --version
```

Check for a newer release:

```bash
llm-council check-update
```

Update with the same install command:

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
```

Or with `pipx`:

```bash
pipx install --force git+https://github.com/Intellimetrics/llm-council.git
```

Releases are tagged as `vX.Y.Z` and recorded in [CHANGELOG.md](CHANGELOG.md).

## More

- [Operator reference](docs/llm-council.md)
- [Changelog](CHANGELOG.md)

---

<sub>MIT licensed. Built for coding agents that should ask before they ship.</sub>
