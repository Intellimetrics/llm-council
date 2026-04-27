# LLM Council

LLM Council is a lightweight, read-only-first multi-agent tool for coding projects.

Baseline assumption: the user usually works inside one of three premium CLIs:

- Claude Code
- Codex CLI
- Gemini CLI

By default, `llm-council` asks the other two CLIs for read-only opinions. OpenRouter and local models are explicit participants, not fallbacks.

## Quick Start

Install directly from GitHub into a target project:

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/your/project
llm-council setup --yes
llm-council doctor
```

If you do not use `uv`, install with `pipx`:

```bash
pipx install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/your/project
llm-council setup --yes
llm-council doctor
```

Do not use `uvx` for project installation. `uvx` is useful for one-off smoke
tests, but `setup` writes the MCP command into the target project's `.mcp.json`,
so it should run from a stable installed `llm-council` executable.

`setup --yes` uses `--preset auto` by default. Auto setup refuses to write a
default config unless it finds a usable route: at least two installed native
CLIs, or `OPENROUTER_API_KEY` in your shell or project env files. This keeps a
one-CLI project from silently getting a broken native-only council. Choose a
preset explicitly when you know what you want:

```bash
llm-council setup --yes --preset tri-cli              # Claude/Codex/Gemini
llm-council setup --yes --preset openrouter           # hosted-only fallback
llm-council setup --yes --preset tri-cli-openrouter   # native CLIs + hosted
llm-council setup --yes --preset local-private        # native CLIs + Ollama
llm-council setup --yes --preset all                  # every built-in preset
```

Then make each CLI agent aware of council. `setup` writes snippets under
`.llm-council/instructions/`, but Claude Code, Codex CLI, and Gemini CLI do not
load those snippets automatically:

- Claude Code: create `CLAUDE.md` in the project root if needed, then append
  the full contents of `.llm-council/instructions/claude.md` to it.
- Codex CLI: create `AGENTS.md` in the project root if needed, then append the
  full contents of `.llm-council/instructions/codex.md` to it.
- Gemini CLI: create `GEMINI.md` in the project root if needed, then append the
  full contents of `.llm-council/instructions/gemini.md` to it.

Restart the relevant CLI session after editing `.mcp.json` or instruction
files so the MCP server and project instructions reload.

From this checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
llm-council run --dry-run --current codex "Should we add an MCP wrapper?"
```

Manual MCP install without `setup`:

```json
{
  "mcpServers": {
    "llm-council": {
      "type": "stdio",
      "command": "/absolute/path/to/llm-council",
      "args": ["mcp-server"],
      "env": {
        "LLM_COUNCIL_MCP_ROOT": "/absolute/path/to/your/project"
      }
    }
  }
}
```

Do not copy `/absolute/path/to/llm-council` literally. Run
`command -v llm-council` after `uv tool install` or `pipx install` to find the
real absolute command path. If the project already has `.mcp.json`, add only
the `llm-council` entry under its existing `mcpServers` object.

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
The generated MCP config does not embed API keys; use one of those local env
files or your MCP client's normal environment configuration for hosted models.

Live call:

```bash
llm-council run --current codex "Review this architecture decision"
```

During live CLI runs, `llm-council` prints progress as each participant starts,
finishes, skips, or errors, then writes the full transcript. MCP calls return
the same participant progress in `metadata.progress_events` when the tool call
completes; the stdio MCP server does not stream incremental progress events.

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

Include a file and git diff. The diff context includes both staged and unstaged
changes when the working directory is a Git repository:

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

Explicit participants still respect `--origin-policy us`; a non-US participant
under that policy fails clearly instead of being silently selected or dropped.

US-origin only:

```bash
llm-council run --mode us-only "Review this plan"
llm-council run --mode diverse --origin-policy us "Review this plan"
```

For users who want this as the default:

```bash
llm-council setup --yes --us-only-default
```

Generated setup configs use `replace_defaults: true` so presets such as
`tri-cli` stay exact instead of re-expanding omitted hosted or local defaults at
load time.

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
Prompts are globally capped by `defaults.max_prompt_chars` before participant
calls are made. Individual participants can also set `max_prompt_chars`; that
skips only that participant when the built prompt is too large for the model or
CLI. Claude has a conservative default participant guard so very large reviews
skip it immediately instead of waiting for a long process timeout.

Use explicit models only when you want exact routing:

```yaml
participants:
  claude:
    model: claude-opus-4-7
    max_prompt_chars: 200000
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

Custom CLI participants inherit a sanitized subprocess environment. Add
`env_passthrough` for the specific secret-looking variables that participant
needs for authentication:

```yaml
participants:
  my_cli:
    type: cli
    command: my-cli
    args: ["--prompt", "{prompt}"]
    env_passthrough: ["MY_CLI_API_KEY"]
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
- Gemini: `--approval-mode plan`, with prompts sent over stdin

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
usage/cost and opt-in deliberation behavior as the CLI. MCP context files are
always restricted to `working_directory`; use the CLI `--allow-outside-cwd` flag
for intentional external files.

MCP runs also apply conservative budget checks before paid hosted calls. Tune
them in config with `defaults.mcp_max_prompt_chars` and
`defaults.mcp_max_estimated_cost_usd`.
MCP `working_directory` values are restricted to the project root configured by
setup, so MCP callers cannot widen context access by setting `/` as the working
directory.

Run it manually:

```bash
LLM_COUNCIL_MCP_ROOT=/path/to/project python -m llm_council mcp-server
```

or:

```bash
LLM_COUNCIL_MCP_ROOT=/path/to/project python -m llm_council.mcp_server
```

When `LLM_COUNCIL_MCP_ROOT` is not set, the MCP root defaults to the server
process current directory.

## Keyword Layer

Setup writes instruction snippets for each CLI under `.llm-council/instructions/`.
They are not loaded automatically by the CLIs. Add the matching snippet to the
project or user instructions for the agent you use, for example Claude project
instructions, Codex `AGENTS.md` guidance, or Gemini custom instructions. The
snippets are plain Markdown so you can copy the wording into your existing
agent memory without changing the project config.

The intended user language is:

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
to share external files through the CLI. MCP does not expose that override.
Subprocess CLI participants do not inherit broad secret-looking environment
variables by default. The built-in Claude, Codex, and Gemini participants pass
through their standard API key variables only, so env-based CLI auth can work
without exposing unrelated credentials.

## When Not To Use Council

Skip council for formatting, exact user-directed edits, obvious syntax fixes,
single-file changes you already understand, or tasks where extra model calls
would add cost without reducing risk.

## Transcripts

Each run writes:

- `.llm-council/runs/<timestamp>_<slug>.md`
- `.llm-council/runs/<timestamp>_<slug>.json`

These are intentionally local run artifacts.

See [LLM Council reference](docs/llm-council.md) for the full config/MCP
operator reference. 
