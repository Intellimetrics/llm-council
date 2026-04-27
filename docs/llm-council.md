# LLM Council Reference

This file is the operator reference. The README is the product overview and
quick-start path.

## Setup

Install from GitHub into a target project:

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/your/project
llm-council setup --yes
llm-council doctor
llm-council check-update
```

If you do not use `uv`, install with `pipx`:

```bash
pipx install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/your/project
llm-council setup --yes
llm-council doctor
```

Do not use `uvx` for project installation. `uvx` is useful for one-off smoke
tests, but `setup` writes the command used by the MCP client. Prefer a stable
installed `llm-council` executable, then run setup from the target project.

`setup --yes` uses `--preset auto` by default. Auto setup refuses to write a
default config unless it finds a usable route: at least two installed native
CLIs, or `OPENROUTER_API_KEY` in your shell or project env files. This is meant
for the common one-CLI case: set an OpenRouter key, run setup, and get hosted
reviewers instead of a broken native-only council.

Preset choices:

- `auto`: choose `tri-cli` when at least two native CLIs exist, otherwise choose
  `openrouter` when `OPENROUTER_API_KEY` is available.
- `tri-cli`: Claude Code, Codex CLI, and Gemini CLI.
- `openrouter`: hosted OpenRouter reviewers only.
- `tri-cli-openrouter`: native CLIs plus hosted OpenRouter reviewers.
- `local-private`: native CLIs plus Ollama.
- `all`: every built-in preset.

`setup` creates `.mcp.json`, `.llm-council.yaml`, and instruction snippets, but
the snippets are deliberately separate so existing project instructions are not
overwritten. Wire each CLI explicitly by appending the full snippet contents:

- Claude Code: create `CLAUDE.md` in the project root if needed, then append
  the full contents of `.llm-council/instructions/claude.md` to it.
- Codex CLI: create `AGENTS.md` in the project root if needed, then append the
  full contents of `.llm-council/instructions/codex.md` to it.
- Gemini CLI: create `GEMINI.md` in the project root if needed, then append the
  full contents of `.llm-council/instructions/gemini.md` to it.

Restart the relevant CLI after changing `.mcp.json` or instruction files.

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
the `llm-council` entry under its existing `mcpServers` object. A project
config is optional for manual installs; without `.llm-council.yaml`, the
built-in default modes and participants are used.

Install from a checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
llm-council setup --yes
llm-council doctor
```

Useful setup variants:

```bash
llm-council setup --yes
llm-council setup --yes --preset openrouter
llm-council setup --yes --preset tri-cli
llm-council setup --yes --preset tri-cli --us-only-default
llm-council setup --yes --no-mcp --no-instructions
llm-council setup --yes --force
```

`setup` writes `.llm-council.yaml`, `.mcp.json`, and optional instruction
snippets under `.llm-council/instructions/`.

## Versioning And Updates

Releases are versioned in `pyproject.toml`, exposed by the package, recorded in
`CHANGELOG.md`, and tagged as `vX.Y.Z` in git. The installed package version is
visible through:

```bash
llm-council --version
llm-council check-update
llm-council doctor --check-update
```

`check-update` compares the installed version to the public `main` branch
`pyproject.toml` version and prints the update command:

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
```

MCP clients can pass `check_update: true` to `council_doctor` to get the same
version and update status in the returned JSON.

For hosted OpenRouter modes, estimate before running:

```bash
llm-council estimate --mode review-cheap "Review this change"
llm-council estimate --mode review-cheap --completion-tokens 2500 "Review this diff"
```

## Config Model

Project config merges over built-in defaults unless `replace_defaults: true` is
set. Generated setup configs use `replace_defaults: true` so presets remain
exact.

Core keys:

- `defaults.mode`: default mode, usually `quick`
- `defaults.origin_policy`: `any` or `us`
- `defaults.transparent`: include usage/cost and comparison details by default
- `defaults.max_concurrency`: maximum concurrent participant calls per round
- `defaults.max_deliberation_rounds`: default cap for opt-in deliberation
- `defaults.max_prompt_chars`: global prompt construction cap, default 200000
- `defaults.mcp_max_prompt_chars`: MCP budget guard for hosted paid calls
- `defaults.mcp_max_estimated_cost_usd`: MCP estimated cost guard
- `participants`: named CLI, OpenRouter, or Ollama participants
- `modes`: named participant routing presets
- `transcripts_dir`: local run transcript directory

Prompt sizing has two layers. `defaults.max_prompt_chars` limits the built
prompt before any participant runs. A participant-level `max_prompt_chars`
skips only that participant when the already-built prompt is too large for that
model or CLI.

## Participants

CLI participants:

```yaml
participants:
  my_cli:
    type: cli
    command: my-cli
    args: ["--prompt", "{prompt}"]
    timeout: 180
    stdin_prompt: false
    env_passthrough: ["MY_CLI_API_KEY"]
```

OpenRouter participants:

```yaml
participants:
  reviewer:
    type: openrouter
    model: provider/model-id
    api_key_env: OPENROUTER_API_KEY
    input_per_million: 0.5
    output_per_million: 1.5
```

Ollama participants:

```yaml
participants:
  local_reviewer:
    type: ollama
    model: qwen3-coder-next:q4_K_M
    base_url: http://localhost:11434
```

Hosted paid MCP calls fail closed when pricing is unknown. Add
`input_per_million` and `output_per_million` for custom paid hosted models.

## Model And Cost Selection

Native CLI participants use the user's installed Claude Code, Codex CLI, or
Gemini CLI account. That cost is external to `llm-council`; it may be a
subscription, a rate limit, or a CLI-specific API setup. OpenRouter participants
are hosted API calls billed by token. Ollama participants are local runtime
calls.

For users with only one native CLI, the simplest expansion path is:

```bash
export OPENROUTER_API_KEY=...
llm-council setup --yes --preset openrouter
llm-council doctor --probe-openrouter
llm-council estimate --mode review-cheap "Review this plan"
```

Use cheap OpenRouter modes first for breadth. Escalate to frontier hosted
models when the extra quality can justify the bill: release gates, architecture
tradeoffs, security-sensitive decisions, or unresolved bugs.

Built-in cheap modes use low-cost paid routes rather than `:free` OpenRouter
routes. Free routes are account-dependent and can reject otherwise valid calls
with `402 Payment Required`. The legacy `qwen_coder_free` participant remains
available for explicit experiments; use `qwen_coder_flash` for reliable cheap
defaults.

The cost shape is roughly:

```text
participants * rounds * ((prompt_tokens * input_price) + (output_tokens * output_price))
```

where prices are dollars per million tokens divided by 1,000,000. A long diff,
many participants, high output token assumptions, or deliberation can multiply
cost quickly.

`estimate` builds the same prompt surface as `run` but does not call
participants:

```bash
llm-council estimate --current codex --mode review --diff "Review the diff"
llm-council estimate --participants deepseek_v4_pro,qwen_coder_plus "Review this"
llm-council estimate --deliberate --max-rounds 2 "Decide between these designs"
```

Use live OpenRouter catalog lookup when considering Claude Opus/Sonnet, Gemini
Pro, OpenAI, or other hosted frontier routes:

```bash
llm-council models openrouter --no-cache --filter claude
llm-council models openrouter --no-cache --filter gemini
llm-council estimate \
  --openrouter-model anthropic/<copy-exact-model-id> \
  --openrouter-model google/<copy-exact-model-id> \
  "Review this architecture"
```

Do not hardcode example model IDs from old docs without checking the live
catalog. OpenRouter model IDs, prices, capacity, and availability change.

Reusable frontier-hosted mode template:

```yaml
participants:
  claude_frontier_openrouter:
    type: openrouter
    model: anthropic/<copy-exact-claude-model-id>
    api_key_env: OPENROUTER_API_KEY
    # Optional: copy prices from the live catalog for MCP fail-closed budget checks.
    # input_per_million: <input price per 1M tokens>
    # output_per_million: <output price per 1M tokens>
  gemini_frontier_openrouter:
    type: openrouter
    model: google/<copy-exact-gemini-model-id>
    api_key_env: OPENROUTER_API_KEY

modes:
  frontier-review:
    participants:
      - claude_frontier_openrouter
      - gemini_frontier_openrouter
```

Prefer `llm-council estimate --openrouter-model ...` and the live model catalog
for current costs.

## Modes

A mode either lists exact participants or uses `strategy: other_cli_peers`.

```yaml
modes:
  quick:
    strategy: other_cli_peers
  review:
    strategy: other_cli_peers
    add: ["qwen_coder_plus"]
  cheap:
    participants: ["deepseek_v4_flash", "qwen_coder_flash"]
```

`origin_policy: us` filters non-US participants. If the filter removes every
participant, the run fails clearly.

## MCP

Generated `.mcp.json` sets:

- `PYTHONPATH` to the project root
- `LLM_COUNCIL_MCP_ROOT` to the project root

Every MCP `working_directory` must be inside `LLM_COUNCIL_MCP_ROOT`. If you run
the server manually without that environment variable, the root is the process
current directory:

```bash
LLM_COUNCIL_MCP_ROOT=/path/to/project python -m llm_council.mcp_server
```

MCP tools:

- `council_run`: run the council
- `council_recommend`: decide whether council is useful for a task
- `council_estimate`: estimate prompt size and hosted OpenRouter cost
- `council_list_modes`: show configured modes and participants
- `council_last_transcript`: read the latest transcript
- `council_doctor`: diagnose local readiness
- `council_models`: list cached OpenRouter models

`council_run` returns `metadata.progress_events`, including participant start,
finish, skip, error, and deliberation events. With the stdio MCP server these
events are returned when the tool call completes, not streamed incrementally.
MCP context files are always restricted to the selected working directory.

## User-Facing Output

CLI runs print:

- selected mode, current agent, participants, and prompt size
- one line when each participant starts
- one line when each participant finishes, skips, or errors
- deliberation status when a second round is considered or run
- transcript paths

With `--transparent`, the transcript includes prompt text, per-model usage and
cost when available, and a compact final-round model comparison.

## Transcripts

Each run writes a Markdown transcript and JSON metadata:

```text
.llm-council/runs/<timestamp>_<slug>.md
.llm-council/runs/<timestamp>_<slug>.json
```

Use:

```bash
llm-council last
llm-council last --path-only
llm-council transcripts list
llm-council transcripts summary
```

## Safety Notes

The built-in Claude, Codex, and Gemini participants are configured for
read-only or plan modes, and prompts are sent over stdin where the CLI supports
it. Subprocess environments are sanitized; only configured `env_passthrough`
variables are forwarded.

Do not include secrets in diffs or context files. OpenRouter participants send
the prompt to OpenRouter and the selected upstream model. Ollama participants
send the prompt to the configured Ollama server.
