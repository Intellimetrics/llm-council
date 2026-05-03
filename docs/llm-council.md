# LLM Council Operator Reference

The [README](../README.md) is the agent-first user guide. This file is the
deeper reference for people and coding agents that need exact setup behavior,
config fields, MCP tool names, cost controls, and custom participant routing.

> [!NOTE]
> In normal use, a person does not run council by hand. They ask their coding
> agent to install it, then ask that same agent to use council during real
> work — the terminal command is for setup, diagnostics, direct runs, and
> transcript inspection.

```text
Install LLM Council from https://github.com/Intellimetrics/llm-council in this project.
```

```text
Use council to review this migration before editing files.
```

## Table of contents

- [Agent-first setup](#agent-first-setup)
- [Presets](#presets)
- [Coding-agent install path](#coding-agent-install-path)
- [Manual MCP install](#manual-mcp-install)
- [Versioning and updates](#versioning-and-updates)
- [Config model](#config-model)
- [Participants](#participants)
- [Model and cost selection](#model-and-cost-selection)
- [Modes](#modes)
- [Images](#images)
- [Timeouts and slow warnings](#timeouts-and-slow-warnings)
- [MCP](#mcp)
- [User-facing output](#user-facing-output)
- [Transcripts](#transcripts)
- [Safety notes](#safety-notes)

## Agent-first setup

Recommended user prompt:

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
8. After setup, read the generated snippet for this CLI from `.llm-council/instructions/`, then append that file's full contents to the correct project instruction file without overwriting existing content.
9. Confirm the destination file now contains the LLM Council routing rules.
10. Run `llm-council doctor` and show me the result.
11. Tell me to restart this CLI session so MCP and project instructions reload.
```

Manual equivalent:

```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/your/project
llm-council setup --plan
llm-council setup --yes --preset <chosen-preset>
llm-council doctor
llm-council check-update
```

If you do not use `uv`, install the same package with `pipx`:

```bash
pipx install --force git+https://github.com/Intellimetrics/llm-council.git
cd /path/to/your/project
llm-council setup --plan
llm-council setup --yes --preset <chosen-preset>
llm-council doctor
```

> [!WARNING]
> Do not use `uvx` for project installation. `uvx` is useful for one-off
> smoke tests, but `setup` writes the command used by the MCP client. Prefer
> a stable installed `llm-council` executable, then run setup from the
> target project.

## Presets

Agent installs should run `llm-council setup --plan` first and ask the user
which preset to write. `setup --yes` uses `--preset auto` only when the user has
explicitly accepted the default. Auto setup writes a default config only when it
finds a usable route: at least two installed native CLIs, or `OPENROUTER_API_KEY`
in your shell or project env files. This protects the common one-CLI case from
getting a native-only council that cannot run.

Explicit presets are validated too. If the chosen preset needs missing CLI tools
or API keys, non-interactive setup stops before writing files, and interactive
setup asks for confirmation. Use `--allow-incomplete` only when you intentionally
want to stage config before installing those dependencies.

Preset choices:

- `auto`: choose `tri-cli` when at least two native CLIs exist, otherwise choose
  `openrouter` when `OPENROUTER_API_KEY` is available.
- `tri-cli`: Claude Code, Codex CLI, and Gemini CLI.
- `openrouter`: hosted OpenRouter reviewers only.
- `tri-cli-openrouter`: native CLIs plus hosted OpenRouter reviewers.
- `local-private`: native CLIs plus Ollama.
- `all`: every built-in preset.

Useful setup variants:

```bash
llm-council setup --plan
llm-council setup --yes --preset <chosen-preset>
llm-council setup --yes --preset openrouter
llm-council setup --yes --preset tri-cli
llm-council setup --yes --preset tri-cli --us-only-default
llm-council setup --yes --preset tri-cli --no-mcp --no-instructions
llm-council setup --yes --preset tri-cli --force
llm-council setup --yes --preset openrouter --allow-incomplete
```

`setup` writes `.llm-council.yaml`, `.mcp.json`, and optional instruction
snippets under `.llm-council/instructions/`.

`.mcp.json` contains local absolute paths for the installed `llm-council`
command and project root. Treat it as machine-local config. Setup adds
`.mcp.json`, `.llm-council/runs/`, `.llm-council/*.log`, and
`.llm-council.env` to the project `.gitignore` when it writes MCP config.
If `.mcp.json` was already committed, remove it from the index with
`git rm --cached .mcp.json` after confirming your team does not intentionally
share local MCP config.

## Coding-agent install path

When a user asks a coding agent to install LLM Council, the agent should do the
same thing a careful human would do:

1. Install with `uv tool install --force git+https://github.com/Intellimetrics/llm-council.git`, or fall back to `pipx install --force git+https://github.com/Intellimetrics/llm-council.git`.
2. Do not use `uvx`; it is not a stable project install path.
3. Run `llm-council setup --plan` from the target project root.
4. Show the plan to the user and ask which preset to write.
5. Run `llm-council setup --yes --preset <chosen-preset>`.
6. If setup reports no usable route, ask the user whether to set
   `OPENROUTER_API_KEY` or install another native CLI.
7. Append the generated instruction snippet to the right project instruction
   file. Do not overwrite existing content.
8. Run `llm-council doctor` and report the result.
9. Tell the user to restart the active coding CLI.

Instruction snippet mapping:

- Claude Code: create `CLAUDE.md` in the project root if needed, then append
  the full contents of `.llm-council/instructions/claude.md` to it.
- Codex CLI: create `AGENTS.md` in the project root if needed, then append the
  full contents of `.llm-council/instructions/codex.md` to it.
- Gemini CLI: create `GEMINI.md` in the project root if needed, then append the
  full contents of `.llm-council/instructions/gemini.md` to it.

Restart the relevant CLI after changing `.mcp.json` or instruction files.

Verification checklist:

```text
.mcp.json exists and has an llm-council MCP server entry.
.mcp.json is ignored by git unless the project intentionally commits local MCP config.
.llm-council.yaml exists.
.llm-council/instructions/<active-cli>.md exists.
CLAUDE.md, AGENTS.md, or GEMINI.md includes the generated snippet.
llm-council doctor passes or gives an actionable missing requirement.
llm-council --version prints a version.
```

## Manual MCP install

Manual MCP install without `setup`:

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

Do not copy the placeholder command literally. Run `command -v llm-council`
after `uv tool install` or `pipx install`, then write that resolved absolute
path into `.mcp.json`. Autonomous agents configuring this file must resolve the
path dynamically instead of copying the example string.

If the project already has `.mcp.json`, add only the `llm-council` entry under
its existing `mcpServers` object. A project config is optional for manual
installs; without `.llm-council.yaml`, the built-in default modes and
participants are used.

Install from a checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
llm-council setup --plan
llm-council setup --yes --preset <chosen-preset>
llm-council doctor
```

## Versioning and updates

Releases are versioned in `pyproject.toml`, exposed by the package, recorded in
`CHANGELOG.md`, and tagged as `vX.Y.Z` in git. The installed package version is
visible through:

```bash
llm-council --version
llm-council check-update
llm-council doctor --check-update
```

`check-update` compares the installed version to the latest public `vX.Y.Z`
release tag and prints the update command:

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

## Config model

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

## Model and cost selection

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
Built-in native modes set `include_current: true`, so `quick` asks Claude,
Codex, and Gemini as explicit read-only participants even when one of them is
the active host. Use `peer-only`, or set `include_current: false`, when you only
want outside perspectives.

```yaml
modes:
  quick:
    strategy: other_cli_peers
    include_current: true
  peer-only:
    strategy: other_cli_peers
    include_current: false
  review:
    strategy: other_cli_peers
    include_current: true
    add: ["qwen_coder_plus"]
  cheap:
    participants: ["deepseek_v4_flash", "qwen_coder_flash"]
```

`origin_policy: us` filters non-US participants. If the filter removes every
participant, the run fails clearly.

The temporary `opus-versions` mode runs `claude_4_6` and `claude_4_7`
side-by-side for a head-to-head comparison of Opus 4.6 vs 4.7 on the same
prompt. Both ship as part of the native preset and can also be added to other
modes via `include: [claude_4_6]`.

## Images

Council can review screenshots and other UI artifacts. Pass `image_paths` (list
of repo-relative paths) on `council_run`/`estimate` or via the CLI's repeatable
`--image PATH` flag. Native CLI participants share the project filesystem and
open staged images themselves with their file-read tool. Hosted participants
need `vision: true` on their config; the OpenRouter adapter then sends a
multimodal `content` array, and the Ollama adapter populates
`messages[].images`. Non-vision participants in a council with images present
get the textual `## Images` section only and the orchestrator emits an
`images_skipped` progress event for them.

For sandboxed hosts that cannot write to disk, `council_run` also accepts
inline `images: [{data, mime, name?}]`. llm-council decodes them under
`.llm-council/inputs/<run-id>/` before participants run, with hard caps of
8 MB per file and 32 MB total enforced both pre-decode (base64 length
heuristic) and post-decode (exact byte count). The same byte caps apply to
`image_paths` via `image_attachment_violations` so an estimate that passes
matches a run that passes. `.llm-council/inputs/` is gitignored by setup.
Stale subdirectories older than `INLINE_INPUTS_RETENTION_DAYS` (default 7
days) are pruned opportunistically before each new staging.

Allowed mime types: `image/png`, `image/jpeg`, `image/webp`, `image/gif`.
`image/svg+xml` is refused.

## Timeouts and slow warnings

Each CLI participant has a per-config `timeout` (default 240s). When the
deadline is hit, the participant returns an actionable error naming the
participant, the timeout, the prompt size, and the config knob to raise:

```
Timeout: `claude` did not respond within 240s (prompt was 9342 chars). To
raise the limit, set `participants.claude.timeout: <seconds>` in
`.llm-council.yaml`. ...
```

Before the hard deadline, the orchestrator emits a `participant_slow`
progress event at `slow_warn_after_seconds` (per-participant config) or, by
default, `max(30.0, timeout * 0.75)`. The event is recorded in
`metadata.progress_events` and rendered as `still running after Xs` in the
CLI. The `participant_finish` event uses `status="timeout"` distinct from
`"error"`/`"skipped"`. Transcripts label the participant section
`(timeout)`, and the CLI run summary calls out timeouts with the same
config-knob hint.

In deliberation mode, participants that timed out (or hit `PromptTooLarge`)
are excluded from subsequent rounds — cumulatively, so a participant that
times out in round 1 stays excluded in round 3 even if round 2 ran
successfully without them.

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

## User-facing output

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

## Safety notes

The built-in Claude, Codex, and Gemini participants are configured for
read-only or plan modes, and prompts are sent over stdin where the CLI supports
it. Subprocess environments are sanitized; only configured `env_passthrough`
variables are forwarded.

> [!WARNING]
> Do not include secrets in diffs or context files. OpenRouter participants
> send the prompt to OpenRouter and the selected upstream model. Ollama
> participants send the prompt to the configured Ollama server.

> [!CAUTION]
> Do not use council for classified, CUI, regulated, customer, production,
> credential, or `DEPLOY_MODE=secret` content unless every configured
> participant is explicitly approved for that data. US-origin participants
> identify model/company origin only; they do not imply GovCloud, FedRAMP,
> or enterprise data-handling approval.
