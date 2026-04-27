# LLM Council Reference

This file is the operator reference. The README is the product overview and
quick-start path.

## Setup

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
llm-council setup --yes --preset tri-cli
llm-council setup --yes --preset tri-cli --us-only-default
llm-council setup --yes --no-mcp --no-instructions
llm-council setup --yes --force
```

`setup` writes `.llm-council.yaml`, `.mcp.json`, and optional instruction
snippets under `.llm-council/instructions/`.

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
    participants: ["deepseek_v4_flash", "qwen_coder_free"]
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
