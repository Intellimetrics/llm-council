# Local OpenAI-compatible models

`llm-council` can talk to any inference server that exposes an OpenAI-style
`/v1/chat/completions` endpoint via the `type: openai_compatible` participant
type. That covers the common open-weights serving stacks:

- [vLLM](https://docs.vllm.ai/) and [sglang](https://docs.sglang.ai/) (default `:8000`)
- [LM Studio](https://lmstudio.ai/) (default `:1234`)
- [llama.cpp `--api`](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) (default `:8080`)
- [TGI (Text Generation Inference)](https://huggingface.co/docs/text-generation-inference) (default `:8080`)
- [Ollama's OpenAI-compatible shim](https://github.com/ollama/ollama/blob/main/docs/openai.md) at `/v1` (port `:11434`)
- [MLX server](https://github.com/ml-explore/mlx-examples) (commonly `:5000`)
- Anything else that returns `{"object": "list", "data": [...]}` from `GET /v1/models`

This page documents how to wire one up. Defaults stay out of `defaults.py` on
purpose — the model identity (and therefore `origin`, `family`) is determined
entirely by what you pointed your endpoint at, so it must come from your
project YAML.

## Discover what's running

Use the doctor probe to see which OpenAI-compatible servers are responding on
your machine:

```bash
# Scan well-known local ports (8000, 1234, 8080, 11434, 5000)
llm-council doctor --probe-local-openai

# Probe a specific endpoint
llm-council doctor --probe-local-openai http://127.0.0.1:8000/v1
```

The probe validates the JSON shape of `/v1/models`, not just that the port
answers. A Django or FastAPI dev server happily listening on `:8000` will be
reported as "HTTP 200 but body is not JSON" rather than mis-identified as a
local LLM.

Output looks like:

```text
ok       probe:local-openai:vLLM/sglang  1 model(s): Qwen/Qwen3.6-27B
ok       probe:local-openai:Ollama /v1   2 model(s): qwen3-coder:30b, llama3.3:70b
```

## The origin invariant (read this first)

When you wire up a local participant, **`origin` describes the model behind
the endpoint, not where the binary is running**. A vLLM serving Qwen on your
laptop is still origin `China / Alibaba Qwen`, because that's what determines
whether it gets included or excluded under `origin_policy: us`.

Existing canonical origin strings (used by built-in participants):

| Origin string | Used for |
|---|---|
| `US / Anthropic` | Claude family |
| `US / OpenAI` | Codex / GPT family |
| `US / Google` | Gemini family |
| `US / Meta` | Llama family |
| `US / Mistral` | (Mistral is French, but Mistral Small / Codestral see US enterprise use) |
| `France / Mistral` | Mistral family — preferred when matching `origin_policy: us` should exclude |
| `China / Alibaba Qwen` | Qwen family |
| `China / DeepSeek` | DeepSeek family |
| `China / Z.ai` | GLM family |
| `China / Moonshot AI` | Kimi family |

Pick the closest match. `origin_policy: us` matches participants whose
`origin` starts with `US /`, so a typo (`us-only`, `US/Meta`, etc.) silently
drops the participant from US-only runs.

## Required env var (the dummy-key gotcha)

The `openai_compatible` adapter always sends an `Authorization: Bearer …`
header, even to local servers that don't check it. That means **you must
export the env var named in `api_key_env`** before running the council, even
for an unauthenticated local server. The convention:

```bash
export LOCAL_OPENAI_API_KEY=dummy
```

If `LOCAL_OPENAI_API_KEY` (or whatever you set in `api_key_env`) is unset, the
adapter fails fast with `Missing LOCAL_OPENAI_API_KEY` rather than sending a
bare token to your local server.

## `allow_private: true` is required for loopback

Config validation rejects participants whose `base_url` resolves to loopback,
private, or link-local addresses unless they explicitly opt in:

```yaml
participants:
  local_vllm:
    type: openai_compatible
    base_url: http://127.0.0.1:8000/v1
    allow_private: true   # required for loopback / RFC1918 base_urls
    # ...
```

This is a defense-in-depth check against a misconfigured `base_url` pointed
at an internal corporate service by accident. For genuine local serving, you
want `allow_private: true`.

## Recipes

All recipes below assume you've exported `LOCAL_OPENAI_API_KEY=dummy` (or a
real key if your server enforces auth) before running the council.

### vLLM

vLLM serves any HuggingFace model with a stable OpenAI-compatible API. The
served model name is whatever you passed to `--model` (or `--served-model-name`):

```yaml
participants:
  local_vllm:
    type: openai_compatible
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3.6-27B            # match your --served-model-name
    family: qwen                        # or llama, deepseek, mistral, …
    origin: "China / Alibaba Qwen"      # MUST reflect the model
    api_key_env: LOCAL_OPENAI_API_KEY
    allow_private: true
    timeout: 360                        # vLLM with long context can exceed 180s
    read_only: true
```

Long-context models (e.g., 131K) can have multi-second TTFT and minute-scale
total time. Default `timeout` of 180s is tight under load — `360` is a safer
floor for local serving.

### LM Studio

LM Studio's OpenAI-compatible server runs on `:1234`. Whatever model you
loaded in the GUI is what `/v1/models` will return:

```yaml
participants:
  local_lmstudio:
    type: openai_compatible
    base_url: http://127.0.0.1:1234/v1
    model: lmstudio-community/Meta-Llama-3.3-70B-Instruct-GGUF   # match the loaded model
    family: llama
    origin: "US / Meta"
    api_key_env: LOCAL_OPENAI_API_KEY
    allow_private: true
    timeout: 240
    read_only: true
```

### llama.cpp `--api`

```yaml
participants:
  local_llamacpp:
    type: openai_compatible
    base_url: http://127.0.0.1:8080/v1
    model: gguf-model
    family: llama
    origin: "US / Meta"
    api_key_env: LOCAL_OPENAI_API_KEY
    allow_private: true
    timeout: 360
    read_only: true
```

Some llama.cpp builds return `404` on `/v1/models` even when
`/v1/chat/completions` works. The doctor probe will surface this as
"HTTP 404 — server reachable but `/v1/models` not implemented" — the
participant is still usable; you just can't auto-enumerate the model id.

### sglang

```yaml
participants:
  local_sglang:
    type: openai_compatible
    base_url: http://127.0.0.1:8000/v1
    model: meta-llama/Llama-3.3-70B-Instruct
    family: llama
    origin: "US / Meta"
    api_key_env: LOCAL_OPENAI_API_KEY
    allow_private: true
    timeout: 360
    read_only: true
```

### TGI (Text Generation Inference)

```yaml
participants:
  local_tgi:
    type: openai_compatible
    base_url: http://127.0.0.1:8080/v1
    model: meta-llama/Llama-3.3-70B-Instruct
    family: llama
    origin: "US / Meta"
    api_key_env: LOCAL_OPENAI_API_KEY
    allow_private: true
    timeout: 360
    read_only: true
```

### Ollama via its `/v1` shim

If you're already running Ollama, you can use the OpenAI-compatible adapter
instead of `type: ollama` — useful when you want to pass through the same
plumbing the hosted reviewers use:

```yaml
participants:
  local_ollama_openai:
    type: openai_compatible
    base_url: http://127.0.0.1:11434/v1
    model: qwen3-coder:30b               # `ollama list` shows the available ids
    family: qwen
    origin: "China / Alibaba Qwen"
    api_key_env: LOCAL_OPENAI_API_KEY    # Ollama ignores it but the adapter requires one
    allow_private: true
    timeout: 240
    read_only: true
```

The dedicated `type: ollama` participant (e.g., the built-in `local_qwen_coder`)
remains the path of least resistance for Ollama users; the openai-compatible
form is for parity with other local stacks.

### MLX server

```yaml
participants:
  local_mlx:
    type: openai_compatible
    base_url: http://127.0.0.1:5000/v1
    model: mlx-community/Llama-3.3-70B-Instruct-4bit
    family: llama
    origin: "US / Meta"
    api_key_env: LOCAL_OPENAI_API_KEY
    allow_private: true
    timeout: 240
    read_only: true
```

## Wiring local participants into modes

### Built-in `local-only` mode

The `local-only` mode auto-discovers every local participant in your config —
both `type: ollama` entries and any `type: openai_compatible` whose `base_url`
resolves to loopback (`127.0.0.1`, `localhost`) or RFC1918. No further
wiring needed:

```bash
llm-council run --mode local-only --diff "Review this change"
```

Hosted-inference CLI peers (claude/codex/gemini) and hosted API peers
(openrouter) are excluded — `local-only` is for offline/private review.

### Custom modes

To mix local and hosted peers, define your own mode:

```yaml
modes:
  # Native triad plus a local pass
  plan-with-local:
    strategy: other_cli_peers
    include_current: true
    add: ["local_vllm"]
    description: "Native triad plus local Qwen on vLLM."

  # Pin a specific local participant
  vllm-only:
    participants: ["local_vllm"]
    description: "Single-peer local review against vLLM."
```

Or use `--include local_vllm` on a single run without modifying modes.

## Cost estimation

Local participants count as `$0` in `--max-cost-usd` and `--max-tokens`
estimates. That's correct for cash cost, but the GPU is real — a council run
with three local peers can spin your machine for tens of minutes against a
`--max-cost-usd 0.10` cap and still pass. Use `--max-tokens` (which does
include local participants) when you want to bound effort, not just spend.

## Concurrent serving

A single vLLM/sglang/TGI/llama.cpp instance serializes incoming requests by
default. If you add three council participants pointing at the same local
endpoint, they will execute sequentially — no parallelism. The council UI
shows three "in flight" requests, but only one is actually running at a time.

Two ways to actually parallelize:

1. **Run multiple inference servers** on different ports (e.g., vLLM on
   `:8000` and sglang on `:8001`), point separate participants at each.
2. **Use a server with batching** that's configured for high concurrency
   (vLLM's continuous batching, TGI's `--max-concurrent-requests`).

For most council use, sequential local execution is fine — the bottleneck
becomes "wait for the slowest peer," same as with hosted reviewers.

## Troubleshooting

### "Missing LOCAL_OPENAI_API_KEY"

You forgot to export the dummy key. Set it before running:
`export LOCAL_OPENAI_API_KEY=dummy`.

### `ConfigError: openai_compatible participant '…' resolves to a private/loopback IP`

You need `allow_private: true` on the participant.

### `RECOMMENDATION:` label missing

The `openai_compatible` adapter requires the model's response to include a
`RECOMMENDATION: yes|no|tradeoff` label. Smaller / weaker local models may
need a one-shot example or stricter system prompt — `llm-council`'s built-in
prompts ask for the label, but a 7B model serving in a quantized config may
not consistently emit it. Verify with `llm-council last` to see the exact
output, and consider stepping up to a larger local model.

### Probe shows the endpoint, but the run hangs

vLLM with very long context windows (≥131K) can take minutes to serve a
single completion. If you're running into the default 180s timeout, bump
`timeout: 360` (or higher) on the participant.

### `origin_policy: us` excludes my local Llama

Check the exact `origin` string. The matching is case-sensitive. `US / Meta`
works; `US/Meta` (no spaces around the slash) does not.

## See also

- [`docs/llm-council.md`](llm-council.md) — operator reference
- [`CLAUDE.md`](../CLAUDE.md) — project-internal architecture notes
- `llm-council doctor --probe-local-openai` — discover servers
- `llm-council doctor --probe-ollama` — separate probe for `type: ollama` participants
