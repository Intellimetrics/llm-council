# LLM Council

[![Tests](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-ready-2f855a)](docs/llm-council.md)
[![Source read-only peers](https://img.shields.io/badge/peers-source--read--only-6b7280)](#read-only-safety)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.21.1-111827)](CHANGELOG.md)

Your coding agent is incredibly fast, capable, and confident. 

That is highly valuable—until it confidently overwrites a critical database migration, introduces a subtle security vulnerability, or refactors a complex module it wasn't supposed to touch.

**LLM Council** is a lightweight, source-read-only multi-agent orchestration harness designed to give your primary coding agent a fast, independent second opinion before committing to risky changes or expensive edits. Peers cannot edit project source; the host still writes council-owned transcripts, caches, and explicitly requested configuration. It runs as a Python 3.11+ MCP server and command-line tool.

---

## Architecture & Workflow

LLM Council acts as an advisory layer between your primary developer agent (e.g. Claude Code, Codex CLI, Antigravity) and a pool of independent peer review models.

```mermaid
graph TD
    User([User Prompt]) --> Agent[Primary Developer Agent]
    Agent -- "1. Trigger (natural language / command)" --> Server[LLM Council MCP Server]
    Server -- "2. Parse prompt / git diff" --> Orchestrator[Orchestrator]
    Orchestrator -- "3. Parallel invocation (isolated stdin)" --> PeerA[Peer A: Claude]
    Orchestrator -- "3. Parallel invocation (isolated stdin)" --> PeerB[Peer B: Codex]
    Orchestrator -- "3. Parallel invocation (isolated stdin)" --> PeerC[Peer C: Antigravity]
    PeerA -- "4. YES/NO/TRADEOFF" --> Consensus[Consensus Evaluator]
    PeerB -- "4. YES/NO/TRADEOFF" --> Consensus
    PeerC -- "4. YES/NO/TRADEOFF" --> Consensus
    Consensus -- "5. Write Markdown/HTML transcript" --> Out[Output Handler]
    Consensus -. "Optional: --synthesize" .-> Synth[Synthesis Chair]
    Synth -. "Decision memo" .-> Out
    Out -. "Optional: --open or config" .-> Browser([Default Browser])
    Out -- "6. Structured JSON & Summary" --> Agent
```

---

## Key Features & Capabilities

*   **Consensus Gates & Vote Contracts**: Every peer response is strictly parsed and must resolve to one of three consensus labels. Vague essays are rejected:
    *   `RECOMMENDATION: yes` — Safe to proceed.
    *   `RECOMMENDATION: no` — Stop; major issues detected.
    *   `RECOMMENDATION: tradeoff` — Plausible, but note critical trade-offs.
*   **Assigned-Stance Review**: `consensus` assigns for/against/neutral stances in the first round. Add `--deliberate` for multi-round critique and `--synthesize` for an optional **Synthesis Chair** decision memo.
*   **Host-Aware Routing**: Detects the active developer agent. `quick` includes that host for an explicit council pass; `peer-only` excludes the host's family when the user wants only outside perspectives.
*   **Rigorous Sandboxing & Read-Only Safety**: Native CLI peers are invoked with binary-level flags disabling file writes (`--permission-mode manual` for Claude, `--sandbox read-only` for Codex, `--mode plan` for Antigravity).
*   **Recursion-Proof**: A council can never start another council. Every peer subprocess carries a `LLM_COUNCIL_NESTED` marker that makes any nested `council_run` refuse outright, and codex peers boot with an empty MCP server table so a globally registered llm-council can't spawn inside a peer. Without this, one prompt-injected `council_run` would fan out exponentially.
*   **Local-CLI-Only Defaults**: Every built-in mode rosters only the native CLIs (Claude, Codex, Antigravity) running under your own accounts — no hosted, per-token-billed peer ever runs unless you explicitly opt in (`--include`, per-mode `add`, or a hosted setup preset).
*   **Cost Controls & Caching**: Pre-flight token and USD cost estimation (`llm-council estimate`) plus response caching prevents unexpected hosted API charges. A hard `--max-cost-usd` / `--max-tokens` gate refuses a run before launch; an optional soft `cost_warn_usd` tier warns (but never blocks) when an estimate gets pricey, and an optional `litellm` fallback prices hosted models missing from the OpenRouter catalog.
*   **Credential Secret Scanning**: Scans all prompt content for API keys, tokens, or private keys, with configurable responses (`warn`, `block`, `redact`, or `off`).
*   **HTML Transcript**: Every run generates a formatted HTML transcript. Opening it in the default browser is opt-in through `--open` or `defaults.auto_open_browser: true`.
*   **Anti-Herding Deliberation**: Round-2 deliberation asks peers to converge toward what is *correct* rather than toward agreement — no capitulating to the group, no digging in out of consistency bias — and to critique each other rather than re-defend their own prior answer (mitigating the multi-agent-debate herding failure mode).
*   **Dissent-Preserving Synthesis** *(opt-in)*: With `--synthesize`, the Synthesis Chair attributes blockers to the peers who raised them, narrates position changes, and names genuine remaining disagreement instead of papering it over.
*   **Cross-Vendor Independence Warning** *(opt-in)*: When every labeled vote comes from a single vendor family, the council flags it (`min_distinct_vendors`) so correlated same-vendor agreement isn't mistaken for independent corroboration.
*   **Independent Review** *(opt-in)*: suppress prior-round verdicts on a continuation (`--independent-review`) so re-reviews aren't anchored to past opinions.
*   **Observable Per-CLI Usage** *(opt-in)*: `usage_from_json` reads real token counts and cost from the Claude/Codex JSON output modes — recovering usage telemetry that is otherwise invisible for native CLI peers — while preserving the read-only invocation.

---

## Built-in Modes

All built-in rosters are local-CLI only; hosted peers join only by explicit opt-in.

| Mode | What it does |
| :--- | :--- |
| `quick` | Ask all available native CLI peers (includes a fresh instance of the host's own family for a clean-context pass). |
| `peer-only` | Exclude the host CLI's family — outside perspectives only. |
| `plan` | Independent planning review. |
| `review` | Code/diff review (pair with `--diff`). |
| `review-with-tools` | *Experimental.* CLI peers use their own read-only file/grep tools to verify claims against the repo before voting, instead of relying solely on pasted context. |
| `deliberate` | Adds a second round when first-round votes disagree. |
| `consensus` | Assigned-stance debate (for / against / neutral) to attack groupthink; 2x timeouts. |
| `fable` | Read-only second opinion from Claude Fable 5, with defensive-review framing and a silent-model-substitution guard. |
| `private-local` | Routes only to same-machine loopback Ollama participants. |

---

## Single-Model Peer Isolation

Even if you only have access to a single LLM (due to offline constraints, local setup, or API key limits), running a council session provides substantial value:

1.  **Fresh Eyes / Context Separation**: In a typical developer agent session, the context window accumulates tools, history, and reasoning, leading to confirmation bias. The council calls a fresh, isolated API process containing *only* the diff and prompt, forcing a clean evaluation.
2.  **Stance & State Splitting**: By invoking multiple calls to the same model concurrently with different stance prompts (e.g., Peer A as the *Attacker*, Peer B as the *Defender*, Peer C as the *Judge*), you isolate their states. Peers see one another's arguments only if deliberation or synthesis is enabled.
3.  **Adversarial Extraction**: Forcing the model to adopt contrarian stances (e.g. "Find 3 security flaws in this diff") bypasses its default cooperative/agreeable bias, extracting a deeper critique than a standard prompt.

---

## Installation

Install `llm-council` globally on your system.

### Option A: Using `uv` (Recommended)
```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
```

### Option B: Using `pipx`
```bash
pipx install --force git+https://github.com/Intellimetrics/llm-council.git
```

> [!WARNING]
> Do not use `uvx` for your primary installation. While convenient for one-off trials, `uvx` does not configure a stable MCP environment or keep the tool persistently available on your path.

---

## Quickstart

Navigate to your active project repository and initialize the council:

```bash
cd /path/to/your/project
llm-council setup --plan
llm-council setup --yes --preset auto
llm-council doctor --probe-native --probe-ollama
```

> [!IMPORTANT]
> Setup writes host-specific snippets under `.llm-council/instructions/`; it does not edit your existing instruction files. Append the active host's full snippet to `CLAUDE.md`, `AGENTS.md`, or `GEMINI.md`, then restart that developer-agent session so MCP and the routing rules reload. `setup` prints the exact paths.

### The 30-Second Example
Once configured, you can talk to your primary developer agent (e.g., Claude Code, Codex, Antigravity CLI) directly in natural language:

*   *"Ask the council to review my current changes before we commit."*
*   *"Take this failing test to council. I need theory and review, not another patch."*
*   *"Run a private-local council on this code. Route it only to my same-machine Ollama daemon."*

---

## CLI Reference

While most interaction happens transparently via the MCP server inside your agent, you can invoke the CLI directly:

| Command | Description / Example |
| :--- | :--- |
| **`llm-council run`** | Run a council query. <br>`llm-council run --mode quick "Why is this test flaky?"` |
| **`llm-council run --diff`** | Review the current git diff. <br>`llm-council run --mode review --diff "Is this migration safe to run?"` |
| **`llm-council run --independent-review`** | On a `--continue` run, suppress the prior council's verdicts/rationales so the round forms its opinion independently. <br>`llm-council run --mode review --continue 20260101_120000 --independent-review --diff "Re-review"` |
| **`llm-council run --continue`** | Continue a prior run. A private-local parent stays private-local when mode is omitted; moving its context to hosted/native peers is refused unless `--allow-privacy-downgrade` is explicit. |
| **`llm-council run --cost-warn-usd`** | Attach a non-fatal warning when the pre-flight estimate exceeds a threshold (complements, never replaces, the hard `--max-cost-usd` gate). <br>`llm-council run --mode consensus --cost-warn-usd 0.50 --diff "Worth a full debate?"` |
| **`llm-council estimate`** | Calculate prompt size and costs before running; reports a `cost_class` (low/moderate/high) plus paid/free peer counts. <br>`llm-council estimate --mode consensus --diff "Should we merge this?"` |
| **`llm-council last`** | Inspect the last run's raw transcripts. <br>`llm-council last` |
| **`llm-council config get`** | Retrieve a configuration value. <br>`llm-council config get defaults.auto_open_browser` |
| **`llm-council config set`** | Set a configuration value. <br>`llm-council config set defaults.auto_open_browser true` |
| **`llm-council install-hook`** | Install a `pre-commit` or `pre-push` council gate. It validates the mode and refuses to replace an existing hook unless `--force` is explicit. <br>`llm-council install-hook --hook-type pre-push --mode consensus` |
| **`llm-council transcripts prune`** | Preview transcript deletion under a retention policy; add `--delete` to remove the listed files. <br>`llm-council transcripts prune --keep-last 20 --delete` |

---

## MCP Server Integration

The `llm-council` server exposes the following tools to your developer agents:

| MCP Tool Name | Description / Inputs |
| :--- | :--- |
| **`council_run`** | Run a council query with custom modes, context files, and optional diffs. Supports `open: true` to open the HTML transcript, `independent_review: true` to suppress prior-council context on a continuation run, and `cost_warn_usd` for a non-fatal cost heads-up. Surfaces advisory signals in the result when present (`independence_warning`, `cost_warning`, `cost_estimate`). |
| **`council_estimate`** | Check token sizes and estimated OpenRouter cost before launching. |
| **`council_recommend`** | Evaluates a task, risk level, and files touched to recommend whether to consult the council. Also returns a mechanical `difficulty_class` and the matched trigger keywords (`suggested_mode_reason_codes`) and an optional LLM-graded `judge` verdict when `recommend_judge` is configured. |
| **`council_doctor`** | Diagnoses connection issues, API key status, and CLI path resolution. |
| **`council_models`** | Lists the cached OpenRouter model catalog. |
| **`council_list_modes`** | Lists configured runtime modes and participants. |
| **`council_last_transcript`** | Returns the path and content of the last recorded run. |
| **`council_stats`** | Aggregates participant metrics (run count, success, tokens, cost) across recorded transcripts. |
| **`council_query_transcripts`** | Searches past transcript history for similar reviews. |
| **`council_config`** | Get or set configuration keys in `.llm-council.yaml` programmatically via the MCP connection. |

---

## Presets & Configuration

The setup wizard (`llm-council setup --plan`) automatically probes your environment to find available tools and recommends the best preset:

| Preset | Description / Use Case |
| :--- | :--- |
| `auto` | Selects `tri-cli` when a Gemini-family CLI and at least one Claude/Codex CLI are available; otherwise selects `openrouter` when its key is set. |
| `tri-cli` | Configures native CLI participants from the Claude/Codex families plus Antigravity (the Gemini-family CLI; Google retired the standalone `gemini` CLI for individual accounts in June 2026). |
| `openrouter` | Uses hosted API models through a single OpenRouter key. |
| `tri-cli-openrouter` | Configures native CLI and hosted OpenRouter participants. |
| `private-local` | Configures Ollama-only participants, excludes hosted and native CLI participants, and makes `private-local` the generated default mode. |
| `all` | Configures every discovered participant route on the host machine. |

Custom participants, runtime modes, and default options can be configured directly in `.llm-council.yaml`.
Older `local-private` setup commands and `local-only` runtime commands remain
accepted as deprecated aliases; new output and generated config use
`private-local`.

`private-local` connects only to an HTTP(S) loopback Ollama endpoint and
ignores proxy environment variables for that connection. It does not sandbox
or firewall the Ollama daemon itself; for a hard offline guarantee, run Ollama
with OS/network egress disabled and configure it to use only local model
artifacts.

### Advisory configuration knobs

These optional keys are **off by default** and **advisory-only** — they sharpen the council's signal without changing the read-only guarantee or gating a run. Set them under `defaults:` (global) or, where noted, per-mode / per-peer.

| Key | Scope | What it does |
| :--- | :--- | :--- |
| `min_distinct_vendors` / `require_distinct_vendors` | `defaults` / per-mode | Emit an `independence_warning` when fewer than N distinct vendor families produced a labeled vote (never affects quorum or `degraded`). |
| `cost_warn_usd` | `defaults` (or `--cost-warn-usd`) | Attach a non-fatal `cost_warning` when the pre-flight estimate exceeds the threshold; complements the hard `--max-cost-usd` gate. |
| `recommend_judge` | `defaults` | Name a hosted peer to add an LLM difficulty grade to `council_recommend`. Fail-open: any error falls back to the mechanical heuristic. |
| `deliberation_early_stop` | `defaults` / per-mode | In multi-round modes (`max_rounds ≥ 3`), stop deliberating early once a round shows no divergence **and** an unchanged vote tally. |
| `usage_from_json` | per-peer | Invoke `claude` / `codex` in their JSON output modes to record real token usage and cost; fails soft to raw text and keeps the read-only flags. |

> `litellm` pricing fallback is automatic when the optional `litellm` package is installed — it prices hosted models absent from the OpenRouter catalog. It is never a hard dependency.

See `CLAUDE.md` for the full invariant notes behind each knob.

---

## Read-Only Safety

Peers act strictly as advisors, not co-authors. How strongly that's enforced differs by peer type — know the difference before reviewing untrusted code:

*   **Flag-enforced (hard) — `claude`, `codex`, `antigravity`**: invoked with flags that disable their write tools at the CLI level (`--permission-mode manual` for Claude, `--sandbox read-only` for Codex, `--mode plan` for Antigravity). A misbehaving model — or a prompt-injected diff — *cannot* write files. The council prompt's read-only directive remains as defense in depth, and `--dangerously-skip-permissions` is deliberately **omitted** for `agy` so residual tool attempts are denied, not auto-approved.
*   **Hosted & local models** (OpenRouter / Ollama): plain API calls with no filesystem access at all — inherently read-only.
*   **Stdin isolation**: peers receive the codebase or diff via standard input (except `agy`, which stopped reading stdin in 1.1.1 and receives the prompt as a command-line argument instead).

---

## Manual Driver Configuration

To integrate the council with your active terminal developer agents, append the appropriate config snippet to their global/project instruction files:

### Claude Code
Append to `CLAUDE.md`:
```bash
cp .llm-council/skills/claude-code/SKILL.md ~/.claude/skills/llm-council/SKILL.md
```

### Codex CLI
Append to `~/.codex/AGENTS.md`:
```bash
cat .llm-council/skills/codex-cli/AGENTS.md >> ~/.codex/AGENTS.md
```

### Antigravity CLI
Append to `~/.gemini/GEMINI.md`:
```bash
cat .llm-council/skills/antigravity/GEMINI.md >> ~/.gemini/GEMINI.md
```

---

## Updates & Maintenance

Check current version:
```bash
llm-council --version
```

Check for updates:
```bash
llm-council check-update
```

Update to latest release:
```bash
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git
```

All releases and changes are logged in [CHANGELOG.md](CHANGELOG.md).

---

<sub>MIT Licensed. Built to help coding agents ask before they ship.</sub>
