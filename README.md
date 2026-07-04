# LLM Council

[![Tests](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-ready-2f855a)](docs/llm-council.md)
[![Read-only](https://img.shields.io/badge/default-read--only-6b7280)](#read-only-safety)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.16.0-111827)](CHANGELOG.md)

Your coding agent is incredibly fast, capable, and confident. 

That is highly valuable—until it confidently overwrites a critical database migration, introduces a subtle security vulnerability, or refactors a complex module it wasn't supposed to touch.

**LLM Council** is a lightweight, read-only multi-agent orchestration harness designed to give your primary coding agent a fast, independent second opinion before committing to risky changes or expensive edits. It runs as a Python 3.11+ MCP server and command-line tool.

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
    Orchestrator -- "3. Parallel invocation (isolated stdin)" --> PeerC[Peer C: Gemini]
    PeerA -- "4. YES/NO/TRADEOFF" --> Consensus[Consensus Evaluator]
    PeerB -- "4. YES/NO/TRADEOFF" --> Consensus
    PeerC -- "4. YES/NO/TRADEOFF" --> Consensus
    Consensus -- "5. Compile stances" --> Synth[Synthesis Chair]
    Synth -- "6. Generate Memo & Markdown/HTML dashboard" --> Out[Output Handler]
    Out -- "7. Auto-open HTML Dashboard (webbrowser)" --> Browser([Default Browser])
    Out -- "8. Structured JSON & Summary" --> Agent
```

---

## Key Features & Capabilities

*   **Consensus Gates & Vote Contracts**: Every peer response is strictly parsed and must resolve to one of three consensus labels. Vague essays are rejected:
    *   `RECOMMENDATION: yes` — Safe to proceed.
    *   `RECOMMENDATION: no` — Stop; major issues detected.
    *   `RECOMMENDATION: tradeoff` — Plausible, but note critical trade-offs.
*   **Assigned-Stance Debates**: Support for multi-round refutation and debates (`consensus` mode) where one peer defends, one opposes, and one remains neutral, followed by a compilation from the **Synthesis Chair**.
*   **Family Exclusion Routing**: Automatically detects the active agent running the session and excludes models from the same family (e.g., Gemini-family) from the peer pool to maximize reviewer diversity.
*   **Rigorous Sandboxing & Read-Only Safety**: Native CLI peers are invoked with binary-level flags disabling file writes (`--permission-mode default` for Claude, `--sandbox read-only` for Codex).
*   **Cost Controls & Caching**: Pre-flight token and USD cost estimation (`llm-council estimate`) plus response caching prevents unexpected hosted API charges. A hard `--max-cost-usd` / `--max-tokens` gate refuses a run before launch; an optional soft `cost_warn_usd` tier warns (but never blocks) when an estimate gets pricey, and an optional `litellm` fallback prices hosted models missing from the OpenRouter catalog.
*   **Credential Secret Scanning**: Scans all prompt content for API keys, tokens, or private keys, with configurable responses (`warn`, `block`, `redact`, or `off`).
*   **Auto-Opening HTML Dashboard**: Automatically generates a beautifully formatted HTML dashboard of the run and opens it in your default web browser upon completion.
*   **Anti-Herding Deliberation**: Round-2 deliberation asks peers to converge toward what is *correct* rather than toward agreement — no capitulating to the group, no digging in out of consistency bias — and to critique each other rather than re-defend their own prior answer (mitigating the multi-agent-debate herding failure mode).
*   **Dissent-Preserving Synthesis**: The Synthesis Chair attributes consensus blockers to the peers who raised them, narrates how positions moved across rounds, and names genuine remaining disagreement instead of papering over it — minority signal is never silently merged away.
*   **Cross-Vendor Independence Warning** *(opt-in)*: When every labeled vote comes from a single vendor family, the council flags it (`min_distinct_vendors`) so correlated same-vendor agreement isn't mistaken for independent corroboration.
*   **Contract-Scoped & Independent Review** *(opt-in)*: Anchor a review on numbered acceptance criteria (`--acceptance-contract`) so only real violations block, and suppress prior-round verdicts on a continuation (`--independent-review`) so re-reviews aren't anchored to past opinions.
*   **Observable Per-CLI Usage** *(opt-in)*: `usage_from_json` reads real token counts and cost from the Claude/Codex JSON output modes — recovering usage telemetry that is otherwise invisible for native CLI peers — while preserving the read-only invocation.

---

## Single-Model Peer Isolation

Even if you only have access to a single LLM (due to offline constraints, local setup, or API key limits), running a council session provides substantial value:

1.  **Fresh Eyes / Context Separation**: In a typical developer agent session, the context window accumulates tools, history, and reasoning, leading to confirmation bias. The council calls a fresh, isolated API process containing *only* the diff and prompt, forcing a clean evaluation.
2.  **Persona & State Splitting**: By invoking multiple calls to the same model concurrently with different persona prompts (e.g., Peer A as the *Attacker*, Peer B as the *Defender*, Peer C as the *Judge*), you isolate their states. The model cannot see its own arguments from the other perspective until the synthesis phase.
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
llm-council doctor
```

> [!IMPORTANT]
> After running setup, restart your active terminal session or reload your developer agent so it detects the newly configured MCP server and project-specific instructions.

### The 30-Second Example
Once configured, you can talk to your primary developer agent (e.g., Claude Code, Codex, Antigravity CLI) directly in natural language:

*   *"Ask the council to review my current changes before we commit."*
*   *"Take this failing test to council. I need theory and review, not another patch."*
*   *"Run a local-only council on this code. It cannot leave this machine."*

---

## CLI Reference

While most interaction happens transparently via the MCP server inside your agent, you can invoke the CLI directly:

| Command | Description / Example |
| :--- | :--- |
| **`llm-council run`** | Run a council query. <br>`llm-council run --mode quick "Why is this test flaky?"` |
| **`llm-council run --diff`** | Review the current git diff. <br>`llm-council run --mode review --diff "Is this migration safe to run?"` |
| **`llm-council run --focus`** | Compose operator-authored review-focus bundles onto any mode (comma-separated). <br>`llm-council run --mode review --focus security-review,test-gaps --diff "Safe to merge?"` |
| **`llm-council run --acceptance-contract`** | Anchor review on stated acceptance criteria (advisory). Pass literal text or a file path inside cwd; a finding blocks only when it violates a numbered criterion. <br>`llm-council run --mode review --acceptance-contract ./CRITERIA.md --diff "Does this meet the contract?"` |
| **`llm-council run --independent-review`** | On a `--continue` run, suppress the prior council's verdicts/rationales so the round forms its opinion independently. <br>`llm-council run --mode review --continue 20260101_120000 --independent-review --diff "Re-review"` |
| **`llm-council run --cost-warn-usd`** | Attach a non-fatal warning when the pre-flight estimate exceeds a threshold (complements, never replaces, the hard `--max-cost-usd` gate). <br>`llm-council run --mode consensus --cost-warn-usd 0.50 --diff "Worth a full debate?"` |
| **`llm-council estimate`** | Calculate prompt size and costs before running; reports a `cost_class` (low/moderate/high) plus paid/free peer counts. <br>`llm-council estimate --mode consensus --diff "Should we merge this?"` |
| **`llm-council last`** | Inspect the last run's raw transcripts. <br>`llm-council last` |
| **`llm-council config get`** | Retrieve a configuration value. <br>`llm-council config get defaults.auto_open_browser` |
| **`llm-council config set`** | Set a configuration value. <br>`llm-council config set defaults.auto_open_browser true` |

---

## MCP Server Integration

The `llm-council` server exposes the following tools to your developer agents:

| MCP Tool Name | Description / Inputs |
| :--- | :--- |
| **`council_run`** | Run a council query with custom modes, context files, and optional diffs. Supports `open: true` to auto-launch the HTML dashboard, `focus: ["security-review", ...]` to compose review-focus bundles, `acceptance_contract: "<text or path>"` to gate blockers on numbered criteria, `independent_review: true` to suppress prior-council context on a continuation run, and `cost_warn_usd` for a non-fatal cost heads-up. Surfaces advisory signals in the result when present (`independence_warning`, `cost_warning`, `cost_estimate`, `applied_focus`). |
| **`council_estimate`** | Check token sizes and estimated OpenRouter cost before launching. |
| **`council_recommend`** | Evaluates a task, risk level, and files touched to recommend whether to consult the council. Also returns a mechanical `difficulty_class` and the matched trigger keywords (`suggested_mode_reason_codes`), an optional LLM-graded `judge` verdict when `recommend_judge` is configured, and a reliability-based `peers_to_consider_dropping` advisory drawn from your recorded outcomes. |
| **`council_doctor`** | Diagnoses connection issues, API key status, and CLI path resolution. |
| **`council_list_modes`** | Lists configured modes, presets, and active peers. |
| **`council_last_transcript`** | Returns the path and content of the last recorded run. |
| **`council_stats`** | Aggregates participant metrics (run count, success, tokens, cost) across recorded transcripts. |
| **`council_query_transcripts`** | Searches past transcript history for similar reviews. |
| **`council_config`** | Get or set configuration keys in `.llm-council.yaml` programmatically via the MCP connection. |

---

## Presets & Configuration

The setup wizard (`llm-council setup --plan`) automatically probes your environment to find available tools and recommends the best preset:

| Preset | Description / Use Case |
| :--- | :--- |
| `auto` | Automatically selects the best mix of local CLIs, hosted keys, and local models. |
| `tri-cli` | Resolves a local 3-member triad (Claude Code, Codex CLI, and Antigravity CLI or Gemini CLI). |
| `openrouter` | Uses hosted API models through a single OpenRouter key. |
| `tri-cli-openrouter` | Runs local CLIs with hosted fallback or additional variety. |
| `local-private` | Strict offline-only review using local Ollama instances. |
| `all` | Configures every discovered route on the host machine. |

Custom presets, modes, and default options can be configured directly in `.llm-council.yaml`.

### Advisory configuration knobs

These optional keys are **off by default** and **advisory-only** — they sharpen the council's signal without changing the read-only guarantee or gating a run. Set them under `defaults:` (global) or, where noted, per-mode / per-peer.

| Key | Scope | What it does |
| :--- | :--- | :--- |
| `min_distinct_vendors` / `require_distinct_vendors` | `defaults` / per-mode | Emit an `independence_warning` when fewer than N distinct vendor families produced a labeled vote (never affects quorum or `degraded`). |
| `cost_warn_usd` | `defaults` (or `--cost-warn-usd`) | Attach a non-fatal `cost_warning` when the pre-flight estimate exceeds the threshold; complements the hard `--max-cost-usd` gate. |
| `recommend_judge` | `defaults` | Name a hosted peer to add an LLM difficulty grade to `council_recommend`. Fail-open: any error falls back to the mechanical heuristic. |
| `deliberation_early_stop` | `defaults` / per-mode | In multi-round modes (e.g. `deep-audit`, `max_rounds ≥ 3`), stop deliberating early once a round shows no divergence **and** an unchanged vote tally. |
| `usage_from_json` | per-peer | Invoke `claude` / `codex` in their JSON output modes to record real token usage and cost; fails soft to raw text and keeps the read-only flags. |

> `litellm` pricing fallback is automatic when the optional `litellm` package is installed — it prices hosted models absent from the OpenRouter catalog. It is never a hard dependency.

See [Review Focus Bundles](#review-focus-bundles) for the `--focus` / `focus:` bundle system, and `CLAUDE.md` for the full invariant notes behind each knob.

---

## Read-Only Safety

Peers act strictly as advisors, not co-authors. How strongly that's enforced differs by peer type — know the difference before reviewing untrusted code:

*   **Flag-enforced (hard) — `claude`, `codex`, `gemini`**: invoked with flags that disable their write tools at the CLI level (`--permission-mode default` for Claude, `--sandbox read-only` for Codex, `--approval-mode plan` for Gemini). A misbehaving model — or a prompt-injected diff — *cannot* write files.
*   **Hosted & local models** (OpenRouter / Ollama): plain API calls with no filesystem access at all — inherently read-only.
*   **Prompt-enforced (soft) — `antigravity` (`agy`)**: `agy` exposes no read-only / approval-mode / tools-allowlist flag, and its `--sandbox` only restricts the *terminal*, not the model's native file-write tool — so `agy` *can* write files. Its read-only behavior is carried by the council prompt's read-only directive, which `agy` reliably honors (it refuses write requests). `--dangerously-skip-permissions` is deliberately **omitted** so a stray write isn't auto-approved. This is a **softer** guarantee than the flag-enforced peers: a determined prompt-injection in reviewed content could in principle override the directive with no hard backstop. If you review untrusted code, prefer `gemini` (hard) over `antigravity`.
*   **Stdin isolation**: peers receive the codebase or diff via standard input.

---

## Review Focus Bundles

Operator-authored "review focus" bundles let you express *what* a council should scrutinize without editing source. A bundle composes onto **any** mode and is **inert prompt text only** — it shapes the review angle but grants **no tool, write, or exec capability**. It rides on top of the same read-only guarantees described above.

**Layout** — drop a bundle under your project's `.llm-council/` directory (discovery walks up from cwd, first match wins, exactly like config discovery):

```
.llm-council/review-skills/<name>/SKILL.md
```

Each `SKILL.md` is YAML-ish frontmatter (`name:` + `description:`) followed by a markdown body of read-only scrutiny directives:

```markdown
---
name: security-review
description: Read-only security scrutiny lens.
---
Scrutinize for authz/authn gaps, injection, secrets in code, unsafe
deserialization. Cite file:line. Do not propose edits, only flag.
```

**Usage** — name one or more bundles; they compose with the active mode and persist across deliberation rounds:

```bash
llm-council run --mode review --focus security-review,test-gaps --diff "Safe to merge?"
```

MCP equivalent: pass `"focus": ["security-review", "test-gaps"]` to `council_run`.

**Validation & discovery semantics**:

*   **Strict names**: `name` must match `^[a-z0-9-]+$`, be ≤ 64 chars, and equal the directory name.
*   **Lenient discovery**: a malformed bundle is *skipped* (with a reason), never fatal — one bad bundle can't break a run. The CLI prints a one-line warning naming skipped bundles.
*   **Fail fast on typos**: an unknown `--focus` name aborts the run with the list of available bundles **before any peer is launched**.
*   **Provenance (M11)**: applied bundles are recorded in the transcript (`metadata.applied_focus`, markdown summary line) and surfaced top-level in the `council_run` MCP response as `applied_focus` (bundle name + content `sha256`).

Ready-to-copy examples live in [`examples/review-skills/`](examples/review-skills/) (`security-review`, `test-gaps`). Copy one into your project's `.llm-council/review-skills/` and edit the body to taste — they are documentation, not auto-applied.

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

### Antigravity CLI / Gemini CLI
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
