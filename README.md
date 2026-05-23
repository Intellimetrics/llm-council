# LLM Council

[![Tests](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-ready-2f855a)](docs/llm-council.md)
[![Read-only](https://img.shields.io/badge/default-read--only-6b7280)](#read-only-safety)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.11.7-111827)](CHANGELOG.md)

Your coding agent is incredibly fast, capable, and confident. 

That is highly valuable—until it confidently overwrites a critical database migration, introduces a subtle security vulnerability, or refactors a complex module it wasn't supposed to touch.

**LLM Council** is a lightweight, read-only multi-agent orchestration harness designed to give your primary coding agent a fast, independent second opinion before committing to risky changes or expensive edits. It runs as a Python 3.11+ MCP server and command-line tool.

---

## Why Use LLM Council?

Coding agents move fast. To prevent single-model blind spots from becoming production incidents, LLM Council lets you pause and request a peer review:

> *"This diff looks plausible, but I want other models to try to break it."*

When you ask the council, peers read the same context (e.g., prompt, codebase, or git diff) and independently evaluate it. To ensure actionable feedback, every peer response is strictly parsed and must resolve to one of three consensus labels:

```text
RECOMMENDATION: yes       # Safe to proceed
RECOMMENDATION: no        # Stop; major issues detected
RECOMMENDATION: tradeoff  # Plausible, but note critical trade-offs
```

If a peer fails to supply one of these labels, the response is rejected as a failure. Vague essays are not permitted.

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

Once installed, navigate to your active project repository and run the setup wizard:

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

```bash
# Get a fast second opinion on a question
llm-council run --mode quick "Why is this test flaky?"

# Review the current git diff
llm-council run --mode review --diff "Is this migration safe to run?"

# Force a detailed debate with assigned stances
llm-council run --mode consensus --diff "Should we merge this auth rewrite?"

# Estimate token sizes and costs before running a council
llm-council estimate --mode consensus --diff "Should we merge this?"

# Inspect the last run's raw transcripts
llm-council last
```

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

---

## Peers & Agent Integration

LLM Council supports three classes of peer participants:

1.  **Native CLIs**: Uses existing command-line interfaces installed on your machine (`claude`, `codex`, `agy`, or `gemini`).
2.  **Hosted Models**: Integrates with remote APIs via **OpenRouter**.
3.  **Local Models**: Communicates with local instances via **Ollama**.

### Dynamic Triad Resolution & Family Exclusions
*   **The 3-CLI Triad**: The `tri-cli` preset dynamically selects exactly three active local CLIs. If both `antigravity` (`agy`) and `gemini` are installed, `llm-council` prioritizes `antigravity` as the active Gemini-family peer.
*   **Family Exclusions**: If the primary driver running your session is a Gemini-family agent (e.g., Antigravity CLI or Gemini CLI), `llm-council` automatically excludes other Gemini-family peers from the voting pool to avoid redundant reviews. It instead recruits independent peers (such as Claude Code and Codex CLI) for a balanced triad.

---

## Read-Only Safety

Peers act strictly as advisors, not co-authors:
*   **No File Modifications**: Every native CLI peer is invoked with read-only or strict approval-only flags (e.g., `agy --sandbox --dangerously-skip-permissions -` or `claude --read-only`). They cannot modify your files.
*   **Isolated Environments**: Peers inspect the codebase or diff via standard input, protecting your repository from accidental writes.

---

## Configured Modes

Modes determine the composition and behaviors of the council. You can customize them in `.llm-council.yaml`:

| Mode | Purpose |
| :--- | :--- |
| `quick` | Fast, lightweight review. Perfect default for general troubleshooting. |
| `peer-only` | Excludes the active driver CLI to hear solely from external peers. |
| `plan` | Structural/architecture questions *before* starting implementation. |
| `review` | Thorough diff evaluation before a merge or release. |
| `review-with-tools` | Experimental mode where peers run file-read/grep tools before voting. |
| `review-cheap` | Budget hosted review using smaller, cheaper models. |
| `diverse` | Broad coverage across different companies and host architectures. |
| `private-local` | Strict offline review pinned to local Ollama models. |
| `consensus` | Multi-round structured debate with assigned stances (Pro, Con, Neutral). |
| `deliberate` | Forces a secondary round of discussion even if peers initially agree. |

### The Consensus Mode
The `consensus` mode is designed for critical decisions (e.g., database schema changes, authentication logic). 
1.  **Assigned Stances**: One peer is assigned to argue in favor of the proposal, one against, and one to remain neutral.
2.  **Refutation Round**: If there is disagreement, a second round is run where peers receive the strongest opposing arguments and are given the opportunity to revise their stance.
3.  **No Forced Unanimity**: If the peers still disagree, the final report outlines the conflicting arguments clearly.

---

## Cost Controls & Data Boundaries

*   **Cost Caps**: Enforce limits to prevent unexpected hosted API charges:
    ```bash
    llm-council run --mode consensus --diff --max-cost-usd 0.50 "Is this migration safe?"
    ```
*   **Pre-flight Estimation**: Run `llm-council estimate` to calculate prompt tokens and project costs before making API calls.
*   **API Credentials**: Put keys in `.env`, `.env.local`, or `.llm-council.env`. Keys are never written directly into the shared `.mcp.json`.

> [!CAUTION]
> Do not use hosted council modes for classified, regulated, or credentialed codebases unless all configured models/providers are compliant with your security standards. For restricted codebases, use `local-private` or `private-local`.

---

## MCP Server Integration

The `llm-council` server exposes the following tools to your developer agents:

*   `council_run`: Run a council query with custom modes and optional diffs.
*   `council_estimate`: Check sizes and estimated costs.
*   `council_recommend`: Ask whether a council review is recommended for a given task.
*   `council_doctor`: Diagnoses connection issues and CLI path status.
*   `council_list_modes`: Lists configured presets, modes, and active peers.
*   `council_last_transcript`: Returns the path/contents of the last run.
*   `council_query_transcripts`: Searches past transcript history for similar reviews.

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
