# LLM Council

[![Tests](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Intellimetrics/llm-council/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-ready-2f855a)](docs/llm-council.md)
[![Source read-only peers](https://img.shields.io/badge/peers-source--read--only-6b7280)](#read-only-safety)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.26.0-111827)](CHANGELOG.md)

Your coding agent grades its own homework. **llm-council** gives it a second opinion that isn't its own: one question or diff, fanned out in parallel to independent, read-only AI peers — the Claude, Codex, and Antigravity CLIs you already have, on your own accounts — and every peer must commit to a verdict: `yes`, `no`, or `tradeoff`. Read-only is enforced by CLI flags, not vibes. Essays without a verdict are rejected.

Runs as an MCP server inside your coding agent, or as a standalone CLI.

**Jump to:** [Quickstart](#quickstart) · [How it works](#how-it-works) · [Modes](#built-in-modes) · [Blast-radius context](#blast-radius-context-okf) · [Built-in coaching](#the-council-coaches-its-operator) · [Safety](#read-only-safety) · [MCP tools](#mcp-tools) · [Configuration](#configuration--presets)

## What a run looks like

A real run from this repo's own development. Upstream `mcp` 2.0.0 had just broken fresh installs, we'd shipped a `<2` pin instead of migrating immediately, and we asked the council whether that was the right call. This is the actual output the host agent received:

> **LLM Council** · mode=quick · 3/3 succeeded · run wall=105.2s · recommendation=**yes**
>
> | peer | label | wall time |
> |---|---|---|
> | claude | yes | 105.2s |
> | codex | yes | 89.8s |
> | antigravity | yes | 74.3s |
>
> Transcript: `.llm-council/runs/20260806_123707_….md`

Each peer's response opens with its verdict, backed by evidence the orchestrator mechanically checks against your repo:

```text
RECOMMENDATION: yes - pinning `mcp>=1.0.0,<2` was the right immediate call.
Fresh installs crashing at startup is a severity-critical regression, the pin
is the standard minimal remediation, and the coupling surface in mcp_server.py
is small and well-localized... The pin should carry a deliberate follow-up
ticket, not sit indefinitely.

EVIDENCE:
- [VERIFIED:llm_council/mcp_server.py:991-1035] Progress bridge depends on
  `session.send_progress_notification`, the deepest 1.x internal coupling...
```

Every run also writes paired Markdown, JSON, and HTML transcripts under `.llm-council/runs/`. (This README was itself council-reviewed — unanimous *yes* on "does this need a rewrite" — and rebuilt to the council's blueprint.)

## Quickstart

```bash
# 1. Install (uv)
uv tool install --force git+https://github.com/Intellimetrics/llm-council.git

# 2. Wire it into your project
cd /path/to/your/project
llm-council setup --plan                 # detect what's available, change nothing
llm-council setup --yes --preset auto    # write .llm-council.yaml + .mcp.json
llm-council doctor --probe-native        # verify the peers actually respond
```

> [!IMPORTANT]
> Setup writes host-specific snippets under `.llm-council/instructions/` — by default it never edits your instruction files for you. Either append the active host's snippet to `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` (setup prints the exact paths), or pass `--write-instructions` to have setup maintain an idempotent marker-delimited block in those files itself (re-runs replace the block; everything outside the markers is untouched). Then restart that agent session so MCP reloads.

Then just talk to your coding agent:

- *"Ask the council to review my current changes before we commit."*
- *"Take this failing test to council. I need theory and review, not another patch."*
- *"Run a private-local council on this — route it only to my same-machine Ollama."*

Or drive it directly:

```bash
llm-council run --mode review --diff "Is this migration safe to run?"
```

<details>
<summary>Requirements · pipx install · updating · why not uvx</summary>

**Requirements.** Python 3.11+. At least one peer route: native CLIs (`claude`, `codex`, and/or `agy`) on PATH under your own accounts, an `OPENROUTER_API_KEY` for hosted models, or a local Ollama daemon. Optional: the [`okf-rs`](https://github.com/jyjeanne/okf-rs) binary for [blast-radius context](#blast-radius-context-okf). `llm-council doctor` tells you exactly which of these it can see — and what to install for the ones it can't.

```bash
# pipx alternative
pipx install --force git+https://github.com/Intellimetrics/llm-council.git

# update = re-run the install command; check what's newer first with:
llm-council check-update
```

After updating, restart the MCP connection and verify it through the connected
`council_doctor` tool. Checkout edits and `uv run` tests do not update a separate
`uv tool` installation. For local builds and refreshing agent instructions, see
[updating an active installation](docs/llm-council.md#updating-an-active-installation).

Avoid `uvx` for your primary installation: it's fine for one-off trials but doesn't keep the tool persistently on PATH or give MCP a stable environment. All releases are logged in [CHANGELOG.md](CHANGELOG.md).

</details>

## How it works

See the [September review and improvement priorities](docs/review-improvements.md)
for recent reliability fixes and the next review-quality improvements.

llm-council sits between your primary agent and a pool of independent peers. Peers answer in parallel from fresh, isolated contexts; votes are parsed, checked, and reduced to one headline verdict.

```mermaid
graph TD
    User([User Prompt]) --> Agent[Primary Developer Agent]
    Agent -- "1. Trigger (natural language / command)" --> Server[LLM Council MCP Server]
    Server -- "2. Parse prompt / git diff" --> Orchestrator[Orchestrator]
    Orchestrator -- "3. Parallel invocation (isolated stdin)" --> PeerA[Peer A: Claude]
    Orchestrator -- "3. Parallel invocation (isolated stdin)" --> PeerB[Peer B: Codex]
    Orchestrator -- "3. Parallel invocation (prompt argument)" --> PeerC[Peer C: Antigravity]
    PeerA -- "4. YES/NO/TRADEOFF" --> Consensus[Consensus Evaluator]
    PeerB -- "4. YES/NO/TRADEOFF" --> Consensus
    PeerC -- "4. YES/NO/TRADEOFF" --> Consensus
    Consensus -- "5. Write Markdown/HTML transcript" --> Out[Output Handler]
    Consensus -. "Optional: --synthesize" .-> Synth[Synthesis Chair]
    Synth -. "Decision memo" .-> Out
    Out -. "Optional: --open or config" .-> Browser([Default Browser])
    Out -- "6. Structured JSON & Summary" --> Agent
```

## Why trust it

- **Hard read-only peers.** Native CLI peers run with write tools disabled at the flag level — `--permission-mode manual` (Claude), `--sandbox read-only` (Codex), `--mode plan` (Antigravity). A misbehaving model — or a prompt-injected diff — *cannot* write. [Details.](#read-only-safety)
- **A vote contract, not vibes.** Responses are strictly parsed for `RECOMMENDATION: yes|no|tradeoff`. A peer that rambles without committing is rejected and retried once, then dropped from quorum.
- **Local-CLI-only by default.** Every built-in mode rosters only the native CLIs running under your own accounts. No hosted, per-token-billed peer ever runs unless you explicitly opt in.
- **Cost gates before launch.** Pre-flight token/USD estimation (`llm-council estimate`), a hard `--max-cost-usd` / `--max-tokens` refusal gate, a soft `cost_warn_usd` heads-up, and response caching.
- **Anti-herding deliberation.** When first-round votes disagree, an optional round 2 asks peers to converge on what's *correct* — critique each other, don't capitulate to the group or dig in — mitigating the multi-agent-debate herding failure mode.

<details>
<summary><b>More capabilities</b> — recursion-proofing, secret scanning, synthesis, independence checks, usage telemetry…</summary>

- **Recursion-proof**: a council can never start another council. Every peer subprocess carries a `LLM_COUNCIL_NESTED` marker that makes any nested `council_run` refuse outright, and codex peers boot with an empty MCP server table so a globally registered llm-council can't spawn inside a peer.
- **Credential secret scanning**: prompt content is scanned for API keys, tokens, and private keys, with configurable responses (`warn`, `block`, `redact`, `off`).
- **Assigned-stance review**: `consensus` mode assigns for/against/neutral stances to attack groupthink and sycophancy.
- **Dissent-preserving synthesis** *(opt-in)*: with `--synthesize`, a Synthesis Chair writes a decision memo that attributes blockers to the peers who raised them and names genuine remaining disagreement instead of papering it over.
- **Cross-vendor independence warning** *(opt-in)*: when every labeled vote comes from a single vendor family, the result carries an `independence_warning` so correlated agreement isn't mistaken for independent corroboration.
- **Independent re-review** *(opt-in)*: `--independent-review` suppresses prior-round verdicts on a continuation, so re-reviews aren't anchored to past opinions.
- **Observable per-CLI usage** *(opt-in)*: `usage_from_json` reads real token counts and cost from the Claude/Codex JSON output modes — telemetry that is otherwise invisible for native CLI peers — while preserving the read-only invocation.
- **Host-aware routing**: detects which agent is asking; `quick` includes a fresh instance of the host's own family for a clean-context pass, `peer-only` excludes it.
- **HTML transcript**: every run renders a formatted HTML dashboard; opening it in the browser is opt-in (`--open` or `defaults.auto_open_browser: true`).
- **Loud context accounting**: context files too large to inline are never silently dropped — the result carries a top-level `context_files_dropped` warning, and the transcript header flags it.
- **A stable failure taxonomy**: every peer failure maps to a machine-readable `error_kind` (`timeout`, `quota_exhausted`, `content_refused`, `abdicated`, `model_substituted`, …) surfaced in transcripts, `--json` output, and stats — so patterns across runs are countable, not anecdotal. The full table lives in [`CLAUDE.md`](CLAUDE.md#failure-taxonomy).

</details>

## Built-in modes

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
| `fable` | Read-only second opinion from Claude Fable 5.1, with defensive-review framing and a model-substitution guard. |
| `private-local` | Routes only to same-machine loopback Ollama participants. |

Any mode composes with `--origin-policy us` to restrict the roster to US-origin participants, and with your own modes defined in `.llm-council.yaml`.

## Blast-radius context (OKF)

A diff shows *what changed*; it doesn't show *who depends on it*. With `--okf-context` (CLI), `okf_context: true` (MCP / per-mode / defaults), a diff review also carries a budgeted one-hop call graph of matched touched symbols:

```bash
llm-council run --mode review --diff --okf-context "Is this refactor safe?"
```

Under the hood, llm-council generates an ephemeral [Open Knowledge Format](https://github.com/jyjeanne/okf-rs) bundle from your working tree (`okf-rs generate -o <tmpdir> --no-cache` — nothing is ever written into your project), maps the diff's line ranges to concepts, and inserts a compact excerpt after the Git Diff section: signatures, `file#Lstart-Lend` locators, callers, and callees.

```text
## OKF Blast Radius (call-graph context)

### functions/llm_council/deliberation/recommendation_label
- signature: `def recommendation_label(text: str) -> str`
- resource: llm_council/deliberation.py#L92-L103
- called_by:
  - `def _is_labeled_vote(r) -> bool` — llm_council/orchestrator.py#L275-L276
  - `def aggregate(records, ...)` — llm_council/stats.py#L177-L394
  ...
```

The graph provides starting points for finding affected callers and tests. In
the September 5 live comparison, all three native peers found the same injected
compatibility bug with OKF off and on (30.1s versus 33.9s). That verifies the
integration, but does not establish an accuracy or speed improvement. Graph
edges can be noisy; verify relationships against source before relying on them.

The default excerpt budget is 12,000 characters. Check `metadata.okf_context`
for status and `concepts` / `matched` coverage: a large diff may include only a
subset. Currently OKF calculates headroom before diff trimming, so it can be
omitted as `excerpt_over_budget` even when trimming later frees space. A saved
fallback bundle is marked `stale_attached`. See [context access and measured
limits](docs/review-context.md).

Everything about it is opt-in and fail-soft: no built-in mode enables it, prompts are byte-identical when it's off, and a missing binary, failed generation, or unmatched diff leaves the run exactly as it would have been — plus a `metadata.okf_context` diagnostic, a transcript-header note, and a stderr warning. Requires the external `okf-rs` binary on PATH (prebuilt binaries on its releases page, or `cargo install --git https://github.com/jyjeanne/okf-rs okf-cli`).

## What peers can access

Native CLI peers can read files beyond the attached diff with their configured
read-only tools. Ask them to inspect affected callers, tests, and requirements;
`review-with-tools` adds this direction automatically. Hosted API and Ollama
peers receive the assembled context without a filesystem tool loop.

Peers do not automatically receive the host conversation, connected apps, or
research results, and live web access should not be assumed. Supply the decision
criteria in the question and relevant documents through `context_files`.

For a tool-assisted diff review with OKF, use MCP `mode: "review-with-tools"`,
`include_diff: true`, and `okf_context: true`, or:

```bash
llm-council run --mode review-with-tools --diff --okf-context \
  "Check compatibility; inspect callers and tests and cite the affected code."
```

Project modes can override the roster and OKF setting. This repository enables
OKF in `review`; its `review-with-tools` mode selects Claude and Codex and needs
the explicit OKF toggle. One mode does not inherit settings from another.

Before acting, check failed peers, `context_files_dropped`, `degraded`, and
`metadata.partial`. Quorum and verified line ranges do not establish complete
coverage or prove that a finding is correct.

## The council coaches its operator

The tuning knowledge isn't buried in docs — the tool applies it to your own telemetry and tells you what to change:

- **`llm-council stats` ends with a `recommendations:` block.** The aggregated numbers are run through the same interpretation rules the maintainers use: timeout walls that retries never rescue → the exact `timeout` / `timeout_multiplier` key to raise; repeated quota walls with no recovery → configure a `fallback_chain`; high missing-label rates → the phrasing or `require_recommendation` fix; content-policy refusals → rephrase as verification. Conservative minimum sample sizes, advisory only, also present in `--json` and MCP `council_stats` output.

  ```text
  recommendations:
    - claude: 6 timeouts on small prompts with 0 terse-retry recoveries —
      raise `participants.claude.timeout` (or the mode's `timeout_multiplier`).
    - deepseek_v4_flash: 20% of successful responses lacked a usable
      RECOMMENDATION label — check custom prompt phrasing; ...
  ```

- **Every failing `doctor` check says what to do next** — missing CLIs carry the install command, missing keys point at `.llm-council.env`, and an `okf:binary` row tracks the optional okf-rs dependency (informational when unused; a real failure only when your config enables `okf_context` and the binary is gone).
- **`llm-council list --verbose`** surfaces the per-peer and per-mode tuning keys that are otherwise invisible — `reasoning_effort`, `usage_from_json`, `env_strict`, `fallback_chain`, mode `timeout_multiplier`s, `stances`, `model_overrides` — plus your configured `tiers`.
- **`llm-council setup --write-instructions`** closes the last manual step: instead of hand-appending snippets, setup maintains an idempotent marker-delimited block in `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` itself. Re-runs replace the block; bytes outside the markers are never touched; writes are atomic; a file with a damaged marker pair is refused, never guessed at.

## Read-Only Safety

Peers are advisors, not co-authors. How strongly that's enforced differs by peer type — know the difference before reviewing untrusted code:

- **Flag-enforced (hard) — `claude`, `codex`, `antigravity`**: write tools are disabled at the CLI level, so even a prompt-injected diff cannot make a peer write files. The council prompt's read-only directive remains as defense in depth, and `--dangerously-skip-permissions` is deliberately **omitted** for `agy` so residual tool attempts are denied, not auto-approved.
- **Hosted & local models** (OpenRouter / Ollama): plain API calls with no filesystem access at all — inherently read-only.
- **Stdin isolation**: peers receive the codebase or diff via standard input (except `agy`, which stopped reading stdin in 1.1.1 and receives the prompt as a command-line argument instead).

## MCP tools

MCP requests default to a 240-second deadline, below the observed 300-second
host timeout. Peer work stops early enough to clean up and save a **partial
result** with completed votes. Unfinished peers do not vote. Longer overrides
require a longer host tool-call timeout too. See
[request deadlines and input limits](docs/request-limits.md).

The server exposes ten tools to your agent. The ones you'll actually see in traffic:

| Tool | What it does |
| :--- | :--- |
| `council_run` | Run a council query with modes, context files, optional diffs, and optional `okf_context` blast-radius enrichment. Returns the verdict, per-peer results, and advisory signals (`cost_warning`, `independence_warning`, `context_files_dropped`, `metadata.okf_context`). |
| `council_recommend` | Should this task even go to council? Returns a difficulty class, matched trigger keywords, and (optionally) an LLM-graded verdict. |
| `council_estimate` | Token sizes and estimated cost before launching. |
| `council_doctor` | Diagnoses connectivity, API keys, and CLI path resolution. |

<details>
<summary>All ten MCP tools</summary>

| Tool | Description |
| :--- | :--- |
| `council_run` | Run a council query with custom modes, context files, and optional diffs. Supports `open: true`, `independent_review: true`, and `cost_warn_usd`. |
| `council_estimate` | Check token sizes and estimated OpenRouter cost before launching. |
| `council_recommend` | Evaluates a task, risk level, and files touched to recommend whether to consult the council. |
| `council_doctor` | Diagnoses connection issues, API key status, and CLI path resolution. |
| `council_models` | Lists the cached OpenRouter model catalog. |
| `council_list_modes` | Lists configured runtime modes and participants. |
| `council_last_transcript` | Returns the path and content of the last recorded run. |
| `council_stats` | Aggregates participant metrics (runs, success, tokens, cost, quota incidents, OKF attach rates) across transcripts — including the advisory `recommendations` list. |
| `council_query_transcripts` | Searches past transcript history for similar reviews. |
| `council_config` | Get or set `.llm-council.yaml` keys over the MCP connection. |

</details>

## Configuration & presets

`llm-council setup --plan` probes your environment and recommends a preset:

| Preset | Use case |
| :--- | :--- |
| `auto` | Selects `tri-cli` when a Gemini-family CLI and at least one Claude/Codex CLI are available; otherwise `openrouter` when its key is set. |
| `tri-cli` | Native CLI participants from the Claude/Codex families plus Antigravity (the Gemini-family CLI; Google retired the standalone `gemini` CLI for individual accounts in June 2026). |
| `openrouter` | Hosted API models through a single OpenRouter key. |
| `tri-cli-openrouter` | Native CLIs plus hosted OpenRouter participants. |
| `private-local` | Ollama-only; excludes hosted and native CLI participants and makes `private-local` the default mode. |
| `all` | Every discovered participant route on the host machine. |

Custom participants, modes, and defaults live in `.llm-council.yaml`.

See [native model configuration](docs/native-models.md) for current Codex,
Claude, and Gemini model options, fallback migration, and account-specific
availability. Native primary peers inherit the operator's CLI model selection;
the `fable` peer explicitly pins `claude-fable-5-1`.

Response caching is enabled for hosted/local API peers. CLI peers run afresh
because they can inspect files outside the pasted prompt. Set per-participant
`cache_response: true` only for a custom CLI whose response depends solely on
the supplied prompt/configuration, or `false` to disable an API peer's cache.
`--cache off` and consensus-mode cache disabling still take precedence.

`private-local` connects only to an HTTP(S) loopback Ollama endpoint and ignores proxy environment variables for that connection. It does not sandbox the Ollama daemon itself; for a hard offline guarantee, run Ollama with OS/network egress disabled.

<details>
<summary>Advisory knobs (off by default) · deprecated aliases</summary>

These optional keys sharpen the council's signal without changing the read-only guarantee or gating a run. Set under `defaults:` (global) or, where noted, per-mode / per-peer.

| Key | Scope | What it does |
| :--- | :--- | :--- |
| `min_distinct_vendors` / `require_distinct_vendors` | `defaults` / per-mode | Emit an `independence_warning` when fewer than N distinct vendor families produced a labeled vote (never affects quorum or `degraded`). |
| `cost_warn_usd` | `defaults` (or `--cost-warn-usd`) | Attach a non-fatal `cost_warning` when the pre-flight estimate exceeds the threshold; complements the hard `--max-cost-usd` gate. |
| `recommend_judge` | `defaults` | Name a hosted peer to add an LLM difficulty grade to `council_recommend`. Fail-open: any error falls back to the mechanical heuristic. |
| `deliberation_early_stop` | `defaults` / per-mode | In multi-round modes (`max_rounds ≥ 3`), stop deliberating early once a round shows no divergence **and** an unchanged vote tally. |
| `usage_from_json` | per-peer | Invoke `claude` / `codex` / `agy` in their JSON output modes to record real token usage and cost (and, for `agy`, its `status` when a response comes back empty); fails soft to raw text and keeps the read-only flags. |
| `skip_terse_retry_when_quorum_met` | `defaults` | Default **on**: when a peer times out after the round already has labeled quorum, skip its timeout retry instead of burning more wall time. Set `false` to always retry. |
| `reasoning_effort` | per-peer (`codex`) | Reasoning effort pinned for the council turn as `-c model_reasoning_effort=…` (default `medium`; `inherit` uses the CLI's own config). Codex otherwise inherits the operator's interactive `~/.codex/config.toml` setting — at `ultra`, 5 KB review prompts blew 600 s timeouts. |
| `okf_context` | `defaults` / per-mode (or `--okf-context`) | With a diff attached, append a one-hop call-graph blast-radius excerpt derived from an OKF knowledge bundle (requires the external [`okf-rs`](https://github.com/jyjeanne/okf-rs) binary; ephemeral tempdir generation, never written into the project). Fail-soft: any OKF problem leaves the run unchanged plus a `metadata.okf_context` diagnostic. |
| `retry_on_empty_response` | per-peer | Default **on**: one same-prompt re-run when a CLI exits 0 with no output. Either way the error records the exit status and a stderr tail, and provider content-policy refusals surface as `content_refused_peers` (rephrase as verification, not attack). |
| `tiers` | `defaults` (applied with `--tier <name>`) | Named per-peer model-swap maps — e.g. a `deep` tier pinning top thinking models, a `fast` tier pinning budget ones. Peers missing from a tier map keep their default model, so a tier can swap a subset. |
| `timeout_multiplier` | per-mode | Layered on the per-peer base timeout for slow modes (built-ins: `consensus` 2.0×, `deliberate` 1.5×). A prompt-size bonus also scales timeouts automatically (~5 s/KB above 4 KB). |
| `fallback_chain` | per-peer | Ordered step-down model ids tried when a peer hits a quota wall (up to 3 steps); recoveries surface as `quota_recoveries`. Claude-family peers delegate to the CLI's own `--fallback-model` instead. |
| `env_strict` | per-peer | Restrict the peer subprocess to a safe env-var allowlist plus its `env_passthrough` — stops ambient `*_MODEL` / `*_BASE_URL` exports from silently steering a CLI peer's model or endpoint. |
| `okf_max_excerpt_chars` / `okf_generate_timeout_seconds` / `okf_binary` | `defaults` | Tuning for [blast-radius context](#blast-radius-context-okf): excerpt budget (default 12 000 chars), generation timeout (default 20 s, hard-capped at 120 s), and the binary name/path. |

`litellm` pricing fallback is automatic when the optional `litellm` package is installed — never a hard dependency. Older `local-private` / `local-only` command aliases remain accepted as deprecated; new output uses `private-local`. See `CLAUDE.md` for the full invariant notes behind each knob.

</details>

<details>
<summary><b>CLI reference</b></summary>

| Command | Description / Example |
| :--- | :--- |
| **`llm-council run`** | Run a council query. <br>`llm-council run --mode quick "Why is this test flaky?"` |
| **`llm-council run --diff`** | Review the current git diff. <br>`llm-council run --mode review --diff "Is this migration safe to run?"` |
| **`llm-council run --continue`** | Continue a prior run. A private-local parent stays private-local when mode is omitted; moving its context to hosted/native peers is refused unless `--allow-privacy-downgrade` is explicit. |
| **`llm-council run --independent-review`** | On a `--continue` run, suppress the prior council's verdicts so the round forms its opinion independently. |
| **`llm-council run --cost-warn-usd`** | Non-fatal warning when the pre-flight estimate exceeds a threshold. <br>`llm-council run --mode consensus --cost-warn-usd 0.50 --diff "Worth a full debate?"` |
| **`llm-council run --okf-context`** | Attach the [OKF blast-radius excerpt](#blast-radius-context-okf) to a `--diff` review. Also on `estimate` for size parity. |
| **`llm-council run --tier`** | Apply a named model-swap tier from `defaults.tiers` for this run. |
| **`llm-council estimate`** | Prompt size and cost before running; reports a `cost_class` plus paid/free peer counts. |
| **`llm-council recommend`** | Zero-cost heuristic: should this task go to council at all, and in which mode? <br>`llm-council recommend "swap the auth middleware" --files-touched 7` |
| **`llm-council last`** | Inspect the last run's raw transcripts. |
| **`llm-council list --verbose`** | Participants and modes, plus the otherwise-invisible tuning keys and configured tiers. |
| **`llm-council stats`** | Aggregate per-peer metrics (incl. quota incidents/recoveries and OKF attach rates) across recorded transcripts, ending with the advisory `recommendations:` block. |
| **`llm-council setup --write-instructions`** | Have setup maintain the instruction block in `CLAUDE.md`/`AGENTS.md`/`GEMINI.md` itself (idempotent markers, atomic writes). |
| **`llm-council models refresh`** | Refresh the cached OpenRouter model/pricing catalog. |
| **`llm-council config get/set`** | Read or write configuration values. <br>`llm-council config set defaults.auto_open_browser true` |
| **`llm-council install-hook`** | Install a `pre-commit` or `pre-push` council gate; refuses to replace an existing hook unless `--force`. |
| **`llm-council transcripts prune`** | Preview transcript deletion under a retention policy; add `--delete` to remove. |
| **`llm-council doctor`** | Diagnostics: `--probe-native`, `--probe-openrouter`, `--probe-ollama`, `--check-update`. |

</details>

<details>
<summary><b>Going deeper</b> — single-model councils · manual host wiring</summary>

### Only have one model? Still worth it

1. **Fresh eyes / context separation**: your agent's session context accumulates history and reasoning that breed confirmation bias. Council peers get a fresh, isolated process containing *only* the diff and prompt.
2. **Stance splitting**: concurrent calls to the same model with different stances (attacker / defender / judge) isolate their states; peers see one another's arguments only if deliberation or synthesis is enabled.
3. **Adversarial extraction**: a contrarian stance ("find 3 security flaws in this diff") bypasses the model's default agreeable bias and extracts deeper critique.

### Manual host wiring

`llm-council setup` generates all of these; run it first. Then:

**Claude Code** — install the generated skill:

```bash
cp .llm-council/skills/claude-code/SKILL.md ~/.claude/skills/llm-council/SKILL.md
```

**Codex CLI** — append the generated snippet:

```bash
cat .llm-council/skills/codex-cli/AGENTS.md >> ~/.codex/AGENTS.md
```

**Antigravity CLI** — append the generated snippet:

```bash
cat .llm-council/skills/antigravity/GEMINI.md >> ~/.gemini/GEMINI.md
```

</details>

---

<sub>MIT Licensed. Built to help coding agents ask before they ship.</sub>
