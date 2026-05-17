# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`llm-council` is a Python 3.11+ MCP server and CLI that lets one coding agent
ask a "council" of other LLMs (Claude Code, Codex CLI, Gemini CLI, OpenRouter
hosted models, local Ollama) for read-only second opinions. It is published as
the `llm-council` console script and as the `llm-council` MCP server.

## Common commands

```bash
# Install editable with dev deps
python -m pip install -e ".[dev]"

# Run the full test suite
pytest -q

# Run a single test file or test
pytest tests/test_config_validation.py -q
pytest tests/test_llm_council.py::test_specific_name -q

# Local CLI invocation without install
python -m llm_council --help

# Setup wizard (writes .llm-council.yaml, .mcp.json, instruction snippets)
llm-council setup --plan                       # detect routes, do not write
llm-council setup --yes --preset tri-cli       # write without prompting

# Diagnostics
llm-council doctor [--probe-openrouter] [--probe-ollama] [--check-update]

# Direct council run (skips MCP)
llm-council run --current codex --mode review --diff "Review this change"

# Run as MCP server over stdio (what `.mcp.json` invokes)
llm-council mcp-server

# Eval harness (v0.8+)
llm-council eval run --mode review --out scorecard.json
llm-council eval run --mode review-with-tools --compare-against scorecard.json

# Outcome tracking (v0.8+)
llm-council outcome mark <run-id-or-prefix> --decision shipped|reverted|rejected|unknown [--bug-found yes|no] [--winning-peer X] [--note "..."]
llm-council outcome list

# Per-peer reliability counters from outcomes (v0.8+)
llm-council stats --reliability [--peer claude]
```

CI (`.github/workflows/test.yml`) runs `pytest -q` on Python 3.11 and 3.12.
There is no separate lint/format step configured.

## Architecture

The codebase is small and single-package (`llm_council/`). Two surfaces share
the same core:

1. `cli.py` (`main` -> `llm-council` script)
2. `mcp_server.py` (`llm-council mcp-server`, exposing `council_run`,
   `council_estimate`, `council_recommend`, `council_doctor`,
   `council_list_modes`, `council_last_transcript`, `council_models`,
   `council_stats`)

Both flow through the same pipeline:

```
load_project_env -> load_config (defaults + project YAML)
                 -> select_participants (mode + current CLI + origin filter)
                 -> build_prompt (question + optional context files / diff / stdin)
                 -> execute_council (orchestrator)
                    -> run_participants (adapters: cli | openrouter | ollama)
                    -> optional deliberation round on disagreement
                 -> write_transcript (markdown + json under transcripts_dir)
```

Key modules:

- `defaults.py` — built-in `DEFAULT_CONFIG`. Project YAML is deep-merged on top
  via `config.deep_merge`. The set of legal participant types (`cli`,
  `openrouter`, `openai_compatible`, `ollama`) and built-in modes (`quick`,
  `peer-only`, `plan`, `review`, `review-cheap`, `diverse`, `private-local`,
  `local-only`, `us-only`, `deliberate`, `consensus`, plus the temporary
  `opus-versions`) live here.
- `config.py` — config discovery (`find_config` walks up from cwd looking for
  `.llm-council.yaml` etc.), validation, the `other_cli_peers` strategy used by
  most modes, `detect_current_agent` (parent-process walk on `/proc`), and
  `migrate_known_cli_defaults` which silently rewrites previously generated
  unsafe Claude/Codex args at load time.
- `adapters.py` — three execution paths. CLI participants run via
  `asyncio.create_subprocess_exec` with `{prompt}`/`{cwd}` template
  substitution and prompt-on-stdin by default; OpenRouter uses `httpx`;
  Ollama hits a local `/api/chat`. Successful CLI output without a
  `RECOMMENDATION: yes|no|tradeoff` label is treated as failure
  (`_response_validation_error`).
- `orchestrator.py` — runs round 1, then optional deliberation rounds (helpers
  live in `deliberation.py`). Emits `progress_events` consumed both by the
  CLI's stream output and by the MCP tool's `metadata.progress_events` field.
- `policy.py` — `should_use_council` heuristic callers use to decide whether
  invoking the council is worth the cost for a given request.
- `update_check.py` — backs `llm-council doctor --check-update` and the
  startup version nag.
- `context.py` — builds the user-facing prompt and enforces `MAX_PROMPT_CHARS`.
- `setup_wizard.py` — writes `.llm-council.yaml`, `.mcp.json`, and the
  per-CLI instruction snippets in `.llm-council/instructions/`. Setup is
  guarded by `_preset_status` in `cli.py`; presets whose required CLIs/keys
  are missing are blocked unless `--allow-incomplete` is passed.
- `budget.py` / `estimate.py` / `model_catalog.py` — token/cost estimation
  and OpenRouter model catalog fetch (cached on disk).
- `transcript.py` — paired markdown + JSON transcripts under
  `.llm-council/runs/`. `latest_transcript` and `transcript_records` back the
  `last` and `transcripts` subcommands.
- `citations.py` — `VerifiedRef`, `parse_verified_tag`, `verify_ref`,
  `verify_evidence_citations`. Run by the orchestrator after every
  round; mutates each result's evidence list to stamp `verified` on
  `[VERIFIED:...]` entries and appends failed refs to
  `ParticipantResult.evidence_verification_failures`.
- `outcomes.py` — `OutcomeRecord` sidecar persistence under
  `.llm-council/outcomes/<run-id>.json`. Read via `read_outcome` /
  `iter_outcomes`; powers the per-peer reliability layer in
  `stats.aggregate_reliability`.
- `findings.py` — `Finding`, `FindingCluster`, `FindingMatrix`,
  `extract_findings`, `cluster_findings`, `build_matrix_from_results`,
  `matrix_to_dict`. Parses the optional `FINDINGS:` envelope block and
  clusters across peers by overlapping verified line ranges + severity.
  Computed post-deliberation in `orchestrator.execute_council`; never
  fed back to peers in-round.
- `eval/` package — `metrics.py` (`blocker_recall`,
  `false_blocker_rate`, `citation_accuracy`, `evidence_density`,
  `signal_to_noise_ratio`), `runner.py` (`load_fixture`, `run_suite`,
  `to_json`, `check_promotion_gate`), bundled minimal `fixtures/`.
  CLI: `llm-council eval run`.

## Invariants worth preserving

- **Read-only by default.** Council participants must not edit files. CLI
  adapters pass flags like `--permission-mode default` (Claude),
  `--sandbox read-only` (Codex), `--approval-mode plan` (Gemini). Don't
  remove these from `defaults.py` without an explicit reason.
- **`RECOMMENDATION:` label.** CLI output is rejected if it lacks the label;
  prompts in `context.py` ask for it. Adapter and prompt changes must keep
  these in sync. The label match is fence-aware: a `RECOMMENDATION:` line
  inside a fenced code block is treated as example syntax, not a real vote
  (`adapters._has_recommendation_label` and `deliberation.recommendation_label`
  both return "no usable label" for fenced-only matches).
  `deliberation.recommendation_line` follows the same fence-aware match
  but, when no out-of-fence label is found, returns the explicit placeholder
  string `(no RECOMMENDATION label emitted)` for round-2 prompt summaries
  rather than falling back to arbitrary prose.
- **Optional response envelope.** Peers may emit `EFFORT:`, `CONFIDENCE:`,
  `RISK:`, `BLOCKERS:`, `EVIDENCE:`, `TESTS_TO_RUN:`, `ASSUMPTIONS:` lines
  alongside the `RECOMMENDATION:` label. Parsed by
  `adapters._extract_response_envelope`, stored on `ParticipantResult`, and
  emitted in transcripts / MCP `structured_results`. All fields are optional
  in the current schema (v2). A peer that says `EFFORT: blocked` with no
  concrete entries in EITHER `BLOCKERS:` OR `ASSUMPTIONS:` is classified
  as abdication (`error_kind=abdicated`, `ok=False`, drops quorum) — no
  repair retry. Naming concrete missing artifacts in either list is
  treated as honest information, not abdication. Track presence via the
  new `envelope_field_present` bucket in `stats.aggregate` before any
  future flip from optional to required.
- **Config migration is silent.** `migrate_known_cli_defaults` rewrites old
  `OLD_CLAUDE_PLAN_ARGS` / `OLD_CODEX_APPROVAL_ARGS` and back-fills
  `peer-only` mode and `include_current` for built-in `other_cli_peers`
  modes. When changing baseline args in defaults, update the migration
  constants too.
- **Prompt-size guard.** `max_prompt_chars` is enforced both globally and
  per-participant before any subprocess launches; preserve this so oversized
  prompts fail fast rather than after a long hosted/CLI timeout.
- **Mode-aware timeouts.** `defaults.py:DEFAULT_CONFIG["modes"]` may carry
  an optional `timeout_multiplier: float`. Resolution:
  `effective = per_participant_timeout * mode_multiplier`. The
  per-participant `timeout` stays the source of truth for the base; the
  multiplier is layered on top so users who raised the base for a stubborn
  host CLI also benefit on consensus/deliberate runs. Defaults: consensus
  2.0x, deliberate 1.5x, diverse 1.5x; all other modes 1.0x (unchanged
  behavior). Helper: `adapters._resolve_effective_timeout`.
- **Terse-retry on timeout.** When a peer times out
  (`is_timeout_error(error)`), the adapter performs one terse-retry with
  `TERSE_RETRY_TIMEOUT_SECONDS` (60s, fixed) and `TERSE_RETRY_INSTRUCTION`
  appended to the prompt. Success sets `ok=True,
  recovered_after_timeout=True`. Failure falls through to normal timeout
  failure — no chained label-retry, no chained section-repair (three
  attempts past the cost ceiling). Mode-multiplier does NOT apply to the
  retry. Disable per-participant with `terse_retry_on_timeout: false`.
- **Section-coverage validator (default on).** When the user prompt
  contains one or more `PART N — TITLE (REQUIRED)` or
  `PART N — TITLE (REQUIRED BY ...)` headers
  (`llm_council/sections.REQUIRED_SECTION_HEADER_RE`), the validator
  scans each peer response for a matching marker (literal `PART N` token
  OR all salient title tokens within a 200-char window). Missing
  sections trigger one repair-retry with `SECTION_REPAIR_RETRY_INSTRUCTION`;
  if the retry also misses, the result is
  `ok=False, error_kind=incomplete_response`. PART 6 (`RECOMMENDATION`)
  is intentionally skipped — the label check covers it. Disable via
  `defaults.require_sections: false` or `--no-require-sections`.
- **Strict evidence-tag enforcement (optional, default off).** Each
  EVIDENCE bullet is parsed for a leading/trailing/inline
  `[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE]/[VERIFIED:path:start-end]`
  tag and stored as `list[{text, tag, ...}]` on `ParticipantResult`.
  When `defaults.strict_evidence: true` (or `--strict-evidence`),
  entries without a tag fail validation with
  `error_kind=untagged_evidence` and trigger the repair-retry path.
  Empty evidence list passes — the gate is FORMAT of entries that
  exist, not PRESENCE. Strict-evidence treats `[VERIFIED:...]` as
  tagged whether mechanical verification passed or failed — the tag is
  present; verification is a separate axis surfaced via
  `evidence_verification_failures`. Watch
  `evidence_tag_distribution["untagged"]` in stats before flipping the
  default. Tag parsing only applies to `evidence` — blockers/assumptions/
  tests_to_run stay `list[str]`.
- **`[VERIFIED:path:start-end]` is a fifth optional evidence tag.**
  Joins `[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE]`. The
  orchestrator runs `citations.verify_evidence_citations` after every
  round (`orchestrator.py:424,547`); failed refs land on
  `ParticipantResult.evidence_verification_failures` as
  `path:start-end` strings but the entry is NOT dropped — coverage >
  filtering. The prompt directive in `context.py`'s envelope block
  asks peers to surface low-confidence findings and explicitly states
  unverifiable cites are kept as-is, not dropped. Cache schema v3
  stays valid; the new field defaults to `[]` on rehydrate via
  `payload.get(...) or []`.
- **Finding matrix is post-deliberation only.**
  `findings.build_matrix_from_results` runs ONCE on the final round's
  results inside `orchestrator.execute_council`. The matrix is
  consumed by `synthesis.build_synthesis_prompt` (rendered as
  "CONSENSUS BLOCKERS" / "SINGLE-PEER CONCERNS") and surfaced in
  transcripts and MCP `structured_results` (`consensus_blockers` +
  `single_peer_concerns`, omitted when no peer emitted findings). It
  is NEVER fed back to peers during round-2 deliberation — MAD
  literature (arxiv 2402.18272) warns that in-round convergence
  forcing depresses signal-to-noise.
- **Per-mode model overrides.**
  `modes.<name>.model_overrides: {peer: model_id}` in
  `.llm-council.yaml`. Resolution order: base
  `participants.<peer>.model` → `--tier` swap → mode override
  (highest priority within a mode). Validated at config-load; honored
  in `config.select_participants`. Built-in modes ship NO overrides —
  operators add their own once eval-harness signal supports the
  affinity. This replaces the cut auto-routed-persona feature (PRISM
  evidence: persona prompting net-negative for knowledge/coding
  accuracy).
- **Experimental mode promotion gate.** A mode marked
  `experimental: true` in `DEFAULT_CONFIG["modes"]` stays experimental
  until the eval harness shows on a canonical fixture set: ≥5pp
  `blocker_recall` lift AND ≤15% SNR collapse vs the baseline mode.
  `eval/runner.py:check_promotion_gate` is the computational gate;
  flipping the flag in `defaults.py` is a manual operator decision
  pinned to scorecard evidence. `review-with-tools` is the first mode
  that ships under this discipline. CLI flags:
  `--compare-against <baseline.json>`, `--promotion-recall-lift`,
  `--promotion-snr-floor-ratio`. The `[EXPERIMENTAL]` marker is
  surfaced in `llm-council list` (`cli.py:845`) and visible via MCP
  `council_list_modes` (the raw `modes` dict carries the flag).
- **Outcome tracking is sidecar.**
  `.llm-council/outcomes/<run-id>.json` is persisted separately from
  transcripts so transcript JSON shape stays immutable. Reliability
  counters in `stats.aggregate_reliability` are mutually exclusive: a
  peer voting `yes`/`tradeoff` on a shipped+no-bug outcome →
  `useful_count`; voting `no` → `false_blocker_count`; no usable
  label → neither. `verified_citation_rate` is the only counter that
  does NOT require user outcome labels — it's mechanical from
  `evidence_verification_failures`.
- **Timeout-by-prompt-size telemetry.** `stats.aggregate` buckets
  timed-out runs into `timeout_by_prompt_size` (small / medium / large /
  xlarge, char cutoffs at 4K / 20K / 60K) and tracks `timeout_recoveries`
  for `recovered_after_timeout=True` successes. Lets the operator see
  whether bigger prompts disproportionately trip the timeout wall — the
  signal for raising `defaults.timeout` or a mode's `timeout_multiplier`
  rather than chunking. Bucket cutoffs are tuned to
  `MAX_PROMPT_CHARS=200_000`; revisit if the global cap changes.
- **`.mcp.json` stays local.** Setup adds it to `.gitignore`. It contains
  absolute paths and must not be committed.
- **Version bumps.** `__version__` in `llm_council/__init__.py` and the
  `version` in `pyproject.toml` and the README badge are kept in sync, with
  a matching `CHANGELOG.md` entry. Releases are tagged `vX.Y.Z`.

## Failure taxonomy

`adapters.classify_error(error)` maps any non-empty result.error to a stable
machine-readable kind (also surfaced as `error_kind` in transcripts and
`--json` stdout). Add new kinds explicitly here when introducing a new
failure path; do not let strings drift.

| `error_kind`         | When                                                                                |
|----------------------|-------------------------------------------------------------------------------------|
| `timeout`            | Participant exceeded its `timeout`. Prefixes: `Timeout:` / `TimeoutError:` (CLI subprocess) or `ReadTimeout:` / `ConnectTimeout:` / `WriteTimeout:` / `PoolTimeout:` / `TimeoutException:` (httpx, used by openai_compatible + ollama). The full set lives in `adapters._TIMEOUT_PREFIXES` and is shared by `is_timeout_error` (terse-retry gate) and `classify_error` (telemetry). |
| `context_overflow`   | Estimated tokens exceed `max_context_tokens`. Prefix: `ContextOverflowExcluded:`    |
| `prompt_too_large`   | Prompt skipped before launch (per-participant `max_prompt_chars`)                    |
| `invalid_response`   | CLI/HTTP succeeded but lacked `RECOMMENDATION:` label after one repair retry         |
| `downstream_error`   | httpx / hosted-API failures other than timeouts (HTTPStatusError, ConnectError, RemoteProtocolError, ReadError, WriteError, ProxyError). httpx timeout class names are classified as `timeout` instead. |
| `cli_nonzero_exit`   | CLI participant exited with a nonzero status and empty stderr. Prefix: `CliExitNonZero:` |
| `preflight_failed`   | Local participant's `base_url` was unreachable at run start. Prefix: `PreflightFailed:` |
| `abdicated`          | Peer emitted `RECOMMENDATION:` and `EFFORT: blocked` with no concrete missing artifact in EITHER `BLOCKERS:` or `ASSUMPTIONS:`. Terminal for the round — no repair retry, drops quorum so consensus doesn't form on a non-vote. The cache DOES persist the raw output; `_with_envelope` re-derives `ok=False` on every read so repeat runs still drop quorum without paying the peer again. Prefix: `AbdicatedResponse:` |
| `incomplete_response` | Response had the `RECOMMENDATION:` label but missed one or more `(REQUIRED)` sections from the prompt after one repair-retry. Prefix: `IncompleteResponse:` |
| `untagged_evidence`  | `defaults.strict_evidence: true` AND one or more EVIDENCE bullets lacked a `[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE]` tag after one repair-retry. Prefix: `UntaggedEvidence:` |
| `unknown`            | Non-empty error that did not match any known prefix — file a dogfood note            |

## Custom CLI participant: minimal template

When defining a one-off CLI participant (in `.llm-council.yaml` or a temp
config) the deep-merge from `defaults.py` only fills keys that exist on a
built-in baseline. For an entirely new family, you generally need:

```yaml
participants:
  my_cli:
    type: cli              # required: routes through the CLI adapter
    family: my_cli         # required when a participant doesn't share a baseline
    origin: us             # `us` | `china` | `unknown` — origin filtering
    command: my-cli        # binary on PATH (or absolute path)
    args: ["--flag"]       # optional; uses {prompt}/{cwd} template substitution
    model: my-model        # optional model identifier
    timeout: 240           # seconds before the participant is killed
    max_prompt_chars: 120000  # per-peer prompt cap (chunking targets this)
    read_only: true        # advisory marker; the read-only invariant is
                           # actually enforced by the per-CLI args baked
                           # into defaults.py (e.g. --permission-mode default
                           # for Claude, --sandbox read-only for Codex), so
                           # custom CLIs need to pass equivalent flags via
                           # `args` for the invariant to hold
    stdin_prompt: true     # whether the prompt is delivered via stdin (default)
                           # vs. {prompt} arg substitution
    env_passthrough:       # secret-named env vars to forward to the child
      - OPENAI_API_KEY     # (e.g. KEY/AUTH/TOKEN). Non-secret env vars
                           # already inherit by default.
    env_strict: false      # when true, the child sees ONLY the names in
                           # _SAFE_ENV_NAMES (PATH/HOME/LANG/…) plus
                           # `env_passthrough`. Use for CLIs that auto-
                           # detect provider config from env (e.g. qwen-
                           # code, a gemini-cli fork that prefers
                           # GEMINI_* over OPENAI_* if both are set).
```

Forget `family` and the participant works but config validation may flag it
as orphaned. Forget `stdin_prompt: true` and an unsubstituted-`{prompt}`
arg gets shipped as literal text. The read-only invariant lives in the
host CLI's own permission flags (passed via `args`), not in the
`read_only:` key — that key is documentation-only today.

## Continuation chain depth

`continuation_id` (CLI `--continue`) prepends a summary of the prior
transcript. Each link summarizes only its immediate parent (not the full
history), so depth growth is linear, not exponential. Still, the default
`max_continuation_depth` of 5 caps how many parents can chain before the
run is refused — set `defaults.max_continuation_depth: <N>` in
`.llm-council.yaml` to override.

## Run-level budget caps

`--max-cost-usd` and `--max-tokens` (CLI) / `max_cost_usd`,
`max_tokens` (MCP `council_run`) gate the run on the **pre-flight
estimate** before any subprocess or HTTP call is made. The estimate sums
known `cost_usd` per participant from the OpenRouter catalog; free/local
peers count as $0 and unknown-cost peers (catalog miss) cannot be
enforced — those raise no error but are visible in the estimate. Use
`llm-council estimate ...` for a per-peer breakdown when a cap fails.

## Per-peer model selection: tiers and per-mode overrides

`defaults.tiers.<name>: {peer: model_id}` defines a named swap applied
globally via `--tier <name>` (CLI) before participant selection. For a
mode-scoped pin, `modes.<name>.model_overrides: {peer: model_id}` in
`.llm-council.yaml` overrides a peer's model only when that mode is
active. Resolution chain: base `participants.<peer>.model` -> `--tier`
swap (when set) -> `modes.<name>.model_overrides` (highest priority).
Overrides naming a peer absent from the resolved roster are silent
no-ops, and only the `model` field is touched (args, timeout, type,
family, origin are untouched). Built-in modes ship without
`model_overrides` on purpose — do NOT add vendor-affinity defaults until
the eval harness shows real lift on the relevant fixture set.
