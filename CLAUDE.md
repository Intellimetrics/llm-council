# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`llm-council` is a Python 3.11+ MCP server and CLI that lets one coding agent
ask a "council" of other LLMs (Claude Code, Codex CLI, Antigravity, OpenRouter
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
llm-council doctor [--probe-openrouter] [--probe-ollama] [--probe-native] [--repair-transcript-permissions] [--check-update]

# Direct council run (skips MCP)
llm-council run --current codex --mode review --diff "Review this change"

# Run as MCP server over stdio (what `.mcp.json` invokes)
llm-council mcp-server

# Aggregate stats over persisted transcripts (incl. per-peer quota telemetry)
llm-council stats
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
   `council_stats`, `council_query_transcripts`)

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
  `peer-only`, `plan`, `review`, `private-local`, `deliberate`, `consensus`,
  `fable`, `review-with-tools`) live here. Near-zero-use modes were removed
  in v0.19.0 after a 518-transcript usage audit; any of them can be
  recreated per-project in `.llm-council.yaml` as plain mode entries.
- `config.py` — config discovery (`find_config` walks up from cwd looking for
  `.llm-council.yaml` etc.), validation, the `other_cli_peers` strategy used by
  most modes, `detect_current_agent` (parent-process walk on `/proc`), and
  `migrate_known_cli_defaults` which silently rewrites previously generated
  unsafe Claude/Codex args at load time.
- `adapters.py` — three execution paths. CLI participants run via
  `asyncio.create_subprocess_exec` with `{prompt}`/`{cwd}` template
  substitution and prompt-on-stdin by default (antigravity is the exception:
  agy 1.1.1+ ignores stdin, so its prompt rides in argv via `{prompt}`);
  OpenRouter uses `httpx`;
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
- `findings.py` — `Finding`, `FindingCluster`, `FindingMatrix`,
  `extract_findings`, `cluster_findings`, `build_matrix_from_results`,
  `matrix_to_dict`. Parses the optional `FINDINGS:` envelope block and
  clusters across peers by overlapping verified line ranges + severity.
  Computed post-deliberation in `orchestrator.execute_council`; never
  fed back to peers in-round.
- `query.py` (v0.9.0) — `SimilarMatch`, `search_similar()`. Jaccard
  token-overlap search over `.llm-council/runs/*.json` for prior
  council questions. Reuses `convergence.tokenize`,
  `deliberation.recommendation_label`, and
  `stats.load_transcript_files`. Surface is MCP only
  (`council_query_transcripts`); no CLI subcommand. No new
  dependencies — sentence-transformers deferred until Jaccard proves
  insufficient.

## Invariants worth preserving

- **Built-in mode rosters are local-CLI only.** No built-in mode seats a
  hosted (openrouter/billed) participant via `participants` or `add` —
  the shipped default council is claude / codex / antigravity everywhere.
  Hosted baselines stay DEFINED in `defaults.py` for explicit opt-in
  (`include`, per-project `modes.<name>.add`, setup presets), but never
  run unless the operator asks. Guarded by
  `tests/test_llm_council.py::test_no_built_in_mode_seats_hosted_peers`.
  Side effect worth keeping: tri-cli setups retain plan/review/deliberate
  (previously pruned because their rosters referenced hosted peers).
- **Read-only by default (hard for all three native CLI peers).**
  Council participants must not edit files. Each native CLI gets a HARD
  guarantee from flags that physically disable the write tool
  (`--permission-mode manual` Claude, `--sandbox read-only` Codex,
  `--mode plan` Antigravity) — a misbehaving
  model or a prompt-injected diff cannot write. Don't remove these from
  `defaults.py` without an explicit reason. Antigravity's `--mode plan` flag
  exists as of agy 1.1.0 (verified live on 1.1.4: an explicitly ordered write
  produces no file, and agy's shell-command fallback is denied in headless
  print mode); before that its read-only-ness was only SOFT (prompt-enforced).
  The prompt directive in `context.build_prompt` ("Do not edit files. Do not
  run write operations.") remains as defense in depth, and we still omit
  `--dangerously-skip-permissions` so residual tool attempts are denied, not
  auto-approved. The opt-in canary `tests/test_live_agy_readonly.py` (gated by
  `LLM_COUNCIL_LIVE_AGY_TEST=1`) guards this across upstream agy releases.
  Note agy also ignores stdin since 1.1.1, so its prompt is delivered via
  argv (`{prompt}`), not stdin.
- **Antigravity per-run isolation (`--new-project`) + matched print
  timeout.** agy keeps a global conversation store under
  `~/.gemini/antigravity-cli/brain/`, and a plain `agy -p` run can recall
  PRIOR runs' content from it (verified live on 1.1.4) — breaking
  fresh-eyes and, under prompt injection, risking cross-project leakage of
  earlier council prompts. The shipped args include `--new-project` so
  every invocation starts with no in-context carryover, and the
  per-family directive (`context.ANTIGRAVITY_READ_TOOL_HINT`) tells the
  peer not to consult prior conversations. Residual risk: agy's native
  read tool can still open the brain dir by absolute path; treat
  isolation as strong, not absolute. Separately,
  `adapters._build_cli_command` injects `--print-timeout
  <effective+30>s` for the antigravity family (agy's internal print cap
  defaults to 5m and would silently truncate longer council timeouts;
  the +30s slack keeps llm-council's own timeout as the one that fires
  and owns the error). Omitted when the operator pins --print-timeout
  in args, and when no effective timeout is passed (doctor probes).
- **`RECOMMENDATION:` label.** CLI output is rejected if it lacks the label;
  prompts in `context.py` ask for it. Adapter and prompt changes must keep
  these in sync. The label match is fence-aware: a `RECOMMENDATION:` line
  inside a fenced code block is treated as example syntax, not a real vote
  (`adapters._has_recommendation_label` and `deliberation.recommendation_label`
  both return "no usable label" for fenced-only matches).
  `deliberation.recommendation_line` follows the same fence-aware match
  but, when no out-of-fence label is found, returns the explicit placeholder
  string `(no RECOMMENDATION label emitted)` for round-2 prompt summaries
  rather than falling back to arbitrary prose. The mandatory label costs one
  repair-retry round-trip (an extra CLI invocation) when a peer's first
  response omits it — even on trivial / non-vote prompts. Per-peer opt-outs:
  `retry_on_missing_label: false` skips the label-repair retry;
  `require_recommendation: false` bypasses the label gate entirely (for
  genuinely non-vote uses). The label is the load-bearing universal vote
  contract — quorum, consensus, and the finding matrix all depend on it — so
  reserve these opt-outs for non-vote peers/prompts, not as a blanket
  cost-trim on real council runs.
- **Optional response envelope.** Peers may emit `EFFORT:`, `CONFIDENCE:`,
  `RISK:`, `BLOCKERS:`, `EVIDENCE:`, `TESTS_TO_RUN:`, `ASSUMPTIONS:`,
  `CONTINUE_DEBATE:` lines alongside the `RECOMMENDATION:` label. Parsed by
  `adapters._extract_response_envelope`, stored on `ParticipantResult`, and
  emitted in transcripts / MCP `structured_results`. All fields are optional
  in the current schema (v2). A peer that says `EFFORT: blocked` with no
  concrete entries in EITHER `BLOCKERS:` OR `ASSUMPTIONS:` is classified
  as abdication (`error_kind=abdicated`, `ok=False`, drops quorum) — no
  repair retry. Naming concrete missing artifacts in either list is
  treated as honest information, not abdication. Track presence via the
  new `envelope_field_present` bucket in `stats.aggregate` before any
  future flip from optional to required. `CONTINUE_DEBATE: yes|no`
  (v0.8.1) is a per-peer vote on whether round-2 deliberation is worth
  running; when ALL label-producing peers emit `no` in round 1
  (denominator excludes abdicated / `invalid_response` / unlabeled),
  the orchestrator skips round-2 and stamps
  `deliberation_status: skipped_continue_debate_unanimous` plus a
  `deliberation_skipped` progress event with `no_votes` + `denominator`
  counts. Unanimity (not 66%) is conservative-until-measured — revisit
  once a transcript corpus exists to audit gaming risk (see the
  `CONTINUE_DEBATE` unanimity skip in `orchestrator.execute_council`).
- **Config migration is silent.** `migrate_known_cli_defaults` rewrites old
  `OLD_CLAUDE_PLAN_ARGS` / `OLD_CLAUDE_DEFAULT_ARGS` /
  `OLD_CODEX_APPROVAL_ARGS` and back-fills `peer-only` mode and
  `include_current` for built-in `other_cli_peers` modes. When changing
  baseline args in defaults, add the outgoing baseline as a new OLD_*
  constant so existing generated configs silently upgrade.
- **Prompt-size guard.** `max_prompt_chars` is enforced both globally and
  per-participant before any subprocess launches; preserve this so oversized
  prompts fail fast rather than after a long hosted/CLI timeout. Both
  `--diff` payloads AND `context_files` (v0.8.1+) route through
  `llm_council/diff_chunking.py`'s hash-aware chunker before assembly so
  large multi-file context drops don't trip the cap. New entry point
  `diff_chunking.chunk_context_files()` reuses the existing scoring
  helpers (filename mentions, extension affinity, smaller-first
  tiebreak). A single file larger than `max_prompt_chars - framing` is
  dropped entirely with a `context_files_chunked` progress event listing
  `oversize_files` — operator-visible rather than silently truncated.
  The round-2 deliberation body budget is DERIVED, not fixed:
  `deliberation.deliberation_body_budget(effective_cap, largest_suffix)`
  returns `min(MAX_DELIBERATION_PROMPT_CHARS, cap) - largest per-peer
  directive suffix`, and both the runtime builder
  (`build_deliberation_prompt(max_chars=...)`, cap threaded via
  `execute_council(deliberation_prompt_cap=...)` — MCP passes
  `min(default_max, mcp_max)`, CLI passes `defaults.max_prompt_chars`)
  and the preflight bound
  (`estimate.deliberation_prompt_char_bounds(effective_prompt_cap=...)`)
  use the same derivation. This keeps `final round-2 prompt = body +
  directive suffix <= effective cap` true by construction for any peer
  family, directive length, or configured cap — never re-introduce a
  fixed body constant compared against an independently defined cap
  (pre-fix, agy's 268-char hint made the worst-case bound 80,268 vs the
  80,000 MCP cap, structurally refusing every deliberate MCP run that
  rostered it; regression tests in `tests/test_deliberation_budget.py`).
- **Mode-aware timeouts.** `defaults.py:DEFAULT_CONFIG["modes"]` may carry
  an optional `timeout_multiplier: float`. Resolution:
  `effective = per_participant_timeout * mode_multiplier`. The
  per-participant `timeout` stays the source of truth for the base; the
  multiplier is layered on top so users who raised the base for a stubborn
  host CLI also benefit on consensus/deliberate runs. Defaults: consensus
  2.0x, deliberate 1.5x; all other modes 1.0x (unchanged
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
  OR all salient title tokens co-occurring within a window anchored on the
  first title token — `-100`/`+200` chars, i.e. ~300 chars total). When two
  REQUIRED sections share a salient title token (e.g. `SECURITY ANALYSIS`
  and `SECURITY HARDENING`), the loose window is ambiguous, so those
  collision-prone sections instead require the full title as a
  near-contiguous phrase (`_section_present(..., ambiguous_anchor=True)`)
  to avoid a sibling's header falsely satisfying them. Missing
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
  round (in `orchestrator.execute_council`); failed refs land on
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
- **Independent-review isolation is an advisory-only flag.**
  `--independent-review` (CLI) / `independent_review` (MCP boolean),
  resolved flag > per-mode `modes.<name>.independent_review` >
  `defaults.independent_review` (validated as booleans in
  `config.validate_config`). Defaults OFF, composes with any mode, and
  changes nothing when unused. For continuation runs only: when ON AND a
  `continuation_id` produced a `prior_context`, that `prior_context`
  is set to `None` so the "independent" round is NOT anchored to the
  prior council's per-peer labels/rationales. Because suppression is a
  pre-run decision (before `execute_council`), it is recorded as
  `metadata["prior_context_suppressed_for_independence"] = True`
  (only when suppression actually occurred) rather than a mid-run
  progress event; the CLI also prints a one-line stderr note and MCP
  mirrors the flag top-level in the response. OFF, or no prior_context
  to suppress, sets nothing and changes nothing — `prior_context`
  flows exactly as before.
- **Tool-call voting is opt-in even within `review-with-tools`.**
  `tool_call_voting: false` by default on the `review-with-tools` mode
  (`DEFAULT_CONFIG["modes"]["review-with-tools"]` in `defaults.py`).
  When flipped to `true`, the orchestrator
  appends a `record_recommendation(verdict, blockers, evidence)` tool
  schema to the per-peer directive and runs a unified
  `_extract_tool_call_recommendation` parser; regex
  `RECOMMENDATION:` parsing remains the fallback. `tool_call_status`
  on `ParticipantResult` (`absent` / `ok` / `malformed` / `None`)
  makes parser behavior operator-visible — serialized in transcripts
  and MCP `structured_results` (Phase 5b fix). Promotion to default
  is a manual operator decision pending real-world signal; no
  family-specific extraction code is added until concrete CLI
  tool-call payloads exist to validate against.
- **Per-mode model overrides.**
  `modes.<name>.model_overrides: {peer: model_id}` in
  `.llm-council.yaml`. Resolution order: base
  `participants.<peer>.model` → `--tier` swap → mode override
  (highest priority within a mode). Validated at config-load; honored
  in `config.select_participants`. Built-in modes ship NO overrides —
  operators add their own with real-world evidence of a per-mode
  affinity. This replaces the cut auto-routed-persona feature (PRISM
  evidence: persona prompting net-negative for knowledge/coding
  accuracy).
- **Experimental mode marker.** A mode marked `experimental: true` in
  `DEFAULT_CONFIG["modes"]` may change or be cut; promoting it to
  non-experimental (flipping the flag in `defaults.py`) is a manual
  operator decision based on real-world usage. `review-with-tools` is
  the one mode currently shipping under this discipline. The
  `[EXPERIMENTAL]` marker is surfaced in `llm-council list`
  (`cli.cmd_list`) and visible via MCP `council_list_modes` (the raw
  `modes` dict carries the flag).
- **Timeout-by-prompt-size telemetry.** `stats.aggregate` buckets
  timed-out runs into `timeout_by_prompt_size` (small / medium / large /
  xlarge, char cutoffs at 4K / 20K / 60K) and tracks `timeout_recoveries`
  for `recovered_after_timeout=True` successes. Lets the operator see
  whether bigger prompts disproportionately trip the timeout wall — the
  signal for raising `defaults.timeout` or a mode's `timeout_multiplier`
  rather than chunking. Bucket cutoffs are tuned to
  `MAX_PROMPT_CHARS=200_000`; revisit if the global cap changes.
- **Quota-fallback chain semantics (multi-step walking).** Each CLI
  participant carries an optional `fallback_chain: list[str]` — ordered
  list of model IDs to step down to on a `quota_exhausted` failure.
  Resolution: if `cfg.model` is in the chain, walk the entries AFTER
  it; if it's None or absent from the chain, walk from `chain[0]`.
  The walker (`_quota_fallback_walk`) is capped at
  `QUOTA_FALLBACK_MAX_STEPS` (currently 3) to bound wall-clock cost
  when multiple chain entries are all throttled. Stops at the first
  success, the first non-quota failure (continuing would spam more
  models with a problem unrelated to quota), or chain exhaustion.
  v0.12.1+ multi-step replaces the v0.11.6 single-step rule. The
  Claude family is special-cased: `_build_cli_command` auto-injects
  `--fallback-model <chain[0]>` so the CLI itself handles overload
  internally; the llm-council-level walker is SKIPPED for Claude to
  avoid double-paying the peer. Antigravity ships with
  `fallback_chain: []` because portable `agy` model ids do not exist and an
  unrecognized `--model` value silently falls back to session state (the peer
  therefore drops with a `quota_throttled_peers` signal). Default chains shipped
  in `DEFAULT_CONFIG` are capability-graceful (same-tier minor version
  back → next-tier-smaller → smallest); wrong model ids just make
  that step fail and the walker proceeds (or the peer drops if chain
  exhausts). Successful recoveries stamp `recovered_after_quota=True`
  + `model_fallback_used=<id>` on the result, emit a
  `peer_quota_recovered` progress event, and surface in top-level
  `quota_recoveries`. On failure, `model_fallback_used` is stamped
  with the LAST attempted model so the transcript shows where the
  walker stopped.
- **Per-CLI token usage is not observable in the default text-mode
  invocation.** CLI participants (claude, codex, antigravity)
  authenticate as the user and burn the user's own account quota. In the
  default invocation we run them in TEXT mode, where there is no metering
  hook — no way to read "X tokens consumed", "Y quota remaining", or "Z
  requests this hour" from the outside. The only observable usage signal
  for text-mode CLI peers is *failures we caught*: `quota_incidents` (how
  many times the peer hit a quota wall) and `quota_recoveries` (how many
  of those the fallback rescued), surfaced per-peer via `llm-council
  stats`. Hosted OpenRouter peers always report real
  token-level usage (the `usage` field on API responses populates
  `prompt_tokens` / `completion_tokens` / `cost_usd` on
  `ParticipantResult`).
  **Opt-in exception (M7): `usage_from_json: true`.** Per-peer config
  that switches the invocation to the CLI's own JSON output mode and
  parses REAL usage/cost into the same `ParticipantResult` fields
  (`prompt_tokens` / `completion_tokens` / `total_tokens` / `cost_usd`,
  plus the CLI-reported `model` id, preferred over the requested one).
  Implemented for **claude** (`-p --output-format json`, single JSON
  object: `result` text + `usage` + `total_cost_usd` + `modelUsage`) and
  **codex** (`exec --json`, JSONL stream: last `agent_message` text +
  `turn.completed` usage; billable `prompt_tokens = max(0, input_tokens -
  cached_input_tokens)`; codex reports no cost so `cost_usd` stays None;
  `item.completed` events supply answer text ONLY when
  `item.type == "agent_message"` — codex 0.143/0.144 added new canonical
  item types that could otherwise overwrite the real answer). Codex has
  NO native turn/tool-call/wall-time cap (verified against the full
  config schema); llm-council's own timeouts are the only backstop, and
  the operator-level lever for shorter runs is
  `-c model_reasoning_effort=low` in the codex peer's args (~3x
  wall-time cut measured; valid values are model-dependent, a bad value
  cleanly fails the turn).
  Default OFF for every built-in peer → byte-identical text-mode command
  and parsing. The JSON flag is PURELY ADDITIVE: it never removes the
  read-only flags (`--permission-mode manual` / `--sandbox read-only`),
  so peers still cannot write. Flag + parser ship together per family —
  `usage_from_json` is a NO-OP for any family without a JSON parser
  (`_USAGE_JSON_FAMILIES = {claude, codex}`; other families add no flag).
  Parsing is fail-soft: `_parse_cli_usage_json` returns None on malformed
  / changed JSON shapes and the adapter falls back to treating raw stdout
  as text (the `RECOMMENDATION:` label check still runs, token fields stay
  None) — a CLI version bump can never crash or silently drop the peer.
  The JSON shapes are version-sensitive; the parser probes key variants
  with `.get()`. Don't promise observability we don't have: even with the
  opt-in, the reported model is what the CLI says, not a server-side
  confirmation, and text mode remains unobservable.
- **Size-scaled timeouts.** `_resolve_effective_timeout` now adds a
  prompt-size bonus on top of the per-participant base: 5s per KB above
  a 4KB threshold by default, capped at +600s. The mode multiplier
  (consensus 2.0x, deliberate 1.5x) layers on top of `(base + bonus)`,
  so a 26KB prompt in consensus mode gets `(240 + 110) * 2 = 700s`
  instead of the prior unconditional 480s. Per-peer override via
  `timeout_per_kb_chars: 0` disables scaling entirely (used in tests
  that need to pin the legacy behavior). Triggered by the v0.11.7
  dogfood showing the same 240s wall hit on prompts of both 4KB and
  26KB — context length scaling is real and prior fixed timeouts
  understated it.
- **Proportional terse-retry budget.** `_terse_retry_budget(original)`
  returns `min(max(original * 0.4, 30), 120)` — floor 30s, ceiling
  120s, 40% of the original timeout in between. Replaces the legacy
  fixed `TERSE_RETRY_TIMEOUT_SECONDS = 60` constant which was
  structurally unlikely to succeed when the original timeout was
  240s+ (the retry nearly always re-timed-out, providing no recovery
  signal). The retry runs with `timeout_per_kb_chars: 0` to prevent
  double-scaling the size bonus already baked into the proportional
  budget.
- **Idle-read timeout (opt-in).** When `cfg.idle_timeout: float`
  is set, `_run_cli_once` switches from `proc.communicate()` to a
  streamed read loop with a per-stream idle deadline. The peer is
  killed when no stdout/stderr data arrives for N seconds, in
  addition to the wall-clock cap. Default OFF (None) for all
  built-in peers since most CLIs (claude `-p`, codex `exec`, agy
  `--print`) buffer everything to the end rather than streaming.
  Operators with a known-streaming CLI can opt in per peer.
- **Missing-key peer pre-drop.** `_drop_missing_key_participants`
  scans hosted peers (openrouter / openai_compatible) at the top
  of `execute_council` and removes any whose `api_key_env` env var
  is unset. Dropped peers emit a `peer_missing_api_key` progress
  event AND land in `metadata.missing_key_peers` for top-level
  surfacing, BUT do NOT count toward the quorum denominator. A
  run that ends with one hosted peer missing its key looks
  identical (for `degraded` / `min_quorum` purposes) to a run that
  never listed that peer at all — a missing key is an operator
  configuration gap, not a council failure. Asymmetry between the
  hosted types: both `openrouter` and `openai_compatible` peers
  without explicit `api_key_env` default to `OPENROUTER_API_KEY`.
  Configure an explicit environment variable for a different hosted
  provider; use the dedicated local participant types for no-auth
  local inference.
- **Independence warning is advisory-only (H2).** The optional
  `defaults.min_distinct_vendors` (global) and per-mode
  `modes.<name>.require_distinct_vendors` (override; resolution: mode
  override first, then global default, else feature OFF) set a floor on
  how many DISTINCT vendor families the final-round labeled votes must
  span. When the resolved threshold is set AND the count of distinct
  families among labeled final-round votes is below it, the orchestrator
  (`execute_council`, immediately after the degraded block) sets a NEW
  `metadata["independence_warning"]` dict (`distinct_vendors`, `required`,
  `families`, `labeled_quorum`) and emits a `single_vendor_quorum`
  progress event. It NEVER drops a peer and must NEVER overload
  `metadata["degraded"]` / `min_quorum` / `labeled_quorum` — `degraded`
  means below-quorum-COUNT only; independence is an orthogonal correlated-
  agreement signal. Both keys validate as positive integers at config-load
  (`config._validate_positive_int`); built-in modes ship WITHOUT either key
  (feature OFF by default — when the threshold is unset OR met, the
  `independence_warning` key is omitted entirely and no event fires).
  Surfaced top-level in MCP `council_run` `structured_results`
  (omit-when-absent, like `quota_throttled_peers`; also left in
  `metadata` like `degraded`), persisted in the transcript JSON via the
  whole-`metadata` serialization, and rendered as a one-line ⚠️ note in
  the markdown transcript near the quorum/degraded summary.
- **Fable peer: reduce false-positive refusals, detect the silent
  Opus fallback (v0.16.0).** Claude Fable 5 runs request-side safety
  classifiers (research-bio + most cybersecurity content) that
  false-positive on benign security-adjacent review; on the Claude Code
  surface a refused request is silently re-served by Opus 4.8, and the
  default text-mode CLI invocation cannot see the swap — so an Opus answer
  would be recorded as a "Fable" opinion. The `claude_fable` peer + `fable`
  mode address this WITHOUT trying to disable the surface's built-in
  fallback (the CLI exposes no such control):
  - **Reduce** — the `fable` mode sets `safe_context: true`, which makes
    `context.build_prompt` inject a defensive-review framing block (resolved
    from the mode config at the CLI and MCP call sites via
    `config["modes"][mode]["safe_context"]` for `build_prompt`; inside
    `execute_council` the flag is re-derived from the same mode config —
    like `timeout_multiplier` — so a caller can't desync the per-round
    framings). The block states ONLY facts the tool can vouch
    for: the review is operator-invoked, read-only, and analysis-only. It
    deliberately does NOT claim the reviewed content is the operator's own
    work or benign — the content may be an untrusted third-party patch, and
    a true-positive refusal on genuinely malicious material must stay
    possible. Instead it redirects suspicion: flagging malicious code as a
    finding IS the requested output. It also tells peers they need not
    expose raw chain-of-thought (heading off the `reasoning_extraction`
    refusal category — the structured format is all the council consumes).
    Factual context, **NOT** an instruction to bypass safety; harmless for
    non-Fable peers; absent unless a mode opts in.
  - **Detect** — `claude_fable` pins `model: claude-fable-5` with
    `usage_from_json: true` (so `_parse_claude_usage_json` reports the model
    that ACTUALLY served the turn — selected as the `modelUsage` key with the
    most `outputTokens`, i.e. the answer's author, NOT the first key: a
    fallback turn can log both models and helper models can appear) and
    `require_pinned_model: true`. When the served model fails the lenient
    variant-tolerant `_model_pin_satisfied` check against the pinned id,
    `_run_cli_once` drops the peer (`ok=False`,
    `error_kind=model_substituted`, `ModelSubstituted:` prefix) so a
    substituted Opus answer never counts as a Fable vote; `result.model`
    still reports the REAL served model. The guard's signal is preserved
    end-to-end: the repair-retry merges (`_merge_cli_retry`,
    `_merge_section_retry`, terse-timeout retry) propagate a substituted
    retry instead of falling through to the original error, and combine
    both attempts' outputs so the original pinned-model response stays
    auditable; detection runs live per round via
    `_detect_and_emit_substitutions` (round 1 and round-2
    deliberation — dedup on peer+served_by, each
    `peer_model_substituted` event stamped with the round the swap actually
    happened in) plus a post-synthesis scan of the chair payload (the chair
    turn never enters `results`; a substituted chair memo gets
    `synthesis_payload["model_substituted"] = True` and a
    `{... synthesis: true}` entry so an Opus-authored memo is never consumed
    as the pinned chair's). Entries land in
    `metadata['model_substituted_peers']`, mirroring `quota_throttled_peers`
    — including the MCP top-level `council_run` payload key + output-schema
    declaration (schema v7); and substituted results are EXCLUDED from
    `build_matrix_from_results` input so an Opus-served FINDINGS block can't
    enter consensus blockers (deliberation + synthesis already skip not-ok
    results). `modes.<name>.safe_context`, `require_pinned_model`, and
    `usage_from_json` validate as booleans at config load (quoted "false"
    would otherwise silently enable them), and `estimate_council` builds the
    prompt with the mode's `safe_context` so estimates keep prompt-size
    parity with the real run.
  - **Known residual risk (documented, not mechanically closable):** the
    served-model attribution picks the `modelUsage` key with the most
    cumulative `outputTokens` across the whole agentic turn. A MID-TURN
    refusal fallback — Fable emits a long tool-use loop, then the final
    answer is served by Opus with fewer total output tokens — can pass the
    pin check, and there is nothing in the CLI's JSON that distinguishes
    the answer's author from a helper model. Treat `model_substituted`
    detection as high-recall for whole-turn swaps, not a proof of
    authorship; text mode remains fully unobservable.
  - `fallback_chain` is intentionally EMPTY on `claude_fable` so
    `_build_cli_command` does not inject `--fallback-model` — an overload swap
    would be a SECOND silent-substitution path. Do not add entries without
    re-reading this risk. More generally, `require_pinned_model` SUPPRESSES
    `--fallback-model` injection for any claude-family peer (the flag's
    whole purpose is serving the answer from a different model — the swap
    the guard rejects), and `config.config_warnings` flags the contradictory
    require_pinned_model + fallback_chain combination as inert. The whole
    feature is opt-in/default-OFF: no built-in mode except `fable` selects
    the peer, and a peer without `require_pinned_model` (or without JSON
    usage) never trips the guard.
- **A council must never start another council.** Two independent layers,
  both required. (1) Guard: `adapters.clean_subprocess_env` unconditionally
  sets `LLM_COUNCIL_NESTED=1` in every CLI child env (sieve AND strict
  modes; it inherits down the whole process tree), and
  `orchestrator.execute_council` refuses to run (`NestedCouncilRefused:`
  ValueError) when the marker is present in its own environment. (2)
  Starvation: the codex baseline args include `-c mcp_servers={}` so a
  council-spawned codex boots NO MCP servers — the observed real-world
  recursion path is an operator's global `~/.codex/config.toml` registering
  llm-council itself as an MCP server, which made every codex council peer
  start a nested llm-council server (plus headless-browser servers) per
  run. Claude peers already isolate via `--strict-mcp-config` with no
  `--mcp-config`. The outgoing codex baseline is preserved as
  `config.OLD_CODEX_EPHEMERAL_ARGS` so previously generated configs
  silently upgrade at load. Without both layers a prompt-injected peer
  calling `council_run` recurses exponentially (each level spawns N peers).
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
| `quota_exhausted`    | Peer hit a known quota / rate-limit signal (`RESOURCE_EXHAUSTED`, `quota_exceeded`, `Individual quota reached`, `rate_limit_exceeded`, `insufficient_quota`, `insufficient credits`, `usage limit`, `5-hour limit`, HTTP 429 with quota-adjacent text). Detected by `adapters.is_quota_exhausted_error` over the raw error string — no prefix synthesized. Surfaced top-level in transcripts and MCP `structured_results` as `quota_throttled_peers: [{peer, family, model, message}]`; orchestrator emits a `peer_quota_throttled` progress event per peer (deduped across rounds). A peer with a non-empty `cfg.fallback_chain` walks up to three next-in-chain models before landing in quorum. On success the peer recovers (`recovered_after_quota=True`, `model_fallback_used=<id>`, appears in `quota_recoveries` not `quota_throttled_peers`). On failure the peer drops with this error kind. Skipped for the Claude family because Claude's own `--fallback-model` CLI flag handles overload natively (auto-injected by `_build_cli_command` when chain non-empty). |
| `model_substituted`  | A CLI peer with `require_pinned_model: true` was served by a model other than its pinned `model` — e.g. Claude Fable 5 refused and the Claude Code surface silently fell back to Opus 4.8. Prefix: `ModelSubstituted:`. Only observable when `usage_from_json: true` surfaces the served model id (the `modelUsage` key with the most `outputTokens` — the answer's author); the served model must fail the lenient variant-tolerant `adapters._model_pin_satisfied` check. Terminal for the peer (ok=False, drops quorum) so a substituted model's answer is never recorded as the requested model's vote; `result.model` still reports the REAL served model, and substituted outputs are excluded from the finding matrix. Detected live per round (round 1, round-2 deliberation) plus the synthesis-chair turn, and preserved through repair-retry merges (with combined original+retry output). Surfaced as `metadata['model_substituted_peers']: [{peer, requested, served_by, synthesis?}]` (omitted when empty), lifted top-level in the MCP `council_run` payload + schema, with a per-round `peer_model_substituted` progress event. Opt-in — a peer without `require_pinned_model` or without JSON usage never trips it. Known limit: attribution is by max cumulative `outputTokens`, so a mid-turn refusal fallback inside a long agentic turn can evade it. |
| `pinned_model_unverified` | A peer required pinned-model verification but the CLI output did not report a served model. Prefix: `PinnedModelUnverified:`. The answer is excluded from quorum rather than being attributed without evidence. |
| `client_ineligible` | A native client is installed but durably ineligible for the configured account/tier — e.g. the retired standalone Gemini CLI's `IneligibleTierError` with `UNSUPPORTED_CLIENT` (Google ended individual-tier service 2026-06-18; the built-in `gemini` peer was removed in v0.20.0). Doctor's opt-in native probe surfaces the compatible fallback. |
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
                           # into defaults.py (e.g. --permission-mode manual
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

**Ambient-env bleed (documented behavior + remedy).** Default CLI peers
use "sieve" env mode: non-secret-named host env vars pass straight through
to the child. So an ambient host export like `GEMINI_MODEL`,
`ANTHROPIC_MODEL`, `OPENAI_MODEL`, or any `*_BASE_URL` can silently steer a
native CLI peer's model/endpoint with no llm-council visibility. The remedy
is per-peer `env_strict: true`, which restricts the child to
`_SAFE_ENV_NAMES` (PATH/HOME/LANG/…) plus the peer's `env_passthrough` list
— so ambient routing vars are dropped unless explicitly forwarded. Use it
when you want deterministic per-peer model/endpoint selection independent of
the operator's shell environment.

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
`model_overrides` on purpose — do NOT add vendor-affinity defaults
without real-world evidence of a per-mode affinity.

**Per-CLI model identity is not observable.** Native CLI peers
(claude/codex/antigravity) ship `model: None` intentionally so each
runs under the user's own account-default model. llm-council has no hook to
read which concrete model the CLI actually executed, so `result.model` stays
`None` — the transcript renders `cli default (unreported)` and the JSON shows
`null`. To get a RECORDED model id, pin `participants.<peer>.model` (or use
`--tier` / `modes.<name>.model_overrides`): `_build_cli_command` then injects
`--model <id>` (or `exec -m <id>` for codex) and `result.model` carries the
REQUESTED id — never a server-side confirmation that the CLI honored it. The
estimate row substitutes the load-bearing `CLI_DEFAULT_MODEL_LABEL`
(`"cli default"`) sentinel for `model: None` peers; `cli.py:cmd_estimate`
compares against that same constant to render the peer NAME in the table.
Antigravity 1.0.x accepts `--model`, so a configured model is injected like
other non-Codex CLIs. Use the exact `agy models` display name: an unrecognized
value silently falls back to Antigravity's session default, so a requested id
is not proof of the model that served the response.
