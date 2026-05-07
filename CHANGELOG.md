# Changelog

## Unreleased

## 0.4.6 - 2026-05-07

### Orchestrator

- New per-run pre-flight ping for local participants. Before the council
  starts round 1, every selected participant whose `base_url` resolves
  loopback or RFC1918 (Ollama and local `openai_compatible`) gets a
  1-second `GET /v1/models` (or `/api/tags` for ollama). Probes run
  concurrently; total pre-flight wall time is bounded by the single-probe
  timeout, not the participant count. Reachable endpoints pass through
  to the normal run path; unreachable ones short-circuit with a synthesized
  `PreflightFailed: local endpoint unreachable for 'name' (base_url='…')`
  result. Hosted participants (CLIs, openrouter, public
  openai_compatible) are skipped silently — pre-flight is solely about
  local-endpoint failure detection.
- Turns the most common local-only failure mode (server stopped, port
  wrong, model not loaded) from a multi-minute opaque `downstream_error`
  at participant timeout into a sub-second legible failure that names
  the participant and the URL.
- New `preflight_failed` event in `progress_events` (with `participant`
  and `error` fields) so the CLI's progress stream and the MCP tool's
  metadata both surface the early failure.
- New `pre_flight_check: false` per-participant opt-out. Useful when an
  intermittently-reachable endpoint is fine for the user to retry but
  shouldn't fail-fast at run start.
- New `preflight_failed` `error_kind` in the failure taxonomy
  (joins `timeout`, `context_overflow`, `prompt_too_large`,
  `invalid_response`, `downstream_error`, `cli_nonzero_exit`,
  `unknown`). Distinguishes "we knew this would fail before trying"
  from "the call failed midway."

## 0.4.5 - 2026-05-07

### Validation

- New `config_warnings(config)` surfaces non-fatal advisories at config-
  load time. The first class shipped is **origin typo detection**:
  participants whose `origin` string normalizes (lowercase + strip
  whitespace + strip punctuation) to a canonical entry in the new
  `KNOWN_ORIGIN_STRINGS` registry but doesn't match it literally trigger
  a warning suggesting the canonical form. `origin_policy: us` uses
  literal-prefix matching (`origin.startswith("US /")`), so spacing or
  case typos (`us/anthropic`, `US/Meta`, `us / meta`) silently exclude a
  participant from US-only runs — the warning catches that class before
  it bites.
- `KNOWN_ORIGIN_STRINGS` (in `defaults.py`) lists every origin used by
  built-in participants plus the ones promised in
  `docs/local-models.md`: `US / Anthropic`, `US / OpenAI`, `US / Google`,
  `US / Meta`, `US / Mistral`, `France / Mistral`, `China / Alibaba Qwen`,
  `China / DeepSeek`, `China / Z.ai`, `China / Moonshot AI`. Origins
  outside the registry are accepted without comment (free-text custom
  origins like `Canada / Ada Lovelace Labs` are not flagged).
- The detection is intentionally normalize-equality only, not edit-
  distance fuzzy match. `US / Anthrpic` (missing 'o') is not flagged —
  the warning class targets the high-impact case/spacing/punctuation
  drift that's almost always a typo, not similarity matching.
- Warnings print to stderr (prefix `llm-council warning:`) at the start
  of `list`, `doctor`, `estimate`, and `run`. Informational only —
  exit codes and behavior are unchanged.

## 0.4.4 - 2026-05-07

### Modes

- New built-in `local-only` mode and `local_only_peers` mode strategy.
  Selects every `type: ollama` participant plus any `type:
  openai_compatible` whose `base_url` resolves to loopback (`127.0.0.1`,
  `localhost`, `[::1]`) or RFC1918 (`10.x`, `172.16-31.x`, `192.168.x`).
  Excludes hosted-inference CLIs (claude/codex/gemini — local binary,
  hosted inference) and hosted API peers (openrouter). Auto-extends as
  users add local participants — no need to update mode wiring when a
  new vLLM/sglang/LM Studio entry shows up in `.llm-council.yaml`.
- `local-only` is distinct from `private-local`: `private-local` stays
  hard-pinned to the built-in `local_qwen_coder` (Ollama) for backcompat,
  while `local-only` picks up any local participant the user has wired
  up. Existing `private-local` callers see no behavior change.
- The `local_only_peers` strategy refuses `include_current` and `add` —
  hybrid modes (local + hosted) must use an explicit `participants:`
  list so the contradiction is visible at the call site rather than
  silently producing a non-local result.
- Setup wizard surfaces the `local-only` mode in generated configs only
  when the project has at least one local participant, mirroring the
  existing pattern that hides `private-local` from setups without
  Ollama.

## 0.4.3 - 2026-05-07

### Diagnostics

- `llm-council doctor --probe-local-openai [BASE_URL]` discovers local
  OpenAI-compatible inference servers (vLLM, sglang, LM Studio,
  llama.cpp `--api`, TGI, Ollama's `/v1` shim, MLX). With no value, it
  scans well-known ports on `127.0.0.1` (`8000`, `1234`, `8080`,
  `11434`, `5000`) with a 500ms per-port timeout. With a URL, it probes
  that endpoint with a 5s timeout. The probe validates the JSON shape
  of `GET /v1/models` — not just that the port answers — so a Django
  or FastAPI dev server on `:8000` is reported as "HTTP 200 but body is
  not JSON," not mis-identified as an LLM server. Connection-refused
  noise is suppressed when scanning defaults so only ports that
  actually responded appear in the report. Mirrors the opt-in pattern
  of `--probe-openrouter` / `--probe-ollama`: not run unless asked.

### Documentation

- New `docs/local-models.md` — copy-paste recipes for wiring
  `type: openai_compatible` participants at vLLM, sglang, LM Studio,
  llama.cpp `--api`, TGI, Ollama `/v1`, and MLX. Calls out the two
  load-bearing gotchas: (1) `origin` describes the model behind the
  endpoint, not the network location (so `origin_policy: us` filters
  correctly), and (2) the adapter requires a non-empty
  `Authorization: Bearer` header even for unauthenticated local
  servers — export `LOCAL_OPENAI_API_KEY=dummy` (or your real key)
  before running. Also documents the `allow_private: true` requirement
  for loopback `base_url`s, the long-context timeout floor (≥360s for
  131K-context vLLM), and the concurrent-serving FAQ ("3 participants
  on one vLLM = serialized").

## 0.4.2 - 2026-05-04

### CLI

- `llm-council estimate` now accepts `--max-cost-usd` and `--max-tokens`,
  mirroring the gates that already exist on `run`. The breakdown is still
  printed; the command exits non-zero when the projected cost or token
  total exceeds the cap. This lets wrappers and CI gate "would this run
  exceed budget" with a single subprocess call (estimate-then-check)
  instead of running an estimate, parsing JSON, and comparing manually.
  Hosted (openrouter / openai_compatible) peers with no catalog price
  refuse rather than slipping past the cap as $0, same as `run`.

## 0.4.1 - 2026-05-03

### Visual identity

- CLI progress and final-result lines now render through a right-aligned
  12-character bold-cyan gutter (verbs `Convening` / `Round` /
  `Deliberating` / `Concluded` for orchestrator events; peer name as the
  gutter token for per-participant lines). Status words inside the
  content are colored separately (`ok` green, `timeout`/`slow` yellow,
  `failed`/`error`/`degraded` red). The layout — not the color — is the
  signature, so it survives `NO_COLOR=1` and non-TTY contexts unchanged.
- Final-result block now ends with a `─` × 12 rule (ASCII `-` × 12
  fallback when `sys.stdout.encoding` is not UTF) above the transcript
  path so the reader's eye lands on the path last.
- New `summary_markdown` field on the MCP `council_run` outputSchema:
  `**LLM Council** · mode=X · N/M succeeded · time · recommendation=Y` +
  per-peer markdown table + blockquoted transcript path. Designed to
  survive host-agent rendering (markdown blockquotes/tables/bold
  headings are reliably preserved when agents quote tool output, even
  if they paraphrase surrounding prose). Also emitted on dry-run.
- **Breaking (greppers):** the orchestrator-level CLI lines are now
  `llm-council starting: ...` and `llm-council complete: ...` (was:
  `Council starting:` / `Council complete:`) so output stays
  identifiable when piped into shared logs or CI artifacts. The MCP
  payload heading is `**LLM Council**` (was: `**Council**`). Any CI
  scripts grepping for the old substrings need to be updated.

## 0.4.0 - 2026-05-02

This release pairs structural fixes for the consensus-stance feature with ergonomics, observability, and budget improvements surfaced during end-to-end testing of the v0.4.0 surface.

### Reliability and recovery

- Repair-retry on missing `RECOMMENDATION:` label for CLI, OpenRouter, and
  Ollama participants — a peer that drops the label gets a single targeted
  retry asking only for the label.
- Launch-retry CLI participants when stderr matches a configured
  `cli_retry_stderr_patterns` regex list (transient ECONNRESETs, daemon
  restarts, etc.). Both retries surface as `recovered_after_launch_retry`
  / `repair_retry_recovered` fields on the result and on the
  `participant_finish` progress event.
- Honor `retries: 0` everywhere — previously `int(cfg.get("retries") or N)`
  silently coerced 0 → N (HTTP) and `_retry_enabled` ignored it (repair
  retry); both are fixed.
- Failure taxonomy: every result now carries an `error_kind` field
  (`timeout`, `context_overflow`, `prompt_too_large`, `invalid_response`,
  `downstream_error`, `unknown`) so callers can branch without parsing
  human-facing strings. Documented in CLAUDE.md.

### Deliberation and consensus

- Slim round-2 deliberation prompt: the bulky `Context:` block is dropped
  on round 2+ since peers reasoned over it in round 1. Per-peer excerpts
  are truncated at line boundaries and label lines are capped.
- `## Remaining disagreement` markdown section + `remaining_disagreement`
  JSON field whenever the final round still has conflicting labels.
- New `consensus` mode with assigned-stance prompting (for/against/neutral)
  and an unconditional ethical-override clause that prevents any peer from
  defending a harmful proposal. Stances now stamp on each result and on
  `metadata.stances` in the transcript.
- `--stance peer=for|against|neutral` CLI flag and `stances` MCP arg let
  callers override or extend stance assignments per-call without forking
  the mode config.
- Convergence detector: per-round Jaccard token-set similarity between
  successive deliberation rounds (states: converged / refining / diverging
  / insufficient when the response is too short to classify).
- Degraded consensus: when fewer than `min_quorum` peers produce a label
  in the final round, the result is marked degraded with a clear
  `## Degraded consensus` section and `degraded_consensus` JSON payload.
  `--min-quorum` CLI flag and `min_quorum` MCP arg.

### Stance bug fixes (the v0.4.0 ship blockers)

The headline consensus-stance feature broke in three independent ways
that the council itself caught during the review pass:

- Stance was silently dropped when no `.llm-council.yaml` existed on disk
  (CLI/MCP didn't pass `mode_cfg["stances"]` to `build_prompt`; the
  fallback YAML lookup returned `({}, {})` for fresh installs).
- Round-2 `_strip_context_payload` truncated everything from
  `\n\nContext:\n` onward, including the `stance_tail` that lived after
  it — so multi-round consensus lost stance after round 1.
- Hard end-truncation chopped `stance_tail` when the prompt exceeded
  `max_prompt_chars`.

Fix: stance now precedes `Context:` in `build_prompt.assemble()`, both
the strip path and the truncation path leave it intact, and CLI/MCP
forward `mode_stances` from the merged config explicitly.

### Scale

- Optional `--diff` chunking strategies: `head`, `tail`, `hash-aware`
  (splits on `^diff --git ` boundaries and prefers files mentioned in the
  question). `fail` (the default) now actually raises on overflow instead
  of silently truncating — the prior behavior could have the council
  answer from a partial diff with no signal to the caller.
- Per-participant context-window budget: peers with a
  `max_context_tokens` smaller than the chunked prompt are excluded
  gracefully via a `context_overflow_excluded` event, the rest of the
  council still runs.
- On-disk per-participant result cache keyed on
  `sha256(name + canonical(cfg) + prompt + image_manifest)`, with a
  `CACHE_SCHEMA_VERSION` and TTL. `--cache {on|off|refresh}` flag.
- Chunking budget mismatch fix: chunking now targets the smallest
  per-peer `max_prompt_chars`, not the global default — adapters used to
  reject the chunked prompt at launch when peers had stricter caps.

### Threading and continuation

- Conversation threading via `--continue <run_id>` (CLI) /
  `continuation_id` (MCP). Prepends a `Prior council context` summary of
  the prior transcript to the new prompt; the new transcript records
  `parent_run_id`.
- Continuation depth cap (default 5) so chained `--continue` runs cannot
  silently eat into `MAX_PROMPT_CHARS`. Configurable via
  `defaults.max_continuation_depth`.

### Transcripts and observability

- `## Round 2 Prompt` (and beyond) section in the markdown transcript +
  `metadata.deliberation_prompts` in JSON, so an operator can audit
  exactly what context peers got each round.
- `from_cache`, `recovered_after_launch_retry`, `repair_retry_recovered`,
  `stance`, and `error_kind` all surface in both the on-disk transcript
  JSON and the `--json` stdout summary.
- `transcripts prune --keep-last N --keep-since DATE` subcommand
  (dry-run by default; `--apply` to actually delete) for cleanup.
- `llm-council stats` aggregator: per-participant runs, success rate,
  recommendation distribution, tokens, cost, last-used time. CLI + MCP
  tool. `--since` accepts both integer days back and ISO date.

### Configuration and deployment

- `openai_compatible` participant type with SSRF-defended `base_url`
  validation (https-only, reject IP-private/loopback/link-local, reject
  reserved-key headers). `type: openrouter` silently migrates to
  `openai_compatible + base_url: https://openrouter.ai/api/v1` for
  backwards compatibility.
- Run-level budget caps: `--max-cost-usd` and `--max-tokens` (CLI) /
  `max_cost_usd`, `max_tokens` (MCP) gate on the pre-flight estimate
  before any subprocess or HTTP call. `estimate` was previously advisory.
- Structured `council_run` outputSchema advertised on the MCP tool with
  typed `recommendation` (yes/no/tradeoff/unknown), `agreement_count`,
  `total_labeled`, `degraded`, `rounds`, `participants`, and per-peer
  records — agents no longer have to grep markdown for `RECOMMENDATION:`.
  Falls back gracefully on older `mcp` SDK versions that don't accept
  the `outputSchema` kwarg.

### CLI ergonomics

- `--question` flag as alias for the positional question, matching the
  MCP `council_run` arg name (mutually exclusive with the positional).
- New `council_stats` CLI subcommand and MCP tool.
- `consensus` mode added to default-config modes alongside the existing
  `quick` / `peer-only` / `plan` / `review` / `diverse` / `private-local`
  / `us-only` / `deliberate`.

### Documentation

- CLAUDE.md gains sections for the failure taxonomy, custom CLI
  participant minimal template, continuation-chain depth cap, and
  run-level budget caps.

## 0.3.2 - 2026-04-29

Closes documentation and test-coverage gaps from the 0.3.1 review pass.

- README now has a "What's New" section covering image passthrough, graceful timeouts, and the temporary Opus version variants. `docs/llm-council.md` gains "Images" and "Timeouts and slow warnings" sections plus an explicit mention of `opus-versions` in the Modes section.
- Add a code comment in `run_participant` explaining that the CLI branch intentionally drops `image_manifest`: CLI subprocesses Read staged images from disk via the `## Images` prompt section, so `vision: true` on a CLI participant has no effect.
- Add tests for: `claude_4_7` model-flag pin (symmetry with the existing `claude_4_6` test), the default 75% slow-warn threshold formula and its 30s floor, `sweep_old_inline_inputs` actually being invoked from `run_council`, and a non-dry-run MCP `run_council` that records the image manifest into both transcript markdown and JSON metadata.

## 0.3.1 - 2026-04-29

Reviewer-driven follow-up. Two Claude council runs (4.6 and 4.7
head-to-head against the 0.3.0 codebase) surfaced a handful of real
issues; this release ships the verified ones.

- Enforce image-attachment budget at estimate time so a passing preflight matches a passing run. `estimate_council` now builds the image manifest and runs `image_attachment_violations`, mirroring `run_council`.
- Auto-generate the MCP `mode` schema description from `DEFAULT_CONFIG["modes"]` so new modes (e.g. `opus-versions`) can't fall off the schema as they did in 0.3.0.
- Make the `## Images` prompt copy audience-agnostic: CLI subprocesses are told to open the file with their file-read tool; vision-capable hosted models are told to refer to the attachments by relative path.
- Sweep stale `.llm-council/inputs/<run-id>/` directories before each new staging (default 7-day retention via `INLINE_INPUTS_RETENTION_DAYS`) so disk usage doesn't grow unbounded with screenshot-heavy councils.
- Single-source `RECOMMENDATION_RE`: `deliberation.py` now re-exports the regex from `adapters.py` instead of carrying a byte-identical copy.
- Promote the deliberation per-peer excerpt cap to a named constant `MAX_DELIBERATION_PEER_EXCERPT_CHARS = 20_000` and raise it from a magic 4 000 so a 3-peer second round actually uses the 80 000-char window.
- Simplify `_build_cli_command`: collapse three identical model-flag branches into one default branch, isolate the Codex `exec -m` case, and lock the shape with regression tests.
- Remove the unused sync `_build_user_content` helper from `adapters.py` (production uses the async variant).

## 0.3.0 - 2026-04-29

- Add image passthrough to council: `council_run` and `estimate` accept `image_paths` (path-first) and inline `images: [{data, mime, name?}]` (sandboxed-host fallback). CLI grows a repeatable `--image PATH` flag. `build_prompt` emits a `## Images` section so CLI participants Read images from disk via their existing tools.
- Add per-participant `vision: true` flag. OpenRouter adapter switches to multimodal content arrays and Ollama adapter populates `messages[].images` for vision-capable participants. Non-vision participants in a council with images present get the text manifest only and surface an `images_skipped` progress event.
- Stage inline images under `.llm-council/inputs/<run-id>/` with 8 MB per-file and 32 MB total caps. Add `.llm-council/inputs/` to runtime and project gitignores. Force the staged extension to match the declared mime so downstream mime detection succeeds.
- Make CLI participant timeouts graceful: actionable error message naming the participant, timeout, prompt size, and the config knob to turn; `participant_slow` watchdog event at 75% of timeout; `status="timeout"` distinct from `"error"`/`"skipped"`; transcript labels timed-out participants `(timeout)`; CLI summary calls out timeouts with the actionable hint and base-name dedupe; deliberation rounds skip timed-out participants cumulatively; `skipped_all_excluded` deliberation status preserved after a round has run.
- Add temporary pinned-version Claude participants `claude_4_6` and `claude_4_7` and an `opus-versions` mode for head-to-head comparison. Setup wizard ships them under the native preset; routing keywords cover "with opus 4.6/4.7", "compare opus versions", and "opus 4.6 vs 4.7".

## 0.2.7 - 2026-04-28

- Make built-in native modes ask Claude, Codex, and Gemini as explicit participants by default.
- Add `peer-only` mode for the old behavior that excludes the current host subprocess.
- Add `include_current` routing support while preserving peer-only behavior for custom `other_cli_peers` modes.
- Update generated instructions, docs, and example config for full-triad default council runs.

## 0.2.6 - 2026-04-28

- Change the generated Claude Code participant from `--permission-mode plan` to `--permission-mode default` while keeping read-only tools.
- Treat successful subprocesses without the required `RECOMMENDATION:` label as invalid participant responses.
- Preserve invalid participant output in transcripts so adapter failures are debuggable.
- Remove the obsolete Codex `--ask-for-approval never` flag from defaults.
- Stop printing successful CLI stderr banners as participant error details.
- Migrate old generated Claude and Codex args at config load time.

## 0.2.5 - 2026-04-27

- Refuse explicit setup presets when required CLIs or API keys are missing.
- Add `--allow-incomplete` for advanced users who intentionally want to stage an incomplete setup.
- Add regression coverage for blocked preset writes.

## 0.2.4 - 2026-04-27

- Add `llm-council setup --plan` so agent installers show detected routes and ask before choosing a preset.
- Update agent-first install instructions to avoid silently accepting `auto` setup.

## 0.2.3 - 2026-04-27

- Rewrite the README around agent-first installation and natural council usage.
- Expand generated project instructions so coding agents know when and how to call council.
- Move direct terminal usage behind the primary MCP/coding-agent workflow.

## 0.2.2 - 2026-04-27

- Treat generated `.mcp.json` as local machine config by adding it to project `.gitignore`.
- Add explicit data-boundary policy text to generated CLI instruction snippets and docs.
- Make generated snippets pass the active CLI identity to council calls.
- Add comparable native CLI prompt caps for Codex and Gemini.
- Avoid generating a duplicate `us-only` mode when `--us-only-default` already applies globally.

## 0.2.1 - 2026-04-27

- Switch update checks to public release tags instead of raw `main` metadata so new releases are visible immediately after tagging.

## 0.2.0 - 2026-04-27

- Add `llm-council estimate` and MCP `council_estimate` for hosted cost previews.
- Add `llm-council check-update`, `doctor --check-update`, and MCP `council_doctor` version reporting.
- Improve beginner setup guidance for native CLIs, OpenRouter, local models, and frontier-model cost tradeoffs.
- Use `qwen_coder_flash` for reliable cheap hosted defaults while retaining `qwen_coder_free` for explicit experiments.
- Handle empty OpenRouter responses gracefully instead of surfacing adapter tracebacks.
- Fix project config discovery so `--cwd` controls config lookup.

## 0.1.0 - 2026-04-25

- Initial clean Intellimetrics `llm-council` project.
- Added CLI and MCP server for read-only multi-agent council runs.
- Added native Claude Code, Codex CLI, Gemini CLI, OpenRouter, and Ollama participants.
- Added transparency mode with per-model token/cost reporting when providers return usage.
- Added opt-in deliberation mode with a second round on detected disagreement.
- Added project setup, doctor checks, OpenRouter model catalog, and transcript storage.
- Added transcript inspection commands and hardened labeled deliberation across multiple rounds.
- Hardened subprocess cleanup, prompt redaction, MCP context boundaries, budget checks, and config validation.
- Added live CLI progress reporting and MCP `metadata.progress_events` so users can see council participant starts, finishes, skips, errors, and deliberation status.
- Added prompt-size preflight guards for CLI participants, including Claude, to skip oversized prompts immediately with a clear message.
- Hardened setup presets with `replace_defaults`, MCP project-root isolation, staged/unstaged diff capture, transcript markdown fencing, and actionable setup parse errors.
- Added fail-closed MCP pricing checks for paid hosted participants and documented custom CLI `env_passthrough`.
- Added an operator reference, manual MCP root guidance, non-run MCP tool tests, setup `--yes` coverage, and configurable global prompt construction limits.
- Kept MCP budget guards independent from global prompt sizing, removed stale timeout defaults, and made `doctor` validate the configured default route instead of always requiring all native CLIs.
