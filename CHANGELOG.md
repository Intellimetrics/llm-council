# Changelog

## 0.15.0 - 2026-06-16

v0.15.0 is a batch of advisory-scoped improvements to the council's deliberation, synthesis, independence, and cost/observability layers. Every change keeps llm-council strictly **advisory and read-only** — no peer gains write/exec capability — and is **default-OFF or purely additive**, so existing behavior is unchanged unless explicitly opted in. Each change was implemented and then reviewed by the council itself (cross-vendor `codex`/`gemini`/`antigravity`/`qwen`, excluding the authoring model — dogfooding the independence this release also adds); review-caught fixes are noted inline.

**Deliberation quality**
*   **Anti-capitulation + de-anchoring in the round-2 prompt.** `build_deliberation_prompt` now tells peers to converge toward what is *actually correct* rather than toward agreement for its own sake (do not capitulate to the group; do not hold a position out of consistency bias; name what convinced you on a change or what you weighed on a hold), and to critique the OTHER peers rather than re-justify their own prior answer. Pure prompt text — no control-flow or contract change; the `RECOMMENDATION:` label and envelope instructions are unchanged. Directly targets the MAD herding failure mode the codebase already cites (arxiv 2402.18272).
*   **Distinct-vendor independence warning (opt-in, default off).** New `defaults.min_distinct_vendors` / per-mode `require_distinct_vendors`: when every labeled vote in the final round shares one vendor `family`, the orchestrator sets a NEW `metadata['independence_warning']` and emits a `single_vendor_quorum` progress event, so correlated same-vendor agreement isn't mistaken for independent corroboration. It NEVER touches `degraded`/`min_quorum`/`labeled_quorum` and drops no peer. (Review fix: guarded against a false warning on a zero-labeled, already-degraded run.)

**Synthesis presentation**
*   **Attribution, dissent, and movement.** The synthesis chair prompt gains a `### How positions moved` section (only on deliberated runs, grounded strictly in the convergence states/similarity the chair actually receives — not round-1 labels it never sees), per-peer attribution on `### Consensus blockers`, a dissent-preservation directive in `### Decision`, and an even-handed "moderator, not a third debater; don't favor any peer including yourself" framing. The markdown transcript's "Remaining disagreement" line now names the minority peers. Presentation only — the headline recommendation still comes from peer votes.

**Composable review focus**
*   **User-authorable review-skill bundles (`--focus`).** Operators can drop reusable directive bundles at `.llm-council/review-skills/<name>/SKILL.md` (frontmatter `name`/`description` + markdown body) and apply them to any run via `--focus name1,name2` (CLI) or `focus: [...]` (MCP `council_run`). Bundles are **inert prompt text** — they shape *what* peers scrutinize and grant no tool/write/exec capability — and **compose onto any mode** (additive `focus_directive`; existing review-with-tools/stance/persona branches were intentionally left intact). Strict name validation (kebab ≤64 matching directory, description ≤1024) with lenient discovery (a malformed bundle is skipped, not fatal); an unknown `--focus` name fails fast listing the available bundles. Provenance is recorded as `metadata['applied_focus'] = [{name, sha256}]`. Ships two example bundles (`security-review`, `test-gaps`). (Review note: a bundle body containing a `PART N — TITLE (REQUIRED)` header is enforced by the section-coverage validator since it's part of the peer prompt — opt-in by the author; shipped examples are guarded against tripping it.)

**Independent, contract-scoped review**
*   **Acceptance contract (`--acceptance-contract <text|path>` / MCP `acceptance_contract`).** Injects an ACCEPTANCE CONTRACT block instructing peers to gate a blocker / `RECOMMENDATION: no` only on a violation of a numbered criterion — sharply cutting drive-by nitpicks. A path is read only when it resolves to an existing file inside cwd (`ensure_inside_cwd`), otherwise treated as literal text; counted against `max_prompt_chars`.
*   **Independent-review context isolation (`--independent-review` / `independent_review`, default off).** On a continuation run, suppresses the prior council's per-peer labels/rationales so the new round forms its verdict without anchoring on prior verdicts; records `metadata['prior_context_suppressed_for_independence']`. (Review fix: None-aware precedence so a higher-priority explicit `false` — a mode opt-out or per-call MCP `false` — overrides a `true` default, rather than an `or`-chain.)

**Cost advisories (the hard `--max-cost-usd`/`--max-tokens` gate is unchanged)**
*   **Soft cost warning.** `defaults.cost_warn_usd` / `--cost-warn-usd` / MCP `cost_warn_usd` attaches a non-fatal `metadata['cost_warning']` when the pre-flight estimate exceeds the threshold but is under the hard cap — computed from the SAME `summarize_preflight_caps` reduction the hard gate uses, so the two can't drift. Never raises, never blocks.
*   **litellm pricing fallback.** For hosted (`openrouter`/`openai_compatible`) peers whose model id is absent from the OpenRouter catalog, an import-guarded litellm local-cost-map lookup fills the price (no network; litellm is not a hard dependency — absent ⇒ unchanged behavior). Native CLI peers are never priced.
*   **`cost_class` + run-path echo.** `estimate_council` gains `cost_class` (low/moderate/high from the retry-safety total) plus `paid_peer_count`/`free_peer_count`, and both run paths echo a compact `metadata['cost_estimate']` so a caller who skipped `council_estimate` still sees the signal. (Review fix: local `openai_compatible` peers — loopback/RFC1918, effectively $0 — count as free, not paid.)

**`council_recommend` enrichment**
*   **Mechanical difficulty grade (always-on).** `policy.recommend()` adds `difficulty_class` (trivial/moderate/hard from documented risk/attempts/files/matched-keyword thresholds) and `suggested_mode_reason_codes` (the list of matched trigger keywords, vs the single prose `reason`). `should_use_council`'s signature/behavior is unchanged.
*   **Optional LLM difficulty judge (default off, fail-open).** When `defaults.recommend_judge` names a hosted peer, `recommend_judge.grade_difficulty()` makes one bounded JSON call returning `{difficulty, rationale, suggested_mode}` as a supplementary `judge` field; it returns None (never raises) on any missing config / non-hosted peer / missing key / timeout / non-2xx / parse failure, and never overrides the mechanical verdict.
*   **Reliability "consider dropping" advisory.** `council_recommend` and `council_estimate` surface `peers_to_consider_dropping` (peers with `false_blocker_count > useful_count` AND `verified_citation_rate < 0.5`, skipping `None`) from `stats.aggregate_reliability`. Advisory only — `select_participants` is untouched.

**Observability**
*   **Opt-in per-CLI token usage/cost (`usage_from_json`, default off).** Partially lifts the documented "per-CLI usage is not observable" limitation for `claude` and `codex`: when enabled, the peer is invoked in its JSON output mode (`--output-format json` / `exec --json`) and real `prompt_tokens`/`completion_tokens`/`total_tokens`/`cost_usd`/`model` are parsed into `ParticipantResult`. The JSON flag is purely additive (read-only flags preserved), the `RECOMMENDATION:` label check runs on the extracted model text, parsing **fails soft** to today's raw-text behavior on any shape change, and no JSON flag is added for families whose JSON we don't parse (e.g. gemini) — which would otherwise break their label check. codex `prompt_tokens` subtracts `cached_input_tokens`.

**Finding matrix + deliberation budget**
*   **Three-tier gating partition.** `findings.matrix_to_dict` derives an additive `gating` object (`blocking` = consensus blocker clusters, `non_blocking` = consensus medium, `suggestion` = nit clusters + all single-peer concerns) from the entries it already builds. No new clustering/severity/parsing; existing `consensus_blockers`/`single_peer_concerns` shapes are unchanged.
*   **No-new-movement early-stop (opt-in, default off).** `defaults.deliberation_early_stop` / per-mode breaks the deliberation loop when a round shows no divergence AND an unchanged vote tally (the tally corroborates the Jaccard signal), setting `deliberation_status: stopped_no_new_movement`. (Review fix: gated on `round_number < max_rounds` so only deep-audit (≥3 rounds) is affected and a `max_rounds=2` run is never relabelled when nothing was actually skipped.)

**Tests**
*   +119 (1166 → 1285 passing, 2 skipped). Every feature carries unit coverage for its default-off / advisory-safe / fail-soft paths, plus regression tests for each council-review-caught fix.

## 0.14.1 - 2026-06-10

*   **Antigravity `--model` injection restored.** `agy` 1.0.x gained a `--model <name>` flag, so the pre-1.0 `family=antigravity` skip in `_build_cli_command` is retired: a pinned model is injected like any standard family, which also makes the `fallback_chain` quota walk effective for agy (the walk rebuilds the command with the fallback model — previously a no-op because injection was skipped). Caveats documented in-code: agy model strings are `agy models` display names (e.g. `Gemini 3.5 Flash (Medium)`), and agy silently falls back to its session default on an unrecognized string (exit 0, no error).
*   *(Release-hygiene note: this entry, the `__version__` bump in `llm_council/__init__.py`, and the README badge were back-filled post-tag — the v0.14.1 release commit only bumped `pyproject.toml`, which `test_package_version_matches_pyproject` caught.)*

## 0.14.0 - 2026-05-30

v0.14.0 is a code-quality batch (a whole-codebase simplification pass — 39 adversarially-verified findings) plus an honesty fix for CLI-peer model reporting. No new features and no breaking changes; the one user-visible behavior change is the transcript/list model placeholder wording. This continues (and partially closes) the "CLI/MCP pipeline de-duplication" thread flagged as deferred in 0.13.0.

**Code quality / de-duplication (behavior-preserving)**
*   **Hosted-adapter consolidation.** `run_openai_compatible_participant` and `run_ollama_participant` were byte-identical (~120 lines of verbatim overflow→cache→inner→terse-retry→section-repair→strict-evidence pipeline); they now delegate to one shared `_run_hosted_participant(inner, section_repair, …)`, and the two `_maybe_section_repair_*` helpers collapse into `_maybe_section_repair_hosted`. Future fixes to the retry-layering invariant land in one place instead of drifting between transports.
*   **New single-source helpers.** `transcript.transcript_dir(cwd, config)` (killed 6 inlined `transcripts_dir` resolutions across `cli`/`mcp_server`), `transcript.iter_run_json(base_dir)` (killed the duplicate glob+sort+JSON-read loop shared by `stats.load_transcript_files` and `transcript.transcript_records`), `display.format_usd` (one USD formatter for `cli` + `stats`), `budget.enforce_preflight_caps` (the two CLI budget-cap blocks), and `mcp_server._lift` (6 metadata pop-and-lift blocks → 1).
*   **In-file collapses.** Reused `_build_strict_evidence_retry_prompt` on the CLI path; added `_within_prompt_cap` for ~10 repeated per-retry cap checks; a `_base_peer_name` helper in `stats`; `orchestrator` now reuses `_base_name`/`_is_labeled_vote` and a shared quota detect-emit helper (round 1 + round 2), and drops `getattr` guards on always-present dataclass fields; `transcript._select_final_round_records` uses `result_round`; plus dedup in `config` (loopback/local URL prefix), `context` (relative-label), `diff_chunking` (greedy-admit loop), `model_catalog` (fetch/refresh), `safety` (pattern-iter + kind tally), `setup_wizard` (read config once), `eval` (`jaccard_similarity` reuse + `_mean`), and `estimate`. All 39 fixes were adversarially verified to preserve behavior before applying.

**CLI model reporting (honesty + hardening)**
*   **Honest unreported-model placeholder.** Native CLI peers (claude/codex/gemini/antigravity) ship `model: None` by design so each uses the user's own account-default model, and llm-council cannot observe which concrete model actually answered (per the "Per-CLI model identity is not observable" invariant). The transcript/`list` views now render `cli default (unreported)` instead of the misleading `cli default`. Introduced `CLI_DEFAULT_MODEL_LABEL` so the estimate-table sentinel (producer in `estimate.py`, comparator in `cli.py`) can no longer drift apart; JSON `results[].model` stays `null`.
*   **Antigravity `--model` footgun closed.** `_build_cli_command` no longer injects `--model` for the `antigravity` family (`agy` has no model flag); a model pinned via `--tier`/`modes.model_overrides` is now silently ignored rather than producing a broken invocation.
*   **Verified, not changed:** `model:` IS already enforced for `type:cli` (the diagnosis that prompted this release was black-box and wrong on that point) — when set, `_build_cli_command` injects `--model <id>` (or `exec -m <id>` for codex) and `result.model` carries the requested id. Documented in CLAUDE.md + README.

**Docs**
*   Documented the ambient-environment bleed (sieve env mode passes non-secret host vars like `GEMINI_MODEL`/`ANTHROPIC_MODEL`/`*_BASE_URL` straight through, silently steering a native CLI's model/endpoint) and the existing per-peer `env_strict: true` remedy.
*   Documented that the mandatory `RECOMMENDATION:` label costs one repair-retry round-trip on non-vote/trivial prompts, and the per-peer `retry_on_missing_label: false` / `require_recommendation: false` opt-outs (for genuinely non-vote peers only — the label is the load-bearing vote contract).

**Tests**
*   +3: an estimate-table sentinel guard (a `model=None` CLI peer renders the peer NAME, proving producer/comparator stay in sync), an honest-transcript model-line assertion, and an antigravity no-`--model` guard.

## 0.13.0 - 2026-05-28

v0.13.0 lands a batch of fixes from a multi-dimension self-review (35 verified findings). Theme: close correctness/safety gaps where a field or config key was added to the data model but the validation, serialization, or concurrency discipline that should accompany it lagged behind.

**Safety**
*   **Antigravity read-only hardening (SOFT / prompt-enforced).** The `antigravity` peer no longer ships `--dangerously-skip-permissions` (which auto-approved every tool call, including Write/Edit). Important nuance discovered via a live canary: `agy` has no read-only / `--approval-mode` / `--tools` flag, and `--sandbox` only restricts the *terminal*, NOT the model's native write tool — so agy *can* write files when ordered with no read-only framing. What actually keeps it read-only is the council prompt's read-only directive (`context.build_prompt`), which agy reliably honors (verified 4/4: it refuses an explicit write request when the directive is present). Dropping `--dangerously-skip-permissions` ensures a stray write isn't auto-approved. This is a **softer** guarantee than the flag-enforced peers (claude/codex/gemini physically can't write); the residual risk is a prompt-injection in reviewed content overriding the directive. New guards: `tests/test_adapters_safety.py` fails if any default CLI peer re-adds an auto-approve flag, and the opt-in `tests/test_live_agy_readonly.py` canary (gated by `LLM_COUNCIL_LIVE_AGY_TEST=1`) verifies agy still honors the read-only directive across upstream releases.

**Correctness**
*   **Config validation gaps closed (fail-fast at load).** `fallback_chain` (CLI), `timeout_multiplier` (mode), `idle_timeout` and `timeout_per_kb_chars` (participant) are now validated at config-load. Previously a string `fallback_chain` was character-sliced into bogus model ids on the quota path, and `timeout_multiplier: "fast"` raised an uncaught `ValueError` mid-run. `timeout_per_kb_chars: 0` (disable-scaling sentinel) validates as non-negative.
*   **`detect_current_agent` PPID parse** no longer breaks when a parent process `comm` contains spaces/parens (e.g. tmux's `(tmux: server)`); it now parses ppid relative to the final `)` in `/proc/<pid>/stat`. Previously the broad except silently aborted the walk and the host CLI wasn't excluded from its own review.
*   **`.llm-council.env` precedence is now nearest-wins.** A subproject's env file beats an ancestor's, mirroring `find_config` — the override=True last-load-wins tie-break previously inverted this so a stale repo-root file shadowed the child.
*   **`stats.aggregate` no longer double-counts `--cross-rank`.** Ranking-pass results (`<peer>:rank`) are excluded from the final-round view and their cost/latency folds into the base peer instead of phantom `<peer>:rank` rows that inflated `total_runs`.
*   **Section-coverage validator** no longer false-accepts a missing section when two REQUIRED sections share a salient title token (e.g. `SECURITY ANALYSIS` / `SECURITY HARDENING`): collision-prone sections now require the full title as a near-contiguous phrase. Also detects separators with no trailing space (`PART 1 —OVERVIEW`).
*   **Cache round-trip** now preserves `terse_retry_attempted`, so a timeout-recovered result no longer rehydrates to the contradictory `recovered_after_timeout=True, terse_retry_attempted=False`.

**Resilience / concurrency**
*   **Idle-read path no longer pipe-deadlocks** on large prompts: the stdin write now interleaves with the stdout/stderr reads (like `proc.communicate()`), and a stream's idle-timeout cancels its siblings instead of leaking them.
*   **`--cross-rank` ranking pass** runs under the same `max_concurrency` semaphore as the primary rounds (no more unbounded subprocess fan-out) and uses `return_exceptions=True` so one ranking failure can't abort the whole council.
*   **`run_participants`** guards the gather with `return_exceptions=True`: an unguarded per-peer setup error degrades that one peer to a failed result instead of aborting the round.

**Surfacing / schema (MCP outputSchema bumped to v6)**
*   `continue_debate`, `evidence_verification_failures`, `terse_retry_attempted`, `section_repair_attempted`, and `is_ranking_round` are now surfaced in transcript JSON and MCP `structured_results` (and declared in the per-result schema). New drift-guard test asserts every emitted key is declared.
*   The MCP `prompt_chars` description no longer falsely claims `null` on success.
*   The MCP budget pre-flight now counts `cross_rank` (an extra ranking round) and a paid-hosted `synthesize` chair, so a run that should trip `mcp_max_estimated_cost_usd` is no longer under-counted into passing.
*   The synthesis chair's decision memo is now rendered in the markdown transcript (previously only in JSON).

**CLI / UX**
*   `transcripts show <bad-path>` and `models openrouter` network failures now print a clean error instead of a raw traceback.
*   `outcome list --last 0` (or negative) is now rejected instead of silently showing ALL records.
*   `outcome mark` honors a relocated `transcripts_dir` (`resolve_run_id` gained a `transcripts_dir` param) instead of hardcoding `.llm-council/runs`.
*   The update-nag no longer steers a stable install onto a prerelease tag (`v…-rc1`).
*   `transcripts prune --keep-last`/`--keep-since` help text now states the two flags form a UNION.

**Docs / cleanup**
*   `secret_scan`'s `scrubbed_count` renamed to `detected_count` (warn mode never scrubbed anything — it counts and logs but ships the prompt verbatim, now documented).
*   Removed dead `_pick_quota_fallback_model`; consolidated the two near-identical section-retry merge helpers into one `_merge_section_retry`.
*   Reconciled the RISK envelope contract (enum-requested, lenient capture), the `_aggregate_fixture` docstring, the section-matcher window docs (`-100`/`+200` ≈ 300 chars), and refreshed stale CLAUDE.md line refs to stable symbol names.
*   `diff_chunking` now lists every file in `oversize_files` when prompt framing alone exhausts the budget.
*   New tests for the promotion-gate `cross_rank_correlation_floor` branch and the strict-evidence `[VERIFIED:...]` invariant.

**Post-review follow-ups (same release)**
*   **New `redact` secret-scan policy.** `secret_scan: redact` masks each detected credential with `[REDACTED:<kind>]` in the prompt sent to peers AND persisted to the transcript — the only policy with transcript-level protection (`warn` never altered the prompt). New `safety.redact_secrets` re-scans and splices without ever exposing the value.
*   **Real MCP stdio integration test.** `tests/test_mcp_stdio_integration.py` spawns the server over stdio and does the full initialize → list_tools → call_tool round trip, asserting the advertised `schema_version` + required fields — automating the manual MCP-restart dogfood ritual that previously caught schema gaps by hand.
*   **CLI/MCP pipeline de-duplication (partial).** Extracted the two drift-prone shared blocks — `transcript.continuation_depth_limit_error` and `budget.summarize_preflight_caps` (cost/token/unpriced-paid reduction) — now used by `cmd_run_async`, `cmd_estimate`, and `run_council`. Each surface keeps its tailored refusal messages.
*   **Idle-read reap race fixed.** The streamed-read path now `await proc.wait()`s after EOF, so `proc.returncode` is set before it's checked (a pipe-EOF-before-reap race surfaced a spurious `CliExitNonZero` under load — caught by the new idle-deadlock regression test).
*   **Antigravity read-only reframed honestly (see Safety above).** A live canary (`tests/test_live_agy_readonly.py`) proved `agy` is *soft* (prompt-enforced) read-only, not flag-enforced — `--sandbox` is terminal-only and does not block writes. Docs/comments corrected; the canary guards against upstream drift.
*   **Eval harness exercised for `review-with-tools`.** Ran review baseline vs review-with-tools through the promotion gate; the gate correctly returns `promoted: false`. The mode stays `experimental` — the lone bundled fixture yields zero structured findings (no usable signal), so promotion is blocked on a real fixture corpus, not on code.
*   **Fixed pre-existing red CI** (`main` had been failing for ~8 days). 11 tests that drive a default/`quick`-mode run drove participant selection, which since the 0.12-era "dynamic triad resolution" change requires a Gemini-family CLI (`agy`/`gemini`) on PATH — absent on GitHub runners, so they raised before reaching their real assertions. New `tests/conftest.py` autouse fixture makes `shutil.which` report a Gemini-family CLI present only when the environment lacks one (no-op when installed; overridable by tests that exercise CLI presence/absence). No product behavior change. Validated by running the full suite with `agy`/`gemini` stripped from PATH (CI simulation).

Known deferred (tracked for a focused follow-up): the full `prepare_council_run` unification of the ~550-line `cmd_run_async` / `mcp_server.run_council` pipelines (the high-value shared blocks are now extracted; the remainder diverges by transport), and the 228-line `run_cli_participant` repair-block boilerplate.

## 0.12.2 - 2026-05-23

v0.12.2 fixes a council-flagged false-drop in `_drop_missing_key_participants`.
*   `openai_compatible` peers without an explicit `api_key_env` are NO LONGER pre-dropped. The previous code defaulted to `OPENROUTER_API_KEY` for both `openrouter` AND `openai_compatible` types, which would falsely drop a local vLLM / llama.cpp / LM Studio peer that legitimately doesn't need auth. New behavior: defer to the adapter, which surfaces its own `Missing X` error if a key was actually required.
*   `openrouter` peers keep the `OPENROUTER_API_KEY` default (well-known convention; OpenRouter peers always need a key).
*   `openai_compatible` peers WITH an explicit `api_key_env` still pre-drop normally — explicit declaration is treated as explicit expectation.
*   New tests: 3 (`drops_openai_compatible_with_explicit_env`, `skips_openai_compatible_without_explicit_env`, `keeps_openai_compatible_with_env_set`).

## 0.12.1 - 2026-05-23

v0.12.1 reshapes fallback chains for capability-graceful step-down and adds multi-step walking.
*   **Better default fallback chains** for built-in CLI peers:
    *   `claude`: `["claude-opus-4-6", "claude-sonnet-4-6"]` (was `["claude-sonnet-4-6"]`) — same-tier one-version-back before dropping to sonnet.
    *   `claude_4_7`: same shape (`opus-4-6` first, then `sonnet-4-6`).
    *   `claude_4_6`: `["claude-sonnet-4-6", "claude-haiku-4-5"]` — sonnet (next tier) then haiku.
    *   `codex`: `["gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini"]` (was `["gpt-5-mini"]`) — minor version back, codex-tuned variant, then small final.
    *   `gemini`: `["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash"]` (was `["gemini-2.5-flash"]`) — pro is more capable than flash within Google's tiering, so a flash→pro fallback is often an *upgrade* that also sidesteps the throttled flash quota.
    *   `antigravity`: still `[]` (no `--model` flag; confirmed via binary probe — agy stores model selection in `~/.gemini/antigravity/antigravity_state.pbtxt` as opaque `MODEL_PLACEHOLDER_M<N>` enums with no external override mechanism).
*   **Multi-step walking** (`_quota_fallback_walk` + `QUOTA_FALLBACK_MAX_STEPS=3`). On quota error the adapter now walks the chain up to MAX_STEPS entries instead of stopping after one retry — a chain of `[pro, mini, nano]` can step through all three within a single council call. Walker stops at: first success, first non-quota failure (continuing would spam more models with an unrelated problem), or chain exhaustion. Claude family still uses CLI-native `--fallback-model` and is excluded from the walker.
*   On walker failure, `model_fallback_used` is stamped with the LAST attempted model so the transcript shows where the walk stopped. `recovered_after_quota` stays False.
*   `_pick_quota_fallback_model` kept as a back-compat helper that returns walk[0]; primary path uses the walker.

## 0.12.0 - 2026-05-23

v0.12.0 lands four timeout-resilience improvements surfaced by the v0.11.7 dogfood (claude timed out at 240s on a 4KB prompt; codex timed out at 240s on a 26KB prompt; the 60s terse-retry on a 240s timeout was structurally doomed).
*   **Size-scaled base timeouts.** `_resolve_effective_timeout` now adds 5s per KB above a 4KB threshold (capped at +600s), with the mode multiplier layered on top of `(base + bonus)`. Per-peer override via `timeout_per_kb_chars` (set to 0 to disable). A 26KB prompt in consensus mode now gets 700s instead of 480s.
*   **Proportional terse-retry budget.** New `_terse_retry_budget(original)` returns `min(max(original * 0.4, 30), 120)` — floor 30s, ceiling 120s, 40% in between. Replaces the legacy fixed 60s constant which was structurally unlikely to succeed when the original timeout was 240s+. Retry runs with `timeout_per_kb_chars: 0` to avoid double-scaling. The failure-annotation suffix now names the real budget, not the legacy 60s.
*   **Idle-read timeout (opt-in).** New per-peer `idle_timeout: float | None` field. When set, `_run_cli_once` switches to a streamed read loop with per-stream idle deadline; the peer is killed when no stdout/stderr arrives for N seconds (in addition to the wall-clock cap). Default OFF for all built-in peers since most CLIs (claude `-p`, codex `exec`, agy `--print`) buffer output rather than stream. Operators with a known-streaming CLI can opt in.
*   **Missing-key peer pre-drop.** Hosted peers (openrouter / openai_compatible) whose `api_key_env` env var is unset are now removed from the run BEFORE preflight, with a `peer_missing_api_key` progress event + top-level `missing_key_peers` metadata field. Crucially, they do NOT count toward the quorum denominator — a missing key is an operator configuration gap, not a council failure that should degrade an otherwise-healthy run.
*   MCP schema bumped to v5. Top-level `missing_key_peers` field added. Cache schema unchanged.

## 0.11.8 - 2026-05-23

v0.11.8 fixes two correctness bugs and one false-negative gap in the v0.11.6 quota fallback work, surfaced by the v0.11.7 dogfood council review.
*   **Retry chain cost ceiling restored.** When the quota-fallback retry fails, `run_cli_participant` now returns early instead of falling through to launch-retry / label-repair / section-repair / strict-evidence branches. Previously, a fallback that returned (e.g.) an unlabeled response would trigger a 3rd `_run_cli_once` call against the ORIGINAL `cfg` — pointing back at the overloaded model AND violating the documented "one extra call per peer per round" budget. Council finding (`adapters.py:751-872`).
*   **Quota regex hardening.** `QUOTA_EXHAUSTED_PATTERNS` is now case-insensitive across all entries and covers shapes the previous set silently missed:
    *   Google Python SDK: `ResourceExhausted` (PascalCase exception class), `Resource has been exhausted` (natural-language).
    *   OpenAI natural-language: `You exceeded your current quota, please check your plan and billing details.`
    *   `rate limit exceeded` with spaces (previously only the underscore form matched).
    *   Bare-429 window widened from 40 to 60 chars + `exhausted` added to the neighbor-word group, so `429 Resource has been exhausted (e.g. queries per minute limit was exceeded)` matches.
*   Defense in depth: the merge-function field-loss the council also flagged (`_merge_cli_retry` / `_merge_cli_section_retry` / `_merge_hosted_*` discard `model_fallback_used` and `recovered_after_quota`) is intentionally **not** patched in this release because the early-return fix above makes those code paths unreachable from a fallback-stamped result. Adding 8 defensive copies would be dead-code today; will revisit if the retry flow ever permits multi-step chain walking.

## 0.11.7 - 2026-05-23

v0.11.7 lands Phase 3 of quota resilience: visibility in `stats --reliability`.
*   `llm-council stats --reliability` now surfaces two new per-peer counters: `quota_incidents` (every quota wall hit, including recovered ones) and `quota_recoveries` (the subset rescued via `fallback_chain`). Derived `quota_recovery_rate` exposed in the JSON payload; text formatter renders `<recovered>/<incidents> (<pct>%)` under new `quotaInc` / `quotaRec` columns.
*   A peer with ONLY quota-incident signal (no operator outcomes, no VERIFIED citations, no rank data) now appears in the reliability table — previously dropped as "no signal". So a quota-throttled antigravity shows up immediately, no `--participant` flag needed.
*   Counters are mechanical from transcripts (no operator labeling needed) — same pattern as `verified_citation_rate`. A recovered call counts as 1 incident + 1 recovery; a hard-failed call counts as 1 incident + 0 recoveries.
*   CLAUDE.md documents the honest gap: per-CLI token usage is NOT observable from outside the CLI (CLIs auth as the user, burn the user's quota, expose no metering hook). Only OpenRouter peers expose real `prompt_tokens` / `completion_tokens` / `cost_usd`. Quota incident counts are the observable proxy for CLI peers.

## 0.11.6 - 2026-05-22

v0.11.6 lands Phase 2 of quota resilience: actual fallback retries, not just detection.
*   New participant config `fallback_chain: list[str]` — ordered model IDs to step down to on quota errors. Default chains shipped for `claude` / `claude_4_6` / `claude_4_7` → `claude-sonnet-4-6`, `codex` → `gpt-5-mini`, `gemini` → `gemini-2.5-flash`. Antigravity stays empty (no `--model` flag).
*   **Claude**: `_build_cli_command` auto-injects `--fallback-model <chain[0]>` into the CLI args — Claude's native overload handling kicks in inside the CLI. llm-council-level retry is skipped for Claude to avoid double-paying the peer.
*   **Codex / Gemini / any non-Claude CLI**: on `quota_exhausted` detection, the adapter retries ONCE with the next-in-chain model substituted into `cfg.model`. Success stamps `recovered_after_quota=True` + `model_fallback_used=<id>` on the result.
*   New top-level field `quota_recoveries: [{peer, family, fallback_model, model}]` on transcript + MCP `structured_results`, disjoint from `quota_throttled_peers`. A peer is in exactly one list per run, keyed on its final state.
*   New progress event `peer_quota_recovered` emitted in real-time, deduped per peer across rounds.
*   New stats counter `quota_recoveries` per peer; together with `error_kind_counts.quota_exhausted` it tells the operator how often a peer absorbs vs. survives quota incidents over time.
*   New per-result fields `model_fallback_used: str | None` and `recovered_after_quota: bool` surfaced in transcript JSON, MCP `structured_results`, and cache rehydrate.
*   MCP `COUNCIL_RUN_OUTPUT_SCHEMA_VERSION` bumped to v4 (`quota_throttled_peers`, `quota_recoveries`, per-result fallback fields). Cache schema unchanged (new fields are write-only optional with default-on-missing reads).
*   Antigravity is intentionally a no-op for Phase 2: `agy` has no `--model` flag, so the peer still drops with `quota_throttled_peers` (Phase 1 signal) — the calling agent's responsibility to re-run later.

## 0.11.5 - 2026-05-22

v0.11.5 adds quota-exhaustion detection across CLI and hosted peers (Phase 1 of resilience work).
*   New `error_kind=quota_exhausted` classifies known rate-limit / quota signals (`RESOURCE_EXHAUSTED`, `quota_exceeded`, `rate_limit_exceeded`, `insufficient_quota`, `insufficient credits`, `usage limit`, `5-hour limit`, contextual HTTP 429) instead of falling through to `cli_nonzero_exit` / `downstream_error` / `unknown`.
*   New top-level field `quota_throttled_peers: [{peer, family, model, message}]` on transcript JSON and MCP `structured_results`, lifted from metadata. Omitted entirely when no peer was throttled (common case).
*   New progress event `peer_quota_throttled` emitted per peer; deduplicated across rounds so a peer throttled in round 1 doesn't re-emit in round 2.
*   `stats.aggregate.error_kind_counts.quota_exhausted` now visible per peer without any stats changes (the bucket auto-populates from the new error_kind).
*   No auto-fallback yet — that's Phase 2. For now the peer drops from quorum like any other failure; the new surfacing makes the cause visible to the calling agent.

## 0.11.4 - 2026-05-22

v0.11.4 makes `llm-council doctor` self-heal a stale OpenRouter catalog.
*   `doctor` now refreshes a missing/stale catalog inline (10s timeout, best-effort) instead of nagging users to run `llm-council models refresh` manually.
*   Fail-soft: a network failure during auto-refresh falls through to the previous stale-warning Check with the underlying error appended.
*   New default `defaults.catalog_auto_refresh: true` (override to `false` to restore the prior manual-refresh-required behavior).
*   Reminder: the catalog check still only fires when at least one OpenRouter participant is configured, so CLI-only users never see catalog warnings (unchanged).

## 0.11.3 - 2026-05-21

v0.11.3 introduces dynamic stance balancing and robust diagnostics for missing tools.
*   Implemented dynamic stance balancing in consensus mode so that debate roles (for, against, neutral) remain evenly distributed when participants are filtered or excluded.
*   Added unit test suite verifying stance balancing across multiple participant count scenarios (N=2, N=3, etc.).
*   Added a clean startup error when neither `antigravity` nor `gemini` CLI is present on PATH when running quick triad modes.

## 0.11.2 - 2026-05-20

v0.11.2 rewrites the README.md to feature Google Antigravity CLI and SDK as first-class citizens.
*   Documented integration points linking to the official [antigravity-cli](https://github.com/google-antigravity/antigravity-cli) and [antigravity-sdk-python](https://github.com/google-antigravity/antigravity-sdk-python) repositories.
*   Clarified the dynamic triad selection behavior (exactly 3 active CLIs) and the fallback/prioritization rules.
*   Documented Antigravity CLI's native support for Claude models (Claude Sonnet & Claude Opus) and how `llm-council` family exclusions prevent redundant voting when using Antigravity CLI as the primary driver.

## 0.11.1 - 2026-05-20

v0.11.1 refines participant logic to dynamically select exactly 3 CLI peers for the quick-select triad (`tri-cli`).
*   Configured the `other_cli_peers` strategy to dynamically choose between `antigravity` and `gemini` based on PATH, resolving to `antigravity` if both are installed.
*   Updated setup verification, auto-preset routing, and next-steps logic to treat `antigravity` and `gemini` as a single slot in the triad.
*   Added friendly warning/recommendation in diagnostics (`doctor`) and setup to prompt users to upgrade to `antigravity` if only `gemini` is installed.

## 0.11.0 - 2026-05-20

v0.11.0 integrates the Antigravity CLI (`agy`) as a native participant and primary driver.
*   Added `antigravity` to baseline CLIs and registered default participant configs.
*   Updated `detect_current_agent` process detection and environment variable parsing.
*   Implemented model-family selection exclusion rules (excluding `gemini` if driver is `antigravity`, and vice versa).
*   Updated the setup wizard, diagnostics (`doctor`), and README guidelines.

## 0.10.2 - 2026-05-18

v0.10.2 fixes three envelope-parser bugs surfaced when the council
reviewed itself, plus the four pre-existing test failures that had
been carried since v0.10.0 (two real test-staleness bugs, two CI/env-
specific). 1041 → 1048 passing tests on the venv suite; CI now green.
No production code changes for the test fixes; the parser fix is the
only behavior change. Cache schema v3 unchanged. No new dependencies.

**Envelope parser correctness (`adapters.py`).** Three bugs in
`_extract_response_envelope` mangled `structured_results` while
leaving the raw markdown intact, so MCP clients reading the parsed
envelope got garbage even though humans reading transcripts didn't
notice:

1. `RISK:` shared the single-word enum value pattern with
   `EFFORT:` / `CONFIDENCE:` / `CONTINUE_DEBATE:`, truncating
   sentences like "The single biggest risk is external-contract
   drift…" to `"the"`. Split into `_ENVELOPE_ENUM_RE` (closed-vocab
   fields) and a dedicated `_ENVELOPE_RISK_RE` that captures
   rest-of-line preserving case and tolerates trailing `**`
   markdown emphasis.
2. The inline list form (`EVIDENCE: a, b, c` on one line) treated
   the whole post-colon string as a single item. Real peers emit
   multiple `[VERIFIED:path:start-end]` cites or pytest commands on
   one line per the prompt contract in `context.py`. Now split on
   comma; per-line bullet form is unchanged.
3. `BLOCKERS: none` stored `["none"]` (truthy), defeating abdication
   detection — a peer with `EFFORT: blocked` + `BLOCKERS: none` +
   `ASSUMPTIONS: none` was treated as a real vote instead of an
   abdication. Added `_LIST_NONE_SENTINELS` so `none`/`n/a`/`-`/`—`
   normalize to no entries. Concrete fallout: abdication now fires
   correctly when both list fields hold only the sentinel.

All three were dogfood-surfaced when the council reviewed itself:
codex and gemini both emitted comma-separated `[VERIFIED:...]` cites
that came back as `text=", , , ,"` with only the first cite's
metadata, and their sentence-form `RISK:` values came back as a
single word.

**Pre-existing test fixes.** Four failures that had ridden along
since v0.10.0 — three real test-staleness bugs and two CI-env-
specific gaps in shutil.which mocking:

- `test_local_only_mode_picks_up_default_ollama_participant` called
  `load_config(None)` which walks up cwd and merges the developer's
  project-level `.llm-council.yaml`. Devs with local-server peers
  (vLLM / llama.cpp) saw `local-only` resolve to multiple
  participants instead of the single Ollama default the test
  asserts. Switched to `load_config(None, search=False)`.
- `test_council_run_emits_summary_markdown` declared an openrouter
  `cheap` participant without inline pricing. `enforce_mcp_budget`
  rejects paid hosted peers that lack `input_per_million` AND have
  no cached catalog entry, so the test tripped a budget violation
  before reaching the summary_markdown assertion. Pinned inline
  pricing on the fixture.
- `test_mcp_doctor_returns_serialized_checks` asserted exact dict
  equality, but `run_doctor` now returns `config_warnings` so MCP
  clients can surface the same advisory the CLI prints. Added
  `"config_warnings": []` to the expected dict.
- `test_setup_interactive_uses_preset_and_suppression_flags` and
  `test_setup_yes_uses_preset_and_suppression_flags` gated on the
  `tri-cli` preset, which requires 2+ of claude/codex/gemini on
  PATH. CI runners have none, so `_preset_status` blocked setup.
  Added the same `cli_module.shutil.which` mock the adjacent passing
  setup tests already use.

## 0.10.1 - 2026-05-18

v0.10.1 fixes two correctness bugs the council surfaced when dogfooding
v0.10.0 — both verified against source by claude + gemini. 1035 → 1037
passing tests (+2 fix-validating); same 3 pre-existing environmental
failures unchanged. Cache schema v3 unchanged. No new dependencies.

Heads-up: the v0.10.0 MCP progress notifications dogfooded green in
the council loop (the new code paths execute and pytest covers the
shape), but Claude Code did NOT visibly render the
`notifications/progress.message` text in the host UI during the
dogfood run. **Confirmed via post-merge investigation: Claude Code does
not currently implement `notifications/progress` rendering.** Tracked
upstream at
[anthropics/claude-code#4157](https://github.com/anthropics/claude-code/issues/4157)
(open) and #3174 (closed as "not planned" — covers the related
`notifications/message` channel with the same UI-silence root cause).
Other MCP hosts (Claude Desktop, Claude web, third-party clients) DO
render the notifications correctly. Our implementation is spec-
compliant; the gap is in Claude Code's feature roadmap. When #4157
ships, v0.10.0 lights up in Claude Code automatically — no code
changes required. The v0.10.1 fixes ship regardless because they're
real correctness bugs even when no host renders the notifications.

**asyncio orphan-task GC risk.** `asyncio.create_task` is what bridges
the orchestrator's sync `emit()` to async
`session.send_progress_notification`. The event loop only holds **weak
references** to tasks (CPython docs warning), so without a strong ref
the GC can collect an in-flight task mid-await and the notification
disappears silently. Fix: closure-local `_pending_tasks: set` keeps
strong refs until each task's `done_callback` discards itself
(`mcp_server.py:765-789`). New burst test fires 50 advancing events,
forces `gc.collect()` mid-burst, asserts all 50 notifications were
delivered. Surfaced by gemini + claude in the v0.10.0 dogfood council.

**Off-by-one in `planned_total` when preflight fails.** Local
participants (Ollama) run a preflight ping before the orchestrator's
work begins; peers that fail preflight are stripped from `run_targets`
(`orchestrator.py:383-397`) and never reach `participant_finish`. The
total stays `peers * rounds + 1`, so a 4-peer run with 1 preflight
failure goes 0/9 → 3/9 → suddenly 9/9 at `council_finish`. Fix: added
`preflight_failed` to `display.PROGRESS_ADVANCING_EVENTS` so those
peers tick the counter once (`display.py:237-251`). Counter-advance
logic also moved BEFORE the `message is None` early-return in the
MCP callback (`mcp_server.py:792-801`) — minor cleanup that makes the
two concerns (does this event advance? does this event emit a
message?) independent, which they should always have been. Surfaced
by gemini in the v0.10.0 dogfood council.

## 0.10.0 - 2026-05-18

v0.10.0 ships two coupled visibility features: MCP progress
notifications so host agents see mid-run progress on the existing
`council_run` tool call, and a brand-identity layer that makes council
output unambiguous in a stream of regular agent chatter. Comes out of
the Symphony-research pass — three candidates were shelved (stall
detection, `PROMPT.md` overlay, `council_active_runs` sidecar), and
this is the one that landed cheaply by extending `display.py`'s
existing brand affordances rather than introducing new state. 1035 →
1053 passing tests (+18 net new); same 3 pre-existing environmental
failures unchanged. Cache schema v3 unchanged. No new dependencies.

**MCP progress notifications.** New `_build_mcp_progress_callback` in
`mcp_server.py:744-799` bridges the orchestrator's sync `emit` callback
to async `session.send_progress_notification` via
`asyncio.create_task` — fire-and-forget so a slow client cannot wedge
the council run; transport errors swallowed for the same reason. In
`call_tool`, `app.request_context.meta.progressToken` is captured and
threaded into `run_council` (`mcp_server.py:1493-1517`); silent no-op
when the client did not set a token. Progress fraction:
`completed_peer_runs / (peers * effective_rounds + 1)`, the `+1`
reserving headroom for synthesis or cross-rank; clamps to total on
`council_finish`. Replaces the rejected `council_active_runs` sidecar
design (one MCP call now surfaces mid-run progress; no second call,
no on-disk state, no GC, no CLI subcommand).

**Event-to-notification mapping.** Only "interesting" events emit
notifications (`display.format_progress_message`); `participant_start`
is suppressed (per-peer noise multiplier when N peers fire
concurrently; `participant_finish` is the visible signal), along with
`images_skipped`, `truncated_for_deliberation`,
`deliberation_skip_participants`, `convergence`, and
`context_files_chunked`. 12–14 notifications per 4-peer 2-round run.
The `PROGRESS_ADVANCING_EVENTS` frozenset enumerates the three events
that advance the counter: `participant_finish`, `cross_rank_complete`,
`synthesis_finish`.

**Brand identity token.** `display.BRAND_TOKEN = "LLM Council"` plus
`BRAND_SEP = " · "`. Every progress message is plain-text prefixed
`LLM Council · …` — no ANSI (hosts strip), no emoji (font-fallback
risk on macOS Terminal default + CI logs), no markdown bold (some
hosts render `**` as literal in progress messages). Matches the
existing `**LLM Council**` header in `render_summary_markdown`
(`display.py:205`) so CLI and MCP say the same word. Plain ASCII is
greppable and survives every rendering path.

**Per-peer color accent (CLI).** New `PEER_ACCENT_PALETTE`
(cyan/magenta/yellow/green/blue/red) rotated deterministically by
roster index in `display.peer_accent()`. CLI `_print_progress_event`
becomes `_make_progress_printer(participants)` — a factory that
closes over the roster so the sync `progress` callback contract stays
single-argument while still giving deterministic per-peer color.
Custom CLIs defined in `.llm-council.yaml` slot into the cycle by
roster position. `format_gutter` gains an optional `token_color`
parameter (`display.py:113-138`); default bold-cyan gutter applies
when no override is provided or when the peer is not in the roster
(stranger peer fallback). MCP `message` field stays plain text — per-
peer color isn't expressible in one-line notifications.

**`LLM_COUNCIL_QUIET=1` opt-out.** Single env switch suppresses (a)
all MCP progress notifications (treated as if no `progressToken` was
sent) and (b) all CLI gutter colorization. Layout still prints under
QUIET — accessibility and pipe-friendliness are the goals, not
silence. Env-only because MCP servers have no per-call CLI flags;
parity with the existing `NO_COLOR` honoring in `display.wants_color`.

**Honest gaps.** Claude Code's exact rendering of
`notifications/progress.message` is unverified; the plain-text-only
design is the safe-under-uncertainty choice and works under
worst-case "strip everything but text" rendering. If Claude Code
turns out to render markdown in progress messages, future patches can
add bold under the same token without breaking older renderers.

**Test surface.** New `tests/test_display_branding.py` (+18 tests):
peer-accent determinism + palette wrap-around, `wants_quiet` truthy
/ falsy parsing, message prefix on every interesting event,
suppression of `participant_start` + noise events, no-token /
no-session / quiet-env no-op, monotonic progress + clamp-to-total,
transport-error swallowing, CLI quiet-mode strips ANSI but preserves
gutter layout, CLI per-peer accent matches palette, stranger peers
fall back to `ANSI_GUTTER`.

## 0.9.0 - 2026-05-18

v0.9.0 ships four items driven by the post-v0.8 competitor-comparison
pass (karpathy/llm-council, massgen/MassGen, blueman82/ai-counsel —
clones at `/development/projects/reference/`): an MCP transcript-search
tool, opt-in tool-call voting, a one-line Phase-5b serialization fix the
dogfood pass caught, and an anonymized cross-ranking flag composable
with any mode. 935 → 1017 passing tests (+82 net: +91 added across
three new test files, -9 from a conservative cleanup pass); same 3
pre-existing environmental failures unchanged. Cache schema v3
unchanged.

**`council_query_transcripts` MCP tool.** Semantic search over
`.llm-council/runs/*.json`. New module `llm_council/query.py` with
`SimilarMatch` dataclass and `search_similar()`
(`query.py:30-38,99-161`). Reuses `stats.load_transcript_files`,
`convergence.tokenize`/`jaccard_similarity`, and the fence-aware
`deliberation.recommendation_label`. Returns top-k matches with
`(run_id, similarity, question_excerpt, recommendation_label,
timestamp)`; timestamp parsed from the run-id `YYYYMMDD_HHMMSS` prefix.
Wired into MCP at `mcp_server.py:1351,1366,1460,1496` — canonical
surface is MCP only; no CLI subcommand. NO new dependencies (Jaccard
MVP — sentence-transformers deferred until Jaccard demonstrably
insufficient). Scope-cut: `find_contradictions` and `trace_evolution`
deferred to v0.9.x. Dogfood-verified: 5 matches returned for a
"v0.8 closed-loop measurement pipeline" query
(`.llm-council/runs/20260518_044924_*`). New test file
`tests/test_query_transcripts.py`. Inspired by ai-counsel's
`query_decisions` MCP tool.

**Tool-call voting (opt-in).** Strictly opt-in via `tool_call_voting:
true` on the `review-with-tools` mode (default `false`,
`defaults.py:397`). When enabled, the
`record_recommendation(verdict, blockers, evidence)` tool-call schema
is appended to the per-peer directive; the orchestrator runs a unified
`_extract_tool_call_recommendation` parser and falls back to the
existing regex `RECOMMENDATION:` parsing when no structured payload is
present. No family-specific extraction code yet — no real CLI
tool-call payloads to validate against, deferred until concrete shapes
appear. New `tool_call_status` field on `ParticipantResult`
distinguishes `absent` / `ok` / `malformed` / `None` so parser bugs
become operator-visible instead of silently masking. Cache round-trip
preserved (the new field rehydrates to `None` on absence; schema
version unchanged). Orchestrator wiring at
`orchestrator.py:416-418,434,593`. Dogfood-verified: with the flag
flipped via a temp yaml override, all three CLI peers reported
`tool_call_status: "absent"` (extraction ran, found no tool call,
regex fallback succeeded) — confirms the no-op safety path end-to-end
(`.llm-council/runs/20260518_052232_*`). New test file
`tests/test_tool_call_voting.py`.

**Phase 5b (dogfood-caught fix).** Phase 5 set `tool_call_status`
internally but never serialized it — same class of latent bug as
v0.8.1's Phase 1b verified-tag schema gap. `result_to_dict`
(`transcript.py:367-370`) and `council_run_output_schema`
(`mcp_server.py:344,1109`) did not include the field, so the operator
couldn't see whether extraction ran. Two-line fix on each surface;
caught precisely because the v0.8.1 lesson ("dogfood the new surface
through MCP immediately after restart") was followed.

**Anonymized cross-ranking (flag, not mode).** New `--cross-rank` CLI
flag (`cli.py:244-249,2644`) and `cross_rank: true` MCP arg
(`mcp_server.py:204-212,1042`); composable with any existing mode (NOT
a new mode — avoids mode proliferation). After round 1, builds a stable
anonymization map `{peer: "Response A|B|C…"}` (Excel-column style for
>26 peers), constructs per-peer ranking prompts with the OTHER peers'
outputs relabeled, runs the ranking pass via existing
`run_participants`, parses `FINAL RANKING:` numbered lists, aggregates
per-peer `rank_position_mean`. Orchestrator wiring at
`orchestrator.py:310,510-632`. Surfaces in transcript JSON top-level
(`cross_rank_scores` + `anonymization_map` + reverse map +
`cross_rank_rankings`; `transcript.py:918-977`), transcript markdown
(`transcript.py:856-877`), MCP `structured_results`
(`mcp_server.py:518,1188-1204`), and `stats.aggregate_reliability` as
a new per-peer counter (`stats.py:630-632`). New `is_ranking_round`
field on `ParticipantResult`; ranking-round results are persisted and
cached but explicitly EXCLUDED from the round-2 deliberation prompt
builder (`transcript.py:470`) — mirrors the v0.8 finding-matrix
invariant (MAD literature, arxiv 2402.18272). Promotion gate in
`eval/runner.py:402-484` accepts an optional
`cross_rank_correlation_floor` for future eval-data-driven default
flip. Dogfood-verified: a `consensus + cross_rank: true` run produced
`cross_rank_scores: {claude: 1.0, codex: 1.5, gemini: 2.0}` with the
anonymization map persisted to metadata
(`.llm-council/runs/20260518_044924_*`). New test file
`tests/test_cross_rank.py`.

**Test cleanup pass.** 9 tests deleted across 8 files (tautologies,
type-system-checks-dressed-as-tests, byte-identical duplicates,
default-value-only checks). Conservative scope — `test_llm_council.py`
(357K, 437 tests) intentionally untouched; a deeper consolidation pass
is deferred. Flagged but did NOT change:
`llm_council/eval/runner.py:392` — `PromotionResult.to_dict()` is just
`return asdict(self)` and adds no value over calling `asdict` directly;
defer to v0.9.x. Test surface: 935 (v0.8.1) → 1017 (v0.9.0) = +82 net.

**Operational gotcha (carry forward).** The v0.7.1 MCP-server-
restart-after-install warning still applies. All four v0.9.0 surfaces
(`council_query_transcripts`, tool-call voting, Phase 5b
serialization, cross-rank) require an MCP-server restart before
MCP-mediated councils pick them up. Phase 5b was caught precisely
because the post-Phase-5 restart was the first moment the MCP server
emitted `tool_call_status` over the wire; dogfood through MCP
immediately after restart to catch this class of issue.

## 0.8.1 - 2026-05-17

v0.8.1 ships three items: hash-aware chunking for `context_files`
(matching `--diff`'s existing pattern); a latent v0.8.0 MCP-schema bug
fix (the new `[VERIFIED:...]` tag was not in the schema's tag enum, so
any MCP-mediated council where a peer emitted a verified citation
failed output validation); and a new optional
`CONTINUE_DEBATE: yes|no` envelope tag that lets peers vote to skip
round-2 deliberation. 906 → 935 passing tests (+29 across two new test
files); same 3 pre-existing environmental failures unchanged.

**`context_files` chunking.** Today's planning-pass dogfood surfaced
that passing many real files trips the 120K per-participant prompt cap
(the original failure was
`Prompt exceeds max_prompt_chars: 375023 > 120000` with 9 context
files). `--diff` already handles this via hash-aware chunking in
`llm_council/diff_chunking.py`; `context_files` did not. v0.8.1 routes
them through the same chunker. New public function
`chunk_context_files()` reuses the existing scoring helpers (filename
mentions, extension affinity, smaller-first tiebreak). Oversize-alone
files — a single file larger than `max_prompt_chars - framing` — are
dropped entirely with a `context_files_chunked` progress event listing
`oversize_files`. Dogfood-verified: the original 375K-char payload
chunks to ~104K and runs cleanly
(`.llm-council/runs/20260517_170455_dogfood-test-1-*`). The behavior
change inverts an old fail-fast test
(`test_long_context_overflow_fails_fast_instead_of_truncating` renamed
and rewritten to assert loud chunking events, not silent truncation).
The chunker-budget test
`test_build_prompt_hash_aware_drops_unrelated_files` was bumped 8K → 9K
chars to leave headroom for natural envelope growth. New test file
`tests/test_context_chunking.py` (14 tests).

**MCP schema accepts `verified` tag.** v0.8.0's Phase A
`[VERIFIED:path:start-end]` citation parsing produces evidence entries
with `tag="verified"` and accompanying `path`/`start_line`/`end_line`/
`verified` fields, but `council_run_output_schema` in `mcp_server.py`
(around line 385) did not include `"verified"` in the evidence-tag
enum. Any MCP-mediated council where a peer emitted a verified
citation crashed with
`'verified' is not one of ['published', 'observable', 'inferred', 'speculative', None]`.
The bug was latent during v0.8 development because the MCP server was
still on v0.7.1 code; it surfaced the first moment the operator
restarted into v0.8 (dogfood test 1). Fixed by adding `"verified"` to
the tag enum and the four accompanying optional structured properties
(`path`, `start_line`, `end_line`, `verified`); description string
updated to document the new shape. Carry-forward to v0.9.0: continue
to test new envelope features through the actual MCP path during
dogfood, not only via pytest.

**`CONTINUE_DEBATE: yes|no` envelope tag (Feature 4 from the
post-v0.8 competitor-comparison pass).** A new optional envelope field
peers may emit alongside `EFFORT`/`CONFIDENCE`/`RISK`. New regex in
`_ENVELOPE_SINGLE_RE` (`adapters.py:2316`); new field on
`ParticipantResult` (`adapters.py:160-169`); unanimity gate in
`orchestrator.py:489-525`; one-line envelope-doc addition in
`context.py:395`. When **all** label-producing peers in round 1 emit
`CONTINUE_DEBATE: no`, the orchestrator skips the optional round-2
deliberation and stamps
`deliberation_status: skipped_continue_debate_unanimous` plus a
`deliberation_skipped` progress event carrying `no_votes` +
`denominator` counts. Denominator mirrors the existing label-producing
semantics: abdicated peers, `error_kind=invalid_response`, and peers
without a usable `RECOMMENDATION` label are excluded. The unanimity
threshold (not 66%) is deliberately conservative — relaxation
deferred to v0.9.x once a transcript corpus exists to audit gaming
risk. Cache round-trip preserved (the new field on `ParticipantResult`
is persisted; schema version unchanged because absence rehydrates to
None). Dogfood-verified: a deliberate-mode council with all three
peers emitting `CONTINUE_DEBATE: no` triggers
`deliberation_status: "skipped_continue_debate_unanimous"` +
`deliberation_skipped` event with `no_votes: 3, denominator: 3`
(`.llm-council/runs/20260517_170522_dogfood-test-2-*`). New test file
`tests/test_continue_debate.py` (15 tests). Inspired by ai-counsel's
`continue_debate: bool` per-vote field; full competitor-comparison
context lives in `reference_council_projects.md` in the auto-memory.

**Operational gotcha (carry forward).** The v0.7.1 MCP-server-
restart-after-install warning still applies. Both v0.8.1 surfaces
(chunking, `CONTINUE_DEBATE`) require a restart before MCP-mediated
councils pick them up. The Fix B bug above was caught precisely
because the restart-after-v0.8.0 was the first moment the MCP server
saw verified citations; dogfood the new surface through MCP
immediately after restart to catch this class of issue.

## 0.8.0 - 2026-05-17

v0.8.0 ships a closed-loop measurement pipeline. The keystone is the new
`[VERIFIED:path:start-end]` evidence tag with mechanical verification;
it generates the signal that powers both an eval harness with
Signal-to-Noise Ratio (SNR) tracking and a minimal per-peer reliability
layer. Architecture direction set by a council-on-itself
meta-consultation (transcript
`20260517_072537_meta-consultation-llm-council-product-roadmap-you-are-being.md`)
plus a 2024–2026 literature review (citations below). 745 → 906 passing
tests (+161); same 3 pre-existing environmental failures unchanged.
Cache schema stays at v3 — the new `evidence_verification_failures`
field defaults gracefully via `payload.get(...) or []` on rehydrate.

**Closed-loop pipeline (Parts 1, 4, 7).** Three features that only
deliver as a bundle:

- *Verified citations.* `[VERIFIED:path:start-end]` joins the existing
  `[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE]` tag set. New
  module `llm_council/citations.py` (`VerifiedRef`,
  `parse_verified_tag`, `verify_ref`, `verify_evidence_citations`);
  `adapters._parse_tagged_entry` extended to recognize the new tag;
  `orchestrator.execute_council` calls `verify_evidence_citations`
  after every round (`orchestrator.py:424,547`). Failed refs are
  recorded on `ParticipantResult.evidence_verification_failures` but
  the entry is NOT dropped — coverage > filtering, per Anthropic Claude
  4 prompting guidance. Prompt directive added in `context.py`
  envelope block asks peers to prefer the tag and explicitly states
  unverifiable cites are kept, not dropped.

- *Eval harness.* New `llm_council/eval/` package — `metrics.py`
  (`blocker_recall`, `false_blocker_rate`, `citation_accuracy`,
  `evidence_density`, `signal_to_noise_ratio`), `runner.py`
  (`load_fixture`, `run_suite`, `to_json`, `check_promotion_gate`),
  bundled minimal `fixtures/` directory. New CLI surface:
  `llm-council eval run [--mode <m>] [--fixtures <path>] [--out <path>]
  [--require-cached] [--compare-against <baseline.json>]`. SNR metric
  matches the CR-Bench convention (true-positive findings / total
  findings emitted). The seed fixture set is intentionally minimal —
  building a real eval suite is a separate ongoing effort. Scorecards
  land under `.llm-council/eval-runs/` for trend aggregation via
  `llm-council stats --eval`.

- *Outcome tracking + per-peer reliability.* New module
  `llm_council/outcomes.py` — `OutcomeRecord` persisted as sidecar
  `.llm-council/outcomes/<run-id>.json` so transcript JSON shape stays
  immutable. New CLI:
  `llm-council outcome mark <run-id-or-prefix> --decision
  shipped|reverted|rejected|unknown [--bug-found yes|no]
  [--winning-peer X] [--note "..."]` and `llm-council outcome list`.
  New `llm-council stats --reliability [--peer <name>]` surfaces
  per-peer counters from `stats.aggregate_reliability`:
  `outcomes_marked`, `useful_count` (peer voted `yes|tradeoff` AND
  outcome was shipped+no-bug), `false_blocker_count` (peer voted `no`
  AND outcome was shipped+no-bug — mutually exclusive with the useful
  bucket), `unique_blocker_catch_count`, `verified_citation_rate`
  (mechanical; no user label required). NO IRT-style scoring yet —
  revisit at ≥200 marked outcomes per the council deliberation.

**CLI-tool-use mode (experimental, Phase E).** New `review-with-tools`
mode in `DEFAULT_CONFIG["modes"]` (`defaults.py:381`). CLI peers only
(`other_cli_peers` strategy; hosted peers do not participate).
`experimental: true` surfaced as `[EXPERIMENTAL]` in `llm-council list`
and included in MCP `council_list_modes` output.
`timeout_multiplier: 1.8` (≈432s against the 240s baseline). The
per-peer tool-use directive is applied in `adapters.run_one`
(`adapters.py:2029`), NOT `context.build_prompt` — family info is
per-peer, so hosted peers do not receive the directive even when
explicitly routed to this mode. *Promotion gate*:
`eval/runner.py:check_promotion_gate` requires
`blocker_recall(review-with-tools) ≥ blocker_recall(review) + 0.05`
AND `signal_to_noise_ratio(review-with-tools) ≥
0.85 × signal_to_noise_ratio(review)` before the mode can graduate
from `experimental: true`. CLI flags: `--compare-against`,
`--promotion-recall-lift`, `--promotion-snr-floor-ratio`. SWE-PRBench
(arxiv 2603.26130) finding that more context monotonically degrades
review quality motivates the gate: don't ship a regression.

**Per-finding agreement matrix (Phase F, synthesis aid only).** New
module `llm_council/findings.py` — `Finding`, `FindingCluster`,
`FindingMatrix`, `extract_findings`, `cluster_findings`,
`build_matrix_from_results`, `matrix_to_dict`. Peers may emit an
optional `FINDINGS:` envelope block (`id`, `severity`, `claim`,
`evidence` as a `[VERIFIED:...]` tag). `cluster_findings` mechanically
clusters across peers by overlapping verified line ranges + severity
class; consensus = ≥2 distinct peers with overlapping verified refs.
The matrix is computed ONCE post-deliberation on the final round's
results (`orchestrator.py:666`) — explicitly NOT fed back to peers
during round-2 deliberation, because the MAD literature
(arxiv 2402.18272) warns that in-round convergence forcing depresses
signal-to-noise. Surfaced in transcript markdown (`## Finding
Matrix`), transcript JSON top-level (`finding_matrix`), and MCP
`structured_results` (`consensus_blockers` + `single_peer_concerns`,
omitted entirely when no peer emitted findings — gated to match
transcript JSON precedent). `synthesis.run_synthesis_chair` accepts
`finding_matrix: FindingMatrix | None = None`; non-empty renders
"CONSENSUS BLOCKERS" / "SINGLE-PEER CONCERNS" sections so the chair
weights agreement properly. `None` produces the v0.7.x prompt
unchanged.

**Per-mode model overrides (Phase D, persona-routing replacement).**
New optional `modes.<name>.model_overrides: {peer_name: model_id}` in
`.llm-council.yaml`. Resolution chain:
`participants.<peer>.model` (base) → `tiers.<tier>.<peer>` (existing
`--tier` swap) → `modes.<name>.model_overrides` (highest priority
within a mode). Validated at config-load (`config.py:251`); honored
during participant selection (`config.py:904`). No built-in modes ship
overrides — operators add their own once eval harness signal supports
the affinity. *Cuts*: persona auto-routing (Plan Phase 5) and cascade
routing (Plan Phase 3) were deliberately dropped. PRISM
(arxiv 2603.18507) finds persona prompting is net-negative for
accuracy on knowledge/coding tasks on GPT-4-class+ models; the useful
sub-feature (model affinity per task) is captured by `model_overrides`
without the persona-theatre risk. Published cascade-routing gains are
on code GENERATION, zero on code REVIEW, and the user dropped cost as
a constraint so the headline benefit no longer applies.

**Cleanup pass + breadcrumb fixes.** Renamed `--cache-only` →
`--require-cached` (honest naming — peers still execute; the flag
detects cache misses post-hoc and exits non-zero so CI can gate on
"all fixtures pre-warmed"). Dropped redundant runtime type checks in
`config.select_participants` (`validate_config` already enforces shape
at config-load). Reverted cache schema bump (v3 stays valid; the new
`evidence_verification_failures` field defaults to `[]` on rehydrate
— see `cache.py:24` rationale comment). Fixed
`useful_count`/`false_blocker_count` mutual-exclusivity in
`stats.aggregate_reliability` (`stats.py:636`) — a peer voting `no` on
a shipped+no-bug PR is a false blocker, not useful. Dropped unused
`PeerScore.error` / `.ok` / `.from_cache` fields and
`blocker_recall_mean` from fixture scorecards (populated but never
consumed by aggregators; `from_cache` is now read off raw results in
`_aggregate_fixture`). Gated MCP `consensus_blockers` /
`single_peer_concerns` arrays — omitted from the payload entirely
when no peer emitted findings, matching transcript JSON precedent
(`mcp_server.py:1089`).

**Second cleanup pass (post-council-review).** A critical code-review
pass (transcript `20260517_125529_critical-code-review-...`) surfaced
4 real bugs and 9 dead-code items missed by the first cleanup. Fixed:
`PromotionResult.snr_ratio` infinity sentinel (was conflating
"trivially passed over zero baseline" with "candidate has zero
signal" — now serializes as `None`); duplicate `finding_matrix`
serialization in transcripts and MCP (now top-level only, removed
from `metadata`); a duplicate-branch JSON print in `cli.py`; ambiguous
metric naming between `_aggregate_fixture` and `_aggregate_suite`
documented via docstrings on `SuiteScorecard` / `_aggregate_suite`
enumerating per-key aggregation rules. Deletions: `FindingMatrix.by_peer`
(write-only), `FindingCluster.consensus` (always True), unused
`verify_ref` import in `findings.py`, `Fixture.to_dict` and
`Fixture.path` (no callers), unused `Awaitable` import, two
function-local imports in `adapters.py` hoisted, duplicate
`final_round_results(results)` call collapsed, `check_promotion_gate`
parameters renamed `scorecards_a/b` → `baseline/candidate`,
`_result_field` getattr-shim inlined at its four call sites.

**Known operational gotcha (carries forward from v0.7.1).** The MCP
server is a long-running stdio process. After `pip install -e .`
brings these changes in, the MCP server must be restarted before the
new `eval run` / `outcome` / `review-with-tools` surface is reachable
from MCP-mediated runs. Editable installs do not auto-reload long-
running child processes.

**Key papers cited.** Verify accessibility before relying on:
- arxiv 2402.18272 — Rethinking the Bounds of LLM Reasoning
  (multi-agent debate often hurts vs strong single-agent on objective
  benchmarks).
- arxiv 2603.26130 — SWE-PRBench (no frontier model detects >31% of
  human-flagged PR issues; more context monotonically degrades review
  quality).
- arxiv 2512.12117 — Citation-Grounded Code Comprehension (92%
  accuracy with verified file:line citations vs 14–18pp worse
  uncited).
- arxiv 2603.18507 — PRISM (persona prompting net-negative for
  knowledge/coding accuracy).
- CR-Bench — Signal-to-Noise Ratio for code review (Reflexion-style
  agents collapse SNR 5.11 → 1.95).
- Anthropic Claude 4 prompting best-practices — coverage > filtering.

## 0.7.1 - 2026-05-17

Patch release bundling 15 council-surfaced fixes (12 from pass-8 of
v0.7.0, 3 from pass-9 of the v0.7.1 integration itself). 20 commits
since v0.7.0; +3038/-181 lines. 745 tests pass; same 3 pre-existing
environmental failures unchanged. Cache schema stays at v3 (one new
field, `section_repair_attempted`, defaults to False on rehydrate);
MCP output schema body now matches the v3 version declared in v0.7.0.

**Retry layer correctness (pass-8 fixes #3, #4, #6, #8, #12; pass-9
fixes A, B).** Section-repair retry is now wired into the
`openai_compatible` and `ollama` adapter wrappers in addition to CLI,
with the documented layering `terse-retry → section-repair →
strict-evidence` and each gate capped at one extra call per peer per
round. Strict-evidence retry is wired across all three transports.
Both hosted/local strict-evidence and section-repair retries pass
`retry_cfg["retry_on_missing_label"] = False` to the inner adapter so
the inner's own label-repair branch cannot chain into a third call.
`_merge_*_section_retry` now has a third branch that preserves an
`UntaggedEvidence:` retry result (with a `sections_then_evidence`
themed transcript header) instead of silently discarding it; the
result carries a new `section_repair_attempted: bool` flag that
gates the strict-evidence wrapper so a section-fixed-but-untagged
response cannot trigger yet another retry. `recovered_after_timeout`
and `section_repair_attempted` are persisted through cache so warm-
cache repeats preserve the operator-visible retry receipts.
`classify_error` and `is_timeout_error` now share a single
`_TIMEOUT_PREFIXES` tuple covering all 7 httpx/CLI timeout prefixes;
hosted timeouts that previously misclassified as `downstream_error`
(because `ReadTimeout` was in the downstream-markers list) now
correctly bucket as `timeout` in `timeout_by_prompt_size`.

**Terse-retry visibility (pass-8 fix #12).** v0.7.0's terse-retry
was firing correctly but its failure was invisible — `result.elapsed_
seconds` reported only the original call's time and the failure path
returned the original result with `recovered_after_timeout=False`,
byte-identical to "retry never fired". New `terse_retry_attempted:
bool` field on `ParticipantResult` set True on both success and
failure of the retry. `_annotate_timeout_retry_failure()` helper
appends `TERSE_RETRY_FAILED_SUFFIX` to the error string on failure,
preserving the `Timeout:` prefix so `classify_error` still returns
`timeout`. Suffix names the mitigation lever (raise per-peer
`timeout` in project YAML). New `terse_retry_attempts` stats counter;
`attempts - timeout_recoveries` = silent failures. `prompt_chars` now
populated on every `ParticipantResult` construction site (CLI,
openai_compatible, ollama, OpenRouter label-repair, retry merges,
`_context_overflow_result`, `PromptTooLarge` skip) — terse-retry
merges record `len(original_prompt)` so recoveries land in the same
size bucket as the timeout that triggered them. New
`timeout_recoveries_by_prompt_size` stats bucket.

**Envelope parser correctness (pass-9 fix C).** This is the largest
silent-correctness fix in v0.7.1: `_ENVELOPE_LIST_HEADER_RE` was
anchored on `$`, so the parser rejected EVERY inline `EVIDENCE:
<text>` line (tagged or untagged). Strict-evidence validation had
been a no-op on any peer using inline form since v0.7.0 shipped —
likely also explains why pass-7's celebrated "codex emitted tagged
evidence because of R3 ground rule" appeared to work without
structural enforcement. New `_ENVELOPE_LIST_INLINE_RE` plus inline
branch in `_extract_response_envelope` captures these entries; bare-
header-plus-bullets form still matches first (no regression on the
canonical shape). Verified end-to-end on qwen's pass-9 response:
parser now captures all 20 evidence entries (14 tagged + 6 untagged),
strict-evidence then correctly produces
`UntaggedEvidence: 6 EVIDENCE entry/entries lack a tag` which
triggers the now-hardened repair retry.

**Section validator scope (pass-8 fixes #2, #9).** `REQUIRED_SECTION
_HEADER_RE` now accepts `##` markdown headers, `**bold**` wrappers,
colon separators, and mixed case (was: em-dash + ALL-CAPS only).
Response-side `_section_present` tightened: a literal `PART N` only
counts when header-shaped (line start optionally with `#`/`##`/`**`)
AND no skip-prose verb (`skipped|omitted|was not|did not|unable|not
addressed|see PART|refer to`) appears in an 80-char window before or
after. Title-token paraphrase route unchanged.
`_is_recommendation_part` loosened from exact-list `["RECOMMENDATION"]
` / `["RECOMMENDATION", "COUNCIL"]` to `tokens[0] == "RECOMMENDATION"`
so titles like `PART 6 — RECOMMENDATION AND RATIONALE` are correctly
excluded from section-coverage (the existing label check still
governs them).

**Synthesis + MCP schema (pass-8 fixes #1, #5).** `synthesis.py:
_format_envelope_item` renders evidence dicts as `[TAG] text` (or
bare `text` when tag is None/missing) into the chair prompt, with
str-passthrough for the other envelope fields. MCP
`council_run_output_schema` evidence items now use `oneOf` (v3 dict
form with required `text` and closed-enum `tag`, plus defensive
string fallback) — pass-8's live MCP-validation crash on its own
response is fixed.

**Test reorganization (pass-8 fix #11).** `test_pass7_regression.py`
now loads pass-7 prompt/responses from verbatim fixture files under
`tests/fixtures/` (the `.llm-council/runs/` transcripts are
gitignored). The exercise exposed that one existing test
(`test_pass7_codex_evidence_all_tagged`) was a false witness — the
real codex pass-7 response has zero top-level `EVIDENCE:` envelope;
all four tag kinds appear inline in concept prose. Test reframed to
assert the real shape plus a separate synthetic-untagged-block
anchor for the validator.

**Known operational gotcha.** The MCP server is a long-running stdio
process. An editable install (`pip install -e .`) does not auto-
reload long-running child processes — code changes to
`llm_council/` will not take effect in an MCP-server-mediated
council run until the server is restarted. Symptoms: stale schema
declarations (cause MCP output validation errors on otherwise-
successful runs), missing terse-retry annotation on timeouts. Three
consecutive pass-8 / pass-9 / pass-10 dogfood runs bit on this.

## 0.7.0 - 2026-05-16

Three council-surfaced changes addressing the failure modes the pass-7
research-question run exposed (transcript:
`.llm-council/runs/20260516_100758_*`). MCP output schema bumps to v3;
cache schema bumps to v3.

**Timeout policy** — `defaults.modes` now accept an optional
`timeout_multiplier: float` (consensus 2.0×, deliberate 1.5×, diverse
1.5×; others unchanged). Layered on top of per-participant `timeout`
so users who already raised the base benefit too. When a peer times out
the adapter performs one terse-retry with a fixed 60s budget and the
`TERSE_RETRY_INSTRUCTION` directive appended; success sets
`recovered_after_timeout=True`. New `timeout_by_prompt_size` and
`timeout_recoveries` stats buckets let operators see whether the
multiplier needs raising or chunking is the actual answer.

**Section-coverage validator** — when a prompt contains
`PART N — TITLE (REQUIRED)` headers, peer responses must reference each
required section (literal `PART N` token OR all salient title tokens
within a 200-char window). Missing sections trigger one repair-retry,
then `error_kind=incomplete_response`. New `llm_council/sections.py`
module. Disable via `defaults.require_sections: false` or
`--no-require-sections`. PART 6 (RECOMMENDATION) is skipped — the
existing label check covers it. Pass-7 anchor: gemini's three-bullet
response that v0.6.0 silently accepted is now flagged.

**Evidence tags as a first-class envelope contract** — each EVIDENCE
bullet is parsed for a leading/trailing/inline
`[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE]` tag and stored as
`list[{text, tag}]` on `ParticipantResult`. New `evidence_tag_distribution`
stats bucket. `defaults.strict_evidence: false` by default;
`--strict-evidence` (CLI) or `strict_evidence: true` (MCP) makes
untagged entries fail with `error_kind=untagged_evidence` after one
repair-retry. Optional → required rollout mirrors v0.5.0 envelope.
Other envelope list fields (blockers/assumptions/tests_to_run) stay
plain strings — tag semantics only apply to evidence claims.

Two new `error_kind` values: `incomplete_response`, `untagged_evidence`.
New `ParticipantResult` fields: `recovered_after_timeout: bool`,
`prompt_chars: int | None`. New tests:
`tests/test_pass7_regression.py` (8 tests anchored to the actual pass-7
failure mode), `tests/test_timeout_policy.py` (23 tests covering
Changes 1a-1c), `tests/test_section_coverage.py` (23 tests), and
`tests/test_evidence_tags.py` (20 tests). Plus updates to the existing
envelope test for the new structured evidence shape.

## 0.6.0 - 2026-05-16

Cleanup chore recommended by pass-6 council review. **No behavior change**;
v0.5.2's 594 functional tests still pass with one consolidation noted below.

- Rewrote 17 `Pass-N fix #M` comments across `adapters.py`,
  `orchestrator.py`, `synthesis.py`, `deliberation.py`, `defaults.py`, and
  `cli.py` as declarative invariants. Each comment now explains WHY the
  code is shaped the way it is (the actual contract) instead of pointing
  at a council review number. Existing regression tests anchor the
  intended behavior.
- Reorganized regression tests from per-pass files (`test_pass4_fixes.py`,
  `test_pass5_fixes.py` — both removed) into topic-based files
  (`test_abdication_detection.py`, `test_synthesis_gating.py`). Same
  coverage; one redundant cache-write test collapsed into a single
  end-to-end write-then-rederive test.
- Moved the detailed 0.5.0 / 0.5.1 / 0.5.2 release notes into
  `CHANGELOG_ARCHIVE.md`; `CHANGELOG.md` now keeps one-paragraph
  summaries per release with a link to the archive. CHANGELOG.md shrunk
  by 152 lines.

Test totals unchanged: 593 passed (was 594; the dropped test was
redundant double-coverage of cache abdication writes). Same 4
pre-existing env-related failures.

## 0.5.2 - 2026-05-16

Pass-5 council fixes on v0.5.1: dropped overbroad `repair_retry_recovered`
abdication guard (parse-source strip is the lone correctness mechanism);
reverted v0.5.1 cache-refuses-abdications change (cache-hit re-derivation
already handled it offline); `should_synthesize` now skips when
`universal_abdication` fired; `deliberation_status` only stamped when
`deliberate=True`. Full detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#052---2026-05-16).

## 0.5.1 - 2026-05-16

Pass-4 council fixes on v0.5.0 (6 correctness bugs):
synthesis chair bypasses RECOMMENDATION label validation;
synthesis sees final-round results only;
universal-abdication short-circuits before round 2;
new `_envelope_parse_source` strips original section from repair-retry transcripts;
`EFFORT: blocked` without label is now terminal (no retry).
Plus UX patches: `recommendation_line` placeholder for fenced-only labels,
CLI parity with MCP for `secret_scan` metadata, `synthesis_error` rendering.
Full detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#051---2026-05-16).

## 0.5.0 - 2026-05-16

Three new capability picks driven by three council planning passes:

- **Pick A** — Effort contract + abdication detection: optional response
  envelope (`EFFORT`/`CONFIDENCE`/`RISK`/`BLOCKERS`/`EVIDENCE`/`TESTS_TO_RUN`/
  `ASSUMPTIONS`) parsed from peer output; new `error_kind: "abdicated"`
  for `EFFORT: blocked` without concrete missing artifacts. Cache schema
  and MCP output schema bumped to v2.
- **Pick B** — Synthesis chair (`--synthesize` / MCP `synthesize`): a
  configured chair writes a decision memo (blockers / dissent /
  verification plan) post-deliberation. Chair output is metadata, not
  a vote. `defaults.synthesizer` required (no silent default to avoid
  requester bias).
- **Tier 2** — Secret scanner (`llm_council.safety`): preflight regex
  sweep over the assembled prompt body (AWS/GH/OpenAI/Anthropic/Google/
  Slack/JWT/PEM). Default `secret_scan: warn` with allowlist file.
  Closes the prompt-body credential-leak gap.

Side-fixes: MCP summary table bug (`":round"` filter inverted post-deliberation);
fence-aware label validation tightened across adapters and deliberation;
`stats.aggregate` buckets by `error_kind` and tracks `envelope_field_present`.
Full detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#050---2026-05-16).

## 0.4.9 - 2026-05-07

Council code-review pass on v0.4.3..v0.4.8. Fixes 13 items the council
flagged when reviewing the actual implementation against the design
consensus from the prior two rounds.

### Correctness fixes

- **Local openai_compatible peers now count as $0 in the budget gate.**
  The wizard scaffolds `type: openai_compatible` participants without
  pricing fields; before this fix, `--max-cost-usd` and
  `--max-tokens` would refuse to run a `local-only` mode entirely
  because the budget gate flagged them as "unknown unpriced paid"
  hosted peers. Now `_estimate_participant_row` detects `base_url`
  resolves loopback or RFC1918 and forces the row to
  `input_per_million=0.0`, `output_per_million=0.0`,
  `pricing_source="local"`. Matches the promise in
  `docs/local-models.md`. Direct contradiction fixed.
- **`local-only` mode now refuses runtime `--include` of hosted
  peers.** Previously, `--mode local-only --include claude` would
  smuggle a hosted CLI into the run because `select_participants`
  appended `include` after strategy selection. Now the selector
  hard-fails with a clear error pointing at the offending peers,
  matching the config-time strict-mode posture.
- **MCP error-kind schema now lists `preflight_failed`.**
  `classify_error` returns it, but `COUNCIL_RUN_VALID_ERROR_KINDS` in
  `mcp_server.py` was missing the entry — schema-driven MCP callers
  would reject otherwise-valid tool output.
- **MCP responses now surface `config_warnings`.** `run_council`,
  `run_doctor`, and `estimate_run` all propagate the warning list
  (in `metadata.config_warnings` for `run_council`, top-level
  `config_warnings` for the others). Origin typos that previously
  went silent for MCP-driven runs (Claude Code / Codex / Gemini calling
  council via MCP) now surface where users actually see them.

### Pre-flight refinements

- **Pre-flight defaults to loopback only.** v0.4.6 ran for any
  `is_local_participant` (loopback + RFC1918), which false-failed
  homelab/VPN endpoints with a 1-second timeout. Now `is_loopback_base_url`
  (new helper) gates the default-on path; RFC1918 endpoints opt in via
  explicit `pre_flight_check: true`. Existing loopback users see no
  change. Documented in `docs/local-models.md`.
- **Credentials redacted from preflight error messages.**
  `allow_private: true` skips the embedded-credentials validator, so
  `http://user:pass@127.0.0.1:8000/v1` is permitted. The preflight
  error message now strips `user:pass@` to `***@` before formatting,
  preventing creds from leaking into transcripts and progress events.
- **Synthesized preflight result carries `model`.** Transcripts now
  identify which model was targeted even though no call was made.

### Architecture

- **`_is_local_participant` and `_is_local_base_url` (config.py)
  promoted to public** (`is_local_participant`, `is_local_base_url`).
  Both are imported by `orchestrator.py`, `estimate.py`, and tests —
  the cross-module dependency makes them part of the project's public
  surface in practice. `_normalize_local_openai_base_url` (doctor.py)
  similarly promoted.
- **New `LocalOpenAIProbe` dataclass + `discover_local_openai()`.**
  Replaces the wizard's parsing-of-`Check.detail`-strings approach
  (fragile if the probe's detail format ever changed; lost models
  past the first three due to truncation) with structured records
  carrying `base_url`, full `models` tuple, `ok`, and `detail`.
  `probe_local_openai` keeps its `list[Check]` return for `cmd_doctor`
  but is now a thin wrapper around `discover_local_openai`.
- **DNS resolution for `is_local_base_url` cached.** `getaddrinfo`
  results memoized in a 64-entry process-lifetime LRU cache. Avoids
  repeated syscalls when the same hostname is classified across
  `select_participants` + preflight. Matches the OS resolver cache's
  semantics — host-classification doesn't change mid-process.

### Polish

- `pre_flight_check: false` documented in `docs/local-models.md`
  troubleshooting section, with a `pre_flight_check: true` recipe
  for opting RFC1918 in to the loopback ping.
- TODO comment on `_derive_default_family` naming the keyword list
  for future maintainers when granite/command-r/nemotron/exaone
  emerge as new prominent families.
- Redundant `from llm_council.doctor import probe_local_openai`
  inside `_probe_and_collect_local_participants` removed (already
  imported at module top).

### Tests

- 10 new regression tests pinning the fixes: local participant cost
  classification, --include rejection for local-only, --include
  acceptance of local peers, is_loopback_base_url / RFC1918
  separation, RFC1918 skipped by default, RFC1918 opt-in honored,
  credential redaction, synthesized model field, MCP error-kind
  schema, MCP doctor surfaces config_warnings.

## 0.4.8 - 2026-05-07

### Setup wizard

- New `llm-council setup --probe-local` flag scaffolds local
  OpenAI-compatible participants from running endpoints. Reuses the
  doctor probe (`probe_local_openai`) to discover servers on the
  well-known ports (vLLM, sglang, LM Studio, llama.cpp `--api`, MLX,
  Ollama `/v1`), then for each found endpoint prompts the user to:
  (1) confirm scaffolding, (2) pick a served model from the
  `/v1/models` enumeration (or enter manually if the server doesn't
  list any), (3) confirm participant name and family, and (4) pick an
  origin from a numbered registry (`KNOWN_ORIGIN_STRINGS` + free-text
  "Other"). The participant block is written to `.llm-council.yaml`
  with sensible defaults: `allow_private: true`, `timeout: 360`,
  `read_only: true`, `api_key_env: LOCAL_OPENAI_API_KEY`.
- Auto-derives sensible defaults for the user-prompted fields:
  `family` from a substring match against known model families
  (qwen / llama / deepseek / mistral / gemma / phi / kimi / glm) with
  a first-segment fallback, and `participant name` as
  `local_<family>_<sluggified-model-id>` (e.g.,
  `Qwen/Qwen3.6-27B` → `local_qwen_qwen3_6_27b`).
- The flag is interactive-only; under `--yes` it warns and is treated
  as a no-op (probing requires prompts for origin/family). Without the
  flag, `setup` does no network probing — pure-config command behavior
  is preserved by default, matching the council's "opt-in, not
  surprise scan" recommendation.
- `project_config()` and `write_setup_files()` accept a new
  `extra_local_participants={name: cfg, …}` keyword. The wizard
  uses this to merge probe-scaffolded participants into the
  generated YAML alongside the preset's defaults.
- `_mode_available()` for `local_only_peers` now checks a `has_local`
  signal (true when either `include_local` is set or any
  wizard-probed local participant is present), so the `local-only`
  mode surfaces in the generated config whenever ANY local
  participant is configured — not just when the built-in
  `local_qwen_coder` is included.
- After scaffolding, the wizard prints a one-line reminder to
  `export LOCAL_OPENAI_API_KEY=dummy` (or a real key) before running
  the council, since the openai_compatible adapter requires a
  non-empty Authorization header even for unauthenticated local
  servers.

## 0.4.7 - 2026-05-07

### Adapters

- New `env_strict: true` opt-in flag on CLI participants. When set, the
  subprocess inherits ONLY the names in `_SAFE_ENV_NAMES` (PATH, HOME,
  LANG, TERM, USER, SHELL, XDG_*, …) plus whatever is listed in
  `env_passthrough` — every other env var the parent shell carries is
  dropped, including non-secret ones like `GEMINI_MODEL`,
  `OPENAI_BASE_URL`, `GOOGLE_CLOUD_PROJECT`. Default is `false`
  (sieve mode — current behavior — drops only secret-named vars).
- The motivating case is qwen-code (a gemini-cli fork): if
  `GEMINI_API_KEY` / `GOOGLE_*` / `GEMINI_MODEL` is set in the parent
  shell (e.g., for the council's `gemini` peer), qwen-code auto-detects
  Gemini auth and silently routes there instead of the configured
  OpenAI-compatible backend. With `env_strict: true` plus an explicit
  `env_passthrough` of the OpenAI-compat vars, the child can no longer
  see anything that would mis-route it. Belt-and-suspenders alongside
  `--auth-type openai` (which only fixes the auth-routing leak; the
  model-routing leak is separate).
- Sieve-mode behavior unchanged for all existing CLI participants.
  Existing configs see no change in env exposure unless they explicitly
  opt in via `env_strict: true`.

Validation
- `env_strict` must be a boolean — config validator rejects strings
  like `"yes"` with a clear error.

CLAUDE.md
- Custom-CLI template extended to document `env_passthrough` and
  `env_strict`.

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
