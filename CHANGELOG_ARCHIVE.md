# Changelog archive

Detailed release notes for pre-1.0 versions. The top-level `CHANGELOG.md`
keeps these as one-line headlines for quick scanning; the full
council-review iteration history lives here.

---

## 0.5.2 - 2026-05-16

Pass-5 council review of v0.5.1 (transcript:
`.llm-council/runs/20260516_091154_*`) returned RECOMMENDATION: tradeoff
with one bug introduced by v0.5.1 itself, one regression to revert, and
several polish items.

### Bug fix (introduced by v0.5.1)

- **Repaired-response abdication was masked (pass-5 codex).** v0.5.1
  added `repair_retry_recovered=True` as a skip-abdication guard in
  `_with_envelope`. Combined with `_envelope_parse_source` (which
  already strips the original section), this belt-and-suspenders guard
  meant a *legitimately* abdicating repaired response slipped through
  as `ok=True`. Dropped the guard; parse-source strip alone is the
  correctness mechanism.

### Reverted

- **Reverted v0.5.1 fix #10 (cache refuses abdications) — pass-5 gemini.**
  Refusing the cache write was a cost regression: every repeat run paid
  the peer to re-abdicate. The pre-v0.5.1 behavior was already correct
  because `run_participant` pipes cache hits through `_with_envelope`,
  which re-derives `ok=False` for abdication shapes **offline** with
  zero API cost. The "failed runs are never cached" invariant is
  preserved at the RESULT layer (re-derivation) rather than at the
  cache-file layer. Documented inline.

### Polish

- `should_synthesize` now returns False when `universal_abdication`
  fired, even with an explicit `--synthesize`. Chair input would be
  empty after final-round + ok-only filtering — the merged-blockers
  payload IS the deliverable in that case.
- `deliberation_status="skipped_universal_abdication"` is only stamped
  when `deliberate=True`. Non-deliberative runs keep
  `deliberation_status="not_requested"` so the metadata isn't misleading.
- `select_synthesizer` docstring documents the `"current"` requirement:
  host CLI must also be a configured participant for the run; peer-only
  modes fail loudly rather than silently falling back to a peer.
- CLAUDE.md envelope invariant now consistently states "BLOCKERS OR
  ASSUMPTIONS" — pass-4 doc fix had only updated the failure-taxonomy
  table, not the invariants bullet.

6 regression tests in `tests/test_pass5_fixes.py` (since consolidated
into `tests/test_abdication_detection.py` and `tests/test_synthesis_gating.py`)
covering the repaired-response-abdication case, abdication cache
round-trip, `should_synthesize` interaction with universal-abdication,
non-deliberative metadata gating, and the `select_synthesizer("current")`
host-excluded case. Full suite at release: 594 passed; same 4
pre-existing env failures unchanged.

## 0.5.1 - 2026-05-16

Pass-4 council review of v0.5.0 (transcript:
`.llm-council/runs/20260516_073739_*`) returned RECOMMENDATION: no with
6 concrete correctness bugs the implementation missed. This patch
addresses each.

- **Synthesis chair label validation (council #1).** Chair output is a
  decision memo by design, NOT a vote. v0.5.0 invoked the chair through
  the standard `run_participant` path which enforces the `RECOMMENDATION:`
  label — chair responses would fail validation and burn a repair retry.
  Fix: `synthesis.run_synthesis_chair` now sets
  `require_recommendation: False` and `retry_on_missing_label: False`
  on the chair's per-run cfg.
- **Synthesis sees final round only (council #2).** v0.5.0 passed the
  cumulative `results` list (round-1 + `:round2`-suffixed entries) to
  the chair, while the prompt headed them "final round." Fix:
  `orchestrator.execute_council` now calls
  `transcript.final_round_results(results)` before invoking the chair.
- **Universal-abdication short-circuits BEFORE round 2 (council #3).**
  v0.5.0 stamped the merged-blockers payload AFTER the deliberation
  loop, defeating the "save spend" intent. Fix: the check now runs
  immediately after round 1; if it fires, the deliberation while-loop
  guard refuses to enter and `deliberation_status` records
  `skipped_universal_abdication`.
- **Repair-retry no longer misclassified as abdication (council #4).**
  A successful repair-retry's `output` is a `_format_retry_transcript`
  block containing BOTH the repaired AND original sections. If the
  original had `EFFORT: blocked`, `_extract_response_envelope` would
  see it on the combined text and flip the (valid) repaired result
  to abdicated. Fix: new `_envelope_parse_source` strips the original
  section before parsing; `_with_envelope` also skips abdication
  detection when `repair_retry_recovered=True` (this belt-and-suspenders
  guard was dropped in v0.5.2 because the parse-source strip is the
  real correctness mechanism).
- **`EFFORT: blocked` without label is terminal (council #6).** A peer
  that self-reported blocked without emitting a label was still
  eligible for the label-only repair retry — wasted spend on a
  definitively blocked peer. Fix: `_is_label_only_failure` now refuses
  retry when `EFFORT: blocked` is present, even without a label.
- **Abdications never cached (council #10).** Cache writes happen
  inside per-adapter functions BEFORE `run_participant` -> `_with_envelope`
  flips abdication to `ok=False`. The `if not result.ok` cache guard
  therefore let abdication shapes slip through. Fix: `_maybe_persist_cache`
  scanned the output shape with `_is_abdication` and refused the write.
  (This fix was reverted in v0.5.2 — cache-hit re-derivation already
  handled the correctness concern offline.)

UX patches bundled with the bug fixes:

- `recommendation_line` for fenced-only labels now returns the explicit
  placeholder `(no RECOMMENDATION label emitted)` instead of falling back
  to arbitrary intro prose. Prevents round-2 deliberation prompts from
  echoing peer intro sentences as if they were positions.
- `select_synthesizer("current")` documents that the host CLI must be a
  configured participant — peer-only modes fail loudly instead of
  silently picking the requester.
- CLI now stamps `metadata["secret_scan"]` in transcripts (parity with
  MCP server) and surfaces `synthesis_error` on stderr so the user sees
  WHY synthesis didn't happen instead of getting no synthesis section
  with no explanation.
- CLAUDE.md updated: failure-taxonomy note for `abdicated` now correctly
  states EITHER `BLOCKERS:` OR `ASSUMPTIONS:` satisfies the bar; the
  `RECOMMENDATION:` label invariant clarifies that `recommendation_line`'s
  fallback differs from the strict label-match helpers.

11 regression tests added in `tests/test_pass4_fixes.py` (since
consolidated into `tests/test_abdication_detection.py` and
`tests/test_synthesis_gating.py`) covering each of the 6 must-fix bugs
plus the UX patches. Full suite at release: 588 passed; same 4
pre-existing env-related failures unchanged (no `mcp` package
installed, no ollama runtime, budget-config drift in a single test
fixture).

## 0.5.0 - 2026-05-16

Post-council-review build (three council passes drove the scope; transcripts in
`.llm-council/runs/20260515_175359_*`, `20260516_065100_*`, `20260516_070132_*`).

### Pick A — Effort contract + abdication detection

- New optional response envelope parsed alongside `RECOMMENDATION:`:
  `EFFORT` / `CONFIDENCE` / `RISK` (scalars) and `BLOCKERS` / `EVIDENCE` /
  `TESTS_TO_RUN` / `ASSUMPTIONS` (bullet lists). Fields are stored on
  `ParticipantResult`, emitted in transcripts (md + json), CLI `--json`
  output, and MCP `structured_results`. All fields stay optional in v1.
- New `error_kind: "abdicated"`. A peer that emits `RECOMMENDATION:` plus
  `EFFORT: blocked` with no `BLOCKERS:` / `ASSUMPTIONS:` is classified as
  abdication: `ok=False`, drops quorum, terminal for the round (no repair
  retry). Closes the silent-junk-verdict path where `RECOMMENDATION: no -
  too complex` previously kept consensus quorum.
- Cache schema bumped to v2 (`cache.CACHE_SCHEMA_VERSION = 2`) so any
  pre-existing cache entries do not bypass the new validation contract.
- MCP `COUNCIL_RUN_OUTPUT_SCHEMA_VERSION` bumped to v2 with envelope
  fields and `abdicated` added to `COUNCIL_RUN_VALID_ERROR_KINDS`.

### Pick B — Synthesis chair

- New `llm_council.synthesis` module + `--synthesize` CLI flag + `synthesize`
  MCP arg. After peers respond, a configured chair writes a structured
  decision memo (consensus blockers / single-peer concerns / dissent /
  verification plan). Chair output is **metadata, not a vote** — the headline
  `recommendation` / `agreement_count` stay derived from peer votes only.
- `defaults.synthesizer` (required when synthesize is on, fail-loudly per
  pass-3 Q4): a participant name, `"neutral_peer"` (auto-pick whichever
  peer has `stance=neutral`), or `"current"`. No silent default — requester
  bias is opt-in only.
- Auto-trigger fires only on `ran_max_rounds_unresolved`. Agreement does
  NOT auto-trigger (avoids burning a peer call to summarize "everyone agreed").
- Universal-abdication short-circuit: when every peer abdicates in round 1,
  return `recommendation: "unknown"` plus merged-and-deduped blockers
  without paying for round 2 or chair invocation.
- Chair consumes pre-computed `metadata["convergence"]` instead of
  re-deriving Jaccard drift state.

### Tier 2 — Secret scanner

- New `llm_council.safety.scan_prompt_for_secrets` + `apply_secret_scan_policy`.
  Preflight regex sweep over the constructed prompt (AWS/GitHub/OpenAI/
  Anthropic/Google/Slack/JWT/PEM block patterns). Default `secret_scan: warn`
  (count + emit a progress event); `block` raises before any participant
  runs; `off` skips. Allowlist file (`.llm-council-secrets-allow`, line-per-
  regex) covers test fixtures.
- Findings include kind + line + truncated preview (`first4...last4`) only.
  Raw matched values never enter transcripts, progress events, or logs.
- Closes the prompt-body credential leak: `_redact_credentials_in_text` at
  `orchestrator.py:76` only redacted URL userinfo in error strings; `--diff`
  / `--context` content shipped to hosted peers with no scanning.

### Side fixes (council-surfaced during plan review)

- **MCP summary bug (mcp_server.py:815)** — fixed. After deliberation,
  `if ":round" not in row["name"]` was selecting round-1 rows (opposite of
  the documented "final round" intent). Now filters against
  `final_round_results(results)` and strips the `:roundN` suffix for display.
- **Fenced-label validation tightened** — across `adapters._has_recommendation_label`,
  `deliberation.recommendation_label`, and `deliberation.recommendation_line`.
  A `RECOMMENDATION:` line inside a code fence is now treated as example
  syntax and does not satisfy the label contract; if no out-of-fence label
  exists, the response fails validation and triggers the standard repair
  retry path.
- **`stats.aggregate` now buckets by `error_kind`** and tracks
  `envelope_field_present` per peer. Prerequisite telemetry before any
  future flip of envelope fields from optional to required.

### Docs / invariants

- `CLAUDE.md` failure-taxonomy table now lists 9 `error_kind`s (added
  `abdicated`).
- `CLAUDE.md` documents the fence-aware label rule and the optional
  envelope contract under the "Invariants worth preserving" section.
