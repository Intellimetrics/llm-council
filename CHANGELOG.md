# Changelog

## 0.24.0 - 2026-08-31

* **Opt-in OKF blast-radius context (`--okf-context` / `okf_context`).**
  With `--diff`, the orchestrator derives the symbols touched by the
  diff, looks up their one-hop caller/callee neighborhood in an OKF
  knowledge bundle, and appends a compact signatures+locators excerpt
  after the Git Diff section so every peer sees the blast radius, not
  just the diff (live A/B on this repo: 11/11 cross-module consumers
  enumerated vs 1/11 from raw source). Bundle acquisition is
  ephemeral-first — `okf-rs generate <root> -o <tmpdir> --no-cache` from
  the working tree, never writing into the project; a pre-existing
  bundle (`okf.toml` `output` key / `knowledge/index.md`) is a fallback,
  marked stale when its `source_revision` is not HEAD. Default OFF
  everywhere; prompts are byte-identical when disabled. Fail-soft:
  missing binary, generation failure/timeout, no matched concepts, or
  no prompt headroom leaves the run exactly as today plus
  `metadata.okf_context` (statuses: `attached` / `stale_attached` /
  `no_diff` / `excerpt_over_budget` / `binary_missing` /
  `generate_failed` / `generate_timeout` / `no_matched_concepts` /
  `internal_error`), a transcript header bullet, and a CLI stderr note.
  The excerpt renders into post-assembly prompt headroom only (capped by
  `defaults.okf_max_excerpt_chars`, default 12 000), so it can never
  trigger overflow chunking. Config: `defaults.okf_context` /
  `modes.<name>.okf_context` (None-aware precedence like
  `independent_review`), `defaults.okf_binary`,
  `defaults.okf_generate_timeout_seconds`. Estimate parity: `estimate`
  / `council_estimate` accept the same flag and build the same enriched
  prompt. MCP surfacing is metadata-only (schema stays v11). Peers gain
  no tools — this is orchestrator-side prompt text only. Opt-in canary
  `tests/test_live_okf_bundle.py` (`LLM_COUNCIL_LIVE_OKF_TEST=1`)
  guards the okf-rs frontmatter/read-only contract against upstream
  drift (verified live against okf-rs 0.7.0). Hardened after a
  self-review council run over this very diff (3 CLI peers, with the
  feature's own excerpt attached) converged on the subprocess edges:
  tempdir lifecycle moved inside a `TemporaryDirectory` context manager
  (the split-ownership version leaked on Ctrl-C), generator output
  capture switched to the disk-backed bounded pattern `_run_git` uses,
  the generate timeout gained a 120s hard ceiling (async call paths),
  a hostile `okf.toml` `output` can no longer aim the concept walk
  outside cwd, status-callback failures became non-fatal, signatures
  are length-capped, and an unknown fallback-bundle revision now reads
  as stale rather than fresh.

## 0.23.0 - 2026-08-28

**Field report 2026-08-28 (security-council, 48 runs): codex timeouts,
antigravity empty responses, refused turns filed as `unknown`**
* **Codex no longer inherits the operator's interactive reasoning
  profile.** `adapters._build_cli_command` injects
  `-c model_reasoning_effort=<participants.codex.reasoning_effort>` (new
  per-peer key, default `medium`; `inherit` / null restores the CLI's own
  config; an explicit `model_reasoning_effort` token in `args` wins) into
  every codex `exec` invocation — builder-level, like the MCP starvation,
  so hand-edited arg lists get it too. Field evidence: the 5.4 KB prompt
  that timed out at 606 s under the operator's `ultra` setting answered in
  151 s at `medium` (68 s at the CLI default). The codex baseline args
  also gain `--skip-git-repo-check` (councils run in scratch copies and
  untrusted directories; codex refused outright without it); the v0.21–
  v0.22 baseline migrates silently (`config.OLD_CODEX_MCP_STARVED_ARGS`).
* **Empty responses get diagnostics and one same-prompt re-run.** A CLI
  that exits 0 with no stdout used to be recorded as the bare
  `InvalidParticipantResponse: empty response` and dropped from quorum
  (17 of 48 field runs, no exit code, no stderr). The error now carries the
  exit status, the CLI's own status field when JSON mode surfaced one, and
  a stderr excerpt; `ParticipantResult` gains `exit_code` / `stderr_tail`
  — the tail is ALSO captured on timeouts, holding what the peer wrote
  before the kill (the only record of what a timed-out codex was doing).
  The CLI pipeline replays the same prompt once (`retry_on_empty_response:
  false` or `retries: 0` opts out); success stamps
  `recovered_after_empty_retry`, and a re-run that comes back unlabeled
  never chains into the label-repair call. New fields flow to transcripts
  (JSON + markdown "Exit status" / "Stderr tail") and MCP
  `structured_results` (output schema v11).
* **`usage_from_json` now supports antigravity.** `agy --print …
  --output-format json` returns `{status, response, usage}`; the parser
  records real token usage (no model id — agy reports none, so a
  `require_pinned_model` agy peer fails closed as
  `pinned_model_unverified`) and surfaces `status` inside the
  empty-response error. Default OFF like the other families.
* **New `error_kind=content_refused`.** Provider content/safety refusals
  (OpenAI "flagged for possible cybersecurity risk" / Trusted Access for
  Cyber, `content_policy_violation`, Gemini `PROHIBITED_CONTENT` /
  `SAFETY`) classify explicitly instead of `unknown`; surfaced as
  `metadata.content_refused_peers` + a `peer_content_refused` progress
  event, lifted top-level in the MCP `council_run` payload, and rendered
  as a ⚠️ line in the markdown transcript header. A nonzero-exit CLI's
  stderr is bounded (2 KB head + 6 KB tail) before it becomes
  `result.error` — one refused codex turn had shipped 542 KB of echoed
  prompt and tool trajectory into the transcript.
* Setup: the generated host instruction snippets tell agents to phrase
  security reviews as verification ("confirm this control holds") rather
  than attack ("find a bypass") — the refusal trigger observed in the
  field; `.llm-council/.gitignore` and the project `.gitignore` template
  now cover `.llm-council/cache/`.

## 0.22.0 - 2026-08-07

**Migrate the MCP server to the mcp 2.x SDK (lifts the `<2` pin)**
* `llm_council.mcp_server._serve` now targets mcp 2.x's
  constructor-callback `Server` API (`on_list_tools=` / `on_call_tool=`
  replacing the removed 1.x decorators and the ambient
  `request_context`). Dependency bound raised from `mcp>=1.0.0,<2` to
  `mcp>=2.0.0,<3` — mcp 1.x is upstream maintenance-only. A 1.x
  install now gets an actionable SystemExit at server start instead of
  an opaque TypeError, and `llm-council doctor` reports the installed
  mcp version, flagging `<2`.
* Wire behavior preserved: identical tool list and JSON schemas
  (`council_run` output schema stays v10), `structuredContent` on
  `council_run`, and handler errors still surface as `isError` tool
  results carrying the real message text — mcp 2.x's default would
  sanitize them into opaque JSON-RPC "Internal server error" frames
  the calling agent can't read. Also absorbed: 2.x delivers absent
  tool `arguments` as `None` and no longer validates them against the
  advertised inputSchema (handlers already do their own checks).
* Progress bridge intact: the per-call token now comes from the
  handler context (`ctx.meta["progress_token"]`);
  `send_progress_notification` is unchanged and undeprecated. A new
  stdio integration test pins the path end-to-end — a real run with a
  client `progressToken` must produce `notifications/progress` frames
  on the wire (dry-run returns before any progress fires, so the test
  runs a real council whose only peer fails preflight instantly
  against a dead local port).
* The server now reports its real version in `serverInfo` (mcp 2.x
  reports `""` for unversioned servers; `__version__` is passed
  explicitly).

## 0.21.2 - 2026-08-06

**Pin `mcp<2` (install-time breakage)**
* mcp 2.0.0 (released upstream) removed the 1.x low-level
  `Server.list_tools` decorator API this server is built on. The
  previous unbounded `mcp>=1.0.0` range let fresh installs resolve
  2.0.0 and crash the MCP server at startup — v0.21.1's CI caught it.
  Existing environments with mcp 1.x are unaffected. The bound is
  raised only together with a migration to the 2.x server API.

## 0.21.1 - 2026-08-06

Fixes driven by the 2026-08-05 field-issues report (two weeks of MCP
usage from a real project). MCP output schema bumps to v10, dogfooded
live through a restarted MCP server before release.

**Quorum-aware terse-retry skip (field issue #1)**
* When a CLI peer times out after the round already holds `min_quorum`
  labeled votes from the other peers, the 30-120s terse-retry is
  skipped: the timeout error gains an explanatory suffix (its
  `Timeout:` prefix — and therefore `error_kind` / quorum math — is
  unchanged) and a `terse_retry_skipped` progress event fires. In the
  field runs that motivated this, the retry re-timed-out every time,
  doubling the wasted wall time on runs that were already viable.
  Default ON; `defaults.skip_terse_retry_when_quorum_met: false`
  restores the old always-retry behavior.

**Dropped context files are loud (field issue #2)**
* Files dropped by context chunking were only visible as a progress
  event — which some MCP hosts (Claude Code) never render. The MCP
  `council_run` payload now carries a top-level `context_files_dropped`
  key (omit-when-empty) naming every dropped path plus a warning that
  the files never reached any peer; the markdown transcript gains a ⚠️
  header bullet and the CLI a second stderr warning line.

**Directional verdict on definite+tradeoff ties (field issue #4)**
* A final-round tie between `no` and `tradeoff` (or `yes` and
  `tradeoff`) with zero votes for the opposing definite label now
  reports `leaning-no` / `leaning-yes` instead of `unknown` — the peers
  agree on posture and differ only in label strength. `unknown` is
  reserved for label-free runs, true yes/no opposition, and three-way
  ties. `recommendation_tied` stays true and `agreement_count` stays 0
  for leaning outcomes (there is still no unique leader).

**Account-default peer models are named (field issue #3)**
* Runs now stamp `metadata.cli_default_model_peers` with every CLI peer
  running `model: None` (the operator's account default — not
  necessarily the host session's model), plus a transcript header note.
  Motivated by a host session that assumed the claude peer matched its
  own pinned model for weeks.

**Unreported telemetry renders n/a, not zero (field issue #5)**
* Transcript headers rendered "Tokens reported: 0 · Cost: $0.000000"
  when every peer's usage was null (text-mode CLI peers have no
  metering hook), reading as "this run was free". All-null totals now
  render `n/a` in both the markdown and HTML transcripts; partial sums
  still render as before.

**Angle brackets dropped from timeout advice (field issue #6)**
* The timeout error's `participants.<name>.timeout: <seconds>` advice
  arrived HTML-escaped (`&lt;seconds&gt;`) through some MCP hosts'
  rendering. Reworded without angle brackets; llm-council's own JSON
  was already clean.

**MCP starvation enforced at command-build time (field bug)**
* `_build_cli_command` now injects `-c mcp_servers={}` into every
  codex-family `exec` invocation unless the operator's args already
  carry an `mcp_servers` override. The v0.21.0 protection relied on the
  baseline args plus an exact-match config migration, and a field config
  with one extra flag (`--skip-git-repo-check`) defeated the match —
  every council codex run from that project silently booted the
  operator's three global MCP servers (two headless browsers and a
  nested llm-council server). The injection closes the gap for any
  arg-list shape; an explicit `mcp_servers` token in args suppresses it,
  and non-`exec` invocations are left untouched.

Council-integrity hardening driven by a full 3-CLI self-review dogfood:
the council reviewed its own codebase, and everything it (or the runs
themselves) surfaced was fixed and re-verified live through MCP.

**A council must never start another council**
* Every CLI child environment now carries `LLM_COUNCIL_NESTED`
  (presence semantics: any value — including `""` or `"0"` — refuses;
  unsetting is the only escape hatch). `execute_council` raises
  `NestedCouncilRefused` when marked, and the CLI converts exactly that
  error to a clean one-line exit while other ValueErrors keep their
  tracebacks for debuggability.
* The codex baseline args add `-c mcp_servers={}` so a council-spawned
  codex boots no MCP servers at all — the observed real-world recursion
  path was an operator's global `~/.codex/config.toml` registering
  llm-council itself, making every codex peer boot a nested council
  server per run. Existing generated configs silently upgrade via the
  new `OLD_CODEX_EPHEMERAL_ARGS` migration.

**Removed: two undocumented subsystems**
* `apply_contextual_persona_recruitment` (filename-substring persona
  injection on any dirty git tree) and `apply_smart_routing`
  (default-ON silent downgrade of premium models on "low-risk" diffs)
  are excised along with `DEFAULT_CHEAPER_MODELS` and all
  `persona`/`persona_prompt` plumbing. Both ran unconditionally with no
  enable flag and neither was documented; the council flagged them as
  trust violations. A leftover `smart_routing:`/`persona` YAML block is
  now an inert no-op.

**Deliberation prompt budget is now derived, not fixed**
* Every `deliberate: true` MCP run rostering antigravity had been
  structurally refused since v0.18.0: the worst-case round-2 bound was
  `MAX_DELIBERATION_PROMPT_CHARS` (80,000) plus agy's 268-char
  directive suffix, compared against the 80,000-char MCP cap —
  refused regardless of actual prompt size. The round-2 body budget is
  now `deliberation_body_budget(effective_cap, largest_suffix)`, used
  by BOTH the runtime builder and the preflight bound, so
  `body + directive suffix <= cap` holds by construction for any peer
  family, directive length, or configured cap. A lowered
  `mcp_max_prompt_chars` now genuinely bounds round-2 prompts too.

**Built-in mode rosters are local-CLI only**
* `plan`, `review`, and `deliberate` no longer seat hosted peers by
  default (previously deepseek/qwen via `add`). The shipped default
  council is claude / codex / antigravity everywhere, guarded by
  `test_no_built_in_mode_seats_hosted_peers`. Hosted baselines stay
  defined for explicit opt-in (`include`, per-project `add`, setup
  presets). Side effect: tri-cli setups now retain plan / review /
  deliberate instead of pruning them for referencing hosted peers.

**Coverage restored**
* Stance-based hard-cap tests replace the coverage lost with the
  persona tests: `--max-tokens` / `max_tokens` caps are verified to be
  computed over directive-decorated per-peer prompts, not the base
  prompt.

## 0.20.0 - 2026-07-19

v0.20.0 removes the retired standalone Gemini CLI peer and adopts
upstream capabilities surfaced by a changelog-plus-live-probe review of
all three remaining native CLIs.

**Removed: built-in `gemini` participant**
* Google retired Gemini CLI for individual accounts on 2026-06-18
  (superseded by Antigravity). The built-in peer only produced
  `client_ineligible` failures, so it is gone from defaults, setup
  detection (`agy` is now the Gemini-family route), generated configs,
  instruction snippets, and the `current` enums. The family machinery
  stays: an enterprise operator can re-add a custom `gemini`
  participant (family `gemini`) and selection/exclusion/preference
  logic still honors it — covered by new tests.

**Antigravity: per-run isolation + matched internal timeout**
* `--new-project` added to the shipped args. Verified live on agy
  1.1.4: without it, a later `agy -p` run can recall PRIOR runs'
  content from the global `~/.gemini/antigravity-cli/brain/` store —
  breaking fresh-eyes and risking cross-project leakage under prompt
  injection. The per-family directive now also tells the peer not to
  consult prior conversations (residual native-tool reads of the brain
  dir remain possible; isolation is strong, not absolute).
* `_build_cli_command` injects `--print-timeout <effective+30>s` so
  agy's internal 5-minute print cap can no longer silently truncate
  longer council timeouts.
* Stale comment fixes: since agy 1.1.2 an unresolvable `--model`
  hard-fails print mode instead of silently falling back.

**Claude Code (2.1.215) invocation refresh**
* `--permission-mode manual` (the current-blessed name for the old
  `default`, renamed in 2.1.200), `--strict-mcp-config` (peer never
  connects to project `.mcp.json` servers — verified live: zero MCP
  tools, no startup wait), and
  `--exclude-dynamic-system-prompt-sections` (byte-stable system prompt
  for cache reuse across same-cwd calls). Applied to both `claude` and
  `claude_fable`. New `OLD_CLAUDE_DEFAULT_ARGS` migration silently
  upgrades existing generated configs.

**Codex (0.144.6) parser hardening + guidance**
* `_parse_codex_usage_json` now accepts `item.completed` answer text
  only when `item.type == "agent_message"` — codex 0.143/0.144 added
  new canonical item types that could otherwise silently overwrite the
  real answer.
* Documented: codex has no native turn/wall-time cap; the lever for
  shorter runs is `-c model_reasoning_effort=low` in the peer args
  (~3x wall-time cut measured live), left as an operator opt-in.

Full suite: 1284 passed; ruff clean; live agy canary 2/2 with the new
invocation.

## 0.19.0 - 2026-07-19

v0.19.0 is a usage-driven simplification release. A 518-transcript audit
across 14 real projects showed several shipped subsystems and modes with
zero-to-negligible real-world use; all of them are removed. Everything cut
remains recoverable from git history, and custom per-project modes can
recreate any removed built-in mode shape.

**Removed: seven near-zero-use built-in modes**
* `single-llm`, `adversarial-red-team`, `test-gap-analysis`, `deep-audit`
  (0 uses each), `us-only` (1), `review-cheap` (1), and `diverse` (3 uses in
  518 runs). Nine built-ins remain: quick, peer-only, plan, review, fable,
  review-with-tools, private-local, deliberate, consensus. The
  single-peer stance-multiplex machinery survives for `consensus` /
  `deliberate` / any user-defined stance mode; the `single_llm_multiplex`
  config key and the red-team/test-gap prompt prose are gone.
  `--us-only-default` (setup) and `origin_policy: us` are unaffected.

**Removed: cross-rank subsystem** (2 uses ever)
* `--cross-rank` / MCP `cross_rank`, the anonymized ranking pass,
  `FINAL RANKING` parsing, `rank_position_mean` aggregation,
  `is_ranking_round`, and the transcript/MCP `cross_rank_*` fields.
  Historical transcripts containing ranking rounds are still read
  tolerantly by stats. MCP `council_run` output schema bumps to v9.

**Removed: review-focus bundles and acceptance contracts** (0 uses each)
* `--focus` / MCP `focus`, `.llm-council/review-skills/` discovery,
  `applied_focus` provenance, `examples/review-skills/`; and
  `--acceptance-contract` / MCP `acceptance_contract` with its prompt
  block. `--independent-review` is KEPT. The separate Claude Code skill
  installer (`.llm-council/skills/`) is unrelated and KEPT.

**Removed: outcome tracking + reliability layer** (1 record ever)
* `llm-council outcome mark/list`, `.llm-council/outcomes/` sidecars,
  `stats --reliability`, `aggregate_reliability`, and the
  `peers_to_consider_dropping` advisory in `council_recommend`. The
  per-peer quota telemetry (`quota_incidents` / `quota_recoveries` /
  `quota_recovery_rate`) — the only observable usage signal for
  text-mode CLI peers — moved into plain `llm-council stats` output.

**Removed: eval harness** (dogfood-only)
* `llm-council eval run`, metrics, fixtures, scorecards, the promotion
  gate and its CLI flags, and `stats --eval`. The `experimental: true`
  mode marker stays; promoting an experimental mode is now an explicit
  manual operator decision.

**Also**
* Inert per-participant `read_only:` config key removed everywhere —
  read-only enforcement lives in each CLI's own flags, which are
  unchanged.
* New per-family prompt hint steers Antigravity to its native file-read
  tool (headless `--sandbox` auto-denies its shell-`cat` fallback,
  observed live on agy 1.1.4).
* Internal-only names dropped from `privacy`/`safety` `__all__` exports.

Full suite: 1281 passed (from 1441 — the delta is deleted feature tests),
ruff clean, sdist build verified without the eval package-data entry.

## 0.18.0 - 2026-07-19

v0.18.0 is a cruft-removal and simplification pass, plus an Antigravity
compatibility/hardening fix driven by upstream agy 1.1.x changes.

**Antigravity (agy 1.1.x): argv prompt delivery + hard read-only**
* agy 1.1.1 stopped reading stdin when print mode is used, so the shipped
  invocation now passes the prompt as the `--print` argument
  (`{prompt}` substitution, `stdin_prompt: false`). Verified live on agy
  1.1.4: the old stdin style sends the literal prompt `-`; the argv style
  works. Note the Linux per-argument size cap (128 KiB `MAX_ARG_STRLEN`)
  documented alongside `max_prompt_chars`.
* agy 1.1.0 added a public `--mode plan` flag, and the shipped args now use
  it: Antigravity's read-only guarantee is upgraded from SOFT
  (prompt-enforced) to HARD (flag-enforced), matching the other native CLI
  peers. Verified live on 1.1.4: an explicitly ordered write produces no file
  under plan mode, while native-tool file reads still work.
  `--dangerously-skip-permissions` stays omitted so residual tool attempts
  are denied, not auto-approved. The live canary
  (`tests/test_live_agy_readonly.py`) was updated for the new invocation and
  passes 2/2; README, docs, and the example config drop the "softer
  guarantee / prefer Gemini" caveats, and setup no longer warns about the
  prompt-enforced compatibility route.

**Removed: temporary Opus-version comparison feature**
* The `opus-versions` mode and the pinned `claude_4_6` / `claude_4_7`
  participants — explicitly marked temporary since 0.3.x — are gone from
  defaults, the setup wizard, docs, and tests. Version drift between Opus
  4.6 and 4.7 is no longer interesting.

**Removed: dead code and stale files**
* Dead config keys `defaults.synthesizer_max_prompt_chars` and the inert
  per-participant `deprecated` marker; unused `display.ANSI_DIM` constant.
* Production-dead shims kept alive only by tests: `transcript.safe_slug`,
  `context.read_git_diff`, `adapters.run_openrouter_participant` (tests now
  exercise the real production functions).
* Stale tracked files: `CLI_MODEL_DIAGNOSIS_2026-05-30.md` (point-in-time
  v0.13.0 diagnosis, preserved in git history), duplicate `requirements.txt`
  / `requirements-council.txt` (pyproject is the source of truth), and the
  `scripts/llm-council` wrapper (`python -m llm_council` covers it).

**Changelog archive**
* Detailed release notes for 0.3.0 through 0.13.0 moved verbatim to
  `CHANGELOG_ARCHIVE.md` per the summary-plus-archive policy the 0.6.0 entry
  established; each entry here keeps a short summary and a "Full detail"
  link. `CHANGELOG.md` shrank from 128 KB to ~54 KB.

**Docs**
* `CLAUDE.md` mode list synced to reality (drops the nonexistent
  `local-only`, adds `single-llm`, `adversarial-red-team`,
  `test-gap-analysis`, `deep-audit`); read-only invariant and adapter notes
  updated for the agy changes.

## 0.17.0 - 2026-07-13

v0.17.0 is a release-hardening pass driven by a full repository review and
multiple dogfood council runs. It closes privacy and project-boundary gaps,
makes MCP resource limits enforceable, improves runtime honesty, and adds the
regression coverage needed to keep those guarantees from drifting.

**Privacy and transcript integrity**
* Secret-scan `redact` now protects the separately persisted question as well
  as the assembled peer prompt, so raw credentials cannot reappear in
  ranking/synthesis prompts or Markdown, JSON, and HTML artifacts.
* Transcript filenames are opaque sortable identifiers and no longer contain
  question text. On POSIX, new transcript directories/files are owner-only
  (`0700` / `0600`) and written through descriptor-anchored atomic replacement;
  Windows uses reparse-safe path checks, inherited ACLs, and atomic replace.
* `doctor --repair-transcript-permissions` safely audits and repairs eligible
  historical artifacts without following symlinks or touching foreign-owned or
  multiply-linked files.
* A continuation from `private-local` inherits that mode when no replacement
  is supplied and refuses hosted/native participants unless the operator
  explicitly passes `--allow-privacy-downgrade`.

**MCP and input boundaries**
* MCP config and dotenv discovery stop at `LLM_COUNCIL_MCP_ROOT`; context,
  acceptance-contract, diff, and transcript paths cannot escape that root.
  Relative working directories are rejected with project-root diagnostics.
* Context files, contracts, and Git output are read with per-item and aggregate
  bounds. Diff capture is disk-backed and visibly truncated instead of being
  accumulated without limit.
* MCP now exposes the same diff chunk strategies as the CLI, applies the MCP
  prompt ceiling to every participant type, and enforces a whole-request
  deadline through `defaults.mcp_request_timeout_seconds` or the per-call
  `request_timeout_seconds` override.
* `council_run` output schema v8 adds participant wall time and the stable
  `client_ineligible` and `pinned_model_unverified` error kinds.

**Runtime honesty and resilience**
* Progress events carry UTC timestamps and run-relative durations. Results now
  separate legacy attempt time from true wall time (including retries), and
  summaries expose run wall time versus participant aggregates.
* Doctor's default native-CLI checks now say explicitly that authentication was
  not probed. `--probe-native` performs a bounded readiness invocation and
  recognizes Gemini `UNSUPPORTED_CLIENT`; Ollama probing verifies that every
  configured model actually exists, not merely that `/api/tags` responds.
* A peer that requires pinned-model verification now fails closed with
  `pinned_model_unverified` when the CLI omits served-model identity, rather
  than accepting an answer whose authorship cannot be verified.
* Antigravity's exact `Individual quota reached` response now classifies as
  `quota_exhausted`. Native routing prefers the compatible Antigravity seat
  when both Gemini-family CLIs are installed; Gemini remains available for
  explicit hard-plan-mode selection.
* Standalone estimates include contextual personas, per-peer directives,
  stances, safe-context framing, and chunking, matching the prompts a real run
  will construct.

**Product and distribution**
* Setup-generated host instructions include the absolute project directory,
  and the documentation now distinguishes destinations from actions through a
  central product vocabulary.
* Package data includes the evaluation fixtures required by installed wheels.
  Source and wheel builds are covered by packaging regressions.

**Validation and dogfood**
* Full suite: **1,446 passed, 2 skipped** on the release checkout, plus full
  repository Ruff, packaging, permission, and diff-integrity checks.
* The final private-local dogfood completed non-degraded with a `yes`
  recommendation after the health probe identified and corrected a stale local
  Ollama model selection. A hosted recheck degraded honestly when two native
  CLIs timed out and Antigravity reached its external quota; its telemetry
  exposed the final quota-classification and wall-time serialization fixes.

## 0.16.0 - 2026-07-04

v0.16.0 wires **Claude Fable 5** in as a read-only council peer, with a "reduce + detect" design for the one hazard Fable adds: its request-side safety classifiers false-positive on benign security-adjacent review, and on the Claude Code surface a refused request is **silently re-served by Opus 4.8** — a model swap the default text-mode CLI invocation can't see. Left unaddressed, an Opus answer would be recorded as a "Fable" opinion. Everything here is **opt-in and default-OFF**; no existing mode's behavior changes.

**New `fable` mode + `claude_fable` peer**
*   **`claude_fable` participant** (`defaults.py`) — the host `claude` CLI pinned to `--model claude-fable-5`, with `usage_from_json: true` (so `_parse_claude_usage_json` surfaces the model that ACTUALLY served the turn) and `fallback_chain: []` (no `--fallback-model` injection — an overload swap would be a second silent-substitution path). Read-only flags unchanged.
*   **`fable` mode** — consults `claude_fable` as a single read-only reviewer (the current host agent is the "orchestrator" seeking Fable's independent second opinion) and sets `safe_context: true`.

**Reduce — defensive-review framing (`safe_context`)**
*   `context.build_prompt(safe_context=True)` injects an authorized, read-only, defensive-review framing block (resolved from the mode config at the CLI/MCP call sites). It states the TRUE defensive nature of the review to lower Fable's false-positive refusal rate and heads off the `reasoning_extraction` category (peers never need to expose raw chain-of-thought — the structured format is all the council consumes). It is factual context, **not** an instruction to bypass any safety behavior, and is harmless for non-Fable peers. Absent unless the mode opts in.

**Detect — pinned-model guard (`require_pinned_model`)**
*   New per-peer `require_pinned_model: true` (`adapters._run_cli_once`): when `usage_from_json` surfaces a served model that doesn't match the pinned `model` (lenient variant-tolerant match via `_model_pin_satisfied`), the peer drops with `ok=False` and the new `error_kind=model_substituted` (`ModelSubstituted:` prefix), so a substituted Opus answer is **never counted as a Fable vote**. `result.model` still reports the REAL served model so the transcript shows the swap. Fires only on a positive, observed mismatch — a peer without JSON usage never trips it, and the guard is strictly opt-in.
*   **Top-level surfacing** — the orchestrator lifts substitutions into `metadata['model_substituted_peers']` (`{peer, requested, served_by}`, omitted when empty) and emits a `peer_model_substituted` progress event, mirroring `quota_throttled_peers`. `model_substituted` is added to the failure taxonomy and the MCP `council_run` error-kind schema.

**Hardening from the multi-agent review of this release** (33 candidates → 20 confirmed; all fixed)
*   **Answer-author model selection.** `_parse_claude_usage_json` no longer takes `modelUsage`'s FIRST key (JSON key order carries no contract; a fallback turn can log BOTH models, and helper models like haiku can appear). It now picks the key with the most `outputTokens` — the model that authored the answer — with first-key tie-break, closing both a false-negative (Opus accepted as Fable) and a false-positive (healthy Fable turn dropped over a helper model) in the guard.
*   **Substitution on a repair retry is no longer swallowed.** `_merge_cli_retry`, `_merge_section_retry`, and the terse-timeout-retry path all previously fell through to the ORIGINAL error when the retry itself was substituted, reclassifying the swap as `invalid_response`/`incomplete_response`/`timeout`. All three now propagate the `ModelSubstituted:` result.
*   **All-rounds surfacing.** The orchestrator's substitution scan covers every round AND the `--cross-rank` ranking pass (dedup on peer+served_by; ranking-pass entries carry `ranking_round: true`) — previously only the final primary round was scanned, so a round-1 swap in a deliberating run vanished.
*   **Substituted output excluded from the finding matrix.** `build_matrix_from_results` ingests raw outputs without an `ok` filter, so an Opus-served `FINDINGS:` block could be attributed to `claude_fable` in `consensus_blockers`. The orchestrator now filters `model_substituted` results out of the matrix input (deliberation and synthesis already skip not-ok results).
*   **MCP top-level parity.** `model_substituted_peers` is now `_lift`-ed to the top-level `council_run` payload and declared in the output schema, matching the sibling signals it mirrors.
*   **`safe_context` persists into the ranking pass** (the most refusal-prone request of the run — it quotes peers' security findings verbatim), matching the focus-directive persistence rule.
*   **Estimate parity.** `estimate_council` builds the prompt with the mode's `safe_context` so an estimate that passes can't be rejected by the run's prompt-size guard.
*   **Loud config validation.** `modes.<name>.safe_context`, `participants.<name>.require_pinned_model`, and `participants.<name>.usage_from_json` are validated as booleans at config load — a quoted `"false"` previously ENABLED them via truthiness.

**Hardening from the second review pass** (the first pass lost 11 verifier agents to a quota cap; this re-run pooled 22 candidates → 21 verified → 10 distinct defects, all addressed)
*   **Synthesis-chair substitution is no longer invisible.** The chair turn never enters `results`, so the substitution scan couldn't see it: a Fable chair refused-and-served-by-Opus produced a memo stored as `metadata['synthesis']` attributed to `claude_fable` with NO `model_substituted_peers` entry. The orchestrator now scans the chair payload after synthesis, stamps `synthesis_payload['model_substituted'] = True`, and surfaces a `{peer, requested, served_by, synthesis: true}` entry + `peer_model_substituted` event.
*   **Strictly factual `safe_context` wording.** The directive previously asserted facts the tool cannot verify ("requested by the maintainer", "the maintainer's own code", "do not infer malicious intent") — false framing when reviewing an untrusted third-party patch, and capable of suppressing a TRUE-positive refusal on genuinely malicious code. Rewritten to state only what llm-council can vouch for (operator-invoked, read-only, analysis-only) and to redirect rather than suppress suspicion: flagging malicious code as a finding IS the requested output.
*   **`require_pinned_model` now suppresses `--fallback-model` injection.** On a claude-family peer with both a pinned model and a non-empty `fallback_chain`, the CLI's designed overload recovery would be dropped as `model_substituted` (with a misleading "safety-refusal fallback" message) after paying for the answer. The pin now wins — no injection — and `config.config_warnings` flags the contradictory combination as inert.
*   **Live, correctly-attributed substitution events.** `peer_model_substituted` events previously fired only in an end-of-run scan stamped with the FINAL round count (a round-1 swap in a deliberating run reported `round: 2`, after the fact). Detection now runs per round via `_detect_and_emit_substitutions` (round 1 / ranking pass / round 2), mirroring the per-round quota events.
*   **Substituted label-repair retries keep the original output.** `_merge_cli_retry`'s substituted branch returned the retry bare, dropping the original (genuinely Fable-authored) response from the transcript; it now combines both attempts via `_format_retry_transcript`, matching `_merge_section_retry`.
*   **`safe_context` resolved inside `execute_council`.** Was a caller-threaded parameter both call sites had to compute; now derived from `config['modes'][mode]` like `timeout_multiplier` / `tool_call_voting`, so a future caller can't get a framed round-1 prompt but an unframed ranking pass.
*   **MCP output schema version bumped to 7** for the `model_substituted_peers` top-level key (the analogous `missing_key_peers` addition bumped v4→v5).
*   **Known residual risk documented (no mechanical fix available):** served-model attribution uses max cumulative `outputTokens` across the agentic turn, so a MID-turn refusal fallback (long Fable tool loop, shorter Opus-served final answer) can pass the pin check. Treat the guard as high-recall for whole-turn swaps, not proof of authorship.
*   **Test gaps closed.** The terse-timeout-retry and `_merge_section_retry` substitution-propagation branches now have regression tests; the finding-matrix exclusion test drives the real `execute_council` (plus a healthy-peer control) instead of re-implementing the filter; the duplicated fake-subprocess stubs are lifted into a shared `tests/proc_stubs.py`.

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

A 35-finding multi-dimension self-review batch. Headlines: antigravity
read-only hardening (dropped `--dangerously-skip-permissions`; agy's
read-only is SOFT/prompt-enforced, now guarded by a live canary test),
fail-fast config validation for `fallback_chain` / `timeout_multiplier` /
`idle_timeout`, nearest-wins `.llm-council.env` precedence, idle-read
pipe-deadlock and cross-rank concurrency fixes, and MCP output schema v6
surfacing of retry/ranking fields. Post-review follow-ups added the `redact`
secret-scan policy, a real MCP stdio integration test, and a fix for ~8 days
of red CI. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0130---2026-05-28).

## 0.12.2 - 2026-05-23

Fixes a council-flagged false-drop in `_drop_missing_key_participants`:
`openai_compatible` peers without an explicit `api_key_env` are no longer
pre-dropped (local vLLM / llama.cpp / LM Studio peers legitimately run
without auth), while `openrouter` peers keep the `OPENROUTER_API_KEY`
default and explicitly declared keys still pre-drop. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0122---2026-05-23).

## 0.12.1 - 2026-05-23

Reshapes the default quota fallback chains for capability-graceful step-down
(claude/codex/gemini gain multi-entry chains; antigravity stays empty — no
`--model` flag) and adds multi-step walking via `_quota_fallback_walk` (max
3 steps), stopping at first success, first non-quota failure, or chain
exhaustion, with `model_fallback_used` stamping the last attempted model on
failure. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0121---2026-05-23).

## 0.12.0 - 2026-05-23

Four timeout-resilience improvements from the v0.11.7 dogfood: size-scaled
base timeouts (5s per KB above 4KB, capped at +600s), a proportional
terse-retry budget (40% of the original timeout, floor 30s / ceiling 120s)
replacing the fixed 60s, an opt-in per-peer `idle_timeout` streamed-read
kill switch, and pre-dropping hosted peers whose `api_key_env` is unset
without counting them toward quorum. MCP schema bumped to v5 with the
top-level `missing_key_peers` field. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0120---2026-05-23).

## 0.11.8 - 2026-05-23

Fixes two correctness bugs and one false-negative gap in the v0.11.6
quota-fallback work: a failed quota-fallback retry now returns early instead
of chaining label/section repairs against the original overloaded model, and
`QUOTA_EXHAUSTED_PATTERNS` becomes case-insensitive with coverage for
Google/OpenAI natural-language quota messages, spaced `rate limit exceeded`,
and a widened bare-429 window. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0118---2026-05-23).

## 0.11.7 - 2026-05-23

Phase 3 of quota resilience: `llm-council stats --reliability` gains
per-peer `quota_incidents` / `quota_recoveries` counters with a derived
recovery rate, peers with only quota-incident signal now appear in the
reliability table, and the docs record that per-CLI token usage is not
observable — quota incidents are the observable proxy for CLI peers. Full
detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0117---2026-05-23).

## 0.11.6 - 2026-05-22

Phase 2 of quota resilience: actual fallback retries via the new
per-participant `fallback_chain` (Claude gets CLI-native `--fallback-model`
injection; codex/gemini retry once with the next-in-chain model), surfaced
through `quota_recoveries`, the `peer_quota_recovered` progress event, and
new per-result `model_fallback_used` / `recovered_after_quota` fields. MCP
output schema bumped to v4. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0116---2026-05-22).

## 0.11.5 - 2026-05-22

Phase 1 of quota resilience: the new `error_kind=quota_exhausted` classifies
known rate-limit/quota signals, surfaced via the top-level
`quota_throttled_peers` field and a deduplicated `peer_quota_throttled`
progress event; no auto-fallback yet — the peer still drops from quorum, but
the cause is now visible to the calling agent. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0115---2026-05-22).

## 0.11.4 - 2026-05-22

`llm-council doctor` now self-heals a missing/stale OpenRouter catalog with
an inline best-effort refresh (10s timeout, fail-soft), governed by the new
`defaults.catalog_auto_refresh: true`. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0114---2026-05-22).

## 0.11.3 - 2026-05-21

Dynamic stance balancing keeps consensus-mode debate roles
(for/against/neutral) evenly distributed when participants are filtered or
excluded, plus a clean startup error when neither `antigravity` nor `gemini`
is on PATH for quick triad modes. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0113---2026-05-21).

## 0.11.2 - 2026-05-20

README rewrite featuring Google Antigravity CLI and SDK as first-class
citizens: documents the integration points, dynamic triad selection (exactly
3 active CLIs) with fallback/prioritization rules, and Antigravity's native
Claude-model support with family exclusions preventing redundant voting.
Full detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0112---2026-05-20).

## 0.11.1 - 2026-05-20

The quick-select triad (`tri-cli`) now dynamically picks between
`antigravity` and `gemini` based on PATH (preferring `antigravity` when both
are installed), treating the pair as a single triad slot across setup
verification, auto-preset routing, and next-steps logic, with `doctor`/setup
nudging gemini-only users toward antigravity. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0111---2026-05-20).

## 0.11.0 - 2026-05-20

Integrates the Antigravity CLI (`agy`) as a native participant and primary
driver: default participant configs, `detect_current_agent` process
detection, gemini/antigravity model-family exclusion rules, and setup wizard
/ doctor / README updates. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0110---2026-05-20).

## 0.10.2 - 2026-05-18

Fixes three dogfood-surfaced envelope-parser bugs (sentence-form `RISK:`
truncated to a single word, inline comma-separated `EVIDENCE:` lists parsed
as one item, and `BLOCKERS: none` stored as truthy `["none"]` defeating
abdication detection) plus four pre-existing test failures carried since
v0.10.0, turning CI green again. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0102---2026-05-18).

## 0.10.1 - 2026-05-18

Two correctness fixes for v0.10.0's MCP progress notifications: strong task
references prevent asyncio GC from silently dropping in-flight
notifications, and preflight-failed peers now advance the progress counter
(fixing the off-by-one total). Also records that Claude Code does not yet
render `notifications/progress` (upstream anthropics/claude-code#4157);
other MCP hosts do. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0101---2026-05-18).

## 0.10.0 - 2026-05-18

Ships MCP progress notifications — bridging orchestrator progress events to
`notifications/progress` with a completed/total fraction and a curated set
of interesting events — plus a brand-identity layer: every progress message
is prefixed `LLM Council · `, CLI output gains deterministic per-peer color
accents, and `LLM_COUNCIL_QUIET=1` suppresses both. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#0100---2026-05-18).

## 0.9.0 - 2026-05-18

Four items from the post-v0.8 competitor-comparison pass: the
`council_query_transcripts` MCP tool (Jaccard search over prior runs),
opt-in tool-call voting on `review-with-tools` with the operator-visible
`tool_call_status` field, a dogfood-caught fix serializing that field, and
the anonymized `--cross-rank` / `cross_rank` flag composable with any mode
(ranking-round outputs excluded from deliberation). Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#090---2026-05-18).

## 0.8.1 - 2026-05-17

Routes `context_files` through the same hash-aware chunker as `--diff`
(oversize files dropped loudly via a `context_files_chunked` event), fixes a
latent v0.8.0 MCP-schema bug where the `verified` evidence tag was missing
from the output schema's enum (crashing MCP runs with verified citations),
and adds the optional `CONTINUE_DEBATE: yes|no` envelope tag — a unanimous
`no` from label-producing peers skips round-2 deliberation. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#081---2026-05-17).

## 0.8.0 - 2026-05-17

Ships a closed-loop measurement pipeline: the `[VERIFIED:path:start-end]`
evidence tag with mechanical verification, an eval harness (`llm-council
eval run`) with SNR metrics and a promotion gate, and sidecar outcome
tracking (`llm-council outcome`) powering per-peer reliability stats. Also
adds the experimental `review-with-tools` mode, the post-deliberation
per-finding agreement matrix, and per-mode `model_overrides` (replacing the
cut persona auto-routing). Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#080---2026-05-17).

## 0.7.1 - 2026-05-17

Patch bundling 15 council-surfaced fixes: section-repair and strict-evidence
retries wired across all three transports with each gate capped at one extra
call, terse-retry visibility (`terse_retry_attempted` plus a failure
annotation), a major envelope-parser fix (inline `EVIDENCE:` lines were
rejected wholesale, making strict-evidence a no-op), broader section-header
matching with tighter response-side checks, and synthesis/MCP-schema
rendering fixes. Also documents the MCP-server-restart-after-install gotcha.
Full detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#071---2026-05-17).

## 0.7.0 - 2026-05-16

Three council-surfaced changes from the pass-7 run: mode-aware
`timeout_multiplier` plus a one-shot 60s terse-retry on timeout with new
timeout stats buckets; the section-coverage validator for `PART N — TITLE
(REQUIRED)` headers (new `error_kind=incomplete_response`); and evidence
tags as a first-class envelope contract with opt-in `strict_evidence`
enforcement (new `error_kind=untagged_evidence`). MCP output and cache
schemas bump to v3. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#070---2026-05-16).

## 0.6.0 - 2026-05-16

No-behavior-change cleanup from pass-6 review: rewrote 17 `Pass-N fix #M`
comments as declarative invariants, reorganized per-pass regression test
files into topic-based ones, and moved the detailed 0.5.0–0.5.2 release
notes into `CHANGELOG_ARCHIVE.md`, establishing the
summary-plus-archive-link changelog policy. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#060---2026-05-16).

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

Council code-review pass on v0.4.3..v0.4.8 fixing 13 flagged items: local
`openai_compatible` peers now count as $0 in the budget gate, `local-only`
refuses runtime `--include` of hosted peers, MCP surfaces `preflight_failed`
in the error-kind schema plus `config_warnings`, preflight defaults to
loopback-only with credential redaction, and the local-detection helpers
were promoted to public API with a structured `discover_local_openai()`
probe. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#049---2026-05-07).

## 0.4.8 - 2026-05-07

The setup wizard gains `--probe-local`, which discovers running local
OpenAI-compatible servers (vLLM, sglang, LM Studio, llama.cpp, MLX, Ollama
`/v1`) and interactively scaffolds them into `.llm-council.yaml` with
sensible defaults, auto-derived family/participant names, and a
`LOCAL_OPENAI_API_KEY` reminder; probing is interactive-only and off by
default. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#048---2026-05-07).

## 0.4.7 - 2026-05-07

New opt-in `env_strict: true` flag on CLI participants restricts the
subprocess to `_SAFE_ENV_NAMES` plus `env_passthrough`, preventing ambient
env vars (e.g. `GEMINI_MODEL`, `OPENAI_BASE_URL`) from silently mis-routing
CLIs like qwen-code; sieve mode stays the default and the flag is validated
as a boolean. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#047---2026-05-07).

## 0.4.6 - 2026-05-07

New per-run pre-flight ping for local participants: loopback/RFC1918
endpoints get a concurrent 1-second reachability probe before round 1,
turning the multi-minute opaque timeout on a stopped local server into a
sub-second `PreflightFailed:` result with a new `preflight_failed` progress
event and `error_kind`, plus a per-participant `pre_flight_check: false`
opt-out. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#046---2026-05-07).

## 0.4.5 - 2026-05-07

New `config_warnings(config)` surfaces non-fatal advisories at config load,
starting with origin-typo detection: origins that normalize to a canonical
`KNOWN_ORIGIN_STRINGS` entry but don't match it literally (e.g.
`us/anthropic`) trigger a stderr warning, catching the case/spacing drift
that silently excludes participants from `origin_policy: us` runs. Full
detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#045---2026-05-07).

## 0.4.4 - 2026-05-07

New built-in `local-only` mode and `local_only_peers` strategy select every
Ollama peer plus any `openai_compatible` peer with a loopback/RFC1918
`base_url`, auto-extending as local participants are added; distinct from
`private-local` (still pinned to `local_qwen_coder`), and the strategy
refuses `include_current`/`add` so hybrid rosters must be explicit. Full
detail: [CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#044---2026-05-07).

## 0.4.3 - 2026-05-07

`llm-council doctor --probe-local-openai [BASE_URL]` discovers local
OpenAI-compatible servers on well-known ports (validating the `/v1/models`
JSON shape, not just that the port answers), and the new
`docs/local-models.md` ships copy-paste recipes for vLLM/sglang/LM
Studio/llama.cpp/TGI/Ollama/MLX with the load-bearing gotchas (origin
semantics, non-empty Authorization header, `allow_private: true`, timeout
floor). Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#043---2026-05-07).

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

Visual-identity release: CLI progress/result lines render through a
right-aligned bold-cyan gutter with colored status words (the layout, not
the color, is the signature — survives `NO_COLOR` and non-TTY), and MCP
`council_run` gains a `summary_markdown` field built to survive host-agent
rendering. Breaking for greppers: orchestrator lines are now `llm-council
starting:` / `llm-council complete:` and the MCP heading is `**LLM
Council**`. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#041---2026-05-03).

## 0.4.0 - 2026-05-02

Big consensus/ergonomics release: repair-retry on missing `RECOMMENDATION:`
labels, launch-retry on transient CLI stderr, the `error_kind` failure
taxonomy, the `consensus` mode with assigned stances (plus fixes for three
stance-dropping ship blockers), convergence detection, degraded-consensus
quorum marking, `--diff` chunking strategies, a per-participant result
cache, conversation threading via `--continue`, run-level budget caps, an
SSRF-defended `openai_compatible` participant type, a structured MCP output
schema, `llm-council stats`, and `transcripts prune`. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#040---2026-05-02).

## 0.3.2 - 2026-04-29

Closes documentation and test-coverage gaps from the 0.3.1 review pass:
README/docs gain What's New, Images, and Timeouts sections, a comment
documents why the CLI branch drops `image_manifest`, and tests cover the
`claude_4_7` model pin, the slow-warn threshold formula, inline-input
sweeping, and image-manifest recording. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#032---2026-04-29).

## 0.3.1 - 2026-04-29

Reviewer-driven follow-up from two Claude council runs: image-attachment
budgets are enforced at estimate time, the MCP `mode` schema description is
auto-generated from `DEFAULT_CONFIG["modes"]`, stale `.llm-council/inputs/`
staging directories are swept (7-day retention), the `## Images` prompt copy
is audience-agnostic, and `RECOMMENDATION_RE` is single-sourced, alongside
deliberation-excerpt and `_build_cli_command` cleanups. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#031---2026-04-29).

## 0.3.0 - 2026-04-29

Adds image passthrough to the council (`image_paths` / inline `images` /
repeatable `--image`, per-participant `vision: true` multimodal support,
staged inline images under size caps), makes CLI participant timeouts
graceful (actionable messages, a `participant_slow` watchdog, a distinct
`timeout` status, cumulative deliberation skips), and ships temporary pinned
`claude_4_6`/`claude_4_7` participants with an `opus-versions` comparison
mode. Full detail:
[CHANGELOG_ARCHIVE.md](CHANGELOG_ARCHIVE.md#030---2026-04-29).

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
