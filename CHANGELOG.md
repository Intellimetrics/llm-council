# Changelog

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
