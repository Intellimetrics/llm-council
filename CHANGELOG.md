# Changelog

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
