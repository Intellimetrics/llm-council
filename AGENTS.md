# LLM Council repository instructions

LLM Council provides independent, read-only second opinions. The host remains
responsible for verifying findings and implementing authorized changes.

## Working in this repository

- Read [CLAUDE.md](CLAUDE.md) for architecture and implementation invariants;
  those maintainer notes apply to every coding agent.
- Use `uv run pytest -q` for the suite and `uv build` for packaging. Run focused
  existing tests for a bounded change; native live canaries are opt-in and use
  the operator's accounts. Preserve unrelated work and project configuration.
- Keep README, the [operator reference](docs/llm-council.md), and relevant
  maintainer notes aligned with runtime changes. Agent instruction templates
  live in `llm_council/setup_wizard.py`: update both project and global output
  through their shared guidance, then verify generated files.
- Follow [product vocabulary](docs/vocabulary.md) for UI labels.
- A checkout, an installed `uv tool`, and the running MCP process are separate.
  Verify the host's configured executable, install the tested build when
  updating it, and restart only when needed to load changed server code.
  Dogfood through the connected MCP tool; local imports alone do not validate
  the active integration. See [installation verification](docs/llm-council.md#updating-an-active-installation).
- Keep released version identifiers synchronized only when making a release.
  Unreleased work belongs in the changelog's Unreleased section.

## Council routing

When the user asks to use council or get a second opinion, call the
`llm-council` MCP tool `council_run`. Pass the absolute reviewed project path as
`working_directory` and the host as `current`: `codex`, `claude`, or
`antigravity`. Use `quick` by default here; honor explicit modes. Use
`consensus` for assigned-stance debate, `peer-only` to exclude this host, and
`private-local` for loopback Ollama-only review. Native CLIs are not an offline
route. A hard offline guarantee also requires daemon network controls.

Treat “current diff” or “review my changes” as `include_diff: true`, an explicit
dollar cap as `max_cost_usd`, and “continue from <run_id>” as `continuation_id`.
Do not launch a nested council from inside a participant turn.

## Context and findings

- Native peers can read files beyond the diff. Ask them to inspect affected
  callers, tests, and requirements and cite evidence. `review-with-tools`
  supplies this direction; honor an explicitly requested mode.
- Attach requirements and relevant research with `context_files`. Peers do not
  inherit the host conversation or connected apps; do not assume web access.
  Hosted API/Ollama peers receive supplied context without filesystem tools.
- OKF needs a diff and `okf_context: true`. This project's `review` enables it;
  `review-with-tools` selects Claude/Codex and needs the explicit OKF toggle.
  Check `metadata.okf_context` for omission, staleness, and concept coverage.
  Graph edges and valid line ranges are not proof that a claim is correct.
- Check failed peers, `context_files_dropped`, `degraded`, `metadata.partial`,
  and transcript-search `search_scope` before claiming complete coverage.
  MCP requests default to 240 seconds; raising a peer timeout alone does not
  extend the request or host limit. Preserve completed findings from partial runs.
- Summarize agreements, real disagreements, and remaining evidence gaps before
  acting. Ask before large or risky edits unless implementation is already
  authorized. Participants must not edit files.
- Do not send classified, regulated, secret, credential, or customer data unless
  the user explicitly confirms the configured participants are approved for it.

See [context access and OKF limits](docs/review-context.md), [request limits](docs/request-limits.md),
and the dated [native model snapshot](docs/native-models.md). Open OKF coverage
and graph-quality issues are documented; do not describe them as fixed.
