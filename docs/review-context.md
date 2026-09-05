# Context access and OKF review guidance

Native CLI peers can inspect source beyond the supplied diff. Their fresh
conversation does not automatically include the host's requirements, research,
connected apps, or conversation history. Give each review a concrete decision,
the relevant constraints, and the evidence needed to assess it.

| Information | How it reaches a peer |
| --- | --- |
| Changed tracked source | `include_diff: true` / CLI `--diff` |
| Other source, callers, tests, local docs | Native read-only tools; explicitly ask for inspection |
| Requirements and prior research | Question plus `context_files`; CLI `--context` |
| Caller/callee map | Diff plus `okf_context: true` / `--okf-context` |
| Earlier council review | `continuation_id` / `--continue`; summarizes the immediate parent |
| Host chat and connected apps | Not inherited; attach relevant approved information |
| Current external documentation | Supply verified research; do not assume peer web access |

Hosted API and Ollama adapters send the assembled context without giving the
model a filesystem tool loop. Native Claude permits Read/Grep/Glob/LS; Codex
runs in its read-only sandbox; Antigravity uses plan mode with its native read
tool. Claude and Codex council subprocesses disable MCP servers. Native CLI
permission flags enforce source read-only behavior, but are not a universal
read boundary around the project. MCP's own attachment containment checks do
not sandbox every native tool read.

## Requesting a grounded review

For MCP, set `mode: "review-with-tools"`, `include_diff: true`, and
`okf_context: true` when that mode is configured. Attach relevant requirements
with `context_files`. Ask peers to identify an affected caller, cite source,
give a concrete failure case, and identify evidence they could not inspect.
Keep any mode or participant choices the user explicitly requested.

Mode configuration is separate: `review-with-tools` does not inherit
`review`'s OKF flag. Built-in modes leave OKF off. This repository's project
config enables it for `review` and limits `review-with-tools` to Claude/Codex
because of previously observed Antigravity headless tool failures.

Check `metadata.okf_context.status`, `concepts`, `matched`, and `stale` for
actual attachment and coverage. Also inspect `context_files_dropped`, failed
peers, `degraded`, and `metadata.partial`. Quorum can survive an incomplete
run. A verified source range proves that the location exists, not that the
peer's claim is true. Native response caching is off by default because files
outside the prompt can change between reviews.

## Measured behavior on September 5, 2026

The installed `okf-rs 0.7.0` passed the opt-in live bundle integration test.
A connected MCP A/B used a small Git fixture whose expiry parser changed from
seconds to milliseconds while an unchanged caller still compared against a
seconds clock. No caller file was attached. Each run also asked the peers to
read an ignored marker file containing a fresh random value absent from the
assembled prompt.

| Check | OKF off | OKF on |
| --- | --- | --- |
| Peers finding the caller and compatibility defect | 3/3 | 3/3 |
| Peers reproducing the fresh marker | 3/3 | 3/3 |
| Run wall time | 30.1 seconds | 33.9 seconds |
| OKF excerpt | None | 516 characters, 1/1 matched symbol |

Claude, Codex, and Antigravity all passed; no response was cached and fixture
source remained unchanged by the peers. This proves file access and integration
in those runs. It does not establish an accuracy or speed improvement, broader
Antigravity reliability, or token savings (CLI usage telemetry was null).

On the current approximately 95,000-character repository diff, standalone OKF
generation took 1.294 seconds and included 10 of 70 matched symbols in 11,965
characters with a 12,000-character allowance. The actual 80,000-character
prompt builder omitted OKF before trimming, then produced a 75,509-character
prompt. These observations identify remaining limitations:

- OKF budgets against the untrimmed prompt and does not retry after trimming.
- Path/source-order selection can leave later changed files uncovered.
- Call edges can be incorrect: dictionary `.get` calls in `_cache_lookup`
  were linked to an unrelated test client's `get` method.
- A harmless Git revision in the fixture excerpt triggered a secret-scan
  warning. The generated revision was the false positive, not a credential.

These are open improvement items. Use the graph to navigate, verify in source,
and measure representative multi-file reviews before changing default rosters,
budgets, or claiming better review quality.
