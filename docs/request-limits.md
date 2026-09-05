# Request deadlines and input limits

MCP `council_run` defaults to **240 seconds** through
`defaults.mcp_request_timeout_seconds`. The host used in the September dogfood
run stopped waiting after 300 seconds while the old 1,200-second server deadline
allowed peers to continue. The new default leaves room before that host limit.

The server reserves up to 10 seconds (20% of a shorter request) for process
cleanup, citation verification, and transcript writing. The peer deadline covers
all retries, queued participants, deliberation rounds, and synthesis. After it:

- Completed votes remain available.
- Unfinished peers receive an ordinary timeout failure and do not vote.
- Queued participants and further rounds are not launched.
- Responses and transcripts explicitly identify a **Partial result**.
- `metadata.partial`, `metadata.request_deadline_reached`, and
  `metadata.deadline_stopped_peers` identify this condition. Quorum/degraded
  calculations keep their existing meaning; a partial run can still have quorum.

Explicit client cancellation propagates after process cleanup. On POSIX, native
CLI and generator invocations own separate process groups, which are killed on
completion, timeout, or cancellation. This includes ordinary tool subprocesses
that inherit the group. Detached processes that create new sessions and Windows
process trees are outside this POSIX group guarantee.

`request_timeout_seconds` can override the default up to 7,200 seconds, but it
must fit the host's tool-call timeout. Increasing only the council setting does
not extend the host deadline. Preparation/finalization that exceeds the outer
deadline returns `CouncilRequestTimeout`; partial-vote preservation applies once
peer execution has begun and can complete finalization within the reserved time.

## Citation checks

Citation verification runs in a cooperative worker. It scans at most 8 MiB of
decoded characters per source file and 32 MiB across a council run. Line counts
are shared across EVIDENCE and FINDINGS checks and invalidated by file changes.
Only complete lines in the checked prefix can verify a range; ranges beyond the
prefix remain unverified. A valid range establishes location, not claim truth.

## Transcript search

Search retains the newest 10,000 eligible JSON files by modification time, with
a filename tie-break. It streams directory enumeration, examines up to 100,000
entries, excludes symlinks and JSON files larger than 8 MiB, and budgets 64 MiB
of cold transcript reads per query. Later queries can fill deferred cache entries.
Token matching uses the first 4,096 characters of each question and query.

MCP `council_query_transcripts` returns `search_scope` alongside `matches`.
`limited`, `entry_limit_reached`, `skipped_oversize_files`, and `deferred_files`
describe incomplete coverage. These limits avoid unbounded reads and memory
growth; absence of a match within a limited search is not proof that no older
transcript covers the topic. Existing transcript files are never modified.

Restart an already-running MCP server after installing these changes; a Python
process does not reload imported modules when source files change on disk.

## Validation

Verified September 5, 2026: `uv run pytest -q --tb=short` completed with
1,460 passed and 3 skipped in 62.31 seconds. Wheel and source builds passed.
Real stdio tests verified partial transcript preservation under a two-second
deadline, and process tests checked successful-generator and cancelled-peer
descendant cleanup. Initial local validation did not establish that the active
host had loaded the same build: its separate `uv tool` installation was still
older. After installing the tested wheel and restarting, the connected MCP
tool preserved the fast peer's vote in a 2.437-second deadline fixture and left
no fixture subprocesses running. A native connectivity run completed in 8.161
seconds with three fresh responses. Subsequent [file-access dogfooding](review-context.md)
confirmed all three peers could inspect unprovided files. Installation and
connected behavior were checked separately from checkout tests.
