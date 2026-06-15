---
name: test-gaps
description: Read-only test-coverage lens — flag logic changed without matching tests and name the missing cases. Composes with any mode.
---

TEST GAP FOCUS (read-only, advisory).

Analyze the proposed changes and identify missing test coverage. Do NOT
write or apply tests — only flag the gaps and name the cases that should
exist.

- When logic is added or modified but no test is added or updated, call it
  out explicitly with the `file:line` of the untested change.
- Identify untested branches, error paths, and boundary/edge conditions
  (empty input, off-by-one, null/None, concurrency, timeouts).
- Note assertions that test the happy path only and would not catch the
  regression the change is meant to prevent.

If the change lacks adequate tests, vote `RECOMMENDATION: no` or
`RECOMMENDATION: tradeoff` and list the concrete missing cases under
`TESTS_TO_RUN:`. This focus is inert prompt guidance only: it grants no
tools and no write capability.

This example mirrors the spirit of the built-in `test-gap-analysis` mode as
an editable template; it does not replace that mode.
