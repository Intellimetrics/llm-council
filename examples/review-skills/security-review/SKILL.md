---
name: security-review
description: Read-only security scrutiny lens — authz/authn gaps, injection, secret leakage, unsafe deserialization. Composes with any mode.
---

SECURITY REVIEW FOCUS (read-only, advisory).

Scrutinize the changes for security weaknesses and flag — do NOT propose or
apply edits. Concentrate on:

- Authentication and authorization gaps: missing/incorrect access checks,
  privilege escalation, IDOR, trust placed in client-supplied identity.
- Injection: SQL/command/template/log injection, unsanitized input reaching
  an interpreter, unsafe string interpolation into queries or shell.
- Secrets: credentials, API keys, tokens, or private keys committed to the
  repo, logged, or echoed into error messages.
- Unsafe deserialization / dynamic code: `pickle`, `eval`, `exec`,
  `yaml.load` (non-safe), arbitrary `__import__`, untrusted format strings.
- Input validation and output encoding: missing bounds checks, path
  traversal, SSRF, open redirects, missing XSS/output escaping.
- Cryptography: weak/absent hashing for secrets, hardcoded IVs/salts, use of
  broken primitives (MD5/SHA1 for passwords), insecure randomness.

For every concern, cite the specific `file:line` and prefer
`[VERIFIED:path:start-end]` evidence tags so the orchestrator can
mechanically verify the location. If you find a hard security blocker, vote
`RECOMMENDATION: no` (or `tradeoff` if it is present but mitigable) and list
it under `BLOCKERS:`. This focus is inert prompt guidance only: it grants no
tools and no write capability — keep all suggestions read-only.
