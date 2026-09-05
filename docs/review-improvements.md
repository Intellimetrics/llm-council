# September 2026 review and improvement priorities

LLM Council's purpose is to give the host independent, grounded second opinions
while keeping the reviewed project read-only. More agreement is useful only
when it improves the decision. The changes below address correctness and
efficiency; the proposed next steps need measured review-quality outcomes.

## Implemented in this patch

| Problem | Result |
| --- | --- |
| Saved CLI responses ignored source edits outside the prompt | Native CLI peers run fresh by default; custom prompt-only CLIs can explicitly opt into response caching. |
| OKF paths could escape the project, and input caps applied after expensive reads/walks | Resolve and check containment, skip linked concept trees, bound reads and enumeration before allocation. |
| Prompt preparation blocked concurrent MCP requests | Worker execution preserves request-local environment and waits for generator cleanup on cancellation. |
| A saved bundle at HEAD looked fresh for uncommitted changes | Only a bundle generated for this run is labeled fresh. |
| Default native fallback models were outdated | Updated Codex/Claude defaults and migration of exact old shipped chains; documented current Gemini routes in [native models](native-models.md). |
| A related model ID satisfied an explicit model pin | Require exact identity or a dated snapshot of that identity. |
| Search exposed the first peer's vote | Use the canonical final-round decision, including ties and directional leanings. |
| Every search reparsed every transcript | Reuse compact records for unchanged files, invalidate on edits/deletions, and bound retained index entries. |
| FINDINGS could claim verified agreement without a successful citation check | Check findings directly against the project even if absent from EVIDENCE; share line counts within the run. |
| Cache receipts were counted as new spend/retry activity | Exclude reused receipts from incurred usage and expose cache-hit counts. |
| Client names overstated vendor diversity | Prefer served/configured model IDs for known vendors, then normalized families; keep the signal advisory. |
| Antigravity lacked a tool-review directive and peers were invited to call an unregistered tool | Give Antigravity verification guidance and describe structured voting as response text. |

Removed the unused deliberation helper. Regression tests use local fixtures and
stub processes, including source edits between repeated reviews, escaped paths,
large concept files, process cancellation, false citations, and routed models.
Native model access still depends on the operator's account. Subsequent live
dogfooding used native CLI accounts; no hosted paid API peers were added. CLI
usage telemetry was unavailable, so these runs do not establish zero cost.

Validation on September 5, 2026:

- `uv run pytest -q`: **1,460 passed, 3 skipped** in 62.31 seconds after the deadline fixes.
- `uv build`: source distribution and wheel built successfully.
- `git diff --check`: passed.
- Local synthetic allocation check on a 32 MiB concept file: old
  `read_text()[:16384]` peaked at 64.005 MiB; the bounded read peaked at
  0.036 MiB, measured with `tracemalloc`.
- Local synthetic search over 200 transcripts with approximately 20 KB of
  response text each: 9.29 ms cold and 1.19 ms warm. These single-machine
  measurements illustrate the mechanism, not production latency guarantees.

## Subsequent dogfooding and open OKF work

The installed MCP server was verified separately from the checkout: deadline
fixtures preserved completed votes, and fresh native runs confirmed connectivity
and file access. See [request limits](request-limits.md) and [review context](review-context.md).

OKF attaches useful navigation context, but a small live A/B found the same
defect with it off and on. Remaining work: reserve space after/before diff
trimming consistently, spread symbol coverage across changed files, reduce
false call edges, and avoid flagging generated Git revisions as secrets. These
are findings, not fixes already implemented.

## Next improvements, in priority order

1. **Measure useful review findings.** Create a small, versioned set of repository
   snapshots with known defects and clean controls. Score defect precision and
   recall, false blockers, valuable minority findings, elapsed time, and incurred
   tokens/cost. Compare one peer, quick council, and deliberate council on the same
   cases. Existing behavioral tests establish orchestration correctness, not
   whether extra peers improve review quality. Use the results before changing
   roster size, model tiers, or default round counts.

2. **Distinguish issue agreement from shared location.**
   [Finding clustering](../llm_council/findings.py) uses compatible severity and
   overlapping verified source ranges. Two different defects in one function
   can therefore look like corroboration. Add deterministic examples of both
   duplicate and distinct claims, then evaluate claim matching or an explicit
   issue-confirmation round. Preserve uncertain matches as separate concerns.
   A valid file range proves location exists; it does not prove the claim.

3. **Spend follow-up rounds on unresolved evidence.**
   [Deliberation](../llm_council/deliberation.py) and
   [convergence](../llm_council/convergence.py) can be evaluated against a narrower
   follow-up: ask dissenting peers to verify one concrete blocker or citation.
   Preserve minority concerns and stop when a further round produces no new
   verified evidence. Compare improvement per extra second/token against the
   current all-peer rounds; keep existing budgets and maximum rounds as limits.

4. **Record the source state reviewed.** Attach a revision, diff digest, and
   start/end source-state check to a run. If files change during the council,
   surface that fact with the decision instead of presenting citations as a
   stable snapshot. This also supplies an auditable basis for any future native
   response-cache opt-in. Do not silently reuse a result based only on prompt
   text or Git HEAD.

5. **Make native capability checks explicit.**
   [Model discovery](../llm_council/model_catalog.py) mainly supports hosted
   catalogs. Add native client-version/model-list metadata where supported and
   record when it was checked. Distinguish unavailable model, expired login,
   quota exhaustion, and missing tool support. Metadata discovery should never
   require a billable review prompt. Model-vendor inference remains a heuristic
   for private IDs and clients without model telemetry.

6. **Present an evidence-oriented decision brief.** Build a compact result from
   existing [finding matrices](../llm_council/findings.py) and
   [synthesis](../llm_council/synthesis.py): supported findings, minority concerns,
   missing evidence, and next verification steps with transcript links. Show
   coverage and substitutions alongside agreement so a majority is not mistaken
   for proof. Keep the existing machine-readable response contract compatible.

These are proposed follow-up features, not claims of functionality added here.
The evaluation set is the first priority because it makes later changes
measurable without assuming more models or more rounds are better.
