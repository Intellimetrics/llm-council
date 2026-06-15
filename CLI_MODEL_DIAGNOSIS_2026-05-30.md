# CLI Participant Model Diagnosis — codex / gemini / Antigravity

**Date:** 2026-05-30
**llm-council version:** 0.13.0 (via `council_doctor`)
**Host repo used as test bed:** `/development/projects/active/nachnitsa` (its `.llm-council.yaml`)
**Method:** live `council_run`s (mode `quick`, explicit participants) plus black-box
inspection of each CLI's config, environment, and runtime logs.

> ⚠️ This file is untracked and was already removed once by a build/`git clean` in
> this repo (2026-05-30 ~06:36). **Commit it** if you want it to survive.

> Scope note: diagnosed **black-box** — I did not read the engine source while writing
> this. Every claim is tied to an observable artifact (a config file, an env var, a log
> line, or a field in the council's own JSON output) so you can confirm each against the
> engine code.

---

## TL;DR

| Participant | Intended model | Actually ran | Verified from | Verdict |
|---|---|---|---|---|
| `codex` | GPT-5.5 | `gpt-5.5` | `~/.codex/config.toml: model = "gpt-5.5"`, no `-m` override in council args | ✅ correct |
| `gemini` (vanilla) | — | `gemini-3.1-pro-preview` | fresh session files a run wrote: `~/.gemini/tmp/nachnitsa/chats/session-*.jsonl` | ⚠️ host-pinned, not what the operator wanted |
| Antigravity (`agy`) | Gemini 3.5 Flash | `Gemini 3.5 Flash (Medium)` | `~/.gemini/antigravity-cli/log/cli-*.log` | ✅ but not wired into the council by default |

**Root cause, one sentence:** for `type: cli` participants, llm-council neither
**controls** nor **reports** the model — it runs whatever ambient config/env each CLI
happens to have, and surfaces `model: null` for the result.

`codex` only "passed" by luck: nothing overrode its `config.toml` default.
The `gemini` participant ran `gemini-3.1-pro-preview` because the host shell pins it
(`~/.bashrc:130 → export GEMINI_MODEL="gemini-3.1-pro-preview"`), and it invokes the
vanilla `@google/gemini-cli` (`bundle/gemini.js`) — a different binary from Antigravity
(`agy`, `~/.local/bin/agy`).

---

## Findings (engine-level, prioritized)

### 1. CLI participant model is invisible in council output — **highest impact**
- **Symptom:** `council_run` `results[].model` was `null` for every CLI peer; progress
  events likewise show `"model": null`; the transcript renders `- Model: cli default`.
- **Why it matters:** a multi-model council whose value is "independent second opinions
  from different models" cannot tell you which model actually answered. To verify, I had
  to grep each CLI's *private* session logs by hand.
- **Fix direction:** populate `results[].model` — either record what you *passed* via a
  model flag (see #2), or parse the session/transcript the CLI just wrote (gemini-cli:
  `~/.gemini/tmp/<project>/chats/session-*.jsonl`; agy:
  `~/.gemini/antigravity-cli/log/cli-*.log` line `Propagating selected model override … label=…`).

### 2. The `model:` config field is **not enforced** for `type: cli`
- **Evidence:** `claude_4_6` / `claude_4_7` set `model: claude-opus-4-6` / `-4-7`, but
  their `args` contain **no `--model` flag** — same for `gemini` / `codex`. So for CLI
  participants `model:` is metadata only; it's a real selector solely for the
  `openrouter` type (e.g. `deepseek_v4_pro`).
- **Consequence:** a CLI peer runs whatever ambient model its own config/env dictates.
- **⚠️ Check `opus-versions`:** if it relies on `model:` to compare 4.6 vs 4.7 but never
  injects `--model`, both peers run the same ambient Claude model — a no-op.
- **Fix:** for `type: cli`, translate `model:` into the family flag — claude `--model`,
  codex `-m`, gemini `-m`. (Caveat: **agy has no model flag** — its model is set in
  `~/.gemini/antigravity-cli/settings.json`. Model-pinning is per-family, not universal.)

### 3. Ambient environment overrides council intent (env bleed)
- **Symptom:** the `gemini` peer ran `gemini-3.1-pro-preview` purely because the host
  exports `GEMINI_MODEL` (`~/.bashrc:130`, mirrored in `~/.gemini/settings.json`).
- **Evidence of bleed:** the participant's `env_passthrough` lists only the API keys, yet
  `GEMINI_MODEL` still reached the subprocess — so `env_passthrough` is **additive over
  the inherited environment**, not an allowlist. Every host env var leaks in.
- **Consequence:** model selection is host-dependent and silent. Same config, two
  machines → two different "gemini" models.
- **Fix:** run participants with a scrubbed env (only explicitly passed vars), or at least
  unset known model-selection vars (`GEMINI_MODEL`, …) and set the model by flag (#2).

### 4. Codex `--ephemeral` leaves no model trace
- **Symptom:** codex ran `exec --sandbox read-only --ephemeral`; afterward nothing under
  `~/.codex` recorded the run's model (only `models_cache.json`, the catalog). Only
  `config.toml`'s `model = "gpt-5.5"` made it verifiable — and only because nothing
  overrode it. Combined with #1, codex's model is doubly unverifiable from council output.
- **Fix:** pass `-m gpt-5.5` explicitly and record it (ties to #1/#2).

### 5. Antigravity needs a real participant kind — and the naive wiring **deadlocks**
- **What it is:** `agy` v1.0.3 (`~/.local/bin/agy`), a Go binary, **separate** from the
  `gemini` participant (`@google/gemini-cli`). `agy` is already on **Gemini 3.5 Flash
  (Medium)** — confirmed in `~/.gemini/antigravity-cli/log/cli-*.log`
  (`label="Gemini 3.5 Flash (Medium)"`).
- **Non-interactive mode:** `agy --print` / `-p`; reads the prompt from **stdin**.
- **The deadlock (verified):**
  - `… | agy --print --sandbox` → **exit 124 (hang)** whenever the agent wants a tool
    (no headless permission approval).
  - `… | agy --print --sandbox --dangerously-skip-permissions` → **exit 0**, correct answer.
  - A context-only / no-tool prompt completes fine under plain `--print --sandbox`.
- **No read-only-tools mode:** unlike gemini (`--approval-mode plan`) or claude
  (`--tools Read,Grep,…`), agy offers only `--sandbox` (terminal restriction) and
  `--dangerously-skip-permissions` (auto-approve *everything*, incl. writes). "Read-only"
  is not CLI-enforceable for agy.
- **Timeout mismatch:** agy `--print-timeout` defaults to **5m**; a participant
  `timeout: 240` would kill a slow tool-run early.
- **Working config (verified live in nachnitsa):**
  ```yaml
  command: /home/clindell/.local/bin/agy   # absolute; ~/.local/bin may be off the server PATH
  args: [--print, --sandbox, --dangerously-skip-permissions]
  stdin_prompt: true
  timeout: 300
  ```
- **NB the engine already has scaffolding:** the test suite writes
  `.llm-council/skills/antigravity/GEMINI.md` and `.llm-council/instructions/antigravity.md`
  (seen under `pytest-*` tmp dirs), so a first-class `antigravity` participant kind looks
  half-built already.

### 6. Participant self-reports are worthless (informational — not a bug)
Self-ID was wrong **3 for 3**:

| Participant | Self-reported | Real model |
|---|---|---|
| codex | `GPT-5` | `gpt-5.5` |
| gemini | `gemini-2.5-pro` | `gemini-3.1-pro-preview` |
| agy | `gemini-3.1-pro-preview` (echoing inherited `GEMINI_MODEL`) | `Gemini 3.5 Flash (Medium)` |

**Implication:** never derive a participant's model from its text. Model must come from
config/flags/logs (#1).

### 7. Envelope strictness forces a retry even on trivial prompts (minor friction)
Both peers logged `[recovered after retry] First attempt was missing the required
RECOMMENDATION label` — an extra CLI round-trip per peer because the lightweight answer
omitted the envelope. Working as designed, but it doubles latency/cost on lightweight
queries; consider relaxing `require_sections` when there's nothing to match.

### 8. ~~Parsed-config cache is never invalidated~~ — **RETRACTED (misdiagnosis)**
**This was wrong — there is no cache bug.** The persistent `Unknown participant
'antigravity'` was a config error, not stale state.
- **Real cause:** `other_cli_peers` builds the triad and **auto-prefers `antigravity`
  when `agy` is on PATH** (`config.py`: `neutral_peer = "antigravity" if (has_agy or not
  has_gemini)`). This test-bed config sets `replace_defaults: true`, which drops the
  built-in `antigravity` default (`defaults.py`). With no `antigravity` participant
  defined, the triad resolves a peer that doesn't exist → `Unknown participant`.
- **Why `list_modes` and `run` "disagreed":** they do different things, not different
  caching. `council_list_modes` echoes the **raw** mode config (`strategy: other_cli_peers`);
  `council_run` **resolves** that strategy into the triad (→ `antigravity`). No shared
  cache is involved.
- **Config is read fresh per call — verified:** adding the `antigravity` participant took
  effect on the **very next** `council_run` with **no server restart**. (The restart in
  this session was only to load v0.14.0 from 0.13.0, unrelated.)
- **Real fix (config-side):** with `replace_defaults: true`, you must define an
  `antigravity` participant mirroring `defaults.py`. v0.14.0 already ships that default;
  only `replace_defaults` configs need to add it back.
- **Possible engine nicety:** when the triad selects a peer with no participant definition,
  the error could name the cause (`other_cli_peers selected 'antigravity' (agy on PATH) but
  no 'antigravity' participant is defined — replace_defaults dropped the default`) instead
  of the bare `Unknown participant`.

> **v0.14.0 status of the other findings (observed live):** #5 (Antigravity support) is
> **shipped** — first-class `antigravity` participant + `agy` mapping + no-`--model`-flag
> handling in `adapters.py`. #1 (model invisible) is **partially** addressed: the
> transcript now honestly reads `Model: cli default (unreported)` for CLI peers rather
> than a misleading value (it does not yet recover the real model from CLI session logs).
> #2/#3/#4 (model-flag injection, env scrub, codex `-m`) still apply for deterministic,
> reported CLI models.

---

## What worked out of the box
- `codex` ran the correct model (`gpt-5.5`).
- Peers responded; quorum met; `recommendation: yes`.
- The repair-retry recovery path functioned.
- Read-only/sandbox dispatch ran without incident.
- `agy --print --sandbox --dangerously-skip-permissions` drives cleanly as a generic
  `cli` participant (verified end-to-end through the council on Gemini 3.5 Flash).

---

## Proposed engine changes (summary)
1. For `type: cli`, inject the family model flag when `model:` is set
   (claude `--model`, codex `-m`, gemini `-m`; agy pinned via its settings file).
2. Scrub/override model-selection env vars before launch (at least `GEMINI_MODEL`).
3. Record the effective model into `results[].model` (stop printing `cli default`).
4. Add a first-class `antigravity`/agy participant kind (the scaffolding is partly there);
   default its invocation to `--print --sandbox --dangerously-skip-permissions`,
   `stdin_prompt`, `timeout >= 300`.
5. Invalidate the parsed-config cache on change and unify the `list_modes` / `run`
   config-load paths.

---

## Reproduction / evidence commands

```bash
council_doctor                                  # → version 0.13.0
readlink -f "$(command -v gemini)"              # @google/gemini-cli/bundle/gemini.js (NOT antigravity)
grep -n GEMINI_MODEL ~/.bashrc                  # line 130: export GEMINI_MODEL="gemini-3.1-pro-preview"
head -1 ~/.codex/config.toml                    # model = "gpt-5.5"
agy --version                                   # 1.0.3
cat ~/.gemini/antigravity-cli/settings.json     # "model": "Gemini 3.5 Flash (Medium)"
grep -i 'selected model override' ~/.gemini/antigravity-cli/log/cli-*.log | tail -1

# model invisible in council output: results[].model == null; transcript "Model: cli default"
ls -t ~/.gemini/tmp/nachnitsa/chats/session-*.jsonl | head -2   # → gemini-3.1-pro-preview

# agy headless behavior
printf 'reply ONE line: ping\n' | agy --print --sandbox                                            # ok (no tool)
printf 'read CLAUDE.md, reply one line\n' | agy --print --sandbox                                   # exit 124 HANG
printf 'read CLAUDE.md, reply one line\n' | agy --print --sandbox --dangerously-skip-permissions    # exit 0 OK

# stale config cache (finding #8): edit .llm-council.yaml, then
council_list_modes        # reflects the edit
council_run mode=quick dry_run   # still resolves the OLD peer set until server restart
```
