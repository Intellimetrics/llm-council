# Native model configuration

Verified September 5, 2026. A model's presence in public documentation does
not establish access for every account. Primary `claude`, `codex`, and
`antigravity` participants keep `model: null` so the operator's CLI selection
remains authoritative. Use an explicit model pin or a named tier when a review
needs a reproducible model choice.

| Participant | Current options | Council defaults |
| --- | --- | --- |
| Codex | GPT-6 Astra, GPT-5.6 Sol, Terra, Luna | Inherit primary; fallbacks `gpt-5.6-terra`, `gpt-5.6-luna` |
| Claude | Fable 5.1, Opus 5, Sonnet 5, Haiku 4.5 | Inherit primary; fallbacks `claude-opus-5`, `claude-sonnet-5` |
| Claude Fable | `claude-fable-5-1` | Explicit pin with JSON model verification; requires Claude Code 2.1.255+ |
| Antigravity | Gemini 3.8 Flash, with account-specific effort variants | Inherit primary; no portable fallback chain |

OpenAI documents the August 31 retirement of GPT-5.4 and GPT-5.4-mini for
Codex with ChatGPT sign-in, with Terra/Luna replacements. API-key authentication
is unaffected. Exact previously shipped fallback chains are migrated on load;
custom chains and explicit pins remain unchanged. [OpenAI models](https://learn.chatgpt.com/docs/models)

Claude's `fable` alias now resolves to Fable 5.1 on current clients, while
`claude-fable-5` remains an explicit older pin. This project uses the full 5.1
ID for its built-in Fable peer and preserves older pins in project config.
Claude Code's native fallback flag accepts only one fallback; a later chain
entry is not a second automatic retry. [Claude model configuration](https://code.claude.com/docs/en/model-config)

Google's current API catalog lists `gemini-3.8-flash`. Antigravity uses its own
model identifiers/display names: run `agy models` on the intended account and
use the exact supported selection. API model names do not establish Antigravity
availability. Consumer Gemini CLI access retired on June 18; Code Assist
Standard/Enterprise subscriptions were not affected.
[Gemini models](https://ai.google.dev/gemini-api/docs/models),
[Google's retirement notice](https://developers.google.com/gemini-code-assist/docs/deprecations/code-assist-individuals)

For a named tier, configure only the peers you intend to change:

```yaml
defaults:
  tiers:
    fast:
      codex: gpt-5.6-luna
      claude: claude-sonnet-5
    deep:
      codex: gpt-6-astra
      claude: claude-opus-5
```

Select with `--tier fast` or `--tier deep`. These examples are opt-in; they do
not change the default roster or guarantee account access. Fallbacks and higher
capability tiers can change latency and usage, so compare them on representative
reviews before choosing a project default.
