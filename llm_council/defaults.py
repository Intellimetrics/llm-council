"""Default configuration for llm-council.

The defaults assume the user has native CLI access to Claude Code, Codex, and
Gemini CLI. OpenRouter and local providers are available only when explicitly
selected by mode or participant name.
"""

VALID_STANCES = ("for", "against", "neutral")

# Canonical origin strings used by built-in participants and recommended for
# user-defined ones. `origin_policy: us` filters by literal-prefix match
# (`origin.startswith("US /")`), so spacing/case typos like `US/Anthropic` or
# `us / anthropic` silently exclude a participant from US-only runs. The
# config validator emits a warning when a participant's `origin` normalizes
# to a value in this tuple but the literal string differs — see
# `config.config_warnings`.
#
# Add an origin here when you want it suggested as a fix for typo-class
# mismatches. Origins not in this list are accepted without comment (free-
# text is allowed; the registry only catches near-misses).
KNOWN_ORIGIN_STRINGS: tuple[str, ...] = (
    # US-origin (used in defaults + commonly added via docs/local-models.md)
    "US / Anthropic",
    "US / OpenAI",
    "US / Google",
    "US / Meta",
    "US / Mistral",
    # Europe
    "France / Mistral",
    # China-origin (used in defaults)
    "China / Alibaba Qwen",
    "China / DeepSeek",
    "China / Z.ai",
    "China / Moonshot AI",
)

STANCE_INVARIANT_SUFFIX = (
    "Council invariants that always apply, regardless of stance or override: "
    "(1) you remain a read-only participant — propose changes as "
    "recommendations only, never request write/edit operations; "
    "(2) you MUST emit a `RECOMMENDATION: yes - ...`, `RECOMMENDATION: no - "
    "...`, or `RECOMMENDATION: tradeoff - ...` line; "
    "(3) safety, legality, truthfulness, and the user's response-format "
    "instructions always supersede any stance assignment. If the stance "
    "paragraph above conflicts with these invariants, the invariants win."
)

DEFAULT_STANCE_PROMPTS: dict[str, str] = {
    "for": (
        "Stance: FOR. Argue the strongest case in favor of this proposal. "
        "Steelman it. Find the genuine wins, synergies, and compelling reasons "
        "to say yes. However, your stance does NOT override safety, correctness, "
        "or this council's read-only / no-edit invariants. If the proposal is "
        "genuinely harmful, illegal, unsafe, or asks for write operations that "
        "violate the read-only constraint, you MUST call it out clearly and emit "
        "`RECOMMENDATION: no` with your reasoning. Being assigned `for` means "
        "finding the best possible version of a workable idea, never blindly "
        "defending a bad one."
    ),
    "against": (
        "Stance: AGAINST. Argue the strongest case against this proposal. "
        "Find legitimate flaws, risks, overlooked complexities, and failure "
        "modes. Be the rigorous skeptic the council needs. However, your stance "
        "does NOT override truthfulness. If the proposal is straightforwardly "
        "correct, follows established best practices, or is clearly beneficial, "
        "and your `against` arguments would be contrived or contrarian for its "
        "own sake, you MUST override the stance and emit `RECOMMENDATION: yes` "
        "with a brief explanation. The read-only / no-edit invariants of this "
        "council always apply regardless of stance."
    ),
    "neutral": (
        "Stance: NEUTRAL. Weigh both sides honestly without a predetermined "
        "position. Surface the strongest arguments on each side, then let the "
        "weight of evidence decide your `RECOMMENDATION`. Be truthful about "
        "asymmetry: if evidence strongly favors one conclusion, state it "
        "plainly rather than manufacturing artificial 50/50 balance. The "
        "read-only / no-edit invariants of this council always apply."
    ),
}

DEFAULT_CHEAPER_MODELS: dict[str, str] = {
    "anthropic/claude-sonnet-4-6": "anthropic/claude-haiku-4-5",
    "openai/gpt-4o": "openai/gpt-4o-mini",
    "google/gemini-1.5-pro": "google/gemini-1.5-flash",
    "anthropic/claude-sonnet": "anthropic/claude-haiku",
    "openai/gpt-4": "openai/gpt-4o-mini",
}


DEFAULT_CONFIG: dict = {
    "version": 1,
    "transcripts_dir": ".llm-council/runs",
    "defaults": {
        "mode": "quick",
        "synthesize": False,
        # When synthesize=True, this names the chair. No silent default —
        # `select_synthesizer` raises ValueError when this is None so the
        # requester does not bias the chair by accident. Valid values:
        # a participant name | "neutral_peer" | "current".
        "synthesizer": None,
        # Entire MCP request wall-clock contract. Individual peer timeouts
        # remain independently configurable, but a stuck multi-round request
        # cannot outlive this top-level deadline unless the caller overrides it.
        "mcp_request_timeout_seconds": 1200,
        # Tier-2 secret scanner. "warn" (default) counts likely credentials
        # in the prompt body and emits a progress event but ships the prompt
        # UNCHANGED (no mitigation); "block" raises before any participant
        # runs; "redact" masks each match with [REDACTED:<kind>] in the
        # prompt sent to peers AND persisted to the transcript (the only
        # policy with transcript-level protection); "off" skips entirely.
        # Allowlist (default .llm-council-secrets-allow) covers test fixtures.
        "secret_scan": "warn",
        "secret_scan_allowlist": ".llm-council-secrets-allow",
        # Section-coverage validator. When the user prompt contains
        # `PART N — TITLE (REQUIRED)` headers, each peer response must
        # reference each required section (literal `PART N` or salient
        # title tokens within a 200-char window). Missing sections
        # trigger one section-repair retry, then `error_kind=incomplete_response`.
        # Opt-out for prompts where REQUIRED markers are used for reader
        # clarity but not validation.
        "require_sections": True,
        # Strict evidence-tag enforcement. When True, each EVIDENCE bullet
        # must carry one of [PUBLISHED]/[OBSERVABLE]/[INFERRED]/
        # [SPECULATIVE] tags. Untagged entries trigger one repair retry,
        # then `error_kind=untagged_evidence`. Default False — staged
        # rollout mirrors v0.5.0 envelope optional→required pattern.
        # Watch `evidence_tag_distribution["untagged"]` in stats for two
        # releases before flipping the default.
        "strict_evidence": False,
        "origin_policy": "any",
        "max_concurrency": 4,
        "transparent": False,
        "max_deliberation_rounds": 2,
        "convergence_thresholds": {"converged": 0.80, "refining": 0.50},
        # When True, `llm-council doctor` refreshes a missing/stale OpenRouter
        # catalog inline (best-effort, 10s timeout) instead of asking the user
        # to run `llm-council models refresh` manually. Fail-soft: a network
        # failure falls through to the existing stale-warning Check so a
        # disconnected user still gets a usable diagnostic.
        "catalog_auto_refresh": True,
        "auto_open_browser": False,
        # M6 soft cost-warning threshold (USD). ADVISORY ONLY — never gates a
        # run. When set and the pre-flight estimate (the same retry-safety
        # reduction the hard --max-cost-usd gate uses) is >= this value, a
        # non-fatal `cost_warning` is stamped into run metadata (and a one-line
        # stderr note on the CLI). Complements the hard --max-cost-usd gate,
        # which still runs first and unchanged. None = feature off. An explicit
        # 0 warns on any estimated spend. CLI: --cost-warn-usd; MCP: cost_warn_usd.
        "cost_warn_usd": None,
    },
    "participants": {
        "claude": {
            "type": "cli",
            "family": "claude",
            "origin": "US / Anthropic",
            "command": "claude",
            "args": [
                "-p",
                "--permission-mode",
                "default",
                "--tools",
                "Read,Grep,Glob,LS",
                "--no-session-persistence",
            ],
            # Opt-in: set `usage_from_json: true` (per-peer, in
            # .llm-council.yaml) to invoke claude as `-p --output-format json`
            # and parse REAL token usage + cost into the result. Default off →
            # text-mode invocation, usage unobservable. Read-only flags above
            # are preserved either way (the JSON flag is purely additive).
            "model": None,
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "stdin_prompt": True,
            "env_passthrough": ["ANTHROPIC_API_KEY"],
            # Quota / overload fallback chain — capability-graceful step-down.
            # For Claude family this is consumed natively via the CLI's
            # `--fallback-model` flag (only chain[0] is used; the CLI
            # handles the retry itself). For other families llm-council
            # walks the chain on quota detection, capped at
            # QUOTA_FALLBACK_MAX_STEPS (v0.12.1+ multi-step walking).
            # Empty chain disables fallback. Chain[0] is a same-tier
            # one-version-back step (opus→opus); subsequent entries
            # progressively step down (opus→sonnet→haiku).
            "fallback_chain": ["claude-opus-4-6", "claude-sonnet-4-6"],
        },
        # Claude Fable 5 as a read-only council peer. Fable runs its own
        # safety classifiers on the INCOMING request (research-bio + most
        # cybersecurity content); on a false-positive refusal the Claude Code
        # surface transparently re-serves the request on Opus 4.8 — a SILENT
        # model swap llm-council cannot disable from the CLI. Two settings make
        # that swap safe here rather than invisible:
        #   * usage_from_json: true  -> `_parse_claude_usage_json` reports the
        #     model that ACTUALLY served the turn (Opus, if it fell back), so
        #     `result.model` differs from the requested id whenever a swap
        #     happened.
        #   * require_pinned_model: true -> a served model that doesn't match
        #     the pinned `model` drops the peer (ok=False,
        #     error_kind=model_substituted) so an Opus answer is NEVER recorded
        #     as a "Fable" opinion. See `adapters._run_cli_once`.
        # `fallback_chain` is intentionally EMPTY so `_build_cli_command` does
        # not inject `--fallback-model` — an overload swap would be a second
        # silent-substitution path. Pair with the `fable` mode
        # (`defaults.modes.fable`) for the defensive-review "safe context"
        # framing that lowers the false-positive refusal rate in the first
        # place. Not selected by any built-in mode except `fable`; opt in by
        # naming it in a mode's participants or via `--participants`.
        "claude_fable": {
            "type": "cli",
            "family": "claude",
            "origin": "US / Anthropic",
            "command": "claude",
            "args": [
                "-p",
                "--permission-mode",
                "default",
                "--tools",
                "Read,Grep,Glob,LS",
                "--no-session-persistence",
            ],
            "model": "claude-fable-5",
            # Observability + guard for the silent Fable->Opus refusal fallback.
            "usage_from_json": True,
            "require_pinned_model": True,
            # Empty on purpose: no `--fallback-model` injection (see comment
            # above). Do NOT add entries here without re-reading the swap risk.
            "fallback_chain": [],
            # Fable turns run longer than Opus at equivalent effort; give the
            # base timeout headroom. The `fable` mode layers a 1.5x multiplier
            # and size-scaling applies on top.
            "timeout": 360,
            "max_prompt_chars": 120_000,
            "stdin_prompt": True,
            "env_passthrough": ["ANTHROPIC_API_KEY"],
        },
        "codex": {
            "type": "cli",
            "family": "codex",
            "origin": "US / OpenAI",
            "command": "codex",
            "args": [
                "exec",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--cd",
                "{cwd}",
                "-",
            ],
            # Opt-in: set `usage_from_json: true` (per-peer) to invoke codex as
            # `exec --json` (JSONL stream) and parse REAL prompt/completion
            # tokens (cache-adjusted) into the result. codex reports no cost, so
            # cost_usd stays None. Default off → text-mode, usage unobservable.
            # The read-only `--sandbox read-only` flag above is preserved.
            "model": None,
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "stdin_prompt": True,
            "env_passthrough": ["OPENAI_API_KEY"],
            # Capability-graceful step-down: same-tier minor version back
            # (gpt-5.4 from gpt-5.5), then codex-tuned variant (still
            # capable for coding), then a small final fallback. Users on
            # a different account tier should override in .llm-council.yaml;
            # an unknown model id just makes that step fail and the walk
            # continues (or the peer drops if chain is exhausted).
            "fallback_chain": ["gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini"],
        },
        "gemini": {
            "type": "cli",
            "family": "gemini",
            "origin": "US / Google",
            "command": "gemini",
            "args": [
                "--approval-mode",
                "plan",
            ],
            "model": None,
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "stdin_prompt": True,
            "env_passthrough": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            # Within Google's tiering, Pro > Flash for capability — so
            # falling from 3.5-flash to 3.1-pro is actually an UPGRADE
            # in many tasks while sidestepping the throttled flash quota.
            # Then progressively step down through older flash variants.
            "fallback_chain": [
                "gemini-3.1-pro-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-flash",
            ],
        },
        "deepseek_v4_pro": {
            "type": "openrouter",
            "family": "deepseek",
            "origin": "China / DeepSeek",
            "model": "deepseek/deepseek-v4-pro",
            "input_per_million": 0.435,
            "output_per_million": 0.87,
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "deepseek_v4_flash": {
            "type": "openrouter",
            "family": "deepseek",
            "origin": "China / DeepSeek",
            "model": "deepseek/deepseek-v4-flash",
            "input_per_million": 0.14,
            "output_per_million": 0.28,
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "qwen_coder_plus": {
            "type": "openrouter",
            "family": "qwen",
            "origin": "China / Alibaba Qwen",
            "model": "qwen/qwen3-coder-plus",
            "input_per_million": 0.65,
            "output_per_million": 3.25,
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "qwen_coder_flash": {
            "type": "openrouter",
            "family": "qwen",
            "origin": "China / Alibaba Qwen",
            "model": "qwen/qwen3-coder-flash",
            "input_per_million": 0.195,
            "output_per_million": 0.975,
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "qwen_coder_free": {
            "type": "openrouter",
            "family": "qwen",
            "origin": "China / Alibaba Qwen",
            "model": "qwen/qwen3-coder:free",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "glm_5_1": {
            "type": "openrouter",
            "family": "glm",
            "origin": "China / Z.ai",
            "model": "z-ai/glm-5.1",
            "input_per_million": 1.05,
            "output_per_million": 3.50,
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "glm_4_7_flash": {
            "type": "openrouter",
            "family": "glm",
            "origin": "China / Z.ai",
            "model": "z-ai/glm-4.7-flash",
            "input_per_million": 0.06,
            "output_per_million": 0.40,
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "kimi_k2_6": {
            "type": "openrouter",
            "family": "kimi",
            "origin": "China / Moonshot AI",
            "model": "moonshotai/kimi-k2.6",
            "input_per_million": 0.7448,
            "output_per_million": 4.655,
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
        },
        "local_qwen_coder": {
            "type": "ollama",
            "family": "qwen",
            "origin": "China / Alibaba Qwen",
            "model": "qwen3-coder-next:q4_K_M",
            "base_url": "http://localhost:11434",
            "timeout": 180,
        },
        "antigravity": {
            "type": "cli",
            "family": "antigravity",
            "origin": "US / Google",
            "command": "agy",
            # READ-ONLY ENFORCEMENT IS HARD as of agy 1.1.x: `--mode plan`
            # disables the model's native file-write tool (verified live on
            # 1.1.4: an explicitly ordered write produces no file; agy falls
            # back to a shell command, which headless print mode auto-denies).
            # `--sandbox` additionally restricts the TERMINAL (shell commands).
            # The council prompt's read-only directive (context.build_prompt)
            # remains as defense in depth. We still OMIT
            # --dangerously-skip-permissions so residual tool attempts are
            # denied, not auto-approved — do NOT re-add it. The live canary
            # tests/test_live_agy_readonly.py guards this across upstream
            # releases.
            #
            # PROMPT DELIVERY IS ARGV, not stdin: agy 1.1.1 stopped reading
            # stdin ("-" is now a literal prompt), so the prompt is passed as
            # the --print value. Linux caps a single argv string at 128 KiB
            # (MAX_ARG_STRLEN); max_prompt_chars 120_000 fits for ASCII but a
            # heavily non-ASCII prompt near the cap can fail exec with E2BIG
            # (fail-fast, surfaced as a peer error).
            "args": [
                "--print",
                "{prompt}",
                "--sandbox",
                "--mode",
                "plan",
            ],
            "model": None,
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "stdin_prompt": False,
            "env_passthrough": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTIGRAVITY_API_KEY"],
            # agy 1.0.x accepts --model, but its display-name ids are not
            # portable and an unknown id silently falls back to session
            # state. Keep the built-in chain empty rather than pretend a
            # cross-install fallback is reliable.
            "fallback_chain": [],
        },
    },
    # Mode shape (recognized optional keys per entry):
    #   strategy: "other_cli_peers" | "local_only_peers"     — selection rule
    #   participants: list[str]                              — explicit roster
    #   include_current: bool                                — keep host CLI
    #   add: list[str]                                       — extra peers
    #   origin_policy: "any" | "us"
    #   stances: dict[peer, "for"|"against"|"neutral"]       — debate roles
    #   deliberate: bool                                     — force round 2
    #   max_rounds: int                                      — round cap
    #   min_quorum: int                                      — quorum floor
    #   timeout_multiplier: float                            — per-mode * base
    #   experimental: bool                                   — surfaces a
    #       warning in list-modes / council_list_modes that the mode may
    #       still change or be cut; promoting it to non-experimental is a
    #       manual operator decision. Mode still runs normally.
    #   model_overrides: dict[peer_name, model_id]           — pin per-peer
    #       model for THIS mode only. Resolution order:
    #       participants.<peer>.model (base) -> tiers.<tier>.<peer>
    #       (--tier <name>) -> modes.<name>.model_overrides.<peer>
    #       (highest priority). Override is silent: a stale entry naming
    #       a peer not in the resolved roster is a no-op. Built-in modes
    #       intentionally ship without model_overrides — users add their
    #       own once real-world usage supports the pin.
    #   description: str                                     — human note
    "modes": {
        "quick": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "description": "Ask available native CLI participants from the Claude/Codex and Gemini families.",
        },
        "peer-only": {
            "strategy": "other_cli_peers",
            "include_current": False,
            "description": "Ask only other available native CLI participants, excluding the current host.",
        },
        "plan": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "add": ["deepseek_v4_pro"],
            "description": "Native CLI participants plus DeepSeek for independent planning.",
        },
        "review": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "add": ["qwen_coder_plus"],
            "description": "Native CLI participants plus the Qwen coding participant.",
        },
        # Consult Claude Fable 5 as a single read-only reviewer. `safe_context:
        # true` injects the defensive-review framing (context.build_prompt) that
        # lowers Fable's false-positive safety-classifier refusals — the refusals
        # that otherwise trigger a silent fall-back to Opus. The `claude_fable`
        # peer is pinned + observable: if a swap still slips through it is
        # DETECTED via the CLI-reported model and the peer is dropped
        # (error_kind=model_substituted), never recorded as a Fable vote.
        # Fable-only by design — the current host agent (Opus or another model)
        # is the "orchestrator" seeking Fable's independent second opinion; add
        # cross-check peers in `.llm-council.yaml` if you want breadth.
        "fable": {
            "participants": ["claude_fable"],
            "safe_context": True,
            "timeout_multiplier": 1.5,
            "description": (
                "Read-only second opinion from Claude Fable 5, with "
                "defensive-review framing to reduce false-positive refusals. A "
                "silent fall-back to Opus is detected and dropped, not recorded "
                "as a Fable vote."
            ),
        },
        # Experimental: CLI peers only, with explicit directive to use their
        # file-read / grep / glob tools before voting. The CLIs already have
        # tool access via their sandbox flags (claude `--permission-mode
        # default` + `--tools Read,Grep,Glob,LS`; codex `--sandbox read-only`;
        # gemini `--approval-mode plan`), but the standard prompt never asks
        # them to use them. This mode activates that latent autonomy.
        #
        # Stays `experimental: true` until an operator, on the strength of
        # observed dogfooding results, decides it is reliable enough to
        # promote to non-experimental. Promotion is a manual operator
        # decision, not this mode's defaults.
        "review-with-tools": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "experimental": True,
            "timeout_multiplier": 1.8,
            # v0.9.0 Feature 3 — strictly opt-in tool-call voting. When
            # True, CLI peers (claude/codex/gemini) additionally receive a
            # directive describing a `record_recommendation(verdict,
            # blockers, evidence)` tool they can invoke. The adapter then
            # tries to parse a structured tool-call payload from each
            # peer's stdout and, on success, populates the envelope from
            # that payload instead of (or alongside) the regex
            # `RECOMMENDATION:` label. Default `false`: flipping to
            # default-on is a manual operator decision based on observed
            # reliability vs the regex-only baseline. Operators flip per
            # their verified CLI schema.
            "tool_call_voting": False,
            "description": (
                "EXPERIMENTAL — Claude/Codex/Gemini directed to use their "
                "file-read / grep / glob tools to verify diff claims before "
                "voting. CLI participants only; hosted participants do not participate."
            ),
        },
        "private-local": {
            "strategy": "local_only_peers",
            "description": (
                "All configured same-machine loopback `type: ollama` "
                "participants. Excludes OpenAI-compatible gateways, LAN "
                "endpoints, hosted-inference CLIs (claude/codex/gemini), and "
                "hosted API participants (openrouter). See "
                "docs/local-models.md for adding local-server participants."
            ),
        },
        "deliberate": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "add": ["deepseek_v4_pro"],
            "deliberate": True,
            "timeout_multiplier": 1.5,
            "description": "Expensive opt-in second round when first-round responses disagree.",
        },
        "consensus": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "stances": {
                "claude": "for",
                "codex": "against",
                "gemini": "neutral",
                "antigravity": "neutral",
            },
            "timeout_multiplier": 2.0,
            "description": (
                "Assigned-stance debate to attack groupthink and sycophancy. "
                "Each native CLI peer takes a for/against/neutral role; the "
                "ethical-override clause keeps any peer from defending a "
                "harmful proposal or contriving false objections."
            ),
        },
    },
}
