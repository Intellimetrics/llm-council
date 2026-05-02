"""Default configuration for llm-council.

The defaults assume the user has native CLI access to Claude Code, Codex, and
Gemini CLI. OpenRouter and local providers are available only when explicitly
selected by mode or participant name.
"""

VALID_STANCES = ("for", "against", "neutral")

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

DEFAULT_CONFIG: dict = {
    "version": 1,
    "transcripts_dir": ".llm-council/runs",
    "defaults": {
        "mode": "quick",
        "read_only": True,
        "synthesize": False,
        "origin_policy": "any",
        "max_concurrency": 4,
        "transparent": False,
        "max_deliberation_rounds": 2,
        "convergence_thresholds": {"converged": 0.80, "refining": 0.50},
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
            "model": None,
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "read_only": True,
            "stdin_prompt": True,
            "env_passthrough": ["ANTHROPIC_API_KEY"],
        },
        # Temporary: pinned-version Claude participants for opt-in head-to-head
        # review when the user wants a specific Opus version's perspective.
        # Remove (or merge back into `claude`) once version-to-version drift is
        # no longer interesting. Both reuse the host `claude` CLI; only the
        # `--model` flag differs from the default `claude` participant.
        "claude_4_6": {
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
            "model": "claude-opus-4-6",
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "read_only": True,
            "stdin_prompt": True,
            "env_passthrough": ["ANTHROPIC_API_KEY"],
        },
        "claude_4_7": {
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
            "model": "claude-opus-4-7",
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "read_only": True,
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
            "model": None,
            "timeout": 240,
            "max_prompt_chars": 120_000,
            "read_only": True,
            "stdin_prompt": True,
            "env_passthrough": ["OPENAI_API_KEY"],
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
            "read_only": True,
            "stdin_prompt": True,
            "env_passthrough": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
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
            "read_only": True,
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
            "read_only": True,
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
            "read_only": True,
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
            "read_only": True,
        },
        "qwen_coder_free": {
            "type": "openrouter",
            "family": "qwen",
            "origin": "China / Alibaba Qwen",
            "model": "qwen/qwen3-coder:free",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout": 180,
            "read_only": True,
            "deprecated": "Account-dependent free route; use qwen_coder_flash for reliable cheap defaults.",
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
            "read_only": True,
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
            "read_only": True,
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
            "read_only": True,
        },
        "local_qwen_coder": {
            "type": "ollama",
            "family": "qwen",
            "origin": "China / Alibaba Qwen",
            "model": "qwen3-coder-next:q4_K_M",
            "base_url": "http://localhost:11434",
            "timeout": 180,
            "read_only": True,
        },
    },
    "modes": {
        "quick": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "description": "Ask Claude, Codex, and Gemini as explicit council participants.",
        },
        "peer-only": {
            "strategy": "other_cli_peers",
            "include_current": False,
            "description": "Ask only the other native CLIs, excluding the current host.",
        },
        "plan": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "add": ["deepseek_v4_pro"],
            "description": "Claude/Codex/Gemini plus DeepSeek for independent planning.",
        },
        "review": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "add": ["qwen_coder_plus"],
            "description": "Claude/Codex/Gemini plus Qwen coding model.",
        },
        "review-cheap": {
            "participants": [
                "deepseek_v4_flash",
                "qwen_coder_flash",
                "glm_4_7_flash",
            ],
            "description": "Cheap hosted breadth reviewers.",
        },
        "diverse": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "add": ["deepseek_v4_pro", "glm_5_1", "kimi_k2_6"],
            "description": "Native triad plus cross-lab planning diversity.",
        },
        "private-local": {
            "participants": ["local_qwen_coder"],
            "description": "Local-only private pass. Requires the model/runtime to exist.",
        },
        "us-only": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "origin_policy": "us",
            "description": "Use only US-origin native CLI participants.",
        },
        "deliberate": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "add": ["deepseek_v4_pro"],
            "deliberate": True,
            "description": "Expensive opt-in second round when first-round responses disagree.",
        },
        "consensus": {
            "strategy": "other_cli_peers",
            "include_current": True,
            "stances": {
                "claude": "for",
                "codex": "against",
                "gemini": "neutral",
            },
            "description": (
                "Assigned-stance debate to attack groupthink and sycophancy. "
                "Each native CLI peer takes a for/against/neutral role; the "
                "ethical-override clause keeps any peer from defending a "
                "harmful proposal or contriving false objections."
            ),
        },
        # Temporary: head-to-head review using both Opus versions. Remove or
        # collapse once version drift between 4.6 and 4.7 is no longer notable.
        "opus-versions": {
            "participants": ["claude_4_6", "claude_4_7"],
            "description": "Compare Claude Opus 4.6 vs 4.7 directly. Temporary; may be removed.",
        },
    },
}
