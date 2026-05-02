"""Setup helpers for llm-council."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_council.config import deep_merge
from llm_council.defaults import DEFAULT_CONFIG


OPENROUTER_PARTICIPANTS = (
    "deepseek_v4_pro",
    "deepseek_v4_flash",
    "qwen_coder_plus",
    "qwen_coder_flash",
    "glm_5_1",
    "glm_4_7_flash",
    "kimi_k2_6",
)


INSTRUCTION_TEXT = """# LLM Council

This project has LLM Council installed. Use it as a read-only second-opinion
system when the user wants more than this single agent's judgment.

Natural triggers:
- "use council"
- "go to council"
- "ask council"
- "take this to council"
- "get another model's opinion"
- "have Claude/Gemini/Codex review this"

When triggered, call the `llm-council` MCP tool `council_run`.

Routing rules:
- Always pass `current` as `{current}` so transcripts show which host will
  synthesize and act on the council output.
- Use `quick` mode unless the user names a mode. `quick` asks Claude, Codex,
  and Gemini as explicit read-only participants. Use `peer-only` only when the
  user specifically wants to exclude this host CLI from subprocess review.
- Treat "on the diff", "current diff", or "review my changes" as
  `include_diff: true`.
- Treat "cheap" or "budget" as `review-cheap`.
- Treat "private", "local", or "offline" as `private-local`.
- Treat "with deepseek" as including `deepseek_v4_pro`.
- Treat "with qwen" as including `qwen_coder_plus`.
- Treat "with glm" as including `glm_5_1`.
- Treat "with opus 4.6" or "with claude 4.6" as including `claude_4_6`.
- Treat "with opus 4.7" or "with claude 4.7" as including `claude_4_7`.
- Treat "compare opus versions", "opus 4.6 vs 4.7", or "both opus versions"
  as `opus-versions` mode (head-to-head; temporary feature).

Reviewing UI, screenshots, or browser state:
- Council CLI participants share the project filesystem, so they can Read any
  file you stage. Inline screenshot bytes that live only in this agent's
  conversation context cannot be seen by council.
- Before calling `council_run`, save each image to
  `.llm-council/inputs/<short-slug>/<name>.png`. Use whatever capture tool
  you already have (Playwright with an explicit `path:` arg, claude-in-chrome
  `gif_creator`, or a `Bash`/`Write` step that decodes base64 from a prior
  tool result).
- Pass the relative paths in `image_paths` when calling `council_run`. CLI
  participants will Read the file with their own tools; do not inline the
  image into the question.
- If your environment cannot write to disk, fall back to passing
  `images: [{{ data: <base64>, mime: "image/png" }}]` in the `council_run`
  call. llm-council stages those bytes under `.llm-council/inputs/<run-id>/`
  before participants run. Per-image cap is 8 MB; total cap is 32 MB.
- Hosted/local LLM participants see images only when their config has
  `vision: true`; otherwise they get the text reference list and council
  emits an `images_skipped` progress event for that participant.
- Treat staged screenshots with the same care as `context_files`: redact or
  omit screens that capture credentials, session tokens in URLs, or customer
  data, and respect `DEPLOY_MODE=secret`.

Use council for:
- architecture or design decisions
- risky or cross-cutting refactors
- security-sensitive code paths
- database migrations
- release-gate reviews
- stubborn bugs after a failed attempt
- plans where independent disagreement would be useful

Do not use council for trivial formatting, obvious syntax fixes, or exact
mechanical edits the user already specified.

Before acting on council feedback:
- Summarize the main agreements, disagreements, and concrete risks.
- Identify which recommendations you will follow.
- Ask before making large or risky edits unless the user already authorized
  implementation.

Data boundary:
- Do not send classified, CUI, regulated, production, secret, credential, or
  customer data to council unless the user explicitly confirms the configured
  participants are approved for that data.
- Do not include files, diffs, logs, or environment content marked secret,
  sensitive, private, or deployment-only.
- US-origin participants mean model/company origin only; that is not the same
  as GovCloud, FedRAMP, or an enterprise data-handling approval.

Council is advisory and read-only by default. Council participants should not
edit files; this agent remains responsible for deciding what to do next.
"""


def project_config(
    include_native: bool = True,
    include_openrouter: bool = True,
    include_local: bool = True,
    us_only_default: bool = False,
) -> dict:
    participant_names: set[str] = set()
    if include_native:
        participant_names.update({"claude", "codex", "gemini"})
        # Temporary: ship pinned-version Claude variants so `opus-versions`
        # mode is reachable without manual config edits. Remove when version
        # drift no longer warrants direct comparison.
        participant_names.update({"claude_4_6", "claude_4_7"})
    if include_openrouter:
        participant_names.update(OPENROUTER_PARTICIPANTS)
    if include_local:
        participant_names.add("local_qwen_coder")

    modes = {
        name: mode
        for name, mode in DEFAULT_CONFIG["modes"].items()
        if _mode_available(mode, participant_names, include_native=include_native)
    }
    if not include_native and include_openrouter:
        modes.update(_openrouter_only_modes())
    config = {
        "version": 1,
        "replace_defaults": True,
        "transcripts_dir": ".llm-council/runs",
        "defaults": dict(DEFAULT_CONFIG["defaults"]),
        "participants": {},
        "modes": modes,
    }
    if not include_native and include_openrouter:
        config["defaults"]["mode"] = "quick"
    if us_only_default:
        config["defaults"]["origin_policy"] = "us"
        modes.pop("us-only", None)
    if include_native:
        for name in ("claude", "codex", "gemini", "claude_4_6", "claude_4_7"):
            config["participants"][name] = DEFAULT_CONFIG["participants"][name]
    if include_openrouter:
        for name in OPENROUTER_PARTICIPANTS:
            config["participants"][name] = DEFAULT_CONFIG["participants"][name]
    if include_local:
        config["participants"]["local_qwen_coder"] = DEFAULT_CONFIG["participants"][
            "local_qwen_coder"
        ]
    return config


def _mode_participants(mode: dict[str, Any]) -> set[str]:
    names = set(mode.get("participants", []))
    names.update(mode.get("add", []))
    return names


def _mode_available(
    mode: dict[str, Any], participant_names: set[str], *, include_native: bool
) -> bool:
    if mode.get("strategy") == "other_cli_peers" and not include_native:
        return False
    return _mode_participants(mode).issubset(participant_names)


def _openrouter_only_modes() -> dict[str, dict[str, Any]]:
    return {
        "quick": {
            "participants": [
                "deepseek_v4_flash",
                "qwen_coder_flash",
                "glm_4_7_flash",
            ],
            "description": "Hosted OpenRouter breadth reviewers.",
        },
        "plan": {
            "participants": [
                "deepseek_v4_pro",
                "qwen_coder_plus",
                "glm_5_1",
            ],
            "description": "Hosted OpenRouter planning reviewers.",
        },
        "review": {
            "participants": [
                "deepseek_v4_flash",
                "qwen_coder_plus",
                "glm_4_7_flash",
            ],
            "description": "Hosted OpenRouter code review.",
        },
    }


def mcp_config(project_root: Path) -> dict[str, Any]:
    local_python = project_root / ".venv" / "bin" / "python"
    installed = shutil.which("llm-council")
    if installed:
        command = installed
        args = ["mcp-server"]
    elif local_python.exists():
        command = str(local_python)
        args = ["-m", "llm_council.mcp_server"]
    else:
        command = sys.executable
        args = ["-m", "llm_council.mcp_server"]
    return {
        "mcpServers": {
            "llm-council": {
                "type": "stdio",
                "command": command,
                "args": args,
                "env": {
                    "PYTHONPATH": str(project_root.resolve()),
                    "LLM_COUNCIL_MCP_ROOT": str(project_root.resolve()),
                },
            }
        }
    }


def write_setup_files(
    root: Path,
    *,
    include_native: bool = True,
    include_openrouter: bool = True,
    include_local: bool = True,
    us_only_default: bool = False,
    write_mcp: bool = True,
    write_instructions: bool = True,
    force: bool = False,
) -> list[Path]:
    written: list[Path] = []
    root.mkdir(parents=True, exist_ok=True)

    config_path = root / ".llm-council.yaml"
    desired_config = project_config(
        include_native=include_native,
        include_openrouter=include_openrouter,
        include_local=include_local,
        us_only_default=us_only_default,
    )
    if config_path.exists() and not force:
        existing_config = _read_yaml_mapping(config_path)
        merged_config = deep_merge(desired_config, existing_config)
    else:
        merged_config = desired_config
    if force or not config_path.exists() or merged_config != (
        _read_yaml_mapping(config_path) if config_path.exists() else None
    ):
        config_path.write_text(
            yaml.safe_dump(merged_config, sort_keys=False),
            encoding="utf-8",
        )
        written.append(config_path)

    if write_mcp:
        mcp_path = root / ".mcp.json"
        if mcp_path.exists() and not force:
            existing = _read_json_mapping(mcp_path)
            servers = existing.setdefault("mcpServers", {})
            servers["llm-council"] = mcp_config(root)["mcpServers"]["llm-council"]
            mcp_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        else:
            mcp_path.write_text(
                json.dumps(mcp_config(root), indent=2) + "\n", encoding="utf-8"
            )
        written.append(mcp_path)

    if write_instructions:
        instructions_dir = root / ".llm-council" / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        for name in ("claude", "codex", "gemini"):
            path = instructions_dir / f"{name}.md"
            if force or not path.exists():
                path.write_text(
                    INSTRUCTION_TEXT.format(current=name),
                    encoding="utf-8",
                )
                written.append(path)

        skills_dir = root / ".llm-council" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill_files = _generate_host_skill_files(skills_dir)
        for path, content in skill_files:
            if force or not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                written.append(path)

    runtime_gitignore = root / ".llm-council" / ".gitignore"
    runtime_gitignore.parent.mkdir(parents=True, exist_ok=True)
    desired_gitignore = "runs/\ninputs/\n*.log\n"
    if (
        force
        or not runtime_gitignore.exists()
        or runtime_gitignore.read_text(encoding="utf-8") != desired_gitignore
    ):
        runtime_gitignore.write_text(desired_gitignore, encoding="utf-8")
        written.append(runtime_gitignore)

    project_gitignore = root / ".gitignore"
    if write_mcp:
        updated = _ensure_project_gitignore(project_gitignore)
        if updated:
            written.append(project_gitignore)

    return written


def _ensure_project_gitignore(path: Path) -> bool:
    desired_lines = [
        "# LLM Council local machine config and runtime data",
        ".mcp.json",
        ".llm-council/runs/",
        ".llm-council/inputs/",
        ".llm-council/*.log",
        ".llm-council.env",
    ]
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    missing = [line for line in desired_lines if line not in existing]
    if not missing:
        return False
    lines = existing[:]
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(missing)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse {path}. Fix it or rerun setup with --force.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping. Fix it or rerun setup with --force.")
    return data


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {path}. Fix it or rerun setup with --force.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object. Fix it or rerun setup with --force.")
    return data


def _generate_host_skill_files(skills_dir: Path) -> list[tuple[Path, str]]:
    """Build host-specific agent-skill files for Claude Code, Codex, Gemini.

    These are generated under `.llm-council/skills/<host>/` so the user can
    install them into their host's *global* skill location (vs. the
    per-project `.llm-council/instructions/` that the README install flow
    appends to project files). Each host has its own format.
    """

    files: list[tuple[Path, str]] = []
    files.append((skills_dir / "README.md", _SKILLS_README))
    files.append(
        (skills_dir / "claude-code" / "SKILL.md", _CLAUDE_CODE_SKILL_MD)
    )
    files.append(
        (skills_dir / "codex-cli" / "AGENTS.md", _CODEX_CLI_AGENTS_MD)
    )
    files.append(
        (skills_dir / "gemini-cli" / "GEMINI.md", _GEMINI_CLI_GEMINI_MD)
    )
    return files


_SKILLS_README = """# Host-installable agent skills for llm-council

This directory holds host-specific skill / instruction files you can install
into your coding agent's *global* config so every project picks up the
council routing rules — not just this project. They are independent of the
per-project files in `.llm-council/instructions/` (which the README install
flow appends to your project's `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`).

Pick the host(s) you use:

## Claude Code

```bash
mkdir -p ~/.claude/skills/llm-council
cp .llm-council/skills/claude-code/SKILL.md ~/.claude/skills/llm-council/SKILL.md
```

Restart Claude Code. The skill becomes discoverable; it has YAML frontmatter
naming it `llm-council` so the host can surface it where it surfaces other
skills.

## Codex CLI

```bash
mkdir -p ~/.codex
cat .llm-council/skills/codex-cli/AGENTS.md >> ~/.codex/AGENTS.md
```

Codex CLI reads `~/.codex/AGENTS.md` as global agent instructions. Append
(don't overwrite) so existing entries are preserved.

## Gemini CLI

```bash
mkdir -p ~/.gemini
cat .llm-council/skills/gemini-cli/GEMINI.md >> ~/.gemini/GEMINI.md
```

Gemini CLI reads `~/.gemini/GEMINI.md` as global instructions.

After installing on any host, the prerequisite is that `llm-council` is on
PATH (via `uv tool install` or `pipx install`) and that an MCP transport is
wired up in the host's MCP config. See the project README's primary install
path for that.
"""


_HOST_SKILL_BODY = """When the user asks for a "council" review — natural triggers include
"use council", "go to council", "ask council", "take this to council",
or "get a second opinion" — call the `llm-council` MCP tool `council_run`.

Routing rules:
- Pass `current` as `{current}` so transcripts record which host will
  synthesize and act on the council output.
- Default `mode` is `quick`. Use `consensus` when the user explicitly
  wants assigned-stance debate (for/against/neutral). Use `peer-only`
  when the user wants to exclude this host from the council. Use
  `private-local` for offline/Ollama-only review.
- Treat "on the diff" / "current diff" / "review my changes" as
  `include_diff: true`.
- Treat "with budget" or "max $X" as setting `max_cost_usd` to that
  value — the council will refuse to run if the pre-flight estimate
  exceeds the cap.
- Treat "continue from <run_id>" as setting `continuation_id` to that
  prior run; the new transcript will record `parent_run_id`.

Council output shape (when the host supports MCP outputSchema):
- `recommendation`: yes / no / tradeoff / unknown — the majority label
- `agreement_count`: peers matching the majority
- `degraded`: true when fewer than `min_quorum` peers labeled
- `transcript`: filesystem path to the markdown transcript

Before acting on council feedback, summarize agreements, surface real
disagreements, and ask the user before making large or risky edits.

Council is read-only by default. Council participants must not edit
files; this host agent remains responsible for deciding what to do next.

Do not send classified, regulated, secret, credential, or customer data
to council unless the user has explicitly confirmed the configured
participants are approved for that data.
"""


_CLAUDE_CODE_SKILL_MD = (
    "---\n"
    "name: llm-council\n"
    "description: >-\n"
    "  Read-only multi-agent council for second opinions before risky\n"
    "  edits. Routes to the local CLI triad (Claude / Codex / Gemini) plus\n"
    "  optional OpenRouter / Ollama peers via the llm-council MCP server.\n"
    "---\n"
    "\n"
    "# LLM Council\n"
    "\n"
    + _HOST_SKILL_BODY.format(current="claude")
)


_CODEX_CLI_AGENTS_MD = (
    "# LLM Council (global agent instructions)\n"
    "\n"
    + _HOST_SKILL_BODY.format(current="codex")
)


_GEMINI_CLI_GEMINI_MD = (
    "# LLM Council (global Gemini CLI instructions)\n"
    "\n"
    + _HOST_SKILL_BODY.format(current="gemini")
)
