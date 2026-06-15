"""User-authorable, composable "review focus" bundles.

Operators can author focus directives that compose onto ANY council mode
without editing source. A focus bundle lives at
``.llm-council/review-skills/<name>/SKILL.md`` and consists of YAML-ish
frontmatter (``name:`` + ``description:``) followed by a markdown body.

The bundle body is **INERT PROMPT TEXT only** — it shapes WHAT peers
scrutinize; it NEVER grants a tool or any write/exec capability. Focus is
advisory/read-only and rides on top of the read-only invariant enforced by
the per-CLI permission flags in ``defaults.py`` and the read-only directive
in ``context.build_prompt``.

Discovery is lenient (a malformed bundle is skipped, never raised) so one
bad bundle cannot break a council run. Name validation is strict (M12):
``^[a-z0-9-]+$``, ≤ 64 chars, equal to the directory name. Resolution
(``resolve_focus``) fails fast with ``FocusNotFound`` BEFORE any subprocess
launches when a requested bundle is missing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


REVIEW_SKILLS_DIRNAME = "review-skills"
LLM_COUNCIL_DIRNAME = ".llm-council"
SKILL_FILENAME = "SKILL.md"

_NAME_RE = re.compile(r"^[a-z0-9-]+$")
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024


@dataclass(frozen=True)
class ReviewSkill:
    """A single resolved focus bundle.

    ``sha256`` is the hex sha256 of the (whitespace-stripped) body text,
    stamped into provenance metadata so a transcript records exactly which
    directive text shaped the run.
    """

    name: str
    description: str
    body: str
    path: str
    sha256: str


class FocusNotFound(Exception):
    """Raised by :func:`resolve_focus` when a requested bundle is missing.

    Carries the unknown name(s) and the sorted list of available names so
    callers can fail fast with an operator-friendly message BEFORE any
    subprocess or HTTP call is made.
    """

    def __init__(self, missing: list[str], available: list[str]) -> None:
        self.missing = list(missing)
        self.available = sorted(available)
        missing_str = ", ".join(self.missing) if self.missing else "(none)"
        available_str = ", ".join(self.available) if self.available else "(none)"
        super().__init__(
            f"Unknown review focus bundle(s): {missing_str}. "
            f"Available: {available_str}."
        )


def _find_review_skills_dir(start: Path | None = None) -> Path | None:
    """Walk UP from ``start`` for the first ``.llm-council/review-skills/``.

    Mirrors :func:`config.find_config`'s stop-at-first behavior: the first
    such directory found walking upward wins.
    """

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / LLM_COUNCIL_DIRNAME / REVIEW_SKILLS_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """Split a SKILL.md into (frontmatter_block, body).

    The file must start (after optional leading whitespace) with a ``---``
    delimiter line, contain a closing ``---`` line, and the body is
    everything after the closing delimiter. Returns ``None`` when the
    frontmatter block is absent or unterminated.
    """

    lines = text.splitlines()
    # Skip leading blank lines.
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "---":
        return None
    open_idx = idx
    close_idx: int | None = None
    for j in range(open_idx + 1, len(lines)):
        if lines[j].strip() == "---":
            close_idx = j
            break
    if close_idx is None:
        return None
    frontmatter = "\n".join(lines[open_idx + 1 : close_idx])
    body = "\n".join(lines[close_idx + 1 :])
    return frontmatter, body


def _parse_frontmatter(block: str) -> dict[str, str]:
    """Parse the frontmatter block into a flat string->string mapping.

    Prefers PyYAML (already a dependency); falls back to a minimal
    ``key: value`` line parser if YAML parsing fails or yields a non-mapping.
    """

    parsed: dict[str, str] = {}
    try:
        loaded = yaml.safe_load(block)
    except yaml.YAMLError:
        loaded = None
    if isinstance(loaded, dict):
        for key, value in loaded.items():
            if isinstance(key, str):
                parsed[key.strip()] = "" if value is None else str(value).strip()
        return parsed
    # Minimal fallback: `key: value` per line.
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_skill(skill_md: Path, dir_name: str) -> ReviewSkill:
    """Parse + strictly validate a single SKILL.md.

    Raises ``ValueError`` (caught by the caller and recorded as a skip
    reason) on any validation failure.
    """

    text = skill_md.read_text(encoding="utf-8", errors="replace")
    split = _split_frontmatter(text)
    if split is None:
        raise ValueError(
            "missing or unterminated frontmatter block (expected leading "
            "'---' ... '---')"
        )
    frontmatter, body = split
    fields = _parse_frontmatter(frontmatter)

    name = (fields.get("name") or "").strip()
    description = (fields.get("description") or "").strip()
    body = body.strip()

    if not name:
        raise ValueError("frontmatter missing 'name'")
    if not description:
        raise ValueError("frontmatter missing 'description'")
    if len(name) > MAX_NAME_CHARS:
        raise ValueError(f"name exceeds {MAX_NAME_CHARS} chars")
    if not _NAME_RE.match(name):
        raise ValueError("name must match ^[a-z0-9-]+$")
    if name != dir_name:
        raise ValueError(
            f"name '{name}' must equal the directory name '{dir_name}'"
        )
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"description exceeds {MAX_DESCRIPTION_CHARS} chars")
    if not body:
        raise ValueError("body is empty")

    sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return ReviewSkill(
        name=name,
        description=description,
        body=body,
        path=str(skill_md),
        sha256=sha256,
    )


def discover_review_skills(
    start: Path | None = None,
) -> tuple[dict[str, ReviewSkill], list[dict]]:
    """Discover focus bundles under the first ``.llm-council/review-skills/``.

    Walks UP from ``start`` (cwd default) exactly like
    :func:`config.find_config`, using the FIRST such directory found. For
    each immediate subdir ``<name>/`` containing ``SKILL.md``, parses and
    strictly validates the bundle.

    Returns ``(skills_by_name, skipped)`` where ``skipped`` is a list of
    ``{"name", "path", "reason"}`` dicts. Discovery is lenient: a malformed
    or invalid bundle is added to ``skipped`` rather than raised.
    """

    skills: dict[str, ReviewSkill] = {}
    skipped: list[dict] = []
    root = _find_review_skills_dir(start)
    if root is None:
        return skills, skipped

    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        skill_md = entry / SKILL_FILENAME
        if not skill_md.is_file():
            skipped.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "reason": f"no {SKILL_FILENAME} in bundle directory",
                }
            )
            continue
        try:
            skill = _parse_skill(skill_md, entry.name)
        except (ValueError, OSError) as exc:
            skipped.append(
                {"name": entry.name, "path": str(skill_md), "reason": str(exc)}
            )
            continue
        skills[skill.name] = skill
    return skills, skipped


def resolve_focus(
    names: list[str], start: Path | None = None
) -> tuple[list[ReviewSkill], list[dict]]:
    """Resolve requested bundle names to :class:`ReviewSkill` in order.

    Discovers bundles, maps each requested name to its bundle in the
    REQUESTED order, and raises :class:`FocusNotFound` (listing available
    names) if ANY requested name is missing — so the CLI/MCP can fail fast
    BEFORE any subprocess launches.

    Returns ``(resolved_skills, skipped)``; ``skipped`` is the discovery
    skip list, surfaced so callers can warn the operator about malformed
    bundles even on a successful resolve.
    """

    skills, skipped = discover_review_skills(start)
    missing = [name for name in names if name not in skills]
    if missing:
        raise FocusNotFound(missing, list(skills.keys()))
    resolved = [skills[name] for name in names]
    return resolved, skipped


def render_focus_directive(skills: list[ReviewSkill]) -> str:
    """Combine resolved bundles into one appendable directive block.

    Each bundle is delimited clearly so peers can tell where one focus
    ends and the next begins. Returns ``""`` for an empty list (the
    no-focus path appends nothing).
    """

    if not skills:
        return ""
    blocks: list[str] = []
    for skill in skills:
        blocks.append(f"=== REVIEW FOCUS: {skill.name} ===\n{skill.body}")
    return "\n\n".join(blocks)
