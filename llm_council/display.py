"""Visual identity for llm-council CLI and MCP-rendered output.

The CLI surface uses a right-aligned 12-character bold-cyan gutter
(borrowed from `cargo`'s stage column — the layout is what makes that
recognizable, not the color). The MCP path emits a `summary_markdown`
field whose contents host agents tend to preserve verbatim when quoting
tool output: a `**Council**` header line, a per-peer markdown table, and
a blockquoted transcript path.

Both surfaces fall back gracefully:
- `NO_COLOR=1` or non-TTY → drop ANSI but keep the right-aligned layout.
  The layout alone carries the signature.
- Encoding without UTF → use ASCII `-` for the horizontal rule rather
  than `─` (U+2500), which fails on CP437 / legacy Windows consoles.

Per the council's design pass, no startup banner: the gutter is on every
line, which is why it doesn't need to announce itself.
"""

from __future__ import annotations

import os
import sys
from typing import IO


GUTTER_WIDTH = 12
RULE_WIDTH = 12

ANSI_RESET = "\x1b[0m"
ANSI_GUTTER = "\x1b[1;36m"  # bold cyan
ANSI_OK = "\x1b[32m"  # green
ANSI_FAIL = "\x1b[31m"  # red
ANSI_WARN = "\x1b[33m"  # yellow
ANSI_DIM = "\x1b[2m"

# Verbs reserved for the gutter on orchestrator-level lines. Peer-name
# lines use the peer name as the gutter token directly.
VERB_CONVENING = "Convening"
VERB_ROUND = "Round"
VERB_DELIBERATING = "Deliberating"
VERB_CONCLUDED = "Concluded"

STATUS_COLORS = {
    "ok": ANSI_OK,
    "success": ANSI_OK,
    "complete": ANSI_OK,
    "succeeded": ANSI_OK,
    "error": ANSI_FAIL,
    "failed": ANSI_FAIL,
    "degraded": ANSI_FAIL,
    "timeout": ANSI_WARN,
    "slow": ANSI_WARN,
    "warn": ANSI_WARN,
    "warning": ANSI_WARN,
}


def wants_color(stream: IO | None = None) -> bool:
    """True iff color is appropriate. Honors NO_COLOR and TTY detection.

    Per the no-color.org convention, *any* non-empty `NO_COLOR` env var
    disables color. We also disable color when the target stream is not
    a TTY so piped/CI output stays clean.
    """
    if os.environ.get("NO_COLOR"):
        return False
    target = stream if stream is not None else sys.stderr
    isatty = getattr(target, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (ValueError, OSError):
        return False


def wants_unicode_rule(stream: IO | None = None) -> bool:
    """True iff U+2500 box-drawing is safe on this stream's encoding.

    Sniffs `stream.encoding` for any UTF variant (`utf-8`, `utf8`,
    `UTF-16`, etc.). Encoding sniffing is enough — we don't need to
    detect specific terminals; CP437 / legacy Windows consoles get the
    ASCII fallback.
    """
    target = stream if stream is not None else sys.stderr
    encoding = getattr(target, "encoding", None) or ""
    return "utf" in encoding.lower()


def format_gutter(
    token: str,
    content: str,
    *,
    color: bool = True,
    width: int = GUTTER_WIDTH,
) -> str:
    """Format a gutter line: right-aligned token, single space, content.

    `token` is right-aligned to `width` columns; longer tokens are
    truncated rather than widening the gutter. The right-alignment is the
    visual signature, so we preserve it even when color is off.
    """
    if len(token) > width:
        token = token[:width]
    aligned = token.rjust(width)
    if color:
        aligned = f"{ANSI_GUTTER}{aligned}{ANSI_RESET}"
    return f"{aligned} {content}"


def colorize_status(word: str, *, color: bool = True) -> str:
    """Wrap a status word in its semantic color when color is enabled."""
    if not color:
        return word
    code = STATUS_COLORS.get(word.lower())
    if code is None:
        return word
    return f"{code}{word}{ANSI_RESET}"


def horizontal_rule(*, unicode_safe: bool = True, color: bool = True) -> str:
    """Return a `─` rule (UTF) or `-` rule (ASCII fallback)."""
    char = "─" if unicode_safe else "-"
    rule = char * RULE_WIDTH
    if color:
        rule = f"{ANSI_GUTTER}{rule}{ANSI_RESET}"
    return rule


def render_summary_markdown(
    *,
    mode: str,
    ok_count: int,
    total: int,
    elapsed_seconds: float,
    recommendation: str,
    per_peer_rows: list[dict],
    transcript_path: str | None,
    deliberated: bool = False,
    rounds: int = 1,
) -> str:
    """Render a markdown payload host agents tend to preserve verbatim.

    Format (council-recommended pattern):
    1. `**Council**` heading with mid-dot separated key=value pairs
    2. Markdown table per peer (label, time, stance if any)
    3. Blockquoted transcript path

    Agents that quote from tool output keep markdown blockquotes, bold
    headings, and tables intact even when they paraphrase surrounding
    prose. ANSI is irrelevant here — this surface is markdown-only.
    """
    deliberation_note = f" · {rounds} rounds" if deliberated else ""
    header = (
        f"**Council** · mode={mode} · {ok_count}/{total} succeeded · "
        f"{elapsed_seconds:.1f}s · recommendation={recommendation}{deliberation_note}"
    )
    lines = [header, ""]
    if per_peer_rows:
        has_stance = any(row.get("stance") for row in per_peer_rows)
        if has_stance:
            lines.append("| peer | label | stance | time |")
            lines.append("|---|---|---|---|")
            for row in per_peer_rows:
                lines.append(
                    f"| {row['name']} | {row.get('label') or '—'} | "
                    f"{row.get('stance') or '—'} | "
                    f"{row.get('elapsed_seconds', 0):.1f}s |"
                )
        else:
            lines.append("| peer | label | time |")
            lines.append("|---|---|---|")
            for row in per_peer_rows:
                lines.append(
                    f"| {row['name']} | {row.get('label') or '—'} | "
                    f"{row.get('elapsed_seconds', 0):.1f}s |"
                )
    if transcript_path:
        lines.extend(["", f"> Transcript: `{transcript_path}`"])
    return "\n".join(lines)
