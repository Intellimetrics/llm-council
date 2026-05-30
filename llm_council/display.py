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

# Identity token shared by CLI and MCP surfaces. Plain ASCII so it
# survives any host's rendering (markdown / plain / ANSI-stripped) and
# matches the `**LLM Council**` header in `render_summary_markdown`.
BRAND_TOKEN = "LLM Council"
BRAND_SEP = " · "

# Per-peer color rotation for the CLI gutter. Indexed by the peer's
# position in the active roster (deterministic via `select_participants`).
# Custom CLI peers defined in `.llm-council.yaml` slot into the cycle by
# their roster index — no per-name registry needed.
PEER_ACCENT_PALETTE = (
    "\x1b[36m",  # cyan
    "\x1b[35m",  # magenta
    "\x1b[33m",  # yellow
    "\x1b[32m",  # green
    "\x1b[34m",  # blue
    "\x1b[31m",  # red
)

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


def format_usd(value: float | None) -> str:
    """Precision-ladder USD formatter shared across CLI and stats surfaces.

    `None` → "n/a", exact zero → "$0", sub-milli-dollar amounts use 6
    decimals so they don't collapse to "$0.0000", everything else uses 4.
    """
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    if value < 0.001:
        return f"${value:.6f}"
    return f"${value:.4f}"


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


def wants_quiet() -> bool:
    """True iff `LLM_COUNCIL_QUIET=1` opt-out is set.

    Single env switch suppresses MCP progress notifications and CLI color
    (layout still prints). MCP servers have no per-call CLI flags, so an
    env-only switch is the only shape that works on both surfaces.
    """
    value = os.environ.get("LLM_COUNCIL_QUIET", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def peer_accent(name: str, ordered_peers: list[str] | tuple[str, ...]) -> str | None:
    """Return ANSI color for `name` from `PEER_ACCENT_PALETTE`.

    Deterministic via roster index. Returns `None` when the peer is not in
    the roster (caller should fall back to the default `ANSI_GUTTER`).
    """
    try:
        idx = list(ordered_peers).index(name)
    except ValueError:
        return None
    return PEER_ACCENT_PALETTE[idx % len(PEER_ACCENT_PALETTE)]


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
    token_color: str | None = None,
) -> str:
    """Format a gutter line: right-aligned token, single space, content.

    `token` is right-aligned to `width` columns; longer tokens are
    truncated rather than widening the gutter. The right-alignment is the
    visual signature, so we preserve it even when color is off.

    `token_color` overrides the default bold-cyan gutter — used by callers
    that want per-peer accent rotation (see `peer_accent`). Ignored when
    `color=False`.
    """
    if len(token) > width:
        token = token[:width]
    aligned = token.rjust(width)
    if color:
        code = token_color if token_color is not None else ANSI_GUTTER
        aligned = f"{code}{aligned}{ANSI_RESET}"
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
        f"**LLM Council** · mode={mode} · {ok_count}/{total} succeeded · "
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


# Events that should advance the progress counter (i.e. represent
# completed work units). All other "interesting" events emit messages
# with delta=0. Events not listed in `format_progress_message` are
# suppressed entirely (noise vs signal — see plan §3 table).
#
# `preflight_failed` peers are stripped from `run_targets` and never
# emit `participant_finish`, so without listing it here the progress
# counter would stall whenever a local peer's preflight ping failed.
# The peer's "work" is morally done — it just failed before it started.
PROGRESS_ADVANCING_EVENTS = frozenset(
    {
        "participant_finish",
        "preflight_failed",
        "cross_rank_complete",
        "synthesis_finish",
    }
)


def format_progress_message(event: dict) -> str | None:
    """Map an orchestrator `progress_events` entry to an MCP message.

    Returns the `message` field for `notifications/progress` (always
    prefixed with the brand token), or `None` to suppress.

    Plain text only. No ANSI (hosts strip), no emoji (font risk), no
    markdown bold (some hosts render `**` as literal in progress
    messages). The literal prefix `LLM Council · ` is the identity
    signal that survives every rendering path.
    """
    kind = event.get("event")
    peer = event.get("participant") or "peer"
    round_no = event.get("round")
    body: str | None = None

    if kind == "council_start":
        n = len(event.get("participants") or [])
        mode = event.get("mode")
        # `mode` is not on the orchestrator's council_start event today
        # (only `participants`, `round`, `max_rounds`, `deliberate`,
        # `image_count`); we accept it if present and degrade gracefully.
        if mode:
            body = f"convening {n} peers · mode={mode}"
        else:
            body = f"convening {n} peers"
    elif kind == "preflight_failed":
        body = f"preflight failed: {peer}"
    elif kind == "participant_slow":
        elapsed = float(event.get("elapsed_seconds") or 0)
        timeout = float(event.get("timeout_seconds") or 0)
        body = f"{peer} slow ({elapsed:.0f}s / {timeout:.0f}s timeout)"
    elif kind == "participant_finish":
        status = event.get("status") or ("ok" if event.get("ok") else "error")
        elapsed = float(event.get("elapsed_seconds") or 0)
        body = f"{peer} {status} ({elapsed:.1f}s)"
    elif kind == "deliberation_pending":
        body = f"disagreement detected, deliberation round {round_no} starting"
    elif kind in ("deliberation_skip", "deliberation_skipped"):
        reason = event.get("reason") or "unspecified"
        body = f"deliberation skipped ({reason})"
    elif kind == "deliberation_round_start":
        body = f"round {round_no} deliberation"
    elif kind == "deliberation_finish":
        rounds_ = event.get("rounds")
        body = f"deliberation finished after {rounds_} rounds"
    elif kind == "cross_rank_start":
        count = event.get("peer_count") or "?"
        body = f"cross-ranking {count} peers"
    elif kind == "cross_rank_complete":
        ranker = event.get("ranker_count")
        body = (
            f"cross-ranking complete ({ranker} rankers)"
            if ranker is not None
            else "cross-ranking complete"
        )
    elif kind == "synthesis_start":
        chair = event.get("chair") or "?"
        body = f"synthesis chair: {chair}"
    elif kind == "synthesis_finish":
        label = event.get("decision_label") or "done"
        body = f"synthesis done ({label})"
    elif kind == "synthesis_error":
        body = f"synthesis error: {event.get('error') or 'unknown'}"
    elif kind == "universal_abdication":
        body = "universal abdication (all peers blocked)"
    elif kind == "degraded_consensus":
        labeled = event.get("labeled_quorum")
        threshold = event.get("min_quorum")
        body = f"degraded consensus: {labeled}/{threshold} peers labeled"
    elif kind == "council_finish":
        ok = event.get("ok")
        total = event.get("total")
        body = f"concluded: {ok}/{total} ok"
    # Suppressed by design: participant_start (per-peer noise multiplier
    # when N peers fire concurrently — participant_finish is the visible
    # signal), images_skipped, truncated_for_deliberation,
    # deliberation_skip_participants, convergence, context_files_chunked.

    if body is None:
        return None
    return f"{BRAND_TOKEN}{BRAND_SEP}{body}"
