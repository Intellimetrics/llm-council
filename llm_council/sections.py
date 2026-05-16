"""Section-coverage validator for structured council prompts.

When a user prompt contains one or more `PART N — TITLE (REQUIRED)` or
`PART N — TITLE (REQUIRED BY ...)` headers, the validator scans each peer
response for a matching marker and flags missing sections as
`error_kind: incomplete_response`.

Prompt-side detection is permissive across header styles: the matcher
tolerates an optional markdown header prefix (`##`, `###`, ...), an
optional `**bold**` wrapper, em-dash / hyphen / en-dash / colon
separators, and mixed case in both the `PART` keyword and the title
(see `REQUIRED_SECTION_HEADER_RE`).

Response-side detection is strict against prose mentions: a literal
`PART N` token only counts when it appears at the start of a line
(optionally preceded by `#`/`##`/`**` markers) OR is accompanied by
the first salient title token within a 200-char window. Prose
mentions like `I skipped PART 2` or `PART 2 was not addressed` are
explicitly rejected via a skip-prose pattern. Paraphrased headers
(e.g. `## Concept Grid` satisfying `PART 2 — CONCEPT-BY-CONCEPT GRID`)
still pass through the salient-title-tokens route.

Anchor failure mode from pass-7 (transcript
`.llm-council/runs/20260516_100758_*`): the gemini peer was asked for a
14-entry grid + 5 binaries + 5 first-person sections + 3 missed/overstated
items and delivered three bullets. The `RECOMMENDATION:` label was
present, so v0.6.0's validation accepted the response. This module is
the v0.7.0 response to that.

PART 6 (RECOMMENDATION) is intentionally NOT detected by this module —
the existing `adapters._has_recommendation_label` check covers it and a
missing label already produces `error_kind: invalid_response`. Section-
coverage on PART 6 would double-fault for the same root cause.
"""

from __future__ import annotations

import re


REQUIRED_SECTION_HEADER_RE = re.compile(
    r"""
    ^                                # start of line
    [\ \t]*                          # leading whitespace
    (?:\#{1,6}[\ \t]+)?              # optional markdown header prefix (#, ##, ###, ...)
    (?:\*\*[\ \t]*)?                 # optional leading markdown bold wrapper
    (?P<label>
        PART\s+(?P<num>\d+)          # PART <n>
        \s*[—\-–:]\s+                # em-dash / hyphen / en-dash / colon separator
        (?P<title>
            [A-Za-z][A-Za-z0-9\ /,&\-–—']+?
        )
    )
    [\ \t]*
    (?:\*\*[\ \t]*)?                 # optional trailing bold wrapper (before parens)
    \(
        REQUIRED
        (?:\s+BY\s+[A-Za-z\ ]+)?     # tolerates "(REQUIRED BY COUNCIL INVARIANTS)"
    \)
    [\ \t]*
    (?:\*\*)?                        # optional trailing bold wrapper (after parens)
    [\ \t]*$
    """,
    re.VERBOSE | re.MULTILINE | re.IGNORECASE,
)


_TITLE_STOPWORDS = {
    "AND", "OR", "OF", "THE", "BY", "TO", "FOR", "WITH",
    "IN", "ON", "AT", "FROM", "A", "AN",
}


def _extract_salient_tokens(title: str) -> list[str]:
    """Pull ALL-CAPS noun-ish tokens from the title for paraphrase matching.

    Keeps tokens of 4+ ASCII letters (so single letters / 2-3 letter words
    that often differ across paraphrases don't gate detection). Drops a
    small stopword set so connector words don't dominate the match window.
    """
    return [
        w
        for w in re.findall(r"[A-Z]{4,}", title.upper())
        if w not in _TITLE_STOPWORDS
    ]


def required_sections(prompt: str) -> list[dict[str, object]]:
    """Parse REQUIRED section markers out of the prompt.

    Returns a list of dicts with `label`, `num`, `title`, `title_tokens`.
    Empty list when the prompt has no markers — the validator becomes a
    no-op for unmarked prompts, so callers don't need to guard.
    """
    out: list[dict[str, object]] = []
    for match in REQUIRED_SECTION_HEADER_RE.finditer(prompt or ""):
        title = match.group("title").strip()
        out.append(
            {
                "label": match.group("label").strip(),
                "num": match.group("num"),
                "title": title,
                "title_tokens": _extract_salient_tokens(title),
            }
        )
    return out


def _section_present(response_upper: str, requirement: dict[str, object]) -> bool:
    """True when a response plausibly satisfies a required section.

    Two acceptance routes, in priority order:

    1. A `PART N` mention that LOOKS LIKE A HEADER, not just a passing
       reference. A header-shaped mention is either at the start of a
       line (optionally after `#`/`##`/`**` markdown markers) OR is
       accompanied by the first salient title token within a 200-char
       window. This rejects prose mentions like `I skipped PART 2`,
       `PART 2 was not addressed`, or `see PART 2 instructions` —
       a peer must STRUCTURALLY commit to the section, not just name
       it. Pure literal mentions like `PART 2.` with nothing else are
       insufficient on their own.
    2. Salient title tokens — the first title token with all remaining
       title tokens within a 200-char window. Catches paraphrased
       headers like `## Concept Grid` for
       `PART 2 — CONCEPT-BY-CONCEPT GRID`.

    A title with no salient tokens (e.g. a one-letter title) falls back
    to the literal `PART N` route alone, since there's nothing else
    distinguishing to anchor on. If both routes fail, the section is
    treated as missing.
    """
    num = requirement.get("num")
    tokens = list(requirement.get("title_tokens") or [])
    if num and _literal_part_is_header_shaped(response_upper, str(num), tokens):
        return True
    if not tokens:
        # No title tokens to fall back on — if the literal route already
        # failed above, the only remaining signal would be the bare
        # number, which is too noisy. Treat as missing.
        return False
    first = tokens[0]
    for m in re.finditer(rf"\b{re.escape(first)}\b", response_upper):
        window_start = max(0, m.start() - 100)
        window_end = m.end() + 200
        window = response_upper[window_start:window_end]
        if all(re.search(rf"\b{re.escape(t)}\b", window) for t in tokens[1:]):
            return True
    return False


# Window in which a literal `PART N` mention must be accompanied by the
# first salient title token to be treated as a section header rather
# than a passing prose reference. Chosen to match the 200-char window
# already used by the paraphrased-title route.
_LITERAL_PART_CONFIRM_WINDOW = 200

# Prefix patterns that mark `PART N` as header-shaped on its own — start
# of line, possibly preceded by markdown header (`#`, `##`, ...) or
# bold (`**`) markers, and (critically) NOT preceded by skip-prose
# verbs on the same line. We look at the 80 characters before the
# `PART N` match to spot skip phrases like `skipped`, `omitted`,
# `not addressed`, etc.
_HEADER_PREFIX_RE = re.compile(r"(?:^|\n)[ \t]*(?:\#{1,6}[ \t]+|\*\*[ \t]*)?$")
_SKIP_PROSE_RE = re.compile(
    r"\b(?:SKIPPED|OMITTED|MISSING|MISSED|UNABLE|COULD\s+NOT|CANNOT|"
    r"DID\s+NOT|WAS\s+NOT|NOT\s+ADDRESSED|NOT\s+COMPLETED|NOT\s+INCLUDED|"
    r"SEE\s+PART|REFER\s+TO|SKIP)\b",
    re.IGNORECASE,
)


def _literal_part_is_header_shaped(
    response_upper: str, num: str, tokens: list[str]
) -> bool:
    """A `PART N` mention counts only when header-shaped.

    Two acceptance routes for the literal mention:

    - Structural: at line start (possibly after `#`/`**` markers) AND
      the same line does not contain skip-prose verbs.
    - Confirmation: accompanied by at least the first salient title
      token within `_LITERAL_PART_CONFIRM_WINDOW` chars. Catches
      `## PART 2 — Concept Grid` even when the structural route is
      relaxed, and rejects bare prose mentions like `I skipped PART 2`.
    """
    pattern = rf"\bPART\s+{re.escape(num)}\b"
    for m in re.finditer(pattern, response_upper):
        # Skip-prose check: look in a window before AND after the
        # match. If skip verbs appear in either direction, this is a
        # prose mention that disclaims the section rather than
        # delivering it.
        before_start = max(0, m.start() - 80)
        before = response_upper[before_start:m.start()]
        after_end = min(len(response_upper), m.end() + 80)
        after = response_upper[m.end():after_end]
        if _SKIP_PROSE_RE.search(before) or _SKIP_PROSE_RE.search(after):
            continue
        # Structural route: line-start (possibly with `#`/`**` prefix).
        prefix = response_upper[:m.start()]
        if _HEADER_PREFIX_RE.search(prefix):
            return True
        # Confirmation route: first salient title token within window.
        if tokens:
            first = tokens[0]
            confirm_start = max(0, m.start() - _LITERAL_PART_CONFIRM_WINDOW)
            confirm_end = m.end() + _LITERAL_PART_CONFIRM_WINDOW
            window = response_upper[confirm_start:confirm_end]
            if re.search(rf"\b{re.escape(first)}\b", window):
                return True
    return False


def _is_recommendation_part(requirement: dict[str, object]) -> bool:
    """PART 6 (RECOMMENDATION) is checked by the existing label validator."""
    tokens = requirement.get("title_tokens") or []
    return tokens == ["RECOMMENDATION"] or tokens == ["RECOMMENDATION", "COUNCIL"]


def required_sections_missing(prompt: str, response: str) -> list[str]:
    """Return a list of required-section labels missing from the response.

    Skips PART 6 (RECOMMENDATION) — that's the existing label check's job.
    Returns labels in prompt order so the repair-retry instruction can
    list them deterministically.
    """
    requirements = required_sections(prompt)
    if not requirements:
        return []
    response_upper = (response or "").upper()
    missing: list[str] = []
    for req in requirements:
        if _is_recommendation_part(req):
            continue
        if _section_present(response_upper, req):
            continue
        missing.append(str(req["label"]))
    return missing


__all__ = [
    "REQUIRED_SECTION_HEADER_RE",
    "required_sections",
    "required_sections_missing",
]
