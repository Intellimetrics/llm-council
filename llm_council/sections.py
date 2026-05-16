"""Section-coverage validator for structured council prompts.

When a user prompt contains one or more `PART N — TITLE (REQUIRED)` or
`PART N — TITLE (REQUIRED BY ...)` headers, the validator scans each peer
response for a matching marker and flags missing sections as
`error_kind: incomplete_response`. The matcher accepts either a literal
`PART N` token in the response OR all salient title tokens within a
200-char window — peers paraphrase headers (e.g. `## Concept Grid`
satisfies `PART 2 — CONCEPT-BY-CONCEPT GRID`).

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
    \s*
    (?P<label>
        PART\s+(?P<num>\d+)          # PART <n>
        \s+[—\-–]\s+            # em-dash / hyphen / en-dash separator
        (?P<title>
            [A-Z][A-Z0-9\ /,&\-–—']+?   # ASCII-uppercase title
        )
    )
    \s*
    \(
        REQUIRED
        (?:\s+BY\s+[A-Z\ ]+)?        # tolerates "(REQUIRED BY COUNCIL INVARIANTS)"
    \)
    \s*$
    """,
    re.VERBOSE | re.MULTILINE,
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

    Two acceptance routes:

    1. Literal `PART N` token anywhere in the response (case-insensitive).
       Catches peers who used the same header style as the prompt.
    2. The first salient title token, with all the rest of the title's
       salient tokens appearing within a 200-char window. Catches
       paraphrased headers like `## Concept Grid` for
       `PART 2 — CONCEPT-BY-CONCEPT GRID`.

    A title with no salient tokens (e.g. a one-letter title) defaults to
    True so the matcher does not false-positive on titles with no
    distinguishing content.
    """
    num = requirement.get("num")
    if num and re.search(rf"\bPART\s+{re.escape(str(num))}\b", response_upper):
        return True
    tokens = list(requirement.get("title_tokens") or [])
    if not tokens:
        return True
    first = tokens[0]
    for m in re.finditer(rf"\b{re.escape(first)}\b", response_upper):
        window_start = max(0, m.start() - 100)
        window_end = m.end() + 200
        window = response_upper[window_start:window_end]
        if all(re.search(rf"\b{re.escape(t)}\b", window) for t in tokens[1:]):
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
