"""Jaccard token-set similarity convergence detector for deliberation rounds.

Pure functions that emit a per-peer convergence signal between consecutive
rounds. Signal-only — does NOT auto-stop deliberation. The orchestrator
tags each round-2+ peer response with `converged`, `refining`,
`diverging`, or `insufficient` so users can observe accuracy on real
transcripts before any stopping policy is enabled.
"""

from __future__ import annotations

import re
from typing import Iterable

DEFAULT_THRESHOLDS: dict[str, float] = {"converged": 0.80, "refining": 0.50}

# Below this content-token floor (per side, after filtering) the Jaccard
# value is too volatile to classify — a single shared word flips the
# bucket. The orchestrator emits `state: insufficient` instead.
MIN_TOKENS_FOR_CLASSIFICATION = 10

STOPWORDS: frozenset[str] = frozenset(
    {
        # articles / determiners
        "the", "a", "an", "this", "that", "these", "those",
        # conjunctions / prepositions
        "and", "or", "of", "to", "in", "for", "on", "with", "as", "by",
        "at", "from", "but", "if", "so", "than", "then",
        # copulas / aux
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        # modals — every LLM prose answer is saturated with these
        "can", "could", "will", "would", "should", "may", "might", "must",
        # pronouns
        "i", "you", "we", "it", "my", "me", "us", "your", "our",
    }
)

# Code-fence language tags every peer emits when showing a snippet —
# common shared tokens that don't reflect substantive agreement.
CODE_FENCE_TAGS: frozenset[str] = frozenset(
    {"python", "json", "bash", "yaml", "typescript", "javascript", "shell", "sh"}
)

_RECOMMENDATION_LINE_RE = re.compile(
    r"^\s*RECOMMENDATION:\s*(?:yes|no|tradeoff)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Normalize smart quotes some CLIs emit so the regex character class
# (which only allows ASCII `-` and `'`) doesn't fragment hyphenated /
# possessive words.
_QUOTE_TRANSLATIONS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _strip_recommendation_lines(text: str) -> str:
    return _RECOMMENDATION_LINE_RE.sub("", text)


def tokenize(text: str) -> set[str]:
    """Lowercase content-word tokens, stopwords and boilerplate stripped.

    The literal `RECOMMENDATION: yes|no|tradeoff` line is excised before
    tokenization (rather than blindly dropping the words `yes/no/tradeoff`
    everywhere) so those words still contribute when they appear in
    substantive prose.
    """
    if not text:
        return set()
    cleaned = _strip_recommendation_lines(text)
    cleaned = cleaned.translate(_QUOTE_TRANSLATIONS).lower()
    tokens = _WORD_RE.findall(cleaned)
    return {
        token
        for token in tokens
        if token not in STOPWORDS and token not in CODE_FENCE_TAGS
    }


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return intersection / union


def classify(similarity: float, thresholds: dict[str, float] | None = None) -> str:
    resolved = resolve_thresholds(thresholds)
    if similarity >= resolved["converged"]:
        return "converged"
    if similarity >= resolved["refining"]:
        return "refining"
    return "diverging"


def resolve_thresholds(thresholds: dict[str, float] | None) -> dict[str, float]:
    """Merge user-supplied thresholds onto the defaults, validating ordering."""
    merged = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for key in ("converged", "refining"):
            if key in thresholds and thresholds[key] is not None:
                value = float(thresholds[key])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"convergence_thresholds.{key} must be between 0.0 and 1.0"
                    )
                merged[key] = value
    if merged["refining"] > merged["converged"]:
        raise ValueError(
            "convergence_thresholds.refining must be <= converged"
        )
    return merged


def tally_states(states: Iterable[str]) -> dict[str, int]:
    counts = {"converged": 0, "refining": 0, "diverging": 0, "insufficient": 0}
    for state in states:
        if state in counts:
            counts[state] += 1
    return counts
