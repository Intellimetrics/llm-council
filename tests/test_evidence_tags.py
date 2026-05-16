"""Evidence-tag parser + strict-evidence validation (Change 3).

For the pass-7 anchor (codex's response with all four tag kinds), see
`tests/test_pass7_regression.py`.
"""

from __future__ import annotations

from llm_council.adapters import (
    EVIDENCE_TAG_RE,
    KNOWN_ERROR_KINDS,
    UNTAGGED_EVIDENCE_PREFIX,
    _extract_response_envelope,
    _parse_tagged_entry,
    _response_validation_error,
    classify_error,
)


# --- Tag regex -----------------------------------------------------------

def test_tag_regex_matches_all_four_canonical_tags():
    for tag in ("PUBLISHED", "OBSERVABLE", "INFERRED", "SPECULATIVE"):
        m = EVIDENCE_TAG_RE.search(f"[{tag}]")
        assert m is not None
        assert m.group("tag").upper() == tag


def test_tag_regex_is_case_insensitive():
    m = EVIDENCE_TAG_RE.search("[observable]")
    assert m is not None
    assert m.group("tag").lower() == "observable"


def test_tag_regex_accepts_qualifier_after_separator():
    """Pass-7 used `[OBSERVABLE — behavioral]` and `[SPECULATIVE — no
    direct access]` — the qualifier after `—` is dropped from the tag
    but the regex still matches."""
    for raw in (
        "[OBSERVABLE — behavioral]",
        "[SPECULATIVE — no direct access]",
        "[INFERRED - reasoning from priors]",
        "[PUBLISHED: cite paper]",
    ):
        m = EVIDENCE_TAG_RE.search(raw)
        assert m is not None, f"expected match on {raw!r}"


def test_tag_regex_rejects_unknown_tag():
    """Made-up tags don't match — keeps the contract closed."""
    assert EVIDENCE_TAG_RE.search("[MADEUP]") is None
    assert EVIDENCE_TAG_RE.search("[citation]") is None


# --- _parse_tagged_entry shape -------------------------------------------

def test_parse_tagged_entry_extracts_leading_tag():
    out = _parse_tagged_entry("[PUBLISHED] Bai et al. 2022")
    assert out == {"text": "Bai et al. 2022", "tag": "published"}


def test_parse_tagged_entry_extracts_trailing_tag():
    out = _parse_tagged_entry("Bai et al. 2022 [PUBLISHED]")
    assert out == {"text": "Bai et al. 2022", "tag": "published"}


def test_parse_tagged_entry_lowercases_tag():
    out = _parse_tagged_entry("Bai 2022 [PUBLISHED]")
    assert out["tag"] == "published"


def test_parse_tagged_entry_untagged_returns_null_tag():
    out = _parse_tagged_entry("Bai et al. 2022")
    assert out == {"text": "Bai et al. 2022", "tag": None}


def test_parse_tagged_entry_strips_separators():
    """A `—` or `:` between tag and text leaves clean text."""
    out = _parse_tagged_entry("[OBSERVABLE — behavioral]")
    assert out["tag"] == "observable"
    # text becomes the empty string after stripping; parser falls back
    # to the original to avoid losing the user's content.
    assert out["text"]  # not empty


def test_parse_tagged_entry_empty_input_safe():
    assert _parse_tagged_entry("") == {"text": "", "tag": None}


# --- Envelope-level evidence structuring ---------------------------------

def test_envelope_evidence_becomes_list_of_dicts():
    """`evidence` parses as `list[{text, tag}]`; other list fields stay
    `list[str]` because tag semantics only apply to evidence."""
    text = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE:\n"
        "- Bai et al. 2022 [PUBLISHED]\n"
        "- direct observation [OBSERVABLE]\n"
        "- untagged claim\n"
        "BLOCKERS:\n"
        "- missing schema\n"
    )
    env = _extract_response_envelope(text)
    assert env["evidence"] == [
        {"text": "Bai et al. 2022", "tag": "published"},
        {"text": "direct observation", "tag": "observable"},
        {"text": "untagged claim", "tag": None},
    ]
    # Blockers stay plain strings.
    assert env["blockers"] == ["missing schema"]


def test_envelope_empty_evidence_stays_empty_list():
    env = _extract_response_envelope("RECOMMENDATION: yes - ok\n")
    assert env["evidence"] == []


# --- strict_evidence validation ------------------------------------------

def test_strict_evidence_disabled_passes_untagged():
    """Default behavior: untagged evidence is fine. No regression for
    v0.6.0-era responses."""
    output = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE:\n"
        "- untagged claim\n"
    )
    cfg = {"strict_evidence": False}
    assert _response_validation_error(output, cfg) == ""


def test_strict_evidence_enabled_rejects_untagged():
    """With strict_evidence=True, even one untagged entry fails."""
    output = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE:\n"
        "- untagged claim\n"
    )
    cfg = {"strict_evidence": True}
    error = _response_validation_error(output, cfg)
    assert error.startswith(UNTAGGED_EVIDENCE_PREFIX)


def test_strict_evidence_enabled_accepts_all_tagged():
    """All-tagged passes even under strict."""
    output = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE:\n"
        "- Bai 2022 [PUBLISHED]\n"
        "- direct observation [OBSERVABLE]\n"
    )
    cfg = {"strict_evidence": True}
    assert _response_validation_error(output, cfg) == ""


def test_strict_evidence_empty_list_passes():
    """A response with NO evidence at all is fine under strict —
    the gate is FORMAT of entries that exist, not PRESENCE."""
    output = "RECOMMENDATION: yes - ok\nNo evidence here.\n"
    cfg = {"strict_evidence": True}
    assert _response_validation_error(output, cfg) == ""


def test_classify_error_routes_untagged_evidence():
    assert classify_error("UntaggedEvidence: 2 entries") == "untagged_evidence"


def test_known_error_kinds_includes_untagged_evidence():
    assert "untagged_evidence" in KNOWN_ERROR_KINDS


# --- Stats: evidence_tag_distribution ------------------------------------

def test_stats_evidence_tag_distribution_counts_each_tag():
    """Each EVIDENCE bullet bumps its tag's counter in stats."""
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "consensus",
                "results": [
                    {
                        "name": "codex",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        "evidence": [
                            {"text": "Bai 2022", "tag": "published"},
                            {"text": "obs", "tag": "observable"},
                            {"text": "untagged"},  # tag absent
                        ],
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    dist = by_peer["codex"]["evidence_tag_distribution"]
    assert dist["published"] == 1
    assert dist["observable"] == 1
    assert dist["untagged"] == 1
    assert dist["inferred"] == 0


def test_stats_legacy_string_evidence_counts_as_untagged():
    """Pre-v0.7 cached transcripts have `evidence: list[str]`. Each
    string entry counts as untagged so old data flows into the new shape
    without skewing toward zero."""
    from llm_council.stats import aggregate

    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "review",
                "results": [
                    {
                        "name": "codex",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        "evidence": ["plain legacy string"],
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    assert by_peer["codex"]["evidence_tag_distribution"]["untagged"] == 1
