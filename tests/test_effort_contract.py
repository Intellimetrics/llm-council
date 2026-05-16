"""Tests for the optional Pick A response envelope and abdication detection."""

from __future__ import annotations

from llm_council.adapters import (
    ABDICATED_ERROR_PREFIX,
    ERROR_KIND_ABDICATED,
    KNOWN_ERROR_KINDS,
    ParticipantResult,
    _extract_response_envelope,
    _is_abdication,
    _with_envelope,
    classify_error,
)


def test_envelope_parses_full_shape():
    text = (
        "RECOMMENDATION: tradeoff - some risk\n"
        "EFFORT: full\n"
        "CONFIDENCE: high\n"
        "RISK: medium\n"
        "\n"
        "BLOCKERS:\n"
        "- missing migration\n"
        "- no tenant test\n"
        "\n"
        "EVIDENCE:\n"
        "- src/auth.py:42\n"
        "\n"
        "TESTS_TO_RUN:\n"
        "- pytest tests/auth/\n"
        "\n"
        "ASSUMPTIONS:\n"
        "- staging mirrors prod schema\n"
    )
    env = _extract_response_envelope(text)
    assert env["effort"] == "full"
    assert env["confidence"] == "high"
    assert env["risk"] == "medium"
    assert env["blockers"] == ["missing migration", "no tenant test"]
    # v0.7: evidence parsed into list[{text, tag}] for tag distribution
    # telemetry. Untagged entries return `tag=None`.
    assert env["evidence"] == [{"text": "src/auth.py:42", "tag": None}]
    assert env["tests_to_run"] == ["pytest tests/auth/"]
    assert env["assumptions"] == ["staging mirrors prod schema"]


def test_envelope_ignores_fenced_blocks():
    text = (
        "RECOMMENDATION: yes - ship it\n"
        "```\n"
        "EFFORT: blocked\n"
        "RISK: critical\n"
        "```\n"
        "EFFORT: full\n"
    )
    env = _extract_response_envelope(text)
    # Fenced lines must not contribute — only the EFFORT outside the fence.
    assert env["effort"] == "full"
    assert env["risk"] is None


def test_envelope_empty_input_safe():
    env = _extract_response_envelope("")
    assert env["effort"] is None
    assert env["blockers"] == []


def test_with_envelope_populates_dataclass():
    r = ParticipantResult(
        "a",
        True,
        "RECOMMENDATION: tradeoff - ok\nEFFORT: limited\nBLOCKERS:\n- need spec",
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True
    assert out.effort == "limited"
    assert out.blockers == ["need spec"]


def test_abdication_classified_and_terminal():
    """Peer says blocked but lists no concrete missing artifact — abdication."""
    r = ParticipantResult(
        "a",
        True,
        "RECOMMENDATION: no - too complex to evaluate\nEFFORT: blocked\n",
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is False, "abdication must flip ok=False so quorum drops"
    assert out.error.startswith(ABDICATED_ERROR_PREFIX)
    assert out.effort == "blocked"
    assert classify_error(out.error) == ERROR_KIND_ABDICATED
    assert ERROR_KIND_ABDICATED in KNOWN_ERROR_KINDS


def test_blocked_with_concrete_blockers_is_not_abdication():
    """Self-reported blocked + named missing artifact = useful, not abdication."""
    r = ParticipantResult(
        "b",
        True,
        (
            "RECOMMENDATION: no - need data\n"
            "EFFORT: blocked\n"
            "BLOCKERS:\n"
            "- missing migration SQL file\n"
            "- no env config\n"
        ),
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True, "blocked-with-blockers is honest, not abdication"
    assert out.blockers == ["missing migration SQL file", "no env config"]
    assert out.error == ""


def test_substantive_no_vote_not_misclassified():
    """A 'no' vote without EFFORT: blocked must remain a valid vote."""
    r = ParticipantResult(
        "c",
        True,
        "RECOMMENDATION: no - real issue here\nThis breaks tenant isolation.",
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True
    assert out.effort is None


def test_abdication_blocked_with_assumptions_not_abdication():
    """Naming assumptions explicitly counts as effort, not abdication."""
    r = ParticipantResult(
        "d",
        True,
        (
            "RECOMMENDATION: tradeoff - depends\n"
            "EFFORT: blocked\n"
            "ASSUMPTIONS:\n"
            "- the deploy script handles secrets\n"
        ),
        "",
        1.0,
    )
    out = _with_envelope(r)
    assert out.ok is True


def test_is_abdication_requires_label():
    """Abdication only fires when there is a RECOMMENDATION label to invalidate."""
    env = {"effort": "blocked", "blockers": [], "assumptions": []}
    assert _is_abdication(env, "EFFORT: blocked\n") is False


def test_classify_error_routes_abdication():
    assert classify_error(f"{ABDICATED_ERROR_PREFIX} sample") == ERROR_KIND_ABDICATED
