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


# --- Strict-evidence repair-retry --------------------------------------
#
# The strict_evidence validator flags untagged EVIDENCE bullets with
# `UntaggedEvidence:`. The repair-retry path re-asks the peer with a
# directive naming the four legal tags. Success on retry → ok=True with
# `repair_retry_recovered=True`. Failure → ok=False, error_kind retained
# as `untagged_evidence`. The retry must compose correctly with (but not
# chain through) terse-retry-on-timeout, label-retry, and section-repair.
#
# These tests mock the inner adapter calls directly so the retry-loop
# logic is exercised without spinning up subprocesses or HTTP clients.

import asyncio
from pathlib import Path
from unittest.mock import patch

import llm_council.adapters as adapters_module
from llm_council.adapters import (
    ParticipantResult,
    STRICT_EVIDENCE_REPAIR_RETRY_INSTRUCTION,
    classify_error,
    run_cli_participant,
    run_ollama_participant,
    run_openai_compatible_participant,
)


_UNTAGGED_OUTPUT = (
    "RECOMMENDATION: yes - looks fine\n"
    "EVIDENCE:\n"
    "- plain bullet with no tag\n"
)
_TAGGED_OUTPUT = (
    "RECOMMENDATION: yes - looks fine\n"
    "EVIDENCE:\n"
    "- plain bullet [PUBLISHED]\n"
)


# ----- CLI path -----------------------------------------------------------

def test_strict_evidence_cli_retry_recovers_on_success():
    """CLI: untagged first attempt → repair-retry returns tagged → ok=True."""
    call_count = {"n": 0}
    captured_prompts: list[str] = []

    async def fake_run_cli_once(name, cfg, prompt, cwd, *, start, mode_multiplier=None, mode=None):
        call_count["n"] += 1
        captured_prompts.append(prompt)
        if call_count["n"] == 1:
            return (
                ParticipantResult(
                    name=name,
                    ok=False,
                    output=_UNTAGGED_OUTPUT,
                    error=(
                        "UntaggedEvidence: 1 EVIDENCE entry/entries lack a "
                        "[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE] tag "
                        "while defaults.strict_evidence is true"
                    ),
                    elapsed_seconds=2.0,
                ),
                {"nonzero_exit": False, "stderr": "", "exited": True},
            )
        # Retry: fully tagged, no validation error.
        return (
            ParticipantResult(
                name=name,
                ok=True,
                output=_TAGGED_OUTPUT,
                error="",
                elapsed_seconds=2.5,
            ),
            {"nonzero_exit": False, "stderr": "", "exited": True},
        )

    with patch("llm_council.adapters._run_cli_once", side_effect=fake_run_cli_once), \
         patch("llm_council.adapters._cache_lookup", return_value=(None, None)), \
         patch("llm_council.adapters._maybe_persist_cache"):
        result = asyncio.run(
            run_cli_participant(
                "peer",
                {"type": "cli", "command": "peer", "strict_evidence": True, "timeout": 60},
                "Original question",
                Path("/tmp"),
            )
        )

    assert call_count["n"] == 2, "expected one repair retry"
    assert result.ok is True
    assert result.repair_retry_recovered is True
    # Retry prompt contains the strict-evidence directive
    assert STRICT_EVIDENCE_REPAIR_RETRY_INSTRUCTION[:40] in captured_prompts[1]


def test_strict_evidence_cli_retry_fails_returns_untagged_evidence():
    """CLI: both attempts untagged → ok=False, error_kind=untagged_evidence."""
    call_count = {"n": 0}

    async def fake_run_cli_once(name, cfg, prompt, cwd, *, start, mode_multiplier=None, mode=None):
        call_count["n"] += 1
        return (
            ParticipantResult(
                name=name,
                ok=False,
                output=_UNTAGGED_OUTPUT,
                error=(
                    "UntaggedEvidence: 1 EVIDENCE entry/entries lack a "
                    "[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE] tag "
                    "while defaults.strict_evidence is true"
                ),
                elapsed_seconds=2.0,
            ),
            {"nonzero_exit": False, "stderr": "", "exited": True},
        )

    with patch("llm_council.adapters._run_cli_once", side_effect=fake_run_cli_once), \
         patch("llm_council.adapters._cache_lookup", return_value=(None, None)), \
         patch("llm_council.adapters._maybe_persist_cache"):
        result = asyncio.run(
            run_cli_participant(
                "peer",
                {"type": "cli", "command": "peer", "strict_evidence": True, "timeout": 60},
                "Original question",
                Path("/tmp"),
            )
        )

    assert call_count["n"] == 2, "expected one repair retry then give up"
    assert result.ok is False
    assert classify_error(result.error) == "untagged_evidence"


# ----- openai_compatible path --------------------------------------------

def _openai_response(content: str, finish: str = "stop") -> dict:
    return {
        "model": "test/model",
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def test_strict_evidence_openai_retry_recovers_on_success(monkeypatch):
    """openai_compatible: untagged → repair-retry → tagged → ok=True."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    call_count = {"n": 0}
    captured_payloads: list[dict] = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_request(client, method, url, **kwargs):
        call_count["n"] += 1
        captured_payloads.append(kwargs.get("json") or {})
        if call_count["n"] == 1:
            return FakeResponse(_openai_response(_UNTAGGED_OUTPUT))
        return FakeResponse(_openai_response(_TAGGED_OUTPUT))

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "endpoint",
            {
                "type": "openai_compatible",
                "model": "test/model",
                "base_url": "https://api.example.com/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "strict_evidence": True,
            },
            "Original question",
        )
    )

    assert call_count["n"] == 2, "expected one repair retry"
    assert result.ok is True
    assert result.repair_retry_recovered is True
    # Retry payload contains the directive in the user message
    retry_messages = captured_payloads[1].get("messages") or []
    user_blob = "\n".join(
        m.get("content", "") for m in retry_messages if isinstance(m.get("content"), str)
    )
    assert STRICT_EVIDENCE_REPAIR_RETRY_INSTRUCTION[:40] in user_blob


def test_strict_evidence_openai_retry_fails_returns_untagged_evidence(monkeypatch):
    """openai_compatible: both untagged → ok=False, error_kind=untagged_evidence."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    call_count = {"n": 0}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_request(client, method, url, **kwargs):
        call_count["n"] += 1
        return FakeResponse(_openai_response(_UNTAGGED_OUTPUT))

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "endpoint",
            {
                "type": "openai_compatible",
                "model": "test/model",
                "base_url": "https://api.example.com/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "strict_evidence": True,
            },
            "Original question",
        )
    )

    assert call_count["n"] == 2, "expected one repair retry then give up"
    assert result.ok is False
    assert classify_error(result.error) == "untagged_evidence"


# ----- ollama path -------------------------------------------------------

def _ollama_response(content: str, done_reason: str = "stop") -> dict:
    return {
        "message": {"content": content},
        "done_reason": done_reason,
    }


def test_strict_evidence_ollama_retry_recovers_on_success(monkeypatch):
    """ollama: untagged → repair-retry → tagged → ok=True."""
    call_count = {"n": 0}
    captured_payloads: list[dict] = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_request(client, method, url, **kwargs):
        call_count["n"] += 1
        captured_payloads.append(kwargs.get("json") or {})
        if call_count["n"] == 1:
            return FakeResponse(_ollama_response(_UNTAGGED_OUTPUT))
        return FakeResponse(_ollama_response(_TAGGED_OUTPUT))

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_ollama_participant(
            "local",
            {
                "type": "ollama",
                "model": "test:local",
                "base_url": "http://localhost:11434",
                "strict_evidence": True,
            },
            "Original question",
        )
    )

    assert call_count["n"] == 2, "expected one repair retry"
    assert result.ok is True
    assert result.repair_retry_recovered is True
    retry_messages = captured_payloads[1].get("messages") or []
    user_blob = "\n".join(
        m.get("content", "") for m in retry_messages if isinstance(m.get("content"), str)
    )
    assert STRICT_EVIDENCE_REPAIR_RETRY_INSTRUCTION[:40] in user_blob


def test_strict_evidence_ollama_retry_fails_returns_untagged_evidence(monkeypatch):
    """ollama: both untagged → ok=False, error_kind=untagged_evidence."""
    call_count = {"n": 0}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_request(client, method, url, **kwargs):
        call_count["n"] += 1
        return FakeResponse(_ollama_response(_UNTAGGED_OUTPUT))

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_ollama_participant(
            "local",
            {
                "type": "ollama",
                "model": "test:local",
                "base_url": "http://localhost:11434",
                "strict_evidence": True,
            },
            "Original question",
        )
    )

    assert call_count["n"] == 2, "expected one repair retry then give up"
    assert result.ok is False
    assert classify_error(result.error) == "untagged_evidence"


# ----- Empty-evidence sanity check (FORMAT not PRESENCE) -----------------

def test_strict_evidence_empty_list_does_not_trigger_retry():
    """A response with no EVIDENCE entries passes strict_evidence and
    therefore never reaches the retry path. The gate is FORMAT of entries
    that exist, not PRESENCE. Asserts the retry path is gated by the
    UntaggedEvidence error string, not by empty evidence."""
    call_count = {"n": 0}

    async def fake_run_cli_once(name, cfg, prompt, cwd, *, start, mode_multiplier=None, mode=None):
        call_count["n"] += 1
        return (
            ParticipantResult(
                name=name,
                ok=True,
                output="RECOMMENDATION: yes - empty evidence is fine\n",
                error="",
                elapsed_seconds=1.0,
            ),
            {"nonzero_exit": False, "stderr": "", "exited": True},
        )

    with patch("llm_council.adapters._run_cli_once", side_effect=fake_run_cli_once), \
         patch("llm_council.adapters._cache_lookup", return_value=(None, None)), \
         patch("llm_council.adapters._maybe_persist_cache"):
        result = asyncio.run(
            run_cli_participant(
                "peer",
                {"type": "cli", "command": "peer", "strict_evidence": True, "timeout": 60},
                "Original question",
                Path("/tmp"),
            )
        )

    assert call_count["n"] == 1, "no retry should fire for empty evidence"
    assert result.ok is True


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
