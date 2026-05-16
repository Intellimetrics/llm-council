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


def test_envelope_inline_evidence_line_form_is_captured():
    """Pass-9 dogfood regression: qwen_coder_plus emitted each evidence
    claim on its own ``EVIDENCE: <content>`` line instead of a bare
    ``EVIDENCE:`` header with ``- bullet`` lines beneath. The original
    parser only recognized the bare-header-plus-bullets form, so every
    inline ``EVIDENCE:`` line was silently dropped. That silently
    disabled strict-evidence validation: the response had an empty
    parsed evidence list, the validator only fires on parsed-but-
    untagged entries, so the untagged inline lines never tripped the
    repair-retry. The fix added ``_ENVELOPE_LIST_INLINE_RE`` so each
    inline line counts as one entry."""
    text = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE: [PUBLISHED] - tagged inline form\n"
        "EVIDENCE: llm_council/adapters.py:305 untagged inline form\n"
        "EVIDENCE: [OBSERVABLE] - another tagged inline\n"
    )
    env = _extract_response_envelope(text)
    assert env["evidence"] == [
        {"text": "tagged inline form", "tag": "published"},
        {
            "text": "llm_council/adapters.py:305 untagged inline form",
            "tag": None,
        },
        {"text": "another tagged inline", "tag": "observable"},
    ]


def test_envelope_inline_evidence_mixed_with_bare_header_and_bullets():
    """Mixed shape: a bare ``EVIDENCE:`` header followed by ``- bullet``
    lines, then later a stand-alone ``EVIDENCE: <content>`` line.
    Both forms must accrue under the same ``evidence`` list."""
    text = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE:\n"
        "- canonical bullet [PUBLISHED]\n"
        "- another bullet [OBSERVABLE]\n"
        "\n"
        "EVIDENCE: [INFERRED] - inline after a break\n"
    )
    env = _extract_response_envelope(text)
    assert env["evidence"] == [
        {"text": "canonical bullet", "tag": "published"},
        {"text": "another bullet", "tag": "observable"},
        {"text": "inline after a break", "tag": "inferred"},
    ]


def test_envelope_inline_tests_to_run_line_form_is_captured():
    """The same single-line form for the non-tag list fields. Qwen
    emitted ``TESTS_TO_RUN: ...`` inline too. Plain strings (no tag
    parsing) for tests_to_run / blockers / assumptions."""
    text = (
        "RECOMMENDATION: yes - ok\n"
        "TESTS_TO_RUN: Check logs for terse_retry_attempted\n"
        "TESTS_TO_RUN: Verify strict_evidence validation\n"
        "BLOCKERS: missing X\n"
        "ASSUMPTIONS: assume Y\n"
    )
    env = _extract_response_envelope(text)
    assert env["tests_to_run"] == [
        "Check logs for terse_retry_attempted",
        "Verify strict_evidence validation",
    ]
    assert env["blockers"] == ["missing X"]
    assert env["assumptions"] == ["assume Y"]


def test_strict_evidence_inline_evidence_form_triggers_validator():
    """End-to-end pass-9 regression: a response that uses qwen's
    inline ``EVIDENCE: <content>`` form with one untagged entry must
    now trip ``UntaggedEvidence:`` under strict_evidence=true. Before
    the fix this returned the empty string (silent pass) because the
    parsed evidence list was empty."""
    output = (
        "RECOMMENDATION: yes - ok\n"
        "EVIDENCE: [PUBLISHED] - tagged inline\n"
        "EVIDENCE: llm_council/adapters.py:305 untagged inline\n"
    )
    cfg = {"strict_evidence": True}
    error = _response_validation_error(output, cfg)
    assert error.startswith(UNTAGGED_EVIDENCE_PREFIX)
    # The validator should count exactly 1 untagged entry, not all 2.
    assert "1 EVIDENCE entry" in error


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


# ----- Pass-9 regression: no chained inner label-repair on strict-evidence
# retry ---------------------------------------------------------------------
#
# Pass-9 finding B: the hosted (`openai_compatible`) and `ollama` wrappers
# passed the ORIGINAL `cfg` to the inner adapter when firing the
# strict-evidence repair retry. If the retry response dropped the
# `RECOMMENDATION:` label (e.g., the peer rewrote focusing only on tags),
# the inner's own label-repair branch (`_run_*_inner` gated by
# `_retry_enabled` + `_is_label_only_failure`) would fire a third outer
# HTTP call. CLAUDE.md invariant: "one extra call per peer per round,
# never two". The CLI path is unaffected — `_run_cli_once` has no inner
# label-repair retry, label-repair is wrapper-level there.
#
# The fix mirrors the section-repair pattern at adapters.py:1089-1090
# (and the ollama analog): build a `retry_cfg = dict(cfg)` and set
# `retry_cfg["retry_on_missing_label"] = False` before calling the inner.
#
# These tests assert that a strict-evidence retry whose response lacks
# the RECOMMENDATION label triggers EXACTLY 2 outer HTTP calls (original
# + strict-evidence retry), NOT 3.


_LABEL_LESS_OUTPUT = (
    "EVIDENCE:\n"
    "- I rewrote the response to focus on tags but forgot the label\n"
    "- [PUBLISHED] some source\n"
    "- [OBSERVABLE] direct measurement\n"
)


def test_strict_evidence_openai_retry_label_less_does_not_chain_inner_label_repair(
    monkeypatch,
):
    """openai_compatible: strict-evidence retry that drops the
    RECOMMENDATION label must NOT trigger the inner's label-repair branch
    — that would be a third outer call. Pass-9 finding B regression."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    call_count = {"n": 0}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_request(client, method, url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Original: has label, has untagged evidence → fails
            # strict_evidence validation, triggers wrapper retry.
            return FakeResponse(_openai_response(_UNTAGGED_OUTPUT))
        # Strict-evidence retry: peer rewrote and lost the label.
        # The inner would normally repair-retry this, but the wrapper
        # must have disabled `retry_on_missing_label` on the retry_cfg.
        return FakeResponse(_openai_response(_LABEL_LESS_OUTPUT))

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

    assert call_count["n"] == 2, (
        f"expected exactly 2 outer HTTP calls (original + strict-evidence retry), "
        f"got {call_count['n']} — inner label-repair chained on the retry"
    )
    assert result.ok is False


def test_strict_evidence_ollama_retry_label_less_does_not_chain_inner_label_repair(
    monkeypatch,
):
    """ollama: same as the openai_compatible regression above."""
    call_count = {"n": 0}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_request(client, method, url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return FakeResponse(_ollama_response(_UNTAGGED_OUTPUT))
        return FakeResponse(_ollama_response(_LABEL_LESS_OUTPUT))

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

    assert call_count["n"] == 2, (
        f"expected exactly 2 outer HTTP calls (original + strict-evidence retry), "
        f"got {call_count['n']} — inner label-repair chained on the retry"
    )
    assert result.ok is False


def test_section_repair_openai_retry_label_less_does_not_chain_inner_label_repair(
    monkeypatch,
):
    """Sanity / no-regression: the section-repair path already disables
    `retry_on_missing_label` on its retry_cfg (adapters.py:1089-1090).
    Confirm that protection is intact — a section-repair retry that
    drops the RECOMMENDATION label still produces exactly 2 outer calls."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    call_count = {"n": 0}

    # Original: has label but missing a REQUIRED section.
    section_prompt = (
        "Question text.\n\n"
        "PART 1 — ANALYSIS (REQUIRED)\n"
        "Please cover the analysis.\n\n"
        "PART 2 — RECOMMENDATION (REQUIRED)\n"
    )
    original_output = (
        "RECOMMENDATION: yes - ok\n"
        "I am skipping PART 1 entirely.\n"
    )
    # Retry: drops the label entirely.
    retry_output = (
        "PART 1 — ANALYSIS\n"
        "Here is the analysis, but I forgot the label.\n"
    )

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_request(client, method, url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return FakeResponse(_openai_response(original_output))
        return FakeResponse(_openai_response(retry_output))

    monkeypatch.setattr(adapters_module, "_request_with_retries", fake_request)

    result = asyncio.run(
        run_openai_compatible_participant(
            "endpoint",
            {
                "type": "openai_compatible",
                "model": "test/model",
                "base_url": "https://api.example.com/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "require_sections": True,
            },
            section_prompt,
        )
    )

    assert call_count["n"] == 2, (
        f"section-repair must produce exactly 2 outer calls; "
        f"got {call_count['n']} — inner label-repair regressed"
    )
    assert result.ok is False
