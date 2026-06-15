"""Tests for the advisory enrichments to council_recommend / council_estimate.

Covers:
  - M10: policy.recommend mechanical difficulty_class + suggested_mode_reason_codes
  - M9:  optional LLM difficulty judge (default OFF, fail-open)
  - L6:  reliability-based "consider dropping" advisory
  - config: defaults.recommend_judge validation
"""

from pathlib import Path

import pytest

from llm_council import policy
from llm_council.config import validate_config
from llm_council.mcp_server import _run_recommend, estimate_run
from llm_council.policy import (
    recommend,
    peers_to_consider_dropping,
    should_use_council,
)
from llm_council.recommend_judge import grade_difficulty


# ---------------------------------------------------------------------------
# M10 — policy.recommend mechanical enrichment
# ---------------------------------------------------------------------------


def test_recommend_preserves_should_use_council_fields():
    task = "architecture decision for auth"
    use, mode, reason = should_use_council(task)
    result = recommend(task)
    assert result["use_council"] == use
    assert result["mode"] == mode
    assert result["reason"] == reason


def test_recommend_difficulty_trivial():
    result = recommend("rename a local variable", risk="low")
    assert result["difficulty_class"] == "trivial"
    assert result["suggested_mode_reason_codes"] == []


def test_recommend_difficulty_moderate():
    # medium risk, no failed attempts, no trigger keywords -> not trivial,
    # not hard.
    result = recommend("tweak the button color", risk="medium")
    assert result["difficulty_class"] == "moderate"
    assert result["suggested_mode_reason_codes"] == []


def test_recommend_difficulty_hard_via_risk():
    result = recommend("ship a tiny copy fix", risk="high")
    assert result["difficulty_class"] == "hard"


def test_recommend_difficulty_hard_via_failed_attempts():
    result = recommend("fix the flaky test", risk="low", failed_attempts=2)
    assert result["difficulty_class"] == "hard"


def test_recommend_difficulty_hard_via_files_touched():
    result = recommend("rename a symbol everywhere", risk="low", files_touched=5)
    assert result["difficulty_class"] == "hard"


def test_recommend_difficulty_hard_via_matched_keywords():
    # 3 trigger keywords: refactor, auth, schema
    result = recommend("refactor the auth schema", risk="low")
    assert result["difficulty_class"] == "hard"


def test_recommend_reason_codes_lists_matched_keywords():
    result = recommend("refactor the auth schema")
    assert result["suggested_mode_reason_codes"] == ["refactor", "auth", "schema"]


def test_recommend_reason_codes_in_catalog_order():
    # "schema" appears in the task before "auth", but catalog order
    # (auth before schema) governs the returned list.
    result = recommend("update the schema and add auth checks")
    assert result["suggested_mode_reason_codes"] == ["auth", "schema"]


# ---------------------------------------------------------------------------
# M9 — optional LLM difficulty judge: resolution + fail-open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_difficulty_none_when_unset():
    # No defaults.recommend_judge -> feature off.
    assert await grade_difficulty("do a thing", {"defaults": {}, "participants": {}}) is None


@pytest.mark.asyncio
async def test_grade_difficulty_none_when_peer_missing():
    config = {
        "defaults": {"recommend_judge": "ghost"},
        "participants": {},
    }
    assert await grade_difficulty("do a thing", config) is None


@pytest.mark.asyncio
async def test_grade_difficulty_none_when_peer_not_hosted():
    config = {
        "defaults": {"recommend_judge": "claude"},
        "participants": {"claude": {"type": "cli", "model": "x"}},
    }
    assert await grade_difficulty("do a thing", config) is None


@pytest.mark.asyncio
async def test_grade_difficulty_none_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = {
        "defaults": {"recommend_judge": "or"},
        "participants": {
            "or": {"type": "openrouter", "model": "some/model"},
        },
    }
    assert await grade_difficulty("do a thing", config) is None


@pytest.mark.asyncio
async def test_grade_difficulty_parses_strict_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    config = {
        "defaults": {"recommend_judge": "or"},
        "participants": {
            "or": {"type": "openrouter", "model": "some/model"},
        },
    }

    async def fake_post(endpoint, headers, payload):
        assert "chat/completions" in endpoint
        assert headers["Authorization"] == "Bearer sk-test"
        return '{"difficulty": "HARD", "rationale": "deep", "suggested_mode": "plan"}'

    monkeypatch.setattr(
        "llm_council.recommend_judge._post_once_with_retry", fake_post
    )
    out = await grade_difficulty("refactor the world", config)
    assert out == {
        "difficulty": "HARD",
        "rationale": "deep",
        "suggested_mode": "plan",
    }


@pytest.mark.asyncio
async def test_grade_difficulty_fail_open_on_exception(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    config = {
        "defaults": {"recommend_judge": "or"},
        "participants": {
            "or": {"type": "openrouter", "model": "some/model"},
        },
    }

    async def boom(endpoint, headers, payload):
        raise RuntimeError("network is down")

    monkeypatch.setattr("llm_council.recommend_judge._post_once_with_retry", boom)
    # grade_difficulty must NEVER raise; it swallows and returns None.
    assert await grade_difficulty("x", config) is None


@pytest.mark.asyncio
async def test_grade_difficulty_strips_code_fences(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    config = {
        "defaults": {"recommend_judge": "or"},
        "participants": {"or": {"type": "openrouter", "model": "m"}},
    }

    async def fenced(endpoint, headers, payload):
        return '```json\n{"difficulty": "moderate", "rationale": "ok"}\n```'

    monkeypatch.setattr("llm_council.recommend_judge._post_once_with_retry", fenced)
    out = await grade_difficulty("x", config)
    assert out["difficulty"] == "MODERATE"
    assert out["suggested_mode"] == ""


# ---------------------------------------------------------------------------
# M9 — council_recommend handler integration
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / ".llm-council.yaml").write_text(body.lstrip(), encoding="utf-8")


@pytest.mark.asyncio
async def test_council_recommend_judge_attached_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    _write_config(
        tmp_path,
        """
defaults:
  recommend_judge: or
participants:
  or:
    type: openrouter
    model: some/model
""",
    )

    async def fake_grade(task, config):
        return {"difficulty": "HARD", "rationale": "r", "suggested_mode": "plan"}

    monkeypatch.setattr("llm_council.mcp_server.grade_difficulty", fake_grade)
    result = await _run_recommend(
        {"task": "refactor auth schema", "working_directory": str(tmp_path)}
    )
    # Mechanical primary verdict still present and unchanged.
    assert result["difficulty_class"] == "hard"
    assert result["use_council"] is True
    # Judge attached as supplementary.
    assert result["judge"] == {
        "difficulty": "HARD",
        "rationale": "r",
        "suggested_mode": "plan",
    }


@pytest.mark.asyncio
async def test_council_recommend_judge_none_leaves_mechanical(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    _write_config(
        tmp_path,
        """
defaults:
  recommend_judge: or
participants:
  or:
    type: openrouter
    model: some/model
""",
    )

    async def fake_grade(task, config):
        return None

    monkeypatch.setattr("llm_council.mcp_server.grade_difficulty", fake_grade)
    result = await _run_recommend(
        {"task": "tweak the button color", "working_directory": str(tmp_path)}
    )
    assert "judge" not in result
    # Mechanical verdict untouched.
    expected = recommend("tweak the button color")
    assert result["use_council"] == expected["use_council"]
    assert result["mode"] == expected["mode"]
    assert result["difficulty_class"] == expected["difficulty_class"]


@pytest.mark.asyncio
async def test_council_recommend_judge_not_invoked_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    _write_config(
        tmp_path,
        """
participants:
  or:
    type: openrouter
    model: some/model
""",
    )

    invoked = {"count": 0}

    async def fake_grade(task, config):
        invoked["count"] += 1
        return {"difficulty": "HARD", "rationale": "", "suggested_mode": ""}

    monkeypatch.setattr("llm_council.mcp_server.grade_difficulty", fake_grade)
    result = await _run_recommend(
        {"task": "refactor auth schema", "working_directory": str(tmp_path)}
    )
    assert invoked["count"] == 0
    assert "judge" not in result


# ---------------------------------------------------------------------------
# L6 — reliability-based "consider dropping" advisory
# ---------------------------------------------------------------------------


def _reliability(peers):
    return {
        "total_outcomes": 0,
        "transcripts_considered": 0,
        "filters": {"peer": None},
        "peers": peers,
    }


def test_peers_to_consider_dropping_flags_bad_peer():
    rel = _reliability(
        [
            {
                "name": "noisy",
                "useful_count": 1,
                "false_blocker_count": 4,
                "verified_citation_rate": 0.2,
            },
            {
                "name": "healthy",
                "useful_count": 5,
                "false_blocker_count": 0,
                "verified_citation_rate": 0.9,
            },
            {
                "name": "no_signal",
                "useful_count": 0,
                "false_blocker_count": 3,
                "verified_citation_rate": None,
            },
        ]
    )
    assert peers_to_consider_dropping(rel) == ["noisy"]


def test_peers_to_consider_dropping_skips_high_rate():
    # false_blocker > useful but verified rate is NOT low -> not flagged.
    rel = _reliability(
        [
            {
                "name": "x",
                "useful_count": 0,
                "false_blocker_count": 2,
                "verified_citation_rate": 0.7,
            }
        ]
    )
    assert peers_to_consider_dropping(rel) == []


def test_peers_to_consider_dropping_empty_on_no_data():
    assert peers_to_consider_dropping(_reliability([])) == []
    assert peers_to_consider_dropping({}) == []
    assert peers_to_consider_dropping(None) == []


@pytest.mark.asyncio
async def test_council_recommend_surfaces_drop_advisory(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    _write_config(
        tmp_path,
        """
participants:
  or:
    type: openrouter
    model: some/model
""",
    )

    def fake_rel(cwd, *, transcripts_dir=None, peer=None):
        return _reliability(
            [
                {
                    "name": "noisy",
                    "useful_count": 0,
                    "false_blocker_count": 3,
                    "verified_citation_rate": 0.1,
                }
            ]
        )

    monkeypatch.setattr("llm_council.mcp_server.aggregate_reliability", fake_rel)
    result = await _run_recommend(
        {"task": "do a thing", "working_directory": str(tmp_path)}
    )
    assert result["peers_to_consider_dropping"] == ["noisy"]


def test_estimate_run_surfaces_drop_advisory(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    _write_config(
        tmp_path,
        """
participants:
  or:
    type: openrouter
    model: some/model
modes:
  solo:
    participants:
    - or
""",
    )

    def fake_rel(cwd, *, transcripts_dir=None, peer=None):
        return _reliability(
            [
                {
                    "name": "noisy",
                    "useful_count": 0,
                    "false_blocker_count": 3,
                    "verified_citation_rate": 0.1,
                }
            ]
        )

    monkeypatch.setattr("llm_council.mcp_server.aggregate_reliability", fake_rel)
    result = estimate_run(
        {
            "question": "do a thing",
            "mode": "solo",
            "current": "codex",
            "working_directory": str(tmp_path),
        }
    )
    assert result["ok"] is True
    assert result["peers_to_consider_dropping"] == ["noisy"]


def test_estimate_run_empty_drop_advisory_on_no_data(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    _write_config(
        tmp_path,
        """
participants:
  or:
    type: openrouter
    model: some/model
modes:
  solo:
    participants:
    - or
""",
    )
    result = estimate_run(
        {
            "question": "do a thing",
            "mode": "solo",
            "current": "codex",
            "working_directory": str(tmp_path),
        }
    )
    assert result["ok"] is True
    assert result["peers_to_consider_dropping"] == []


# ---------------------------------------------------------------------------
# config — defaults.recommend_judge validation
# ---------------------------------------------------------------------------


def _minimal_config(defaults: dict) -> dict:
    return {
        "defaults": defaults,
        "participants": {"or": {"type": "openrouter", "model": "m"}},
        "modes": {"solo": {"participants": ["or"]}},
    }


def test_config_rejects_non_string_recommend_judge():
    with pytest.raises(ValueError, match="recommend_judge must be a string"):
        validate_config(_minimal_config({"recommend_judge": 123}))


def test_config_accepts_string_recommend_judge():
    # No raise.
    validate_config(_minimal_config({"recommend_judge": "or"}))


def test_config_absent_recommend_judge_ok():
    validate_config(_minimal_config({}))
