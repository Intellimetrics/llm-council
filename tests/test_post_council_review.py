"""Regression tests for the post-council-review side-fixes.

Side-fix #4: MCP summary table at mcp_server.py:815 used to select
round-1 rows after deliberation (`if ":round" not in row["name"]`),
which is the opposite of the documented "final-round" intent. Now
filters against `final_round_results(results)`.

Side-fix #1: stats.aggregate now buckets by error_kind and envelope
field presence so an optional->required envelope flip has telemetry.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_council.adapters import ABDICATED_ERROR_PREFIX, ParticipantResult
from llm_council.stats import _new_peer_bucket, aggregate


def test_peer_bucket_has_error_kind_and_envelope_buckets():
    bucket = _new_peer_bucket()
    assert "error_kind_counts" in bucket
    assert bucket["error_kind_counts"] == {}
    assert "envelope_field_present" in bucket
    # All envelope fields should start at zero so optional->required
    # rollout can read presence rates without conditional defaults.
    for field in ("effort", "confidence", "risk", "blockers", "evidence"):
        assert bucket["envelope_field_present"][field] == 0


def test_aggregate_counts_error_kinds_per_peer():
    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "review",
                "results": [
                    {
                        "name": "claude",
                        "ok": False,
                        "error_kind": "timeout",
                        "output": "",
                    },
                    {
                        "name": "codex",
                        "ok": False,
                        "error_kind": "abdicated",
                        "output": "RECOMMENDATION: no",
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    by_peer = {row["name"]: row for row in result["participants"]}
    assert by_peer["claude"]["error_kind_counts"] == {"timeout": 1}
    assert by_peer["codex"]["error_kind_counts"] == {"abdicated": 1}


def test_aggregate_counts_envelope_field_presence():
    records = [
        {
            "mtime": 1.0,
            "data": {
                "mode": "review",
                "results": [
                    {
                        "name": "claude",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        "effort": "full",
                        "confidence": "high",
                        "blockers": ["one"],
                    },
                    {
                        "name": "claude",
                        "ok": True,
                        "output": "RECOMMENDATION: yes - ok",
                        # No envelope fields present at all this time.
                    },
                ],
            },
        }
    ]
    result = aggregate(records)
    bucket = next(row for row in result["participants"] if row["name"] == "claude")
    presence = bucket["envelope_field_present"]
    assert presence["effort"] == 1
    assert presence["confidence"] == 1
    assert presence["blockers"] == 1
    assert presence["risk"] == 0


def test_mcp_summary_uses_final_round_after_deliberation(
    monkeypatch, tmp_path: Path
):
    """Regression for mcp_server.py:815 — filter must keep final-round rows.

    Builds a fake council run where deliberation happened, so round-1
    results have plain names and round-2 results have a `:round2` suffix.
    The summary table must show the round-2 labels, not round-1.
    """
    from llm_council.mcp_server import run_council

    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  mode: review-cheap
participants:
  cheap_a:
    type: openrouter
    model: openai/gpt-4o-mini
    api_key_env: X
    input_per_million: 0.1
    output_per_million: 0.4
  cheap_b:
    type: openrouter
    model: openai/gpt-4o-mini
    api_key_env: X
    input_per_million: 0.1
    output_per_million: 0.4
modes:
  review-cheap:
    participants: [cheap_a, cheap_b]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("X", "secret")

    async def fake_execute_council(*args, **kwargs):
        return (
            [
                # Round 1 — peer A says yes, peer B says no.
                ParticipantResult("cheap_a", True, "RECOMMENDATION: yes - r1", "", 1.0),
                ParticipantResult("cheap_b", True, "RECOMMENDATION: no - r1", "", 1.0),
                # Round 2 — both flip to tradeoff (final position).
                ParticipantResult(
                    "cheap_a:round2", True, "RECOMMENDATION: tradeoff - r2", "", 1.0
                ),
                ParticipantResult(
                    "cheap_b:round2", True, "RECOMMENDATION: tradeoff - r2", "", 1.0
                ),
            ],
            {
                "rounds": 2,
                "deliberated": True,
                "min_quorum": 2,
                "labeled_quorum": 2,
                "degraded": False,
            },
        )

    import llm_council.mcp_server as mcp_module

    monkeypatch.setattr(mcp_module, "execute_council", fake_execute_council)

    result = asyncio.run(
        run_council({"question": "ping", "working_directory": str(tmp_path)})
    )
    payload = result["summary_markdown"]
    # The summary table must reflect the final round (tradeoff), not round 1.
    assert "tradeoff" in payload
    assert "| cheap_a | tradeoff |" in payload
    assert "| cheap_b | tradeoff |" in payload
    # The `:round2` suffix must be stripped for display.
    assert ":round2" not in payload


def test_mcp_lifts_quota_throttled_peers_to_top_level(
    monkeypatch, tmp_path: Path
):
    """When `metadata.quota_throttled_peers` is set, the MCP payload should
    expose it as a top-level field (parallel to consensus_blockers) and
    strip it from metadata to avoid double-serialization."""
    from llm_council.mcp_server import run_council

    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  mode: quota-test
participants:
  cheap_a:
    type: openrouter
    model: openai/gpt-4o-mini
    api_key_env: X
    input_per_million: 0.1
    output_per_million: 0.4
  cheap_b:
    type: openrouter
    model: openai/gpt-4o-mini
    api_key_env: X
    input_per_million: 0.1
    output_per_million: 0.4
modes:
  quota-test:
    participants: [cheap_a, cheap_b]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("X", "secret")

    quota_record = {
        "peer": "cheap_a",
        "family": "openrouter",
        "model": "openai/gpt-4o-mini",
        "message": "HTTPStatusError: 429 Too Many Requests",
    }

    async def fake_execute_council(*args, **kwargs):
        return (
            [
                ParticipantResult(
                    "cheap_a",
                    False,
                    "",
                    "HTTPStatusError: 429 Too Many Requests",
                    1.0,
                ),
                ParticipantResult(
                    "cheap_b", True, "RECOMMENDATION: yes - ship", "", 1.0
                ),
            ],
            {
                "rounds": 1,
                "deliberated": False,
                "min_quorum": 1,
                "labeled_quorum": 1,
                "degraded": False,
                "quota_throttled_peers": [quota_record],
            },
        )

    import llm_council.mcp_server as mcp_module

    monkeypatch.setattr(mcp_module, "execute_council", fake_execute_council)

    result = asyncio.run(
        run_council({"question": "ping", "working_directory": str(tmp_path)})
    )
    assert result.get("quota_throttled_peers") == [quota_record]
    # The field must be stripped from metadata to avoid double-encoding.
    assert "quota_throttled_peers" not in (result.get("metadata") or {})


def test_mcp_omits_quota_throttled_peers_when_empty(
    monkeypatch, tmp_path: Path
):
    """No quota issues -> the key must be absent from the payload, not an empty list."""
    from llm_council.mcp_server import run_council

    monkeypatch.setenv("LLM_COUNCIL_MCP_ROOT", str(tmp_path))
    (tmp_path / ".llm-council.yaml").write_text(
        """
defaults:
  mode: quota-test
participants:
  cheap_a:
    type: openrouter
    model: openai/gpt-4o-mini
    api_key_env: X
    input_per_million: 0.1
    output_per_million: 0.4
  cheap_b:
    type: openrouter
    model: openai/gpt-4o-mini
    api_key_env: X
    input_per_million: 0.1
    output_per_million: 0.4
modes:
  quota-test:
    participants: [cheap_a, cheap_b]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("X", "secret")

    async def fake_execute_council(*args, **kwargs):
        return (
            [
                ParticipantResult("cheap_a", True, "RECOMMENDATION: yes - ok", "", 1),
                ParticipantResult("cheap_b", True, "RECOMMENDATION: yes - ok", "", 1),
            ],
            {
                "rounds": 1,
                "deliberated": False,
                "min_quorum": 1,
                "labeled_quorum": 2,
                "degraded": False,
            },
        )

    import llm_council.mcp_server as mcp_module

    monkeypatch.setattr(mcp_module, "execute_council", fake_execute_council)

    result = asyncio.run(
        run_council({"question": "ping", "working_directory": str(tmp_path)})
    )
    assert "quota_throttled_peers" not in result
