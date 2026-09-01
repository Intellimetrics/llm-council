"""Tests for the v0.25 operator-guidance surfaces: prescriptive stats
recommendations, doctor remediation + okf row, setup --write-instructions
marker blocks, and list --verbose tuning-key discovery."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from llm_council.stats import (
    aggregate,
    derive_recommendations,
    format_stats_text,
)


# ---------------------------------------------------------------------------
# stats: derive_recommendations


def _row(**overrides) -> dict:
    base = {
        "name": "codex",
        "runs": 10,
        "successes": 8,
        "invalid_label_rate": 0.0,
        "timeout_by_prompt_size": {},
        "timeout_recoveries_by_prompt_size": {},
        "timeout_recoveries": 0,
        "terse_retry_attempts": 0,
        "quota_incidents": 0,
        "quota_recoveries": 0,
        "error_kind_counts": {},
    }
    base.update(overrides)
    return base


def test_recommendation_timeout_wall_no_recoveries():
    stats = {
        "participants": [
            _row(
                timeout_by_prompt_size={"large": 3},
                timeout_recoveries_by_prompt_size={"large": 0},
            )
        ]
    }
    recs = derive_recommendations(stats)
    assert len(recs) == 1
    assert "3 timeouts on large prompts" in recs[0]
    assert "participants.codex.timeout" in recs[0]
    assert "timeout_multiplier" in recs[0]


def test_recommendation_timeout_silent_below_threshold_or_with_recoveries():
    below = {
        "participants": [
            _row(timeout_by_prompt_size={"large": 2})
        ]
    }
    assert derive_recommendations(below) == []
    recovered = {
        "participants": [
            _row(
                timeout_by_prompt_size={"large": 5},
                timeout_recoveries_by_prompt_size={"large": 2},
            )
        ]
    }
    assert derive_recommendations(recovered) == []


def test_recommendation_terse_retry_never_recovers():
    stats = {
        "participants": [
            _row(terse_retry_attempts=4, timeout_recoveries=0)
        ]
    }
    recs = derive_recommendations(stats)
    assert len(recs) == 1
    assert "terse-retry fired 4x" in recs[0]
    assert "terse_retry_on_timeout" in recs[0]


def test_recommendation_quota_without_fallback():
    stats = {
        "participants": [
            _row(quota_incidents=2, quota_recoveries=0)
        ]
    }
    recs = derive_recommendations(stats)
    assert len(recs) == 1
    assert "fallback_chain" in recs[0]
    # Recovering chains stay silent.
    stats_ok = {
        "participants": [_row(quota_incidents=3, quota_recoveries=2)]
    }
    assert derive_recommendations(stats_ok) == []


def test_recommendation_invalid_label_rate():
    stats = {
        "participants": [
            _row(runs=6, invalid_label_rate=0.5)
        ]
    }
    recs = derive_recommendations(stats)
    assert len(recs) == 1
    assert "RECOMMENDATION label" in recs[0]
    assert "require_recommendation" in recs[0]
    # Too few SUCCESSES → silent even at a high rate (the rate's
    # denominator is successes, so the gate must be too — a peer with one
    # unlabeled success must not print "100%" from a sample of one).
    small = {"participants": [_row(runs=8, successes=3, invalid_label_rate=1.0)]}
    assert derive_recommendations(small) == []


def test_recommendation_content_refused():
    stats = {
        "participants": [
            _row(error_kind_counts={"content_refused": 2})
        ]
    }
    recs = derive_recommendations(stats)
    assert len(recs) == 1
    assert "verification" in recs[0]


def test_recommendation_okf_binary_missing_and_unmatched():
    missing = {
        "participants": [],
        "okf_context_status_counts": {"binary_missing": 2},
    }
    recs = derive_recommendations(missing)
    assert len(recs) == 1
    assert "OKF binary (default `okf-rs`) was not on PATH" in recs[0]

    unmatched = {
        "participants": [],
        "okf_context_status_counts": {"no_matched_concepts": 4, "attached": 1},
    }
    recs = derive_recommendations(unmatched)
    assert len(recs) == 1
    assert "matched no concepts" in recs[0]

    healthy = {
        "participants": [],
        "okf_context_status_counts": {"attached": 5, "no_matched_concepts": 1},
    }
    assert derive_recommendations(healthy) == []


def test_aggregate_counts_okf_statuses_and_attaches_recommendations():
    records = [
        {
            "mtime": 1000.0,
            "data": {
                "mode": "quick",
                "metadata": {"okf_context": {"status": "attached"}},
                "results": [],
            },
        },
        {
            "mtime": 1001.0,
            "data": {
                "mode": "quick",
                "metadata": {"okf_context": {"status": "binary_missing"}},
                "results": [],
            },
        },
        {
            "mtime": 1002.0,
            "data": {"mode": "quick", "results": []},  # feature off: no key
        },
    ]
    stats = aggregate(records)
    assert stats["okf_context_status_counts"] == {
        "attached": 1,
        "binary_missing": 1,
    }
    assert any("was not on PATH" in r for r in stats["recommendations"])


def test_format_stats_text_renders_recommendations_and_okf_line():
    stats = {
        "transcripts_considered": 1,
        "total_runs": 1,
        "total_successes": 1,
        "mode_counts": {},
        "okf_context_status_counts": {"attached": 3},
        "participants": [
            {
                "name": "codex",
                "runs": 1,
                "success_rate": 1.0,
                "avg_elapsed_seconds": 1.0,
                "label_counts": {"yes": 1, "no": 0, "tradeoff": 0, "unknown": 0},
                "invalid_label_rate": 0.0,
                "tokens_total": None,
                "cost_total": None,
                "last_used": None,
                "quota_incidents": 0,
                "quota_recoveries": 0,
            }
        ],
        "filters": {},
        "recommendations": ["codex: do the thing"],
    }
    text = format_stats_text(stats)
    assert "okf-context: attached=3" in text
    assert "recommendations:" in text
    assert "  - codex: do the thing" in text
    # Empty recommendations → no block.
    stats["recommendations"] = []
    stats["okf_context_status_counts"] = {}
    text = format_stats_text(stats)
    assert "recommendations:" not in text
    assert "okf-context:" not in text


# ---------------------------------------------------------------------------
# doctor: remediation + okf row


def test_doctor_cli_missing_carries_install_hint(monkeypatch: pytest.MonkeyPatch):
    import llm_council.doctor as doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda cmd: None)
    config = {
        "participants": {
            "claude": {"type": "cli", "family": "claude", "command": "claude"},
        }
    }
    checks = doctor.check_environment(config)
    cli_check = next(c for c in checks if c.name == "cli:claude")
    assert cli_check.ok is False
    assert "npm install -g @anthropic-ai/claude-code" in cli_check.detail


def test_doctor_env_missing_mentions_project_env_file(
    monkeypatch: pytest.MonkeyPatch,
):
    import llm_council.doctor as doctor

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = {
        "participants": {
            "hosted": {
                "type": "openrouter",
                "model": "x/y",
                "api_key_env": "OPENROUTER_API_KEY",
            },
        },
        "defaults": {"catalog_auto_refresh": False},
    }
    checks = doctor.check_environment(config)
    env_check = next(c for c in checks if c.name == "env:OPENROUTER_API_KEY")
    assert env_check.ok is False
    assert ".llm-council.env" in env_check.detail


def test_doctor_okf_row_states(monkeypatch: pytest.MonkeyPatch):
    import llm_council.doctor as doctor

    # Absent + feature disabled → informational ok row with install hint.
    monkeypatch.setattr(doctor.shutil, "which", lambda cmd: None)
    checks = doctor.check_environment({"participants": {}})
    okf = next(c for c in checks if c.name == "okf:binary")
    assert okf.ok is True
    assert "optional" in okf.detail
    assert "cargo install" in okf.detail

    # Absent + feature enabled → failing row.
    checks = doctor.check_environment(
        {"participants": {}, "defaults": {"okf_context": True}}
    )
    okf = next(c for c in checks if c.name == "okf:binary")
    assert okf.ok is False
    assert "okf_context is enabled" in okf.detail

    # Present → ok with usage hint.
    monkeypatch.setattr(doctor.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    checks = doctor.check_environment({"participants": {}})
    okf = next(c for c in checks if c.name == "okf:binary")
    assert okf.ok is True
    assert "--okf-context" in okf.detail


def test_okf_context_enabled_anywhere():
    from llm_council.doctor import okf_context_enabled_anywhere

    assert okf_context_enabled_anywhere({}) is False
    assert okf_context_enabled_anywhere({"defaults": {"okf_context": True}}) is True
    assert (
        okf_context_enabled_anywhere({"modes": {"review": {"okf_context": True}}})
        is True
    )
    assert (
        okf_context_enabled_anywhere({"modes": {"review": {"okf_context": False}}})
        is False
    )


# ---------------------------------------------------------------------------
# setup: --write-instructions marker blocks


def test_upsert_instruction_blocks_create_append_idempotent(tmp_path: Path):
    from llm_council.setup_wizard import (
        INSTRUCTION_BLOCK_BEGIN,
        INSTRUCTION_BLOCK_END,
        upsert_instruction_blocks,
    )

    # Existing CLAUDE.md content must be preserved outside the markers.
    (tmp_path / "CLAUDE.md").write_text(
        "# My project\n\nHouse rules here.\n", encoding="utf-8"
    )

    written = upsert_instruction_blocks(tmp_path, default_mode="review")
    names = {p.name for p in written}
    assert names == {"CLAUDE.md", "AGENTS.md", "GEMINI.md"}

    claude_text = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert claude_text.startswith("# My project")
    assert "House rules here." in claude_text
    assert claude_text.count(INSTRUCTION_BLOCK_BEGIN) == 1
    assert claude_text.count(INSTRUCTION_BLOCK_END) == 1
    # The created files hold only the block.
    agents_text = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert agents_text.startswith(INSTRUCTION_BLOCK_BEGIN)

    # Second run with the same inputs: byte-identical, nothing rewritten.
    second = upsert_instruction_blocks(tmp_path, default_mode="review")
    assert second == []
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == claude_text

    # Changed input: block replaced in place, outside content untouched,
    # still exactly one marker pair.
    third = upsert_instruction_blocks(tmp_path, default_mode="quick")
    assert {p.name for p in third} == {"CLAUDE.md", "AGENTS.md", "GEMINI.md"}
    updated = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert updated.startswith("# My project")
    assert "House rules here." in updated
    assert updated.count(INSTRUCTION_BLOCK_BEGIN) == 1
    assert updated.count(INSTRUCTION_BLOCK_END) == 1


def test_upsert_instruction_blocks_refuses_broken_markers(tmp_path: Path):
    from llm_council.setup_wizard import (
        INSTRUCTION_BLOCK_BEGIN,
        upsert_instruction_blocks,
    )

    (tmp_path / "CLAUDE.md").write_text(
        f"prefix\n{INSTRUCTION_BLOCK_BEGIN}\nno end marker\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unusable llm-council marker"):
        upsert_instruction_blocks(tmp_path)


def test_setup_write_instructions_conflicts_with_no_instructions(tmp_path: Path):
    from llm_council.cli import cmd_setup

    args = argparse.Namespace(
        root=str(tmp_path),
        plan=False,
        preset="tri-cli",
        yes=True,
        force=False,
        allow_incomplete=True,
        us_only_default=False,
        no_mcp=True,
        no_instructions=True,
        write_instructions=True,
        probe_local=False,
    )
    with pytest.raises(SystemExit, match="conflict"):
        cmd_setup(args)


def test_setup_write_instructions_end_to_end(tmp_path: Path):
    from llm_council.cli import cmd_setup
    from llm_council.setup_wizard import INSTRUCTION_BLOCK_BEGIN

    args = argparse.Namespace(
        root=str(tmp_path),
        plan=False,
        preset="tri-cli",
        yes=True,
        force=False,
        allow_incomplete=True,
        us_only_default=False,
        no_mcp=True,
        no_instructions=False,
        write_instructions=True,
        probe_local=False,
    )
    assert cmd_setup(args) == 0
    for entry in ("CLAUDE.md", "AGENTS.md", "GEMINI.md"):
        text = (tmp_path / entry).read_text(encoding="utf-8")
        assert INSTRUCTION_BLOCK_BEGIN in text
        assert "LLM Council" in text


# ---------------------------------------------------------------------------
# list --verbose


def test_participant_and_mode_verbose_notes():
    from llm_council.cli import _mode_verbose_notes, _participant_verbose_notes

    notes = _participant_verbose_notes(
        {
            "reasoning_effort": "medium",
            "usage_from_json": True,
            "env_strict": True,
            "fallback_chain": ["a", "b"],
            "require_recommendation": False,
        }
    )
    joined = "\n".join(notes)
    assert "reasoning_effort=medium" in joined
    assert "usage_from_json=true" in joined
    assert "env_strict=true" in joined
    assert "fallback_chain: a → b" in joined
    assert "require_recommendation=false" in joined
    # Defaults produce no notes.
    assert _participant_verbose_notes({"type": "cli"}) == []

    mode_notes = _mode_verbose_notes(
        {
            "timeout_multiplier": 2.0,
            "okf_context": True,
            "stances": {"a": "for"},
            "model_overrides": {"a": "m1"},
        }
    )
    joined = "\n".join(mode_notes)
    assert "timeout_multiplier=2.0" in joined
    assert "okf_context=true" in joined
    assert "stances: a=for" in joined
    assert "model_overrides: a→m1" in joined
    # timeout_multiplier of exactly 1.0 is the silent default.
    assert _mode_verbose_notes({"timeout_multiplier": 1.0}) == []


def test_cmd_list_hint_and_verbose(tmp_path: Path, capsys: pytest.CaptureFixture):
    import yaml

    from llm_council.cli import cmd_list

    config_path = tmp_path / ".llm-council.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "replace_defaults": True,
                "defaults": {
                    "mode": "custom",
                    "tiers": {"fast": {"a": "cheap-model"}},
                },
                "participants": {
                    "a": {"type": "cli", "command": "true", "env_strict": True}
                },
                "modes": {
                    "custom": {
                        "participants": ["a"],
                        "okf_context": True,
                        "description": "test mode",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert cmd_list(argparse.Namespace(config=str(config_path), verbose=False)) == 0
    out = capsys.readouterr().out
    assert "list --verbose" in out  # discovery hint
    assert "env_strict=true" not in out  # verbose notes absent

    assert cmd_list(argparse.Namespace(config=str(config_path), verbose=True)) == 0
    out = capsys.readouterr().out
    assert "env_strict=true" in out
    assert "okf_context=true" in out
    assert "Tiers (apply with --tier <name>):" in out
    assert "a→cheap-model" in out
    assert "run `llm-council list --verbose`" not in out


def test_recommendation_timeout_rules_deduped_per_peer():
    stats = {
        "participants": [
            _row(
                timeout_by_prompt_size={"large": 3},
                timeout_recoveries_by_prompt_size={"large": 0},
                terse_retry_attempts=3,
                timeout_recoveries=0,
            )
        ]
    }
    recs = derive_recommendations(stats)
    # Bucket rule wins; the terse-retry rule reads the same failures and
    # must not print near-duplicate advice for the same peer.
    assert len(recs) == 1
    assert "timeouts on large prompts" in recs[0]


def test_format_stats_text_advice_shown_with_no_participant_rows():
    stats = {
        "transcripts_considered": 2,
        "total_runs": 0,
        "total_successes": 0,
        "mode_counts": {},
        "okf_context_status_counts": {"binary_missing": 2},
        "participants": [],
        "filters": {},
        "recommendations": ["install okf-rs"],
    }
    text = format_stats_text(stats)
    assert "(no participants in selection)" in text
    assert "okf-context: binary_missing=2" in text
    assert "  - install okf-rs" in text


def test_upsert_refuses_duplicate_and_reversed_markers(tmp_path: Path):
    from llm_council.setup_wizard import (
        INSTRUCTION_BLOCK_BEGIN,
        INSTRUCTION_BLOCK_END,
        upsert_instruction_blocks,
    )

    (tmp_path / "CLAUDE.md").write_text(
        f"{INSTRUCTION_BLOCK_BEGIN}\nx\n{INSTRUCTION_BLOCK_BEGIN}\nx\n"
        f"{INSTRUCTION_BLOCK_END}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unusable llm-council marker"):
        upsert_instruction_blocks(tmp_path)

    (tmp_path / "CLAUDE.md").write_text(
        f"{INSTRUCTION_BLOCK_END}\nx\n{INSTRUCTION_BLOCK_BEGIN}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unusable llm-council marker"):
        upsert_instruction_blocks(tmp_path)


def test_upsert_ignores_midline_marker_mentions(tmp_path: Path):
    from llm_council.setup_wizard import (
        INSTRUCTION_BLOCK_BEGIN,
        upsert_instruction_blocks,
    )

    prose = (
        f"Docs: the `{INSTRUCTION_BLOCK_BEGIN}` marker delimits the block.\n"
    )
    (tmp_path / "CLAUDE.md").write_text(prose, encoding="utf-8")
    upsert_instruction_blocks(tmp_path)
    text = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    # The prose mention survives untouched and a real block was appended.
    assert text.startswith(prose)
    # Idempotent on re-run despite the prose mention.
    assert upsert_instruction_blocks(tmp_path) == []


def test_upsert_preserves_crlf_outside_block(tmp_path: Path):
    from llm_council.setup_wizard import upsert_instruction_blocks

    crlf_content = "# Project\r\n\r\nWindows house rules.\r\n"
    (tmp_path / "CLAUDE.md").write_bytes(crlf_content.encode("utf-8"))
    upsert_instruction_blocks(tmp_path)
    raw = (tmp_path / "CLAUDE.md").read_bytes()
    assert raw.startswith(crlf_content.encode("utf-8"))
    # Re-run stays idempotent with the mixed endings.
    assert upsert_instruction_blocks(tmp_path) == []


def test_upsert_preflights_all_targets_before_writing(tmp_path: Path):
    from llm_council.setup_wizard import (
        INSTRUCTION_BLOCK_BEGIN,
        upsert_instruction_blocks,
    )

    original = "# My project\n"
    (tmp_path / "CLAUDE.md").write_text(original, encoding="utf-8")
    (tmp_path / "GEMINI.md").write_text(
        f"{INSTRUCTION_BLOCK_BEGIN}\nno end\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unusable llm-council marker"):
        upsert_instruction_blocks(tmp_path)
    # Nothing was written anywhere: CLAUDE.md untouched, AGENTS.md absent.
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == original
    assert not (tmp_path / "AGENTS.md").exists()
    # No stray temp files either.
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_upsert_symlinked_entry_points_written_once(tmp_path: Path):
    from llm_council.setup_wizard import (
        INSTRUCTION_BLOCK_BEGIN,
        upsert_instruction_blocks,
    )

    (tmp_path / "CLAUDE.md").write_text("# shared\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").symlink_to(tmp_path / "CLAUDE.md")
    written = upsert_instruction_blocks(tmp_path)
    # CLAUDE.md (shared) once + GEMINI.md; AGENTS.md skipped as duplicate.
    assert {p.name for p in written} == {"CLAUDE.md", "GEMINI.md"}
    shared = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert shared.count(INSTRUCTION_BLOCK_BEGIN) == 1
    # The symlink itself survives (write went through os.replace on the
    # resolved target, not over the link).
    assert (tmp_path / "AGENTS.md").is_symlink()


def test_cmd_doctor_exit_gating_for_okf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import shutil as _shutil

    import llm_council.cli as cli_module
    import llm_council.doctor as doctor
    from llm_council.cli import cmd_doctor

    real_which = _shutil.which

    def _no_okf(cmd, *a, **k):
        if cmd == "okf-rs":
            return None
        if cmd == "true":
            return "/usr/bin/true"
        return real_which(cmd, *a, **k)

    monkeypatch.setattr(doctor.shutil, "which", _no_okf)
    base = {
        "replace_defaults": True,
        "defaults": {"mode": "custom"},
        "participants": {"a": {"type": "cli", "command": "true"}},
        "modes": {
            "custom": {"participants": ["a"], "description": "d"}
        },
    }
    args = argparse.Namespace(
        config=None, json=True, probe_openrouter=False, probe_ollama=False
    )

    from llm_council.config import resolve_config_data

    # Feature disabled: missing binary is informational, doctor exits 0.
    monkeypatch.setattr(
        cli_module, "load_config",
        lambda *a, **k: resolve_config_data(dict(base)),
    )
    assert cmd_doctor(args) == 0

    # Feature enabled in defaults: missing binary fails the doctor.
    enabled = dict(base)
    enabled["defaults"] = {"mode": "custom", "okf_context": True}
    monkeypatch.setattr(
        cli_module, "load_config",
        lambda *a, **k: resolve_config_data(dict(enabled)),
    )
    assert cmd_doctor(args) == 1
