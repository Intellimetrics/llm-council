"""Secret-scanner (Tier-2) tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_council.safety import (
    DEFAULT_ALLOWLIST_FILENAME,
    apply_secret_scan_policy,
    redact_secrets,
    scan_prompt_for_secrets,
)


def test_clean_prompt_yields_no_findings(tmp_path: Path):
    assert scan_prompt_for_secrets("hello world", cwd=tmp_path) == []


def test_detects_aws_access_key(tmp_path: Path):
    findings = scan_prompt_for_secrets(
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", cwd=tmp_path
    )
    kinds = [f["kind"] for f in findings]
    assert "aws_access_key" in kinds
    # Preview must never contain the full value.
    for finding in findings:
        assert "AKIAIOSFODNN7EXAMPLE" not in finding["preview"]


def test_detects_github_token(tmp_path: Path):
    findings = scan_prompt_for_secrets(
        "GH_TOKEN=ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJK", cwd=tmp_path
    )
    assert any(f["kind"] == "github_token" for f in findings)


def test_detects_anthropic_key_without_double_matching_openai(tmp_path: Path):
    text = "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
    findings = scan_prompt_for_secrets(text, cwd=tmp_path)
    kinds = [f["kind"] for f in findings]
    assert "anthropic_key" in kinds
    # sk-ant- must NOT also be flagged as a generic openai_key.
    assert "openai_key" not in kinds


def test_sk_test_placeholder_not_flagged(tmp_path: Path):
    """Test fixtures using sk-test-* must not generate false positives."""
    findings = scan_prompt_for_secrets(
        "OPENAI_API_KEY=sk-test-fake-key-for-tests-12345", cwd=tmp_path
    )
    assert findings == []


def test_allowlist_file_silences_matches(tmp_path: Path):
    (tmp_path / DEFAULT_ALLOWLIST_FILENAME).write_text("AKIAIOSFODNN7EXAMPLE\n")
    findings = scan_prompt_for_secrets(
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", cwd=tmp_path
    )
    assert findings == []


def test_findings_omit_raw_secret_value(tmp_path: Path):
    findings = scan_prompt_for_secrets(
        "GH_TOKEN=ghp_supersecrettoken_NEVER_LOG_THIS_VALUE_abcdef", cwd=tmp_path
    )
    for finding in findings:
        assert "NEVER_LOG_THIS_VALUE" not in str(finding)


def test_policy_off_returns_empty(tmp_path: Path):
    payload = apply_secret_scan_policy(
        "AKIAIOSFODNN7EXAMPLE", policy="off", cwd=tmp_path
    )
    assert payload["detected_count"] == 0
    assert payload["policy"] == "off"


def test_policy_warn_returns_findings_without_raising(tmp_path: Path):
    payload = apply_secret_scan_policy(
        "AKIAIOSFODNN7EXAMPLE and ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        policy="warn",
        cwd=tmp_path,
    )
    assert payload["detected_count"] == 2
    assert payload["kinds"] == {"aws_access_key": 1, "github_token": 1}


def test_policy_block_raises_on_finding(tmp_path: Path):
    with pytest.raises(ValueError, match="SecretsBlocked"):
        apply_secret_scan_policy(
            "AKIAIOSFODNN7EXAMPLE", policy="block", cwd=tmp_path
        )


def test_policy_block_no_findings_is_clean(tmp_path: Path):
    payload = apply_secret_scan_policy(
        "harmless prompt body", policy="block", cwd=tmp_path
    )
    assert payload["detected_count"] == 0


def test_invalid_policy_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid secret_scan policy"):
        apply_secret_scan_policy("x", policy="rage", cwd=tmp_path)


# --- redact policy -------------------------------------------------------

def test_redact_secrets_masks_value_and_returns_value_free_findings(tmp_path: Path):
    prompt = "here is the key AKIAIOSFODNN7EXAMPLE in the config"
    redacted, findings = redact_secrets(prompt, cwd=tmp_path)
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "[REDACTED:aws_access_key]" in redacted
    assert len(findings) == 1
    # Findings must never carry the raw value.
    assert "AKIAIOSFODNN7EXAMPLE" not in str(findings)
    assert findings[0]["kind"] == "aws_access_key"


def test_redact_secrets_clean_prompt_unchanged(tmp_path: Path):
    redacted, findings = redact_secrets("just a normal prompt", cwd=tmp_path)
    assert redacted == "just a normal prompt"
    assert findings == []


def test_redact_secrets_masks_multiple_distinct_secrets(tmp_path: Path):
    prompt = (
        "aws AKIAIOSFODNN7EXAMPLE and gh ghp_"
        + "a" * 36
        + " end"
    )
    redacted, findings = redact_secrets(prompt, cwd=tmp_path)
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "ghp_" + "a" * 36 not in redacted
    assert "[REDACTED:aws_access_key]" in redacted
    assert "[REDACTED:github_token]" in redacted
    assert {f["kind"] for f in findings} == {"aws_access_key", "github_token"}


def test_policy_redact_returns_redacted_prompt(tmp_path: Path):
    payload = apply_secret_scan_policy(
        "key AKIAIOSFODNN7EXAMPLE here", policy="redact", cwd=tmp_path
    )
    assert payload["policy"] == "redact"
    assert payload["detected_count"] == 1
    assert "AKIAIOSFODNN7EXAMPLE" not in payload["redacted_prompt"]
    assert "[REDACTED:aws_access_key]" in payload["redacted_prompt"]


def test_policy_redact_clean_prompt_has_no_changes(tmp_path: Path):
    payload = apply_secret_scan_policy(
        "nothing secret here", policy="redact", cwd=tmp_path
    )
    assert payload["detected_count"] == 0
    assert payload["redacted_prompt"] == "nothing secret here"
