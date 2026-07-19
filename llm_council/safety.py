"""Prompt-body credential scanner (Tier 2 of the post-council-review build).

`_redact_credentials_in_text` in orchestrator.py only redacts URL userinfo in
error-reporting text. The constructed prompt itself — including --diff content
and --context files — ships to hosted peers (OpenRouter, etc.) with zero
scanning. This module adds an opt-out (warn-by-default) preflight scan that
counts and surfaces likely credentials without leaking the values themselves
into transcripts or progress events.

Default policy is ``warn`` (count + emit a progress event, do not block) to
keep the false-positive blast radius low. Set ``defaults.secret_scan: block``
to raise on the first finding when the cost of an exfiltrated credential
outweighs the cost of a halted run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


VALID_POLICIES = ("off", "warn", "block", "redact")
DEFAULT_ALLOWLIST_FILENAME = ".llm-council-secrets-allow"

# Pattern catalogue. Each entry is (kind, regex, min_match_len). Keep the
# patterns tight to minimize false positives; users with niche key formats
# can extend via the allowlist or set policy=off for one run.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(r"\b(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}\b(?=[^A-Za-z0-9/+=]|$)")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b")),
    ("openai_key", re.compile(r"\bsk-(?!ant-|test|fake|example|dummy)[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
)


def _load_allowlist(cwd: Path, allowlist_filename: str) -> list[re.Pattern[str]]:
    """Load per-line regex patterns from an allowlist file. Lines starting
    with ``#`` and blank lines are ignored. Invalid regex lines are skipped
    silently rather than crashing the council run."""
    path = cwd / allowlist_filename
    if not path.exists():
        return []
    patterns: list[re.Pattern[str]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append(re.compile(line))
        except re.error:
            continue
    return patterns


def _is_allowlisted(value: str, allowlist: list[re.Pattern[str]]) -> bool:
    return any(pat.search(value) for pat in allowlist)


def _iter_secret_matches(text: str, allowlist: list[re.Pattern[str]]):
    """Yield ``(kind, match)`` for every non-allowlisted pattern hit in ``text``.

    Iterates ``_PATTERNS`` in order and ``pattern.finditer`` in order, applying
    the ``_is_allowlisted`` filter. Both :func:`scan_prompt_for_secrets` and
    :func:`redact_secrets` consume this so the match set, ordering, and
    allowlist filtering stay identical between them.
    """
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if _is_allowlisted(match.group(0), allowlist):
                continue
            yield kind, match


def _tally_kinds(findings: list[dict[str, Any]]) -> dict[str, int]:
    """Tally findings by ``kind`` into a ``{kind: count}`` dict."""
    kinds: dict[str, int] = {}
    for f in findings:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    return kinds


def scan_prompt_for_secrets(
    prompt: str,
    *,
    cwd: Path | None = None,
    allowlist_filename: str = DEFAULT_ALLOWLIST_FILENAME,
) -> list[dict[str, Any]]:
    """Scan ``prompt`` and return one finding per match.

    Findings DO NOT contain the matched secret value — just its kind, a
    SHA-style truncated fingerprint (first 4 + last 4 chars), and the
    1-based line number. Callers can surface counts/kinds without leaking
    the credential into transcripts or progress events.
    """
    if not prompt:
        return []
    allowlist = _load_allowlist(cwd or Path("."), allowlist_filename)
    findings: list[dict[str, Any]] = []
    for kind, match in _iter_secret_matches(prompt, allowlist):
        value = match.group(0)
        line_number = prompt.count("\n", 0, match.start()) + 1
        preview = value[:4] + "..." + value[-4:] if len(value) > 12 else "***"
        findings.append(
            {
                "kind": kind,
                "line": line_number,
                "preview": preview,
                # NEVER include the raw value; this dict is safe to log
                # and to embed in transcript metadata.
            }
        )
    # Stable ordering: by line, then by kind so transcript diffs are clean.
    findings.sort(key=lambda f: (f["line"], f["kind"]))
    return findings


def redact_secrets(
    prompt: str,
    *,
    cwd: Path | None = None,
    allowlist_filename: str = DEFAULT_ALLOWLIST_FILENAME,
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(redacted_prompt, findings)`` with every detected secret span
    replaced by ``[REDACTED:<kind>]``.

    Unlike :func:`scan_prompt_for_secrets`, this MUST locate the matched spans
    to splice them out — but it still never returns the raw value: the findings
    it returns carry only ``kind`` + ``line`` (preview is the literal
    ``"[redacted]"``). Overlapping matches across patterns are coalesced
    (earliest start wins, longest on ties) so a span is never double-replaced or
    partially mangled.
    """
    if not prompt:
        return prompt, []
    allowlist = _load_allowlist(cwd or Path("."), allowlist_filename)
    spans: list[tuple[int, int, str]] = []
    for kind, match in _iter_secret_matches(prompt, allowlist):
        spans.append((match.start(), match.end(), kind))
    if not spans:
        return prompt, []
    # Earliest start first; on equal starts the longest match wins so a shorter
    # overlapping match is skipped rather than truncating the longer one.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out: list[str] = []
    findings: list[dict[str, Any]] = []
    cursor = 0
    for start, end, kind in spans:
        if start < cursor:
            continue  # overlaps an already-redacted span
        out.append(prompt[cursor:start])
        out.append(f"[REDACTED:{kind}]")
        findings.append(
            {"kind": kind, "line": prompt.count("\n", 0, start) + 1, "preview": "[redacted]"}
        )
        cursor = end
    out.append(prompt[cursor:])
    findings.sort(key=lambda f: (f["line"], f["kind"]))
    return "".join(out), findings


def apply_secret_scan_policy(
    prompt: str,
    *,
    policy: str = "warn",
    cwd: Path | None = None,
    allowlist_filename: str = DEFAULT_ALLOWLIST_FILENAME,
) -> dict[str, Any]:
    """Run the scanner and apply the configured policy.

    Returns a dict with ``findings`` (list), ``detected_count`` (int),
    ``policy`` (the effective policy), and ``kinds`` (dict mapping kind→count).
    Raises ``ValueError`` with the leading prefix ``SecretsBlocked:`` when
    policy is ``block`` and findings are non-empty.

    NOTE: ``warn`` mode does NOT alter the prompt — it counts and logs matches
    but ships the original prompt to peers AND persists it verbatim to the
    on-disk transcript. The count is named ``detected_count`` (not "scrubbed")
    to avoid implying mitigation that does not happen; use ``block`` to refuse
    a run that contains likely credentials, or ``redact`` to mask them.

    ``redact`` returns an extra ``redacted_prompt`` key: each detected secret
    span is replaced with ``[REDACTED:<kind>]``. The caller is responsible for
    using ``redacted_prompt`` as the prompt actually sent to peers / persisted
    (so the secret reaches neither). This is the only policy that provides
    transcript-level protection.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(
            f"Invalid secret_scan policy '{policy}'. "
            f"Expected one of: {', '.join(VALID_POLICIES)}"
        )
    if policy == "off":
        return {"findings": [], "detected_count": 0, "policy": "off", "kinds": {}}
    if policy == "redact":
        redacted, findings = redact_secrets(
            prompt, cwd=cwd, allowlist_filename=allowlist_filename
        )
        return {
            "findings": findings,
            "detected_count": len(findings),
            "policy": "redact",
            "kinds": _tally_kinds(findings),
            "redacted_prompt": redacted,
        }
    findings = scan_prompt_for_secrets(
        prompt, cwd=cwd, allowlist_filename=allowlist_filename
    )
    kinds = _tally_kinds(findings)
    payload = {
        "findings": findings,
        "detected_count": len(findings),
        "policy": policy,
        "kinds": kinds,
    }
    if policy == "block" and findings:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        raise ValueError(
            "SecretsBlocked: prompt contains likely credentials "
            f"({summary}). Set defaults.secret_scan: warn to ship the "
            "prompt with a logged warning instead, or add the matching "
            f"patterns to ./{allowlist_filename} if these are test fixtures."
        )
    return payload


__all__ = [
    "DEFAULT_ALLOWLIST_FILENAME",
    "apply_secret_scan_policy",
    "redact_secrets",
    "scan_prompt_for_secrets",
]
