"""Privacy-boundary helpers shared by CLI and MCP entry points."""

from __future__ import annotations

from typing import Any, Mapping

from llm_council.config import is_private_local_participant


PRIVATE_MODE_ALIASES = frozenset({"private-local", "local-only", "local-private"})


def transcript_was_private_local(transcript: Mapping[str, Any]) -> bool:
    """Return whether a prior transcript declares the private-local boundary."""

    mode = str(transcript.get("mode") or "").strip().lower()
    metadata = transcript.get("metadata")
    privacy_scope = (
        str(metadata.get("privacy_scope") or "").strip().lower()
        if isinstance(metadata, Mapping)
        else ""
    )
    return mode in PRIVATE_MODE_ALIASES or privacy_scope == "private-local"


def participants_are_private_local(
    participants: list[str], participant_cfg: Mapping[str, Any]
) -> bool:
    """Return true only when every selected peer is a loopback Ollama peer."""

    if not participants:
        return False
    for name in participants:
        cfg = participant_cfg.get(name)
        if not isinstance(cfg, dict) or not is_private_local_participant(cfg):
            return False
    return True


def privacy_downgrade_error(
    prior_transcript: Mapping[str, Any],
    *,
    participants: list[str],
    participant_cfg: Mapping[str, Any],
    allow_privacy_downgrade: bool,
) -> str | None:
    """Describe an unsafe private-to-hosted continuation, or return ``None``."""

    if allow_privacy_downgrade or not transcript_was_private_local(prior_transcript):
        return None
    if participants_are_private_local(participants, participant_cfg):
        return None
    return (
        "PrivacyDowngradeRefused: the prior run used private-local routing, "
        "but this continuation selects participants that are not same-machine "
        "loopback Ollama peers. Keep mode private-local, or explicitly pass "
        "--allow-privacy-downgrade (CLI) / allow_privacy_downgrade=true (MCP) "
        "after confirming the prior question and summaries may be sent to "
        "hosted participants."
    )


__all__ = [
    "PRIVATE_MODE_ALIASES",
    "participants_are_private_local",
    "privacy_downgrade_error",
    "transcript_was_private_local",
]
