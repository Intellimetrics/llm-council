"""Version and update checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx


TAGS_URL = "https://api.github.com/repos/Intellimetrics/llm-council/tags?per_page=50"
INSTALL_COMMAND = "uv tool install --force git+https://github.com/Intellimetrics/llm-council.git"


@dataclass
class UpdateStatus:
    current_version: str
    latest_version: str | None
    update_available: bool | None
    source: str
    install_command: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "update_available": self.update_available,
            "source": self.source,
            "install_command": self.install_command,
            "error": self.error,
        }


def check_for_update(current_version: str, *, timeout: float = 10) -> UpdateStatus:
    """Compare the installed version to the latest public release tag."""
    try:
        response = httpx.get(
            TAGS_URL,
            headers={"Accept": "application/vnd.github+json"},
            timeout=timeout,
        )
        response.raise_for_status()
        latest_version, latest_tag = _latest_tag_version(response.json())
        return UpdateStatus(
            current_version=current_version,
            latest_version=latest_version,
            update_available=_compare_versions(current_version, latest_version) < 0,
            source=TAGS_URL,
            install_command=_install_command(latest_tag),
        )
    except Exception as exc:
        return UpdateStatus(
            current_version=current_version,
            latest_version=None,
            update_available=None,
            source=TAGS_URL,
            install_command=INSTALL_COMMAND,
            error=f"{type(exc).__name__}: {exc}",
        )


def _latest_tag_version(tags: Any) -> tuple[str, str]:
    if not isinstance(tags, list):
        raise ValueError("Expected GitHub tags response to be a list")
    best_version: str | None = None
    best_tag: str | None = None
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        name = tag.get("name")
        if not isinstance(name, str):
            continue
        version = _version_from_tag(name)
        if version is None:
            continue
        if best_version is None or _compare_versions(best_version, version) < 0:
            best_version = version
            best_tag = name
    if best_version is None or best_tag is None:
        raise ValueError("Could not find a semantic version tag")
    return best_version, best_tag


def _version_from_tag(tag: str) -> str | None:
    match = re.fullmatch(r"v?(\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?)", tag)
    return match.group(1) if match else None


def _install_command(tag: str | None) -> str:
    if not tag:
        return INSTALL_COMMAND
    return f"{INSTALL_COMMAND}@{tag}"


def _compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    max_len = max(len(left_parts), len(right_parts))
    left_parts.extend([0] * (max_len - len(left_parts)))
    right_parts.extend([0] * (max_len - len(right_parts)))
    if left_parts < right_parts:
        return -1
    if left_parts > right_parts:
        return 1
    return 0


def _version_parts(version: str) -> list[int]:
    base = version.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for part in base.split("."):
        match = re.match(r"(\d+)", part)
        parts.append(int(match.group(1)) if match else 0)
    return parts
