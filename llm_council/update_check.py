"""Version and update checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx


PYPROJECT_URL = (
    "https://raw.githubusercontent.com/Intellimetrics/llm-council/main/pyproject.toml"
)
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
    """Compare the installed version to the public main-branch package version."""
    try:
        response = httpx.get(PYPROJECT_URL, timeout=timeout)
        response.raise_for_status()
        latest_version = _extract_pyproject_version(response.text)
        return UpdateStatus(
            current_version=current_version,
            latest_version=latest_version,
            update_available=_compare_versions(current_version, latest_version) < 0,
            source=PYPROJECT_URL,
            install_command=INSTALL_COMMAND,
        )
    except Exception as exc:
        return UpdateStatus(
            current_version=current_version,
            latest_version=None,
            update_available=None,
            source=PYPROJECT_URL,
            install_command=INSTALL_COMMAND,
            error=f"{type(exc).__name__}: {exc}",
        )


def _extract_pyproject_version(text: str) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if not match:
        raise ValueError("Could not find project version in pyproject.toml")
    return match.group(1)


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
