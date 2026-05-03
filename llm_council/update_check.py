"""Version and update checks."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO

import httpx


TAGS_URL = "https://api.github.com/repos/Intellimetrics/llm-council/tags?per_page=50"
INSTALL_COMMAND = "uv tool install --force git+https://github.com/Intellimetrics/llm-council.git"

NAG_CACHE_TTL_SECONDS = 24 * 60 * 60
NAG_OPT_OUT_ENV = "LLM_COUNCIL_NO_UPDATE_CHECK"
NAG_NETWORK_TIMEOUT_SECONDS = 2.0


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


def _default_nag_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "llm-council" / "update-check.json"


def maybe_print_update_nag(
    current_version: str,
    *,
    stream: IO[str] | None = None,
    cache_path: Path | None = None,
    now: float | None = None,
    checker: Callable[[str], UpdateStatus] | None = None,
) -> bool:
    """Print one stderr line if a newer version exists; cached for 24h.

    Returns True if a nag was printed (an update is available); False
    otherwise. Designed for hot paths: every error swallowed, opt-out via
    env var, network timeout kept tight, cache prevents per-run latency.
    The nag fires from CLI entry points only — never from `mcp-server`,
    which speaks structured stdio that a stray stderr nag would not break
    but is still etiquette to leave clean.
    """
    if os.environ.get(NAG_OPT_OUT_ENV, "").strip():
        return False
    out = stream if stream is not None else sys.stderr
    cache_file = cache_path if cache_path is not None else _default_nag_cache_path()
    timestamp = now if now is not None else time.time()
    cached = _read_nag_cache(cache_file)
    if cached and (timestamp - cached.get("checked_at", 0)) < NAG_CACHE_TTL_SECONDS:
        latest = cached.get("latest_version")
        if (
            isinstance(latest, str)
            and latest
            and _compare_versions(current_version, latest) < 0
        ):
            _print_nag(out, current_version, latest, cached.get("install_command"))
            return True
        return False
    do_check = checker if checker is not None else (
        lambda version: check_for_update(version, timeout=NAG_NETWORK_TIMEOUT_SECONDS)
    )
    try:
        status = do_check(current_version)
    except Exception:
        return False
    _write_nag_cache(
        cache_file,
        {
            "checked_at": timestamp,
            "current_version": current_version,
            "latest_version": status.latest_version,
            "install_command": status.install_command,
        },
    )
    if status.update_available and status.latest_version:
        _print_nag(
            out, current_version, status.latest_version, status.install_command
        )
        return True
    return False


def hydrate_nag_cache_from_status(
    status: UpdateStatus,
    *,
    cache_path: Path | None = None,
    now: float | None = None,
) -> None:
    """Write the nag cache from an explicit UpdateStatus.

    Lets `llm-council check-update` count as the daily check — without this,
    a user who runs check-update sees "no update" but the next `run` still
    fires the nag because the cache wasn't refreshed.
    """
    if status.error or not status.latest_version:
        # Failed checks shouldn't stamp a "fresh" cache; leave the file alone
        # so the next nag attempt re-checks.
        return
    cache_file = cache_path if cache_path is not None else _default_nag_cache_path()
    timestamp = now if now is not None else time.time()
    _write_nag_cache(
        cache_file,
        {
            "checked_at": timestamp,
            "current_version": status.current_version,
            "latest_version": status.latest_version,
            "install_command": status.install_command,
        },
    )


def _print_nag(
    stream: IO[str], current: str, latest: str, install_command: str | None
) -> None:
    install = install_command or INSTALL_COMMAND
    try:
        print(
            f"note: llm-council {latest} is available (you have {current}); "
            f"upgrade with `{install}`",
            file=stream,
            flush=True,
        )
    except Exception:
        # Never let a broken stderr take down the run.
        pass


def _read_nag_cache(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_nag_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        # A read-only home or full disk shouldn't block the run; the
        # next invocation will just re-check (and probably also fail).
        pass
