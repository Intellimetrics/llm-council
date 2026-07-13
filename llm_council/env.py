"""Environment loading helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Mapping

from dotenv import dotenv_values, load_dotenv
from dotenv.variables import parse_variables


# `.llm-council.env` is project-specific to llm-council and is treated as
# authoritative — values there override anything inherited from the parent
# shell or MCP-host process. This avoids a class of bug where a stale
# OPENROUTER_API_KEY (or similar) sitting in the MCP host's environment
# silently shadows the correct value the user just put in the project file.
# `.env` and `.env.local` follow the conventional dotenv "shell wins"
# semantic so we don't surprise users sharing a `.env` with other tools.
OVERRIDING_ENV_FILES = (".llm-council.env",)
NON_OVERRIDING_ENV_FILES = (".env.local", ".env")
ENV_FILE_NAMES = OVERRIDING_ENV_FILES + NON_OVERRIDING_ENV_FILES


# MCP is a long-running, concurrent process.  Mutating ``os.environ`` for one
# project request leaks those values into later requests and lets concurrent
# runs race over credentials.  A ContextVar gives each async request (and all
# tasks it spawns) its own effective environment while preserving the existing
# process-global ``load_project_env`` behavior for the one-shot CLI.
_PROJECT_ENV: ContextVar[Mapping[str, str] | None] = ContextVar(
    "llm_council_project_env", default=None
)
_PROJECT_ENV_FILES: ContextVar[tuple[Path, ...]] = ContextVar(
    "llm_council_project_env_files", default=()
)


def project_directories(
    start: Path | str | None = None,
    *,
    stop_at: Path | str | None = None,
) -> tuple[Path, ...]:
    """Return nearest-first project directories, optionally within a boundary.

    ``stop_at`` is an inclusive trust boundary: its directory is returned, but
    none of its parents are. Both paths are resolved before comparison so a
    symlink cannot make an apparently nested start escape the trusted tree.
    Omitting ``stop_at`` preserves the historical walk to the filesystem root.
    """

    current = Path(start or ".").expanduser().resolve()
    if current.is_file():
        current = current.parent
    if stop_at is None:
        return (current, *current.parents)

    boundary = Path(stop_at).expanduser().resolve()
    if boundary.is_file():
        boundary = boundary.parent
    try:
        current.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(
            f"Project path is outside stop_at boundary: {current} is not under "
            f"{boundary}"
        ) from exc

    directories: list[Path] = []
    directory = current
    while True:
        directories.append(directory)
        if directory == boundary:
            break
        parent = directory.parent
        if parent == directory:  # Defensive; relative_to above makes this unreachable.
            break
        directory = parent
    return tuple(directories)


def _dotenv_path_allowed(path: Path, boundary: Path | None) -> bool:
    """True when a dotenv path does not escape an active trust boundary."""

    if boundary is None:
        return True
    try:
        path.resolve().relative_to(boundary)
    except (OSError, ValueError):
        return False
    return True


def _dotenv_mapping(
    path: Path,
    current: Mapping[str, str],
    *,
    override: bool,
) -> dict[str, str]:
    """Parse one dotenv file using ``load_dotenv``-compatible interpolation.

    ``python-dotenv`` normally resolves interpolation against ``os.environ``.
    Request-local MCP loading cannot mutate that mapping, so resolve the same
    variable atoms against the accumulated request environment instead.
    """

    raw_values = dotenv_values(path, interpolate=False)
    parsed: dict[str, str] = {}
    for key, raw_value in raw_values.items():
        if raw_value is None:
            continue
        interpolation_env: dict[str, str] = {}
        if override:
            interpolation_env.update(current)
            interpolation_env.update(parsed)
        else:
            interpolation_env.update(parsed)
            interpolation_env.update(current)
        parsed[key] = "".join(
            atom.resolve(interpolation_env) for atom in parse_variables(raw_value)
        )
    return parsed


def resolve_project_env(
    start: Path | str | None = None,
    *,
    base_env: Mapping[str, str] | None = None,
    stop_at: Path | str | None = None,
) -> tuple[dict[str, str], list[Path]]:
    """Return an isolated effective environment and the dotenv files loaded.

    Precedence exactly mirrors :func:`load_project_env`, but no process-global
    state is changed.  This is the loader MCP request scopes should use.
    """

    directories = project_directories(start, stop_at=stop_at)
    boundary = Path(stop_at).expanduser().resolve() if stop_at is not None else None
    if boundary is not None and boundary.is_file():
        boundary = boundary.parent
    effective = dict(os.environ if base_env is None else base_env)
    loaded: list[Path] = []

    # First value wins for conventional dotenv files: inherited shell values,
    # then the nearest project file, then increasingly distant ancestors.
    for directory in directories:
        for name in NON_OVERRIDING_ENV_FILES:
            path = directory / name
            if not path.exists() or not _dotenv_path_allowed(path, boundary):
                continue
            values = _dotenv_mapping(path, effective, override=False)
            for key, value in values.items():
                effective.setdefault(key, value)
            loaded.append(path)

    # Authoritative council env files override inherited/conventional values;
    # load farthest first so the nearest project wins last.
    for directory in reversed(directories):
        for name in OVERRIDING_ENV_FILES:
            path = directory / name
            if not path.exists() or not _dotenv_path_allowed(path, boundary):
                continue
            effective.update(_dotenv_mapping(path, effective, override=True))
            loaded.append(path)

    return effective, loaded


@contextmanager
def project_env_context(
    start: Path | str | None = None,
    *,
    stop_at: Path | str | None = None,
) -> Iterator[list[Path]]:
    """Install a request-local project environment for the current context."""

    effective, loaded = resolve_project_env(start, stop_at=stop_at)
    env_token = _PROJECT_ENV.set(effective)
    files_token = _PROJECT_ENV_FILES.set(tuple(loaded))
    try:
        yield loaded
    finally:
        _PROJECT_ENV_FILES.reset(files_token)
        _PROJECT_ENV.reset(env_token)


def env_get(name: str, default: str | None = None) -> str | None:
    """Read from the request-local environment, falling back to the process."""

    effective = _PROJECT_ENV.get()
    if effective is None:
        return os.environ.get(name, default)
    return effective.get(name, default)


def env_items() -> tuple[tuple[str, str], ...]:
    """Return a stable snapshot of the effective environment's items."""

    effective = _PROJECT_ENV.get()
    source: Mapping[str, str] = os.environ if effective is None else effective
    return tuple(source.items())


def load_project_env(
    start: Path | str | None = None,
    *,
    stop_at: Path | str | None = None,
) -> list[Path]:
    """Load local env files from start and its parents.

    `.llm-council.env` overrides existing env vars (project-authoritative);
    `.env` and `.env.local` are loaded only when the var is not already set
    (parent-shell-authoritative, the conventional dotenv semantic).

    Both file classes resolve **nearest-wins**: a value in a `start`-adjacent
    file beats the same key in an ancestor directory, mirroring `find_config`'s
    nearest-config rule. Achieving that requires opposite iteration orders for
    the two classes because `load_dotenv` has opposite tie-breaks per
    ``override``: with ``override=False`` (``.env`/`.env.local``) the *first*
    load wins, so we visit nearest-first; with ``override=True``
    (``.llm-council.env``) the *last* load wins, so we visit farthest-first.
    Without this split a stale ancestor `.llm-council.env` would silently
    shadow the value the user just put in a subproject's file — the exact
    failure this module exists to prevent.
    """

    # An MCP request scope already resolved the project files without touching
    # process state.  Nested legacy calls from existing handlers must be a
    # no-op, otherwise they would defeat the isolation guarantee.
    if _PROJECT_ENV.get() is not None:
        return list(_PROJECT_ENV_FILES.get())

    directories = project_directories(start, stop_at=stop_at)
    boundary = Path(stop_at).expanduser().resolve() if stop_at is not None else None
    if boundary is not None and boundary.is_file():
        boundary = boundary.parent
    loaded: list[Path] = []

    # Non-overriding (.env / .env.local): conventional "first load wins", so
    # iterate nearest -> farthest to let the nearest file set the value first.
    for directory in directories:
        for name in NON_OVERRIDING_ENV_FILES:
            path = directory / name
            if path.exists() and _dotenv_path_allowed(path, boundary):
                load_dotenv(path, override=False)
                loaded.append(path)

    # Overriding (.llm-council.env): "last load wins", so iterate farthest ->
    # nearest and let the nearest file be applied last. Loaded after the
    # non-overriding class so a project file stays authoritative over `.env`.
    for directory in reversed(directories):
        for name in OVERRIDING_ENV_FILES:
            path = directory / name
            if path.exists() and _dotenv_path_allowed(path, boundary):
                load_dotenv(path, override=True)
                loaded.append(path)

    return loaded
