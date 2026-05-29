"""Environment loading helpers."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


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


def load_project_env(start: Path | str | None = None) -> list[Path]:
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

    current = Path(start or ".").resolve()
    if current.is_file():
        current = current.parent

    directories = [current, *current.parents]
    loaded: list[Path] = []

    # Non-overriding (.env / .env.local): conventional "first load wins", so
    # iterate nearest -> farthest to let the nearest file set the value first.
    for directory in directories:
        for name in NON_OVERRIDING_ENV_FILES:
            path = directory / name
            if path.exists():
                load_dotenv(path, override=False)
                loaded.append(path)

    # Overriding (.llm-council.env): "last load wins", so iterate farthest ->
    # nearest and let the nearest file be applied last. Loaded after the
    # non-overriding class so a project file stays authoritative over `.env`.
    for directory in reversed(directories):
        for name in OVERRIDING_ENV_FILES:
            path = directory / name
            if path.exists():
                load_dotenv(path, override=True)
                loaded.append(path)

    return loaded
