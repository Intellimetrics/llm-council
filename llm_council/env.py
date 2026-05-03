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
    """

    current = Path(start or ".").resolve()
    if current.is_file():
        current = current.parent

    loaded: list[Path] = []
    for directory in (current, *current.parents):
        for name in ENV_FILE_NAMES:
            path = directory / name
            if path.exists():
                load_dotenv(path, override=name in OVERRIDING_ENV_FILES)
                loaded.append(path)
    return loaded
