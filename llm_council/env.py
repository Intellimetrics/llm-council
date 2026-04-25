"""Environment loading helpers."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


ENV_FILE_NAMES = (".llm-council.env", ".env.local", ".env")


def load_project_env(start: Path | str | None = None) -> list[Path]:
    """Load local env files from start and its parents without overriding env."""

    current = Path(start or ".").resolve()
    if current.is_file():
        current = current.parent

    loaded: list[Path] = []
    for directory in (current, *current.parents):
        for name in ENV_FILE_NAMES:
            path = directory / name
            if path.exists():
                load_dotenv(path, override=False)
                loaded.append(path)
    return loaded
