"""Shared pytest fixtures.

CI environment compatibility: GitHub runners have no native council CLIs on
PATH. The default `quick` / triad participant selection requires at least one
Claude- or Codex-family CLI plus a Gemini-family CLI and raises otherwise — so
any test that drives a default-mode run/estimate (image-path validation,
context-file errors, budget checks, …) would fail during participant selection
before reaching its real assertion. Developer machines with those CLIs
installed can otherwise mask that fixture gap.

The autouse fixture below makes `shutil.which` report the built-in native CLIs
present only when the real environment lacks them, so triad selection resolves
deterministically everywhere. It is deliberately narrow:
  * every lookup outside the built-in native roster passes straight through;
  * when a native CLI is installed it is a no-op (the real path is returned);
  * tests that deliberately exercise CLI presence/absence (the triad-resolution
    and doctor tests) set their own `shutil.which` monkeypatch in-body, which
    overrides this fixture for the duration of that test.
"""

from __future__ import annotations

import shutil

import pytest


@pytest.fixture(autouse=True)
def _native_council_resolvable(monkeypatch):
    real_which = shutil.which

    def which(cmd, *args, **kwargs):
        found = real_which(cmd, *args, **kwargs)
        if found is None and cmd in ("claude", "codex", "gemini", "agy"):
            return f"/usr/bin/{cmd}"
        return found

    monkeypatch.setattr(shutil, "which", which)
