"""Shared pytest fixtures.

CI environment compatibility: GitHub runners have no Gemini-family CLI
(`agy` / `gemini`) on PATH. Since the 0.12-era "dynamic triad resolution"
change, the default `quick` / triad participant selection REQUIRES at least one
Gemini-family CLI and raises otherwise — so any test that drives a default-mode
run/estimate (image-path validation, context-file errors, budget checks, …)
blew up on participant selection long before reaching its real assertion. That
left `main` CI red for ~8 days even though the suite was green on dev machines
(which have `agy`/`gemini` installed).

The autouse fixture below makes `shutil.which` report a Gemini-family CLI
present ONLY when the real environment lacks it, so triad selection resolves
deterministically everywhere. It is deliberately minimal:
  * every non-`agy`/`gemini` lookup passes straight through;
  * when `agy`/`gemini` ARE installed it is a no-op (real path returned);
  * tests that deliberately exercise CLI presence/absence (the triad-resolution
    and doctor tests) set their own `shutil.which` monkeypatch in-body, which
    overrides this fixture for the duration of that test.
"""

from __future__ import annotations

import shutil

import pytest


@pytest.fixture(autouse=True)
def _gemini_family_resolvable(monkeypatch):
    real_which = shutil.which

    def which(cmd, *args, **kwargs):
        found = real_which(cmd, *args, **kwargs)
        if found is None and cmd in ("gemini", "agy"):
            return f"/usr/bin/{cmd}"
        return found

    monkeypatch.setattr(shutil, "which", which)
