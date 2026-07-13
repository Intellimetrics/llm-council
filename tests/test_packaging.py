"""Release-artifact regressions that editable installs cannot catch."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_bundled_eval_fixtures(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(out_dir.glob("llm_council-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    fixture_root = "llm_council/eval/fixtures/seed_missing_tenant_filter"
    assert f"{fixture_root}/prompt.md" in names
    assert f"{fixture_root}/expected_blockers.json" in names
