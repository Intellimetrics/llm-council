"""Opt-in live canary for the okf-rs machine interface we depend on.

The OKF enrichment feature parses okf-rs's generated concept-file
frontmatter directly (`resource: path#Lstart-Lend`, `relationships`) and
relies on `generate -o <dir> --no-cache` writing nothing into the source
project. Those are upstream contracts okf-rs could drift on in a release;
this canary catches that the same way tests/test_live_agy_readonly.py
guards the agy read-only flags.

Run with:
    LLM_COUNCIL_LIVE_OKF_TEST=1 pytest tests/test_live_okf_bundle.py -q
(okf-rs must be on PATH.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_LIVE = os.environ.get("LLM_COUNCIL_LIVE_OKF_TEST") == "1"
_HAS_OKF = shutil.which("okf-rs") is not None

pytestmark = pytest.mark.skipif(
    not (_LIVE and _HAS_OKF),
    reason=(
        "live okf-rs canary: set LLM_COUNCIL_LIVE_OKF_TEST=1 and put "
        "okf-rs on PATH"
    ),
)


def test_okf_rs_generate_flags_and_frontmatter_contract(tmp_path: Path):
    from llm_council.okf_context import (
        generate_ephemeral_bundle,
        load_concepts,
        match_concepts,
    )

    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (project / "app.py").write_text(
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def entry():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    before = sorted(p.name for p in project.iterdir())

    import tempfile

    with tempfile.TemporaryDirectory(prefix="llm-council-okf-live-") as tmp:
        bundle_dir, status, detail = generate_ephemeral_bundle(
            project, Path(tmp), binary="okf-rs", timeout=60
        )
        assert status is None, f"generate failed: {status} ({detail})"
        assert bundle_dir is not None

        # Bundle root contract: index.md with parseable frontmatter
        # carrying source_revision.
        index_text = (bundle_dir / "index.md").read_text(encoding="utf-8")
        assert index_text.startswith("---")
        assert "source_revision" in index_text or "okf_version" in index_text

        # Concept frontmatter contract: our parser must extract at least
        # the two functions with usable resource line ranges, and the
        # call edge between them.
        concepts = load_concepts(bundle_dir)
        by_title = {cid.rsplit("/", 1)[-1]: c for cid, c in concepts.items()}
        assert "helper" in by_title, f"concept ids: {sorted(concepts)[:10]}"
        assert "entry" in by_title
        assert by_title["helper"].path == "app.py"
        assert by_title["helper"].start >= 1
        assert by_title["entry"].calls, "entry->helper call edge missing"

        # Range matching works against the real resource encoding.
        matched = match_concepts(
            concepts, {"app.py": [(by_title["entry"].start, by_title["entry"].end)]}
        )
        assert any(c.concept_id == by_title["entry"].concept_id for c in matched)

    # Read-only contract: nothing appeared in the source project — no
    # .okf-cache.json, no knowledge/ (".git" was there before generate).
    after = sorted(p.name for p in project.iterdir())
    assert after == before
