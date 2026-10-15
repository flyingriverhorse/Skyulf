"""Rendered-output snapshots of the template defaults and the published example answer files.

Each snapshot lists every rendered file with a digest of its LF-normalized content,
so a template change that alters one of these common projects fails here until the
snapshot is refreshed on purpose. Only refreshing and comparing need the CLI:

    SKYULF_BUNDLE_OFFLINE_CLI=1 pytest \
        skyulf-core/tests/integration/platforms/test_databricks_render_snapshots.py \
        --snapshot-update

Review the resulting `.ambr` diff: it names every rendered file whose content changed.
"""

import hashlib
import json
from pathlib import Path

import pytest
from test_databricks_bundle_generation import (
    CLI,
    OFFLINE_CLI,
    PROFILE,
    TEMPLATE_ROOT,
    _generate_project,
)

pytestmark = pytest.mark.skipif(
    not CLI or not (PROFILE or OFFLINE_CLI), reason="Requires opt-in installed CLI."
)

EXAMPLES = sorted(path.name for path in (TEMPLATE_ROOT / "examples").glob("*.example.json"))


def _digests(project: Path) -> dict[str, str]:
    """Map each rendered file to a short digest that ignores checkout line endings."""
    return {
        path.relative_to(project).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()[:16]
        for path in sorted(project.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("answers", ["defaults", *EXAMPLES])
def test_rendered_project_matches_snapshot(tmp_path, answers, snapshot):
    """A template change that alters a common project must arrive with a refreshed snapshot."""
    overrides = {}
    if answers != "defaults":
        overrides = json.loads((TEMPLATE_ROOT / "examples" / answers).read_text(encoding="utf-8"))
        overrides.pop("project_name", None)
    assert _digests(_generate_project(tmp_path, **overrides)) == snapshot
