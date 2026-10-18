"""The ``affected`` mapping of scripts/test_targets.py selects tests by what changed."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/test_targets.py"
_SPEC = importlib.util.spec_from_file_location("skyulf_test_targets", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
targets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(targets)

PLATFORMS = "skyulf-core/tests/integration/platforms"
HELPER = f"{PLATFORMS}/test_databricks_bundle_generation.py"
TESTS = {
    "skyulf-core/tests/unit/test_spark_a.py": "from skyulf.inference.spark import run\n",
    "skyulf-core/tests/unit/test_spark_b.py": "from skyulf.inference import bundle, spark\n",
    "skyulf-core/tests/unit/test_spark_x.py": "import skyulf.inference.spark_extra\n",
    HELPER: "def _generate_project(tmp_path): ...\n",
    f"{PLATFORMS}/test_uses_helper.py": "    from test_databricks_bundle_generation import CLI\n",
    f"{PLATFORMS}/test_preview.py": 'run(project / "src/tools/preview.py")\n',
    "tests/unit/test_docs.py": 'read("docs/user_guide/databricks_spark_scoring.md")\n',
}


def test_library_module_selects_its_importers_only():
    """Both import spellings select a test; a longer sibling module name does not."""
    assert targets.map_to_tests("skyulf-core/skyulf/inference/spark.py", TESTS) == [
        "skyulf-core/tests/unit/test_spark_a.py",
        "skyulf-core/tests/unit/test_spark_b.py",
    ]


def test_shared_test_module_runs_itself_and_its_importers():
    """Changing the render helper module reruns the test files that import it."""
    assert targets.map_to_tests(HELPER, TESTS) == [HELPER, f"{PLATFORMS}/test_uses_helper.py"]


def test_template_file_selects_tests_naming_its_rendered_path_and_snapshots():
    """A template edit runs the tests reading the rendered file plus the render snapshots."""
    template = (
        "skyulf-core/templates/databricks/template/{{.project_name}}/src/tools/preview.py.tmpl"
    )
    assert targets.map_to_tests(template, TESTS) == sorted(
        [f"{PLATFORMS}/test_preview.py", targets.SNAPSHOTS]
    )
    schema = targets.map_to_tests("skyulf-core/templates/databricks/schema/training.json", TESTS)
    assert set(targets.SCHEMA_TESTS) | {targets.SNAPSHOTS} <= set(schema)


def test_docs_map_by_name_and_conftest_or_unknown_files_are_reported():
    """Unmappable changes are listed for a manual choice, never widened to a directory."""
    changed = [
        "docs/user_guide/databricks_spark_scoring.md",
        f"{PLATFORMS}/conftest.py",
        "frontend/ml-canvas/src/App.tsx",
    ]
    assert targets.map_to_tests(changed[0], TESTS) == ["tests/unit/test_docs.py"]
    _, unmapped = targets.select_affected(changed, TESTS)
    assert unmapped == changed[1:]
