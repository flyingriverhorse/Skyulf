"""Pick the local tests to run: ``smoke`` (default), ``affected`` or ``full``.

    python scripts/test_targets.py                 # smoke
    python scripts/test_targets.py affected --list # show the mapping, run nothing
    python scripts/test_targets.py affected -- -x  # extra pytest arguments after --

smoke     Template schema ``build_schema.py --check``, the documentation contract
          tests, the node admission table and the rendered-template snapshots
          (a handful of renders; skipped when the Databricks CLI is missing).
affected  Changed files from ``git diff --name-only <base>`` plus untracked files,
          mapped to test files by text search (a heuristic, see ``map_to_tests``):
          - a changed test file runs itself, and any module in a tests directory
            also selects the test files importing it by name
            (``from test_databricks_bundle_generation import``);
          - a Python module under skyulf-core/skyulf/ or backend/ selects the test
            files that name its dotted path (``skyulf.inference.spark`` or
            ``from skyulf.inference import spark``);
          - a template file selects the tests naming its rendered path
            (``src/tools/preview.py``) plus the render snapshots; schema sources
            also select the schema build test;
          - any other file (docs, workflows) selects the tests naming its file name.
          A conftest.py maps to nothing. Nothing maps to a whole directory;
          unmapped files are listed so you can
          choose tests by hand. pytest-xdist (``-n auto``) is used when installed.
full      Prints the CI commands. Full suites and coverage run in GitHub CI; run
          them locally only when the user asks (AGENTS.md, local test scope).
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOTS = ("tests/", "skyulf-core/tests/")
PLATFORMS = "skyulf-core/tests/integration/platforms"
TEMPLATE = "skyulf-core/templates/databricks/"
RENDERED = "template/{{.project_name}}/"
SNAPSHOTS = f"{PLATFORMS}/test_databricks_render_snapshots.py"
SCHEMA_TESTS = (f"{PLATFORMS}/test_databricks_schema_build.py",)
SMOKE_TESTS = (
    *SCHEMA_TESTS,
    f"{PLATFORMS}/test_databricks_documentation_contracts.py",
    f"{PLATFORMS}/test_training_config_documentation.py",
    "skyulf-core/tests/unit/test_registry_admission.py",
    SNAPSHOTS,
)
FULL_COMMANDS = (
    "python skyulf-core/templates/databricks/build_schema.py --check",
    "ruff check . && ruff format --check backend skyulf-core tests run_skyulf.py celery_worker.py",
    "ty check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py",
    "pytest tests  # backend, once per SKYULF_ENGINE=polars|pandas",
    "python .github/scripts/run_core_shard.py --shard-index N --shard-count 4 skyulf-core/tests",
    "pytest skyulf-core/tests/spark -o addopts=  # with requirements-spark.txt",
    "python skyulf-core/templates/databricks/build_example_project.py --check",
    f"SKYULF_BUNDLE_OFFLINE_CLI=1 pytest {SNAPSHOTS}",
)


def _git(*args: str) -> list[str]:
    """Return the non-empty output lines of a git command run at the repository root."""
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, encoding="utf-8"
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def changed_files(base: str) -> list[str]:
    """Tracked changes against ``base`` (staged and unstaged) plus untracked files."""
    changed = _git("diff", "--name-only", base) + _git("ls-files", "--others", "--exclude-standard")
    return sorted(set(changed))


def _is_test_file(path: str) -> bool:
    """Recognize pytest files inside the two test trees."""
    return path.startswith(TEST_ROOTS) and Path(path).name.startswith("test_")


def read_test_files() -> dict[str, str]:
    """Read every tracked or untracked test file once, keyed by repository path."""
    paths = _git("ls-files", "--cached", "--others", "--exclude-standard", "*.py")
    return {
        path: (ROOT / path).read_text(encoding="utf-8", errors="replace")
        for path in sorted(set(paths))
        if _is_test_file(path) and (ROOT / path).is_file()
    }


def _module_pattern(path: str) -> re.Pattern[str] | None:
    """Match the dotted import of a library or backend module, or None for other files."""
    parts = Path(path).with_suffix("").parts
    if parts[:2] == ("skyulf-core", "skyulf"):
        parts = parts[1:]
    elif parts[0] != "backend":
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    package, name = ".".join(parts[:-1]), parts[-1]
    dotted = re.escape(".".join(parts))
    return re.compile(
        rf"\b{dotted}\b|\bfrom {re.escape(package)} import\b[^\n]*\b{re.escape(name)}\b"
    )


def _needle(path: str) -> tuple[str, list[str]]:
    """Return the text to search for a non-module file and any always-selected tests."""
    if path.startswith(TEMPLATE):
        inner = path.removeprefix(TEMPLATE)
        if inner.startswith(RENDERED):
            return inner.removeprefix(RENDERED).removesuffix(".tmpl"), [SNAPSHOTS]
        if inner.startswith(("schema/", "library/", "build_schema.py", "databricks_template")):
            return Path(inner).name, [*SCHEMA_TESTS, SNAPSHOTS]
        return Path(inner).name, [SNAPSHOTS]
    return Path(path).name, []


def map_to_tests(path: str, tests: dict[str, str]) -> list[str]:
    """Map one changed file to test files; an empty list means choose by hand.

    conftest.py maps to nothing: it affects a whole directory, which this tool never runs.
    """
    if Path(path).name == "conftest.py":
        return []
    extra: list[str] = []
    if path.endswith(".py") and path.startswith(TEST_ROOTS):
        # A test file runs itself; any test-tree module also runs the files importing it.
        extra = [path] if path in tests else []
        stem = re.escape(Path(path).stem)
        pattern = re.compile(rf"^\s*(from {stem} import|import {stem}\b)", re.MULTILINE)
    else:
        pattern = _module_pattern(path) if path.endswith(".py") else None
    if pattern is None:
        needle, extra = _needle(path)
        pattern = re.compile(re.escape(needle))
    return sorted({*extra, *(test for test, text in tests.items() if pattern.search(text))})


def select_affected(changed: list[str], tests: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return the deduplicated selected tests and the changed files that mapped to none."""
    selected: set[str] = set()
    unmapped = []
    for path in changed:
        found = map_to_tests(path, tests)
        selected.update(found)
        if not found:
            unmapped.append(path)
    return sorted(test for test in selected if (ROOT / test).is_file()), unmapped


def _run(command: list[str]) -> int:
    """Run one command from the repository root, echoing it first."""
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    # Templates render offline when the CLI is installed and no profile was chosen.
    if shutil.which("databricks") and not environment.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"):
        environment.setdefault("SKYULF_BUNDLE_OFFLINE_CLI", "1")
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


def _pytest(tests: list[str], extra: list[str]) -> int:
    """Run explicit test files, in parallel when pytest-xdist is available."""
    parallel = ["-n", "auto"] if len(tests) > 1 and importlib.util.find_spec("xdist") else []
    return _run([sys.executable, "-m", "pytest", *tests, *parallel, *extra])


def main(argv: list[str] | None = None) -> int:
    """Run the chosen target and return its exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", nargs="?", default="smoke", choices=("smoke", "affected", "full"))
    parser.add_argument("--base", default="HEAD", help="git ref the affected target diffs against")
    parser.add_argument("--list", action="store_true", help="print the selection, run nothing")
    parser.add_argument("pytest_args", nargs="*", help="extra pytest arguments after --")
    options = parser.parse_args(argv)
    if options.target == "full":
        print("Full suites run in GitHub CI (.github/workflows/pr_check.yml):")
        print("\n".join(f"  {command}" for command in FULL_COMMANDS))
        return 0
    if options.target == "smoke":
        tests = list(SMOKE_TESTS)
    else:
        tests, unmapped = select_affected(changed_files(options.base), read_test_files())
        for path in unmapped:
            print(f"no test mapped: {path}")
    print("\n".join(tests) if tests else "no tests selected")
    if options.list or not tests:
        return 0
    if options.target == "smoke":
        schema = _run([sys.executable, f"{TEMPLATE}build_schema.py", "--check"])
        if schema:
            return schema
    return _pytest(tests, options.pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
