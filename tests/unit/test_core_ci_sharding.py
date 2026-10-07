"""Core CI partitions whole files without losing or repeating collected tests."""

import os
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
from coverage import CoverageData
from coverage.exceptions import CoverageException

RUNNER = Path(__file__).resolve().parents[2] / ".github/scripts/run_core_shard.py"
COMBINER = RUNNER.with_name("combine_core_coverage.py")


@pytest.fixture
def shard_plugin():
    """Load the actual CI entrypoint without running its command-line main function."""
    assert RUNNER.is_file(), "The Core CI shard runner must exist."
    return runpy.run_path(str(RUNNER))["CoreShard"]


def _select(plugin, root, paths):
    """Apply pytest's collection hook to items grouped into parametrized test files."""
    items = [
        SimpleNamespace(path=root / path, nodeid=f"{path}::test_case[{i}]")
        for path in paths
        for i in range(3)
    ]
    excluded = []
    config = SimpleNamespace(
        rootpath=root,
        hook=SimpleNamespace(pytest_deselected=lambda items: excluded.extend(items)),
    )
    plugin.pytest_collection_modifyitems(config, items)
    return [item.nodeid for item in items], [item.nodeid for item in excluded]


def test_shards_partition_every_item_and_keep_files_together(shard_plugin, tmp_path):
    """Parallel CI must retain all tests exactly once and preserve module fixtures."""
    paths = [f"skyulf-core/tests/unit/test_case_{i}.py" for i in range(40)]
    expected = {f"{path}::test_case[{i}]" for path in paths for i in range(3)}
    assigned = set()
    owners = {}
    for index in range(4):
        selected, excluded = _select(shard_plugin(index, 4), tmp_path, paths)
        assert selected
        assert not assigned.intersection(selected)
        assert set(selected) | set(excluded) == expected
        assert not set(selected).intersection(excluded)
        assigned.update(selected)
        for node in selected:
            owners.setdefault(node.split("::")[0], set()).add(index)
    assert assigned == expected
    assert all(len(indices) == 1 for indices in owners.values())


def test_shard_assignment_ignores_checkout_and_collection_order(shard_plugin, tmp_path):
    """Runner directories and unrelated test additions must not move existing files."""
    paths = [f"skyulf-core/tests/integration/test_case_{i}.py" for i in range(20)]
    for index in range(4):
        first, _ = _select(shard_plugin(index, 4), tmp_path / "first", paths)
        second, _ = _select(shard_plugin(index, 4), tmp_path / "second", paths[::-1])
        extended, _ = _select(shard_plugin(index, 4), tmp_path, [*paths, "test_new.py"])
        assert set(first) == set(second)
        assert set(first) == {node for node in extended if not node.startswith("test_new.py::")}
        assert first == [node for path in paths for node in first if node.startswith(path + "::")]


@pytest.mark.parametrize("index,count", [(-1, 4), (4, 4), (0, 0), (0, -1)])
def test_invalid_shard_selection_is_rejected(shard_plugin, index, count):
    """A mistyped matrix index must fail instead of silently dropping a test partition."""
    with pytest.raises(ValueError, match="shard"):
        shard_plugin(index, count)


def test_runner_hooks_real_pytest_collection(tmp_path):
    """The command used by CI must activate partitioning in a real pytest process."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    suite = tmp_path / "cases"
    suite.mkdir()
    expected = set()
    for index in range(8):
        path = suite / f"test_sample_{index}.py"
        path.write_text("def test_sample():\n    assert True\n", encoding="utf-8")
        expected.add(f"cases/{path.name}::test_sample")
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    found = []
    for index in range(4):
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--shard-index",
                str(index),
                "--shard-count",
                "4",
                "cases",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode in {0, 5}, result.stdout + result.stderr
        found.extend(line for line in result.stdout.splitlines() if line.startswith("cases/"))
    assert set(found) == expected
    assert len(found) == len(expected)


@pytest.fixture
def coverage_checkout(tmp_path, monkeypatch):
    """Use real coverage databases and a nested source root like the CI checkout."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".coveragerc").write_text("[run]\nrelative_files = true\n", encoding="utf-8")
    source = Path("skyulf-core/skyulf/demo.py")
    source.parent.mkdir(parents=True)
    source.write_text(
        "def choose(value):\n    if value:\n        return 1\n    return 0\n", encoding="utf-8"
    )
    for index in range(4):
        _write_branch_shard(index, source.as_posix(), taken=3 if index % 2 else 4)
    _write_branch_shard("spark", source.as_posix(), taken=3)
    _write_branch_shard("delta", source.as_posix(), taken=4)
    return source


def _write_branch_shard(index, source, taken):
    """Store complementary measured function arcs using coverage's public data API."""
    data = CoverageData(basename=f".coverage.shard-{index}")
    data.add_arcs({source: [(-1, 1), (1, -1), (-1, 2), (2, taken), (taken, -1)]})
    data.write()


def _coverage_functions():
    """Load the same strict aggregation entrypoint that the workflow executes."""
    assert COMBINER.is_file(), "The strict Core coverage combiner must exist."
    return runpy.run_path(str(COMBINER))


def test_coverage_combines_complementary_arcs_and_preserves_xml_paths(coverage_checkout):
    """Both branch outcomes must combine and Sonar must add the source prefix only once."""
    functions = _coverage_functions()
    assert functions["combine_core_coverage"]() == 100.0
    combined = CoverageData(basename=".coverage")
    combined.read()
    assert combined.has_arcs()
    assert combined.measured_files() == {coverage_checkout.as_posix()}
    arcs = combined.arcs(coverage_checkout.as_posix())
    assert arcs is not None
    assert {(2, 3), (2, 4)} <= set(arcs)
    classes = ET.parse("coverage-core.xml").findall(".//class")
    assert [item.attrib["filename"] for item in classes] == ["demo.py"]
    assert f"skyulf-core/skyulf/{classes[0].attrib['filename']}" == coverage_checkout.as_posix()


@pytest.mark.parametrize("native", ["spark", "delta"])
def test_coverage_includes_native_runtime_branch_contributions(coverage_checkout, native):
    """Each optional runtime lane must contribute branches absent from the base shards."""
    for index in [0, 1, 2, 3, "spark", "delta"]:
        Path(f".coverage.shard-{index}").unlink()
        _write_branch_shard(index, coverage_checkout.as_posix(), taken=4 if index == native else 3)
    functions = _coverage_functions()
    assert functions["combine_core_coverage"]() == 100.0
    assert ET.parse("coverage-core.xml").getroot().attrib["branch-rate"] == "1"


@pytest.mark.parametrize("index", [3, "spark", "delta"])
@pytest.mark.parametrize("invalid", ["missing", "corrupt", "empty", "statement_only"])
def test_coverage_rejects_incomplete_or_invalid_shard_data(coverage_checkout, index, invalid):
    """An unusable shard must fail before a partial aggregate can satisfy the gate."""
    shard = Path(f".coverage.shard-{index}")
    shard.unlink()
    if invalid == "corrupt":
        shard.write_bytes(b"not a coverage database")
    elif invalid in {"empty", "statement_only"}:
        data = CoverageData(basename=str(shard))
        if invalid == "empty":
            data.add_arcs({})
        else:
            data.add_lines({coverage_checkout.as_posix(): [1, 2, 3, 4]})
        data.write()
    functions = _coverage_functions()
    with pytest.raises((ValueError, CoverageException)):
        functions["combine_core_coverage"]()
    assert not Path("coverage-core.xml").exists()
    assert not Path(".coverage").exists()


def test_coverage_rejects_unexpected_shard_data(coverage_checkout):
    """A stray coverage input must not silently change the combined measurements."""
    _write_branch_shard(4, coverage_checkout.as_posix(), taken=3)
    functions = _coverage_functions()
    with pytest.raises(ValueError):
        functions["combine_core_coverage"]()
    assert not Path("coverage-core.xml").exists()
    assert not Path(".coverage").exists()


def test_coverage_cli_keeps_the_90_percent_branch_floor(coverage_checkout):
    """Valid base and native data cannot pass when every lane misses the same branch."""
    for index in [0, 1, 2, 3, "spark", "delta"]:
        shard = Path(f".coverage.shard-{index}")
        shard.unlink()
        _write_branch_shard(index, coverage_checkout.as_posix(), taken=3)
    result = subprocess.run(
        [sys.executable, str(COMBINER)], text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "90%" in result.stdout + result.stderr
    assert ET.parse("coverage-core.xml").getroot().attrib["branch-rate"] == "0.5"


def test_coverage_includes_unmeasured_core_sources(coverage_checkout):
    """Rebuilding the report must retain uncovered source files in the denominator."""
    source = coverage_checkout.with_name("unmeasured.py")
    source.write_text("untouched = 1\n", encoding="utf-8")
    # pytest-cov records source files that never ran as measured files with no arcs.
    shard = CoverageData(basename=".coverage.shard-0")
    shard.read()
    shard.touch_file(source.as_posix())
    shard.write()
    functions = _coverage_functions()
    assert functions["combine_core_coverage"]() < 90.0
    classes = ET.parse("coverage-core.xml").findall(".//class")
    assert {item.attrib["filename"] for item in classes} == {"demo.py", "unmeasured.py"}
