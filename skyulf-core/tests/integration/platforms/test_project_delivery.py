"""Project assets and dependency pins belong to the saved feature package."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pytest

from skyulf.inference.project_code import load_project_module, project_source_digest
from skyulf.integrations.databricks.projects._project_files import project_source


def test_competition_captured_source_retains_pins_without_executing_code():
    """Composed competition source must retain literal wheel pins during metadata inspection."""
    from skyulf.inference.project_dependencies import source_project_requirements

    common = (
        "raise RuntimeError('must not execute')\n"
        "install_project_package(__name__, {}, requirements=('numpy==1.2.3',))\n"
    )
    source = f"__skyulf_common_source__ = {common!r}\nexec(__skyulf_common_source__)\n"
    assert source_project_requirements(source) == ("numpy==1.2.3",)


def _package(tmp_path):
    """Create a small asset consumer whose relative helper imports need saved data."""
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text(
        "from skyulf.inference.project_package import read_project_asset\n"
        "VALUE = read_project_asset(__package__, 'lookup.json')\n",
        encoding="utf-8",
    )
    (root / "lookup.json").write_bytes(b'{"weight": 2}')
    (root / "assets.json").write_text('["lookup.json"]', encoding="utf-8")
    return root


def test_saved_assets_load_in_fresh_process_without_project_files(tmp_path):
    """Saved asset consumers must not depend on a mutable source checkout."""
    root = _package(tmp_path)
    version = importlib.metadata.version("numpy")
    (root / "requirements.txt").write_text(f"numpy=={version}\n", encoding="utf-8")
    source = project_source(root)
    saved = tmp_path / "saved.py"
    saved.write_text(source, encoding="utf-8")
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    code = (
        "import sys; from pathlib import Path\n"
        "from skyulf.inference.project_code import load_project_module\n"
        "module = load_project_module(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "print(module.VALUE.decode())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(saved)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"weight": 2}


def test_asset_changes_isolate_saved_package_identity(tmp_path):
    """An edited lookup table must not replace the data of an already loaded model."""
    root = _package(tmp_path)
    first_source = project_source(root)
    first = load_project_module(first_source)
    (root / "lookup.json").write_bytes(b'{"weight": 3}')
    second_source = project_source(root)
    second = load_project_module(second_source)
    assert project_source_digest(first_source) != project_source_digest(second_source)
    assert first.VALUE == b'{"weight": 2}'
    assert second.VALUE == b'{"weight": 3}'


def test_documented_asset_manifest_preserves_snapshot_and_ignores_help(tmp_path):
    """Inline instructions must not become assets or change a model's executable snapshot."""
    root = _package(tmp_path)
    original = project_source(root)
    manifest = {"_help": ["Declare paths in files."], "files": ["lookup.json"]}
    (root / "assets.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert project_source(root) == original
    assert load_project_module(project_source(root)).VALUE == b'{"weight": 2}'


@pytest.mark.parametrize(
    "manifest",
    [
        {"_help": ["Missing files must not silently become empty."]},
        {"files": [], "file": ["lookup.json"]},
        {"files": "lookup.json"},
        {"files": [], "_help": 42},
        {"files": ["../outside"]},
        {"files": ["lookup.json", "lookup.json"]},
    ],
)
def test_documented_assets_reject_ambiguous_declarations(tmp_path, manifest):
    """The documented format must retain strict path and declaration validation."""
    root = _package(tmp_path)
    (root / "assets.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="asset"):
        project_source(root)


@pytest.mark.parametrize(
    "asset",
    ["../outside", "/outside", "C:/outside", "a/../lookup.json", "a\\b", "missing", "__init__.py"],
)
def test_assets_reject_unsafe_or_missing_paths(tmp_path, asset):
    """Only explicitly declared regular data files inside the package are captured."""
    root = _package(tmp_path)
    (root / "assets.json").write_text(json.dumps([asset]), encoding="utf-8")
    with pytest.raises(ValueError, match="asset"):
        project_source(root)


@pytest.mark.parametrize("manifest", ["{}", '["lookup.json", "lookup.json"]', "[3]", "invalid"])
def test_assets_reject_invalid_declarations(tmp_path, manifest):
    """Malformed declarations cannot silently omit requested data from the artifact."""
    root = _package(tmp_path)
    (root / "assets.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="asset"):
        project_source(root)


def test_assets_reject_symlinks(tmp_path):
    """Even internal symlinks make the intended snapshot boundary ambiguous."""
    root = _package(tmp_path)
    try:
        (root / "link.json").symlink_to(root / "lookup.json")
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode or privilege.")
    (root / "assets.json").write_text('["link.json"]', encoding="utf-8")
    with pytest.raises(ValueError, match="symlink"):
        project_source(root)


def test_assets_reject_symlink_independent_of_host_privilege(tmp_path, monkeypatch):
    """The symlink rejection branch remains covered on Windows without link privilege."""
    root = _package(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path.name == "lookup.json" or original(path)
    )
    with pytest.raises(ValueError, match="symlink"):
        project_source(root)


@pytest.mark.parametrize("name", ["assets.json", "requirements.txt"])
def test_broken_metadata_symlinks_are_not_treated_as_absent(tmp_path, monkeypatch, name):
    """A broken declaration symlink must fail rather than silently remove saved inputs."""
    root = _package(tmp_path)
    (root / name).unlink(missing_ok=True)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == name or original(path))
    with pytest.raises(ValueError, match="symlink"):
        project_source(root)


def test_asset_reader_accepts_nested_package_names(tmp_path):
    """Helpers nested below the feature root resolve the same declared immutable bytes."""
    root = _package(tmp_path)
    (root / "nested").mkdir()
    (root / "nested/__init__.py").write_text(
        "from skyulf.inference.project_package import read_project_asset\n"
        "VALUE = read_project_asset(__package__, 'lookup.json')\n",
        encoding="utf-8",
    )
    (root / "__init__.py").write_text("from .nested import VALUE\n", encoding="utf-8")
    assert load_project_module(project_source(root)).VALUE == b'{"weight": 2}'


def test_assets_enforce_full_snapshot_bound(tmp_path):
    """Binary assets count toward the same bounded saved source contract."""
    root = _package(tmp_path)
    (root / "lookup.json").write_bytes(b"x" * (64 * 1024))
    with pytest.raises(ValueError, match="64 KiB"):
        project_source(root)


@pytest.mark.parametrize(
    "requirement",
    [
        "numpy",
        "numpy>=1",
        "numpy==1.*",
        "-r other.txt",
        "numpy==1; python_version>'3'",
        "numpy @ https://example.com/a.whl",
        "numpy[extra]==1",
        "numpy==1\nNumPy==2",
    ],
)
def test_dependencies_require_distinct_exact_distribution_pins(tmp_path, requirement):
    """Saved dependencies must identify reproducible versions without install directives."""
    root = _package(tmp_path)
    (root / "requirements.txt").write_text(requirement, encoding="utf-8")
    with pytest.raises(ValueError, match="requirement|dependency"):
        project_source(root)


@pytest.mark.parametrize("requirement", ["skyulf-missing-test-distribution==1.0", "numpy==0.0.0"])
def test_dependency_missing_or_changed_fails_before_project_exec(tmp_path, requirement):
    """Package loading must reject missing and mismatched dependencies before user code."""
    root = _package(tmp_path)
    (root / "__init__.py").write_text("raise AssertionError('executed')", encoding="utf-8")
    (root / "requirements.txt").write_text(requirement, encoding="utf-8")
    source = project_source(root)
    with pytest.raises(ValueError, match="dependency"):
        load_project_module(source)


def test_dependency_pins_exposed_and_part_of_identity(tmp_path):
    """Artifact and MLflow writers can retrieve the same canonical saved dependency pins."""
    from skyulf.inference.project_dependencies import source_project_requirements
    from skyulf.inference.project_package import get_project_requirements

    root = _package(tmp_path)
    before = project_source(root)
    version = importlib.metadata.version("numpy")
    (root / "requirements.txt").write_text(f"# pinned\nNumPy=={version}\n", encoding="utf-8")
    after = project_source(root)
    module = load_project_module(after)
    assert before != after
    assert source_project_requirements(after) == (f"numpy=={version}",)
    assert get_project_requirements(module.__name__) == (f"numpy=={version}",)
    assert source_project_requirements("VALUE = 1\n") == ()


def test_undeclared_asset_is_unavailable(tmp_path):
    """A file left in the original checkout cannot leak into a saved package reader."""
    from skyulf.inference.project_package import read_project_asset

    root = _package(tmp_path)
    (root / "private.txt").write_text("not declared", encoding="utf-8")
    module = load_project_module(project_source(root))
    with pytest.raises(ValueError, match="asset"):
        read_project_asset(module.__name__, "private.txt")
