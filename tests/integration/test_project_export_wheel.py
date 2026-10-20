"""Exported projects carry the exact Core source used by the application."""

import base64
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import PathDistribution, version
from pathlib import Path

import pytest

from backend.ml_pipeline.project_export.service import build_project
from tests.integration.test_project_export import export_job, fitted_bundle


def test_project_bundles_matching_core_wheel():
    """An unreleased Core version must install without relying on an index release."""
    content, _ = build_project(fitted_bundle(), export_job(), "batch")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        wheels = [name for name in archive.namelist() if name.endswith(".whl")]
        assert len(wheels) == 1
        assert wheels[0].startswith("wheels/skyulf_core-")
        requirements = archive.read("requirements.txt").decode()
        assert "--find-links ./wheels\n" in requirements
        assert f"./{wheels[0]}\n" in requirements
        assert f"skyulf-core=={version('skyulf-core')}\n" in requirements
        assert f"scikit-learn=={version('scikit-learn')}\n" in requirements
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["core_installation"]["bundled_wheel"] == wheels[0]


@pytest.fixture
def core_installation(tmp_path, monkeypatch):
    """Use real distribution metadata with small isolated development source files."""
    from backend.ml_pipeline.project_export import core_wheel

    package = tmp_path / "skyulf"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'original'\n")
    (package / "py.typed").write_text("")
    (package / "resources").mkdir()
    (package / "resources" / "settings.json").write_text('{"active": true}')
    metadata = tmp_path / "skyulf_core-0.9.5.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: skyulf-core\nVersion: 0.9.5\n"
        "License-File: LICENSE\nRequires-Dist: numpy>=1.24\n"
    )
    (metadata / "WHEEL").write_text(
        "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    (metadata / "top_level.txt").write_text("skyulf\n")
    (metadata / "entry_points.txt").write_text("[console_scripts]\nskyulf = skyulf:main\n")
    (metadata / "licenses").mkdir()
    (metadata / "licenses" / "LICENSE").write_text("TEST LICENSE")
    (metadata / "direct_url.json").write_text('{"url": "file:///PRIVATE_LOCAL_PATH"}')
    (metadata / "RECORD").write_text("skyulf_core-0.9.5.dist-info/METADATA,,\n")
    installed = PathDistribution(metadata)
    monkeypatch.setattr(core_wheel, "distribution", lambda name: installed)
    monkeypatch.setattr(core_wheel, "_package_root", lambda: package)
    return package, metadata


def test_wheel_retains_resources_license_metadata_and_safe_record(core_installation):
    """A portable wheel must retain runtime resources and verifiable packaging metadata."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    wheel = bundled_core_wheel()
    with zipfile.ZipFile(io.BytesIO(wheel.content)) as archive:
        assert archive.read("skyulf/resources/settings.json") == b'{"active": true}'
        assert archive.read("skyulf_core-0.9.5.dist-info/licenses/LICENSE") == b"TEST LICENSE"
        assert b"Requires-Dist: numpy>=1.24" in archive.read("skyulf_core-0.9.5.dist-info/METADATA")
        assert b"skyulf = skyulf:main" in archive.read(
            "skyulf_core-0.9.5.dist-info/entry_points.txt"
        )
        assert not any("direct_url" in name for name in archive.namelist())
        records = list(
            csv.reader(io.StringIO(archive.read("skyulf_core-0.9.5.dist-info/RECORD").decode()))
        )
        assert {row[0] for row in records} == set(archive.namelist())
        for name, checksum, size in records:
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            if name.endswith("/RECORD"):
                assert checksum == size == ""
                continue
            content = archive.read(name)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
            assert checksum == "sha256=" + encoded.decode()
            assert size == str(len(content))


@pytest.mark.parametrize(
    "relative_path,new_content",
    [
        ("skyulf/__init__.py", "VALUE = 'modified'\n"),
        ("skyulf/resources/settings.json", '{"active": false}'),
        ("skyulf_core-0.9.5.dist-info/licenses/LICENSE", "REVISED LICENSE"),
        ("skyulf_core-0.9.5.dist-info/entry_points.txt", "[console_scripts]\nx = skyulf:main"),
    ],
)
def test_wheel_cache_invalidates_changed_source_and_metadata(
    core_installation, relative_path, new_content
):
    """Same-version source, data and metadata edits must never return a stale wheel."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    package, _ = core_installation
    first = bundled_core_wheel()
    assert bundled_core_wheel() is first
    (package.parent / relative_path).write_text(new_content)
    second = bundled_core_wheel()
    assert second.source_sha256 != first.source_sha256
    with zipfile.ZipFile(io.BytesIO(second.content)) as archive:
        assert archive.read(relative_path) == (package.parent / relative_path).read_bytes()
    assert second.filename != first.filename


def test_wheel_rejects_native_distribution(core_installation):
    """The pure-Python repacker must refuse platform wheels rather than mislabel binaries."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    _, metadata = core_installation
    (metadata / "WHEEL").write_text(
        "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-cp312-win_amd64\n"
    )
    with pytest.raises(ValueError, match="pure-Python"):
        bundled_core_wheel()


@pytest.mark.parametrize("suffix", [".so", ".pyd", ".dll", ".dylib"])
def test_wheel_rejects_native_package_content(core_installation, suffix):
    """Pure-Python metadata cannot hide platform binaries in package resources."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    package, _ = core_installation
    (package / f"native{suffix}").write_bytes(b"native library")
    with pytest.raises(ValueError, match="pure-Python"):
        bundled_core_wheel()


def test_wheel_rejects_missing_package_source(core_installation):
    """A broken installation must not produce a wheel containing only resources."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    package, _ = core_installation
    (package / "__init__.py").unlink()
    with pytest.raises(ValueError, match="missing its Python source"):
        bundled_core_wheel()


@pytest.mark.parametrize("license_path", ["../private.txt", r"nested\license"])
def test_wheel_rejects_unsafe_license_metadata(core_installation, license_path):
    """Distribution metadata cannot add server files through license paths."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    _, metadata = core_installation
    path = metadata / "METADATA"
    path.write_text(
        path.read_text().replace("License-File: LICENSE", f"License-File: {license_path}")
    )
    with pytest.raises(ValueError, match="unsafe relative path"):
        bundled_core_wheel()


def test_wheel_rejects_absolute_license_metadata(core_installation):
    """Absolute license paths cannot copy real files outside distribution metadata."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    _, metadata = core_installation
    private = metadata.parent / "private.txt"
    private.write_text("PRIVATE")
    path = metadata / "METADATA"
    path.write_text(
        path.read_text().replace("License-File: LICENSE", f"License-File: {private.as_posix()}")
    )
    with pytest.raises(ValueError, match="unsafe relative path"):
        bundled_core_wheel()


def test_wheel_rejects_missing_declared_license(core_installation):
    """A declared license must be included rather than silently omitted from exports."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    _, metadata = core_installation
    (metadata / "licenses" / "LICENSE").unlink()
    with pytest.raises(ValueError, match="metadata is missing LICENSE"):
        bundled_core_wheel()


def test_wheel_rejects_missing_distribution_record(core_installation):
    """Incomplete installation metadata must stop export before creating an archive."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    _, metadata = core_installation
    (metadata / "RECORD").unlink()
    with pytest.raises(ValueError, match="requires installed .dist-info metadata"):
        bundled_core_wheel()


def test_wheel_rejects_additional_top_level_package(core_installation):
    """Export must refuse distributions whose additional packages would be lost."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    _, metadata = core_installation
    (metadata / "top_level.txt").write_text("skyulf\nextra_package\n")
    with pytest.raises(ValueError, match="pure-Python"):
        bundled_core_wheel()


def test_wheel_rejects_unsafe_distribution_version(core_installation):
    """Installed metadata must not inject path separators into wheel member names."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    _, metadata = core_installation
    path = metadata / "METADATA"
    path.write_text(path.read_text().replace("Version: 0.9.5", "Version: ../0.9.5"))
    with pytest.raises(ValueError, match="cannot be represented in a wheel filename"):
        bundled_core_wheel()


def test_wheel_rejects_package_symlink_escape(core_installation, tmp_path):
    """Package data links cannot copy server files outside the authorized package root."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    package, _ = core_installation
    secret = tmp_path / "private.txt"
    secret.write_text("PRIVATE")
    try:
        (package / "resource.txt").symlink_to(secret)
    except OSError:
        pytest.skip("Creating symbolic links requires OS permission.")
    with pytest.raises(ValueError, match="outside|symbolic"):
        bundled_core_wheel()


def test_package_reader_rejects_outside_root(core_installation, tmp_path):
    """Resolved containment must reject files even when they have an ordinary suffix."""
    from backend.ml_pipeline.project_export.core_wheel import _contained_file

    package, _ = core_installation
    private = tmp_path / "private.py"
    private.write_text("SECRET = True")
    with pytest.raises(ValueError, match="outside"):
        _contained_file(package, private)


def test_wheel_keeps_actual_source_and_installs_offline(tmp_path):
    """A clean environment installs the bundled Core without resolving a remote release."""
    from backend.ml_pipeline.project_export.core_wheel import (
        _core_distribution,
        bundled_core_wheel,
    )

    wheel = bundled_core_wheel()
    wheel_path = tmp_path / wheel.filename
    wheel_path.write_bytes(wheel.content)
    installed = _core_distribution()
    import skyulf

    source = Path(skyulf.__file__).parent
    with zipfile.ZipFile(io.BytesIO(wheel.content)) as archive:
        for filename in source.rglob("*.py"):
            assert archive.read("skyulf/" + filename.relative_to(source).as_posix()) == (
                filename.read_bytes()
            )
        assert (
            archive.read(f"skyulf_core-{installed.version}.dist-info/METADATA")
            .decode()
            .splitlines()
        ) == installed.read_text("METADATA").splitlines()
    environment = tmp_path / "clean-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    uv = shutil.which("uv")
    assert uv is not None, "CI and development environments require uv."
    result = subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--offline", "--no-deps", str(wheel_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    check = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.metadata as m; from pathlib import Path; "
            "d=m.distribution('skyulf-core'); "
            "assert Path(d.locate_file('skyulf/preprocessing/pipeline.py')).is_file(); "
            f"assert d.version == {installed.version!r}; print(d.version)",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert check.stdout.strip() == installed.version, result.stderr


def test_changed_development_source_imports_from_offline_wheel(core_installation, tmp_path):
    """A clean interpreter must import changed editable source rather than an older release."""
    from backend.ml_pipeline.project_export.core_wheel import bundled_core_wheel

    package, _ = core_installation
    bundled_core_wheel()
    (package / "__init__.py").write_text("VALUE = 'changed development source'\n")
    wheel = bundled_core_wheel()
    wheel_path = tmp_path / wheel.filename
    wheel_path.write_bytes(wheel.content)
    environment = tmp_path / "changed-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    uv = shutil.which("uv")
    assert uv is not None, "CI and development environments require uv."
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--offline", "--no-deps", str(wheel_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    check = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import skyulf; from pathlib import Path; import sys; "
            "assert Path(skyulf.__file__).is_relative_to(sys.prefix); print(skyulf.VALUE)",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert check.stdout.strip() == "changed development source"
