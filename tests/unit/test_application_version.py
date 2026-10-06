"""Application version metadata remains accurate in installed and source deployments."""

import runpy
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

from backend.config.base import Settings
from backend.config.mixins import core

CORE_MODULE = Path(core.__file__)


def _missing_distribution(distribution):
    """Represent the virtual project without an installed application distribution."""
    raise metadata.PackageNotFoundError(distribution)


@pytest.fixture(autouse=True)
def missing_distribution(monkeypatch):
    """Keep source-checkout coverage independent of locally installed distributions."""
    monkeypatch.setattr(metadata, "version", _missing_distribution)


@pytest.fixture
def source_module(tmp_path):
    """Import a fresh copy without changing settings classes cached by other tests."""
    module_path = tmp_path / "backend" / "config" / "mixins" / "core.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(CORE_MODULE.read_text(encoding="utf-8"), encoding="utf-8")
    return module_path


def test_current_checkout_version_ignores_working_directory(monkeypatch, tmp_path):
    """Launching the backend elsewhere must still read its own checkout version."""
    expected = tomllib.loads(
        (CORE_MODULE.resolve().parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "99.0.0"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    module = runpy.run_path(str(CORE_MODULE))
    assert expected == module["CoreMixin"].APP_VERSION


def test_installed_distribution_precedes_source_manifest(monkeypatch, source_module, tmp_path):
    """Installed release metadata must not be replaced by a nearby source checkout."""
    monkeypatch.setattr(metadata, "version", lambda distribution: "7.8.9")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    module = runpy.run_path(str(source_module))
    assert module["CoreMixin"].APP_VERSION == "7.8.9"


@pytest.mark.parametrize("installed_version", [None, "", "   "])
def test_missing_metadata_version_uses_source(
    monkeypatch, source_module, tmp_path, installed_version
):
    """Incomplete distribution metadata must not hide a usable project version."""
    monkeypatch.setattr(metadata, "version", lambda distribution: installed_version)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    module = runpy.run_path(str(source_module))
    assert module["CoreMixin"].APP_VERSION == "1.2.3"


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        b"[project",
        b"\xff",
        b"[tool]\n",
        b"[project]\n",
        b'project = "invalid"\n',
        b"[project]\nversion = 92\n",
        b'[project]\nversion = ""\n',
        b'[project]\nversion = "   "\n',
    ],
)
def test_unusable_source_version_retains_development_fallback(source_module, tmp_path, manifest):
    """Missing or malformed source metadata must not prevent the backend from starting."""
    if manifest is not None:
        (tmp_path / "pyproject.toml").write_bytes(manifest)
    module = runpy.run_path(str(source_module))
    assert module["CoreMixin"].APP_VERSION == "0.0.0-dev"


def test_app_version_environment_override_is_preserved(monkeypatch):
    """Deployments must retain the existing explicit version override."""
    monkeypatch.setenv("APP_VERSION", "4.5.6-deployment")
    monkeypatch.setenv("FASTAPI_ENV", "testing")
    assert Settings(_env_file=None).APP_VERSION == "4.5.6-deployment"
