"""Model packages can install from a private index or from wheels saved in the model."""

import sys

import pytest

from skyulf.integrations.mlflow.spark import _package_sources as sources

PINS = ["pandas==2.2.3", "mlflow==3.1.0"]


def _fake_download(names):
    """Return a downloader that writes the given wheel names and records its call."""
    calls = []

    def download(requirements, target, index_args):
        """Pretend pip downloaded wheels for ``requirements``."""
        calls.append((list(requirements), list(index_args)))
        for name in names:
            (target / name).write_bytes(b"wheel")

    return download, calls


def test_unset_policy_keeps_requirements_unchanged(tmp_path):
    """Without the settings, logged models must stay byte-identical to before."""
    assert sources.apply_package_sources(tmp_path, list(PINS), environ={}) == (PINS, [])
    assert not (tmp_path / "wheels").exists()


def test_index_mode_writes_index_lines_before_the_pins(tmp_path):
    """Serving builds the container from requirements.txt, so the index must be in it."""
    environ = {
        "SKYULF_MODEL_PIP_INDEX_URL": "https://pypi.example.invalid/simple",
        "SKYULF_MODEL_PIP_EXTRA_INDEX_URLS": "https://a.example.invalid/simple https://b.example.invalid/simple",
    }
    requirements, paths = sources.apply_package_sources(tmp_path, list(PINS), environ=environ)
    assert requirements == [
        "--index-url https://pypi.example.invalid/simple",
        "--extra-index-url https://a.example.invalid/simple",
        "--extra-index-url https://b.example.invalid/simple",
        *PINS,
    ]
    assert paths == []


@pytest.mark.parametrize(
    "url", ["https://user:secret@pypi.example.invalid/simple", "ftp://pypi.example.invalid", "x"]
)
def test_index_urls_with_credentials_or_bad_schemes_are_rejected(tmp_path, url):
    """Requirement lines are stored in the registry, so credentials must never be written."""
    with pytest.raises(ValueError, match="SKYULF_MODEL_PIP_INDEX_URL"):
        sources.apply_package_sources(
            tmp_path, list(PINS), environ={"SKYULF_MODEL_PIP_INDEX_URL": url}
        )


def test_bundle_mode_saves_wheels_and_installs_without_an_index(tmp_path):
    """An endpoint whose build network reaches no index installs only the saved wheels."""
    download, calls = _fake_download(["pandas-2.2.3-cp312-manylinux.whl", "mlflow-3.1.0.whl"])
    environ = {
        "SKYULF_MODEL_PIP_BUNDLE_WHEELS": "true",
        "SKYULF_MODEL_PIP_INDEX_URL": "https://pypi.example.invalid/simple",
    }
    requirements, paths = sources.apply_package_sources(
        tmp_path, list(PINS), environ=environ, download=download
    )
    assert requirements == [
        "--no-index",
        "code/wheels/mlflow-3.1.0.whl",
        "code/wheels/pandas-2.2.3-cp312-manylinux.whl",
    ]
    assert paths == [str(tmp_path / "wheels")]
    pins, index_args = calls[0]
    assert pins == PINS
    # The index is used only to download; it is not written into the model.
    assert index_args == ["--index-url", "https://pypi.example.invalid/simple"]


def test_bundle_mode_keeps_the_saved_skyulf_wheel_and_downloads_its_dependencies(tmp_path):
    """The certified source wheel stays in code/, and pip resolves its dependencies."""
    local = "code/skyulf_core-0.9.6-0abc-py3-none-any.whl"
    (tmp_path / local.removeprefix("code/")).write_bytes(b"skyulf")
    download, calls = _fake_download(["skyulf_core-0.9.6-0abc-py3-none-any.whl", "numpy-2.1.0.whl"])
    requirements, _ = sources.apply_package_sources(
        tmp_path,
        [*PINS, local],
        environ={"SKYULF_MODEL_PIP_BUNDLE_WHEELS": "true"},
        download=download,
    )
    assert requirements == ["--no-index", local, "code/wheels/numpy-2.1.0.whl"]
    assert calls[0][0] == [*PINS, str(tmp_path / local.removeprefix("code/"))]
    assert not (tmp_path / "wheels/skyulf_core-0.9.6-0abc-py3-none-any.whl").exists()


def test_bundle_mode_rejects_oversized_downloads(tmp_path, monkeypatch):
    """A runaway dependency set must fail with a clear limit, not an unusable model."""
    monkeypatch.setattr(sources, "MAX_BUNDLE_BYTES", 3)
    download, _ = _fake_download(["big-1.0.whl"])
    with pytest.raises(ValueError, match="wheels exceed"):
        sources.apply_package_sources(
            tmp_path,
            list(PINS),
            environ={"SKYULF_MODEL_PIP_BUNDLE_WHEELS": "true"},
            download=download,
        )


@pytest.mark.parametrize("value", ["yes", "1", "TRUE "])
def test_bundle_flag_must_be_true_or_false(tmp_path, value):
    """A misspelled switch must fail instead of silently logging an online-only model."""
    with pytest.raises(ValueError, match="SKYULF_MODEL_PIP_BUNDLE_WHEELS"):
        sources.apply_package_sources(
            tmp_path, list(PINS), environ={"SKYULF_MODEL_PIP_BUNDLE_WHEELS": value}
        )


def test_pip_download_targets_linux_and_the_logging_python(tmp_path, monkeypatch):
    """Serving runs Linux x86_64 with the Python version MLflow records at logging."""
    seen = {}

    def run(command, **kwargs):
        """Record the pip command instead of downloading."""
        seen.update(command=command)
        return sources.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sources.subprocess, "run", run)
    sources.pip_download(["pandas==2.2.3"], tmp_path, ["--index-url", "https://x.invalid"])
    command = seen["command"]
    assert command[:4] == [sys.executable, "-m", "pip", "download"]
    assert "--only-binary=:all:" in command
    assert f"{sys.version_info.major}{sys.version_info.minor}" in command
    assert "manylinux2014_x86_64" in command and "pandas==2.2.3" in command


def test_pip_download_failure_shows_pip_output(tmp_path, monkeypatch):
    """A blocked or missing package must name pip's reason instead of a bare exit code."""
    monkeypatch.setattr(
        sources.subprocess,
        "run",
        lambda command, **kwargs: sources.subprocess.CompletedProcess(command, 1, "", "No match"),
    )
    with pytest.raises(RuntimeError, match="No match"):
        sources.pip_download(["skyulf-core==0.9.5+local"], tmp_path, [])


def test_model_save_options_use_the_configured_sources(tmp_path, monkeypatch):
    """Every logger goes through pyfunc_environment, so the policy reaches all models."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.spark import _spark_environment

    monkeypatch.setenv("SKYULF_MODEL_PIP_INDEX_URL", "https://pypi.example.invalid/simple")
    options, _ = _spark_environment.pyfunc_environment(tmp_path, list(PINS), spark_certified=False)
    assert options["pip_requirements"][0] == "--index-url https://pypi.example.invalid/simple"
    monkeypatch.delenv("SKYULF_MODEL_PIP_INDEX_URL")
    options, _ = _spark_environment.pyfunc_environment(tmp_path, list(PINS), spark_certified=False)
    assert options["pip_requirements"] == PINS
