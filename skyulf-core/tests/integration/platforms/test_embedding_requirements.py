"""Saved encoder dependencies must travel through the existing local manifest."""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from typing import Any

import pytest

from skyulf.inference import fitted_pipeline, project_dependencies


def _pipeline(*artifacts: dict, source_pins: tuple[str, ...] = ()) -> Any:
    """Represent fitted records without importing any optional encoder dependency."""
    config = {}
    if source_pins:
        config["project_python_source"] = f"install_project_package(requirements={source_pins!r})"
    return SimpleNamespace(
        config=config,
        feature_engineer=SimpleNamespace(
            fitted_steps=[{"type": "sentence_embedder", "artifact": state} for state in artifacts]
        ),
    )


@pytest.fixture
def installed(monkeypatch):
    """Use distribution metadata alone so the base CI environment needs no NLP packages."""
    versions = {"sentence-transformers": "6.0.0", "torch": "2.9.0", "numpy": "2.1.3"}

    def version(name):
        """Match the installed-metadata error contract for undeclared test distributions."""
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(project_dependencies, "version", version)
    return versions


def test_saved_model_pins_merge_with_source_without_mutation(installed):
    """Repeated fitted encoders must add one canonical pin per dependency to the manifest."""
    pipeline = _pipeline(
        {"model_requirements": ("Torch==2.9.0", "sentence_transformers==6.0.0")},
        {"model_requirements": ("torch==2.9.0",)},
        source_pins=("numpy==2.1.3", "torch==2.9.0"),
    )
    before = deepcopy(vars(pipeline))
    assert fitted_pipeline._project_contract(pipeline) == (
        "numpy==2.1.3",
        "sentence-transformers==6.0.0",
        "torch==2.9.0",
    )
    assert vars(pipeline) == before


@pytest.mark.parametrize("from_source", [False, True])
def test_conflicting_saved_model_pins_are_rejected(installed, from_source):
    """One executable package cannot claim two versions of the same fitted dependency."""
    artifacts = [{"model_requirements": ("torch==2.8.0" if from_source else "torch==2.9.0",)}]
    source_pins = ("torch==2.9.0",) if from_source else ()
    if not from_source:
        artifacts.append({"model_requirements": ("torch==2.8.0",)})
    with pytest.raises(ValueError, match="dependency"):
        fitted_pipeline._project_contract(_pipeline(*artifacts, source_pins=source_pins))


@pytest.mark.parametrize("pin", ["torch==2.8.0", "missing-encoder-package==1.0"])
def test_saved_model_pins_verify_the_installed_runtime(installed, pin):
    """A package with unavailable encoder code must fail before native model restoration."""
    with pytest.raises(ValueError, match="dependency"):
        fitted_pipeline._project_contract(_pipeline({"model_requirements": (pin,)}))


@pytest.mark.parametrize("pins", [None, "torch==2.9.0", ["torch==2.9.0"], (1,), ("torch>=2.9.0",)])
def test_saved_model_pins_require_a_tuple_of_exact_strings(installed, pins):
    """Malformed saved dependency metadata must not silently disappear from the package."""
    with pytest.raises(ValueError, match="requirement"):
        fitted_pipeline._project_contract(_pipeline({"model_requirements": pins}))


@pytest.mark.parametrize("source_pins", [(), ("numpy==2.1.3",)])
def test_legacy_and_unrelated_states_keep_existing_contract(installed, source_pins):
    """Only saved sentence-encoder records may extend legacy project requirements."""
    pipeline = _pipeline({}, {"model_name": "legacy"}, source_pins=source_pins)
    pipeline.feature_engineer.fitted_steps.append(
        {"type": "count_vectorizer", "artifact": {"model_requirements": ("missing==1.0",)}}
    )
    assert fitted_pipeline._project_contract(pipeline) == source_pins
