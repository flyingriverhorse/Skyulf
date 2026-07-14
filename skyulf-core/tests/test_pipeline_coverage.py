"""Coverage-focused tests for skyulf.pipeline (the top-level SkyulfPipeline).

Exercises branches not hit by the existing end-to-end test in test_pipeline.py:
* `_artifact_digest` pickle-failure fallback to `repr`.
* `_init_model_estimator` early return, registry-miss fallback to the manual
  model-type dispatch (including the `hyperparameter_tuner` base-model branch),
  and the final "unknown model type" error.
* `fit()` taking the `SplitDataset`-passthrough branch and the evaluation
  failure branch.
* `predict()` raising when the pipeline has no fitted model.
"""

from typing import Any, cast

import pandas as pd
import pytest

from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.pipeline import SkyulfPipeline, _artifact_digest
from skyulf.registry import NodeRegistry

# ---------------------------------------------------------------------------
# _artifact_digest
# ---------------------------------------------------------------------------


class _Unpicklable:
    """An object whose pickling always fails, forcing the `repr` fallback."""

    def __reduce__(self):
        raise TypeError("cannot pickle this object")

    def __repr__(self) -> str:
        return "<_Unpicklable sentinel>"


def test_artifact_digest_falls_back_to_repr_when_pickle_fails() -> None:
    """_artifact_digest must hash `repr(obj)` when pickling raises."""
    import hashlib

    obj = _Unpicklable()
    digest = _artifact_digest(obj)
    assert digest == hashlib.sha256(repr(obj).encode("utf-8")).digest()


# ---------------------------------------------------------------------------
# _init_model_estimator: early return, registry miss + manual dispatch
# ---------------------------------------------------------------------------


def test_init_model_estimator_returns_early_without_type() -> None:
    """A truthy modeling config with no 'type' key must leave model_estimator unset."""
    pipeline = SkyulfPipeline({"modeling": {"some_other_key": 1}})
    assert pipeline.model_estimator is None


@pytest.fixture
def force_registry_miss(monkeypatch):
    """Force NodeRegistry.get_calculator/get_applier to always raise ValueError.

    This simulates a model type that isn't in the registry, forcing
    SkyulfPipeline's manual model-type dispatch (elif chain) to run.
    """

    def _raise_calculator(_name):
        raise ValueError("forced registry miss for test")

    def _raise_applier(_name):
        raise ValueError("forced registry miss for test")

    monkeypatch.setattr(
        NodeRegistry, "get_calculator", classmethod(lambda cls, name: _raise_calculator(name))
    )
    monkeypatch.setattr(
        NodeRegistry, "get_applier", classmethod(lambda cls, name: _raise_applier(name))
    )


@pytest.mark.parametrize(
    "model_type,calc_import_name",
    [
        ("logistic_regression", "LogisticRegressionCalculator"),
        ("random_forest_classifier", "RandomForestClassifierCalculator"),
        ("ridge_regression", "RidgeRegressionCalculator"),
        ("random_forest_regressor", "RandomForestRegressorCalculator"),
    ],
)
def test_init_model_estimator_manual_dispatch_for_known_types(
    force_registry_miss, model_type: str, calc_import_name: str
) -> None:
    """With the registry forced to miss, each known model_type must resolve via
    the manual elif dispatch to the matching Calculator/Applier classes."""
    import skyulf.pipeline as pipeline_module

    pipeline = SkyulfPipeline({"modeling": {"type": model_type}})
    assert pipeline.model_estimator is not None
    expected_calc_cls = getattr(pipeline_module, calc_import_name)
    assert isinstance(pipeline.model_estimator.calculator, expected_calc_cls)


@pytest.mark.parametrize(
    "base_model_type,base_calc_import_name,base_applier_import_name",
    [
        ("logistic_regression", "LogisticRegressionCalculator", "LogisticRegressionApplier"),
        (
            "random_forest_classifier",
            "RandomForestClassifierCalculator",
            "RandomForestClassifierApplier",
        ),
        ("ridge_regression", "RidgeRegressionCalculator", "RidgeRegressionApplier"),
        (
            "random_forest_regressor",
            "RandomForestRegressorCalculator",
            "RandomForestRegressorApplier",
        ),
    ],
)
def test_init_model_estimator_hyperparameter_tuner_manual_base_dispatch(
    force_registry_miss,
    base_model_type: str,
    base_calc_import_name: str,
    base_applier_import_name: str,
) -> None:
    """hyperparameter_tuner must wrap a manually-dispatched base model when the
    registry lookup fails for the base_model type too — for every known base type."""
    import skyulf.pipeline as pipeline_module

    pipeline = SkyulfPipeline(
        {
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": base_model_type},
            }
        }
    )
    assert pipeline.model_estimator is not None
    assert isinstance(pipeline.model_estimator.calculator, TuningCalculator)
    assert isinstance(pipeline.model_estimator.applier, TuningApplier)
    expected_calc_cls = getattr(pipeline_module, base_calc_import_name)
    expected_applier_cls = getattr(pipeline_module, base_applier_import_name)
    calculator = cast(TuningCalculator, pipeline.model_estimator.calculator)
    applier = cast(TuningApplier, pipeline.model_estimator.applier)
    assert isinstance(calculator.model_calculator, expected_calc_cls)
    assert isinstance(applier.base_applier, expected_applier_cls)


def test_init_model_estimator_hyperparameter_tuner_uses_registry_for_base_model() -> None:
    """When the registry *does* resolve the base_model type (no forced miss),
    the tuner must use the registry-provided calculator/applier directly.

    Compares against live `NodeRegistry` lookups (rather than a statically
    imported class) because other test modules legitimately reload
    `skyulf.modeling.regression` (see test_modeling_regression_gaps.py), which
    re-registers a fresh `RidgeRegressionCalculator` class object — a static
    import bound at collection time could then point to a stale class.
    """
    expected_calc_cls = NodeRegistry.get_calculator("ridge_regression")
    expected_applier_cls = NodeRegistry.get_applier("ridge_regression")

    pipeline = SkyulfPipeline(
        {
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": "ridge_regression"},
            }
        }
    )
    assert pipeline.model_estimator is not None
    assert isinstance(pipeline.model_estimator.calculator, TuningCalculator)
    calculator = cast(TuningCalculator, pipeline.model_estimator.calculator)
    applier = cast(TuningApplier, pipeline.model_estimator.applier)
    assert isinstance(calculator.model_calculator, expected_calc_cls)
    assert isinstance(applier.base_applier, expected_applier_cls)


def test_init_model_estimator_hyperparameter_tuner_unknown_base_raises(
    force_registry_miss,
) -> None:
    """An unresolvable base_model type must raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown base model type for tuner"):
        SkyulfPipeline(
            {
                "modeling": {
                    "type": "hyperparameter_tuner",
                    "base_model": {"type": "no_such_base_model"},
                }
            }
        )


def test_init_model_estimator_unknown_model_type_raises(force_registry_miss) -> None:
    """A model type unresolvable via registry or the manual dispatch must raise."""
    with pytest.raises(ValueError, match="Unknown model type"):
        SkyulfPipeline({"modeling": {"type": "no_such_model_at_all"}})


@pytest.fixture
def force_partial_registration(monkeypatch):
    """Simulate a partially-registered node: get_calculator resolves fine but
    get_applier raises ValueError. Regression test for a bug where the
    hardcoded fallback map was skipped whenever *any* calculator resolved
    from the registry, even if its applier didn't, producing a misleading
    "Unknown model type" error instead of falling back correctly.
    """

    def _raise_applier(_name):
        raise ValueError("forced partial-registration miss for test")

    monkeypatch.setattr(
        NodeRegistry, "get_applier", classmethod(lambda cls, name: _raise_applier(name))
    )


def test_init_model_estimator_falls_back_when_applier_registry_lookup_fails(
    force_partial_registration,
) -> None:
    """If the registry resolves a calculator but not its applier, the pipeline
    must still fall back to the hardcoded type map instead of raising."""
    import skyulf.pipeline as pipeline_module

    pipeline = SkyulfPipeline({"modeling": {"type": "logistic_regression"}})
    assert pipeline.model_estimator is not None
    assert isinstance(
        pipeline.model_estimator.calculator, pipeline_module.LogisticRegressionCalculator
    )
    assert isinstance(pipeline.model_estimator.applier, pipeline_module.LogisticRegressionApplier)


def test_init_model_estimator_tuner_falls_back_when_base_applier_registry_lookup_fails(
    force_partial_registration,
) -> None:
    """Same partial-registration guard for the hyperparameter_tuner base-model
    resolution path."""
    import skyulf.pipeline as pipeline_module

    pipeline = SkyulfPipeline(
        {
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": "ridge_regression"},
            }
        }
    )
    assert pipeline.model_estimator is not None
    calculator = cast(TuningCalculator, pipeline.model_estimator.calculator)
    applier = cast(TuningApplier, pipeline.model_estimator.applier)
    assert isinstance(calculator.model_calculator, pipeline_module.RidgeRegressionCalculator)
    assert isinstance(applier.base_applier, pipeline_module.RidgeRegressionApplier)


# ---------------------------------------------------------------------------
# fit(): SplitDataset passthrough + evaluation-failure branch
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_classification_data() -> pd.DataFrame:
    """A purely-numeric classification dataset (no categorical columns), so
    plain sklearn estimators can fit directly without an encoding step."""
    import numpy as np

    rng = np.random.default_rng(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n_samples),
            "feature2": rng.normal(2, 1, n_samples),
            "target": rng.choice([0, 1], n_samples),
        }
    )


def test_fit_uses_split_dataset_produced_by_preprocessing(numeric_classification_data) -> None:
    """When preprocessing yields a SplitDataset, fit() must use it directly
    (the `isinstance(transformed_data, SplitDataset)` branch), not wrap it."""
    config: dict[str, Any] = {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "Split",
                "params": {
                    "target_column": "target",
                    "test_size": 0.2,
                    "random_state": 42,
                },
            },
        ],
        "modeling": {"type": "logistic_regression"},
    }
    pipeline = SkyulfPipeline(config)
    metrics = pipeline.fit(numeric_classification_data, target_column="target")
    assert "preprocessing" in metrics
    assert "modeling" in metrics or "modeling_error" in metrics


def test_fit_records_modeling_error_when_evaluation_fails(numeric_classification_data) -> None:
    """If model_estimator.evaluate() raises, fit() must catch it and record
    metrics['modeling_error'] instead of propagating the exception."""
    config: dict[str, Any] = {
        "preprocessing": [],
        "modeling": {"type": "logistic_regression"},
    }
    pipeline = SkyulfPipeline(config)
    assert pipeline.model_estimator is not None

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("evaluation exploded")

    pipeline.model_estimator.evaluate = _boom  # ty: ignore[invalid-assignment]

    metrics = pipeline.fit(numeric_classification_data, target_column="target")
    assert "modeling_error" in metrics
    assert "evaluation exploded" in metrics["modeling_error"]
    assert "modeling" not in metrics


# ---------------------------------------------------------------------------
# predict(): raises when unfitted / no model configured
# ---------------------------------------------------------------------------


def test_predict_raises_without_model_configured() -> None:
    """predict() must raise ValueError when there is no model configured."""
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {}})
    with pytest.raises(ValueError, match="Pipeline not fitted or no model configured"):
        pipeline.predict(pd.DataFrame({"a": [1, 2, 3]}))


def test_predict_raises_when_model_not_yet_fitted() -> None:
    """predict() must raise ValueError when a model is configured but never fit."""
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    assert pipeline.model_estimator is not None
    assert pipeline.model_estimator.model is None
    with pytest.raises(ValueError, match="Pipeline not fitted or no model configured"):
        pipeline.predict(pd.DataFrame({"a": [1, 2, 3]}))
