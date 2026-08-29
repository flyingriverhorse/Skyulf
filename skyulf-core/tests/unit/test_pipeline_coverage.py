"""Coverage-focused tests for skyulf.pipeline (the top-level SkyulfPipeline).

Exercises branches not hit by the existing end-to-end test in test_pipeline.py:
* `artifact_digest` (moved to `skyulf.pipeline_seal` in the F-19 split): the
  F-15 semantic digest's determinism, weight sensitivity, tree structures,
  tuned ``(model, TuningResult)`` tuples, and the fail-loud ``TypeError``
  that replaced the old ``repr`` fallback.
* `_init_model_estimator` early return, registry-only model resolution
  (including the `hyperparameter_tuner` base-model branch), and the
  "unknown model type" / "partially registered" error paths.
* `fit()` taking the `SplitDataset`-passthrough branch and the evaluation
  failure branch.
* `predict()` raising when the pipeline has no fitted model.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling._tuning.schemas import TuningResult
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline_seal import artifact_digest
from skyulf.registry import NodeRegistry

# ---------------------------------------------------------------------------
# artifact_digest (skyulf.pipeline_seal)
# ---------------------------------------------------------------------------


def _small_xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_artifact_digest_is_deterministic_for_same_estimator() -> None:
    X, y = _small_xy()
    est = LogisticRegression(random_state=42).fit(X, y)
    assert artifact_digest(est) == artifact_digest(est)


def test_artifact_digest_detects_different_weights_same_hyperparams() -> None:
    """Identical hyperparameters but different fitted weights must digest
    differently — this is the collision the old ``repr`` fallback caused."""
    X, y = _small_xy()
    same = LogisticRegression(random_state=42).fit(X, y)
    flipped = LogisticRegression(random_state=42).fit(X, 1 - y)
    assert same.get_params() == flipped.get_params()
    assert not (same.coef_ == flipped.coef_).all()
    assert artifact_digest(same) != artifact_digest(flipped)


def test_artifact_digest_raises_on_undigestible_object() -> None:
    """No silent ``repr`` fallback: an artifact without a canonical
    representation must fail the seal, not pass it."""

    class _Opaque:
        def __init__(self) -> None:
            self.gen = (i for i in range(3))

    with pytest.raises(TypeError):
        artifact_digest(_Opaque())


def test_artifact_digest_is_key_order_insensitive_for_dicts() -> None:
    """Preprocessing artifacts are config dicts; insertion order must not
    matter for the seal."""
    assert artifact_digest({"a": 1, "b": 2.0, "c": [1, 2]}) == artifact_digest(
        {"c": [1, 2], "b": 2.0, "a": 1}
    )
    assert artifact_digest({"a": 1}) != artifact_digest({"a": 2})


def test_artifact_digest_covers_tree_structures() -> None:
    """sklearn Tree objects are C extensions without ``__dict__``; the digest
    must walk their node arrays."""
    X, y = _small_xy()
    rf = RandomForestClassifier(n_estimators=3, random_state=42).fit(X, y)
    assert artifact_digest(rf) == artifact_digest(rf)
    other = RandomForestClassifier(n_estimators=3, random_state=43).fit(X, y)
    assert artifact_digest(rf) != artifact_digest(other)


def test_artifact_digest_covers_tuned_model_tuples() -> None:
    """Tuned models are stored as ``(estimator, TuningResult)``; both halves
    must feed the digest."""
    X, y = _small_xy()
    est = LogisticRegression(random_state=42).fit(X, y)
    result = TuningResult(best_params={"C": 1.0}, best_score=0.9, n_trials=1, trials=[])
    assert artifact_digest((est, result)) == artifact_digest((est, result))
    changed = TuningResult(best_params={"C": 10.0}, best_score=0.9, n_trials=1, trials=[])
    assert artifact_digest((est, result)) != artifact_digest((est, changed))


def test_fingerprint_of_tree_pipeline_is_deterministic_and_data_sensitive() -> None:
    """End-to-end: a RandomForest pipeline's fingerprint must be stable across
    identical fits and change when the training data changes."""
    config = {
        "preprocessing": [],
        "modeling": {"type": "random_forest_classifier", "node_id": "m1", "n_estimators": 5},
    }
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    p1 = SkyulfPipeline(config)
    p1.fit(frame, target_column="target")
    p2 = SkyulfPipeline(config)
    p2.fit(frame, target_column="target")
    assert p1.fingerprint() == p2.fingerprint()

    flipped = frame.copy()
    flipped["target"] = 1 - flipped["target"]
    p3 = SkyulfPipeline(config)
    p3.fit(flipped, target_column="target")
    assert p1.fingerprint() != p3.fingerprint()


# ---------------------------------------------------------------------------
# _init_model_estimator: early return, registry-only resolution
# ---------------------------------------------------------------------------


def test_init_model_estimator_returns_early_without_type() -> None:
    """A truthy modeling config with no 'type' key must leave model_estimator unset."""
    pipeline = SkyulfPipeline({"modeling": {"some_other_key": 1}})
    assert pipeline.model_estimator is None


@pytest.fixture
def force_registry_miss(monkeypatch):
    """Force NodeRegistry.get_calculator/get_applier to always raise ValueError.

    This simulates a model type that isn't in the registry. Since F-10 the
    registry is the only resolution path, so a miss must surface as an error.
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
    "model_type",
    [
        "logistic_regression",
        "random_forest_classifier",
        "ridge_regression",
        "random_forest_regressor",
    ],
)
def test_init_model_estimator_resolves_known_types_from_registry(model_type: str) -> None:
    """Each known model_type must resolve its calculator/applier from the
    NodeRegistry — since F-10 there is no hardcoded fallback map."""
    expected_calc_cls = NodeRegistry.get_calculator(model_type)
    expected_applier_cls = NodeRegistry.get_applier(model_type)

    pipeline = SkyulfPipeline({"modeling": {"type": model_type}})
    assert pipeline.model_estimator is not None
    assert isinstance(pipeline.model_estimator.calculator, expected_calc_cls)
    assert isinstance(pipeline.model_estimator.applier, expected_applier_cls)


@pytest.mark.parametrize(
    "base_model_type",
    [
        "logistic_regression",
        "random_forest_classifier",
        "ridge_regression",
        "random_forest_regressor",
    ],
)
def test_init_model_estimator_hyperparameter_tuner_wraps_registry_base_model(
    base_model_type: str,
) -> None:
    """hyperparameter_tuner must wrap the registry-resolved base model for
    every known base type."""
    expected_calc_cls = NodeRegistry.get_calculator(base_model_type)
    expected_applier_cls = NodeRegistry.get_applier(base_model_type)

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
    """An unknown model should retain the registry's diagnostic context."""
    with pytest.raises(
        ValueError,
        match="Unknown model type: no_such_model_at_all.*forced registry miss for test",
    ):
        SkyulfPipeline({"modeling": {"type": "no_such_model_at_all"}})


@pytest.fixture
def force_partial_registration(monkeypatch):
    """Simulate a partially-registered node: get_calculator resolves fine but
    get_applier raises ValueError. Since F-10 deleted the hardcoded fallback
    map, this must surface as an explicit "partially registered" error rather
    than a misleading "Unknown model type".
    """

    def _raise_applier(_name):
        raise ValueError("forced partial-registration miss for test")

    monkeypatch.setattr(
        NodeRegistry, "get_applier", classmethod(lambda cls, name: _raise_applier(name))
    )


def test_init_model_estimator_partial_registration_raises_explicit_error(
    force_partial_registration,
) -> None:
    """If the registry resolves a calculator but not its applier, the pipeline
    must raise a descriptive error — there is no hardcoded fallback anymore."""
    with pytest.raises(ValueError, match="only partially registered"):
        SkyulfPipeline({"modeling": {"type": "logistic_regression"}})


def test_init_model_estimator_tuner_partial_base_registration_raises(
    force_partial_registration,
) -> None:
    """Same partial-registration guard for the hyperparameter_tuner base-model
    resolution path: an unresolvable base pair is an error, not a fallback."""
    with pytest.raises(ValueError, match="Unknown base model type for tuner"):
        SkyulfPipeline(
            {
                "modeling": {
                    "type": "hyperparameter_tuner",
                    "base_model": {"type": "ridge_regression"},
                }
            }
        )


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
