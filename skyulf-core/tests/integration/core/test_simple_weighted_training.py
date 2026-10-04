"""Exercise positional sample-weight transport through public pipeline training."""

from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from skyulf.data.dataset import SplitDataset
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.modeling._sample_weights import SampleWeightError
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.regression import LinearRegressionApplier, LinearRegressionCalculator
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing._weight_policy import (
    _SAFE_MODULES,
    validate_weighted_preprocessing,
    validate_weighted_steps,
)
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.preprocessing.fold_adapter import AuditedFoldPreprocessor, FeatureEngineerFoldAdapter
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.scaling.standard import StandardScalerApplier, StandardScalerCalculator
from skyulf.preprocessing.split import DataSplitter
from skyulf.registry import NodeRegistry


def _frame() -> pd.DataFrame:
    """Give every position a distinguishable weight despite duplicate index labels."""
    return pd.DataFrame({"row": np.arange(40), "target": np.arange(40) % 7}, index=[0] * 40)


@pytest.mark.parametrize("polars", [False, True])
@pytest.mark.parametrize("xy", [False, True])
def test_split_weights_follow_actual_two_stage_positions(polars: bool, xy: bool) -> None:
    """Validation carving must use the actual training row positions for weights."""
    frame: Any = pl.from_pandas(_frame()) if polars else _frame()
    weights = np.arange(40) + 1.0
    splitter = DataSplitter(test_size=0.2, validation_size=0.25, random_state=11)
    if xy:
        features = frame.drop("target") if polars else frame.drop(columns="target")
        result = splitter.split_xy(features, frame["target"], sample_weight=weights)
        train = cast(Any, result.train)[0]
    else:
        result = splitter.split(frame, sample_weight=weights)
        train = cast(Any, result.train)
    np.testing.assert_array_equal(result.train_sample_weight, np.asarray(train["row"]) + 1)
    validation = cast(Any, result.validation)
    assert len(validation[0] if xy else validation) == 10


@pytest.mark.parametrize("target_factory", [list, np.asarray, pd.Series, pl.Series])
@pytest.mark.parametrize("target_rows", [9, 11])
@pytest.mark.parametrize("polars", [False, True])
def test_weighted_split_rejects_mismatched_targets_before_partition(
    monkeypatch, target_factory, target_rows: int, polars: bool
) -> None:
    """Positional gathers must never truncate extra targets or discover missing targets late."""
    frame: Any = pl.DataFrame({"row": range(10)}) if polars else pd.DataFrame({"row": range(10)})
    splitter = DataSplitter(validation_size=0.2)

    def unexpected_split(*args):
        """Fail if invalid paired data reaches random partitioning."""
        pytest.fail("Mismatched targets reached partitioning")

    monkeypatch.setattr(splitter, "_split_indices", unexpected_split)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        splitter.split_xy(frame, target_factory(range(target_rows)), sample_weight=np.ones(10))


@pytest.mark.parametrize("polars", [False, True])
def test_weighted_split_still_accepts_absent_target(polars: bool) -> None:
    """Length validation must preserve the existing optional-target weighted contract."""
    frame: Any = pl.DataFrame({"row": range(10)}) if polars else pd.DataFrame({"row": range(10)})
    result = DataSplitter().split_xy(frame, None, sample_weight=np.arange(10) + 1)
    train = cast(tuple[Any, Any], result.train)
    assert train[1] is None
    np.testing.assert_array_equal(result.train_sample_weight, np.asarray(train[0]["row"]) + 1)


@pytest.mark.parametrize("split", [False, True])
def test_pipeline_weighted_regression_matches_reference(split: bool) -> None:
    """Public raw weights must reach the same regression fit as sklearn."""
    frame = _frame()
    weights = np.arange(40) + 1.0
    steps = [{"name": "split", "transformer": "TrainTestSplitter", "params": {}}] if split else []
    pipeline = SkyulfPipeline({"preprocessing": steps, "modeling": {"type": "linear_regression"}})
    pipeline.fit(frame, "target", sample_weight=weights)
    train = cast(pd.DataFrame, DataSplitter().split(frame).train) if split else frame
    model = LinearRegression().fit(train[["row"]], train["target"], sample_weight=train["row"] + 1)
    np.testing.assert_allclose(pipeline.predict(frame[["row"]]), model.predict(frame[["row"]]))


def test_split_copy_and_transform_preserve_independent_weights() -> None:
    """Safe transforms and copies must keep the training-only weight payload."""
    dataset = SplitDataset(train=_frame(), test=_frame(), train_sample_weight=np.arange(40) + 1.0)
    copied = dataset.copy()
    copied.train_sample_weight[0] = 900
    assert dataset.train_sample_weight[0] == 1
    steps = [{"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["row"]}}]
    transformed, _ = FeatureEngineer(steps).fit_transform(dataset, target_column="target")
    np.testing.assert_array_equal(transformed.train_sample_weight, dataset.train_sample_weight)


def test_weighted_pipeline_requires_explicit_synthetic_policy_before_model_fit() -> None:
    """New synthetic rows must not receive guessed weights before model training."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "sampling", "transformer": "Oversampling"}],
            "modeling": {"type": "linear_regression"},
        }
    )
    with pytest.raises(SampleWeightError, match="synthetic_weight"):
        pipeline.fit(_frame(), "target", sample_weight=np.ones(40), on_leakage="ignore")
    assert pipeline.model_estimator is not None
    assert pipeline.model_estimator.model is None


def test_presplit_public_weights_are_ambiguous() -> None:
    """Already-split datasets accept only their explicit training weight slot."""
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    with pytest.raises(SampleWeightError, match="SplitDataset"):
        pipeline.fit(
            SplitDataset(train=_frame(), test=_frame()), "target", sample_weight=np.ones(40)
        )


@pytest.mark.parametrize("name", list(_SAFE_MODULES))
def test_weighted_allowlist_resolves_real_builtin_classes(name: str) -> None:
    """Every admitted name must resolve to the actual built-in registry pair."""
    validate_weighted_steps([{"transformer": name}])
    assert NodeRegistry.get_calculator(name).__module__.startswith("skyulf.preprocessing.")


@pytest.mark.parametrize("polars", [False, True])
def test_weighted_split_accepts_engine_wrappers(polars: bool) -> None:
    """Wrapper inputs keep the positional contract of their native frames."""
    frame = (
        SkyulfPolarsWrapper(pl.from_pandas(_frame())) if polars else SkyulfPandasWrapper(_frame())
    )
    result = DataSplitter(validation_size=0.2).split(frame, sample_weight=np.arange(40) + 1)
    np.testing.assert_array_equal(
        result.train_sample_weight, np.asarray(cast(Any, result.train)["row"]) + 1
    )


def test_weighted_adapter_rejects_subclass_reordering() -> None:
    """A trusted adapter subclass cannot declare an unverified row-order contract."""

    class ReorderingAdapter(FeatureEngineerFoldAdapter):
        """Simulate a same-length custom transformation."""

        def fit_transform(self, X, y, sample_weight=None):
            """Reverse rows while preserving their count."""
            return X.iloc[::-1], y.iloc[::-1]

    with pytest.raises(SampleWeightError, match="Custom preprocessing"):
        validate_weighted_preprocessing(AuditedFoldPreprocessor(ReorderingAdapter([], "target")))


def test_weighted_builtin_name_rejects_custom_registry_replacement(monkeypatch) -> None:
    """Known names cannot conceal custom calculators behind the registry."""

    class CustomScaler(StandardScalerCalculator):
        """Stand in for a user-provided node replacement."""

    monkeypatch.setitem(NodeRegistry._calculators, "StandardScaler", CustomScaler)
    with pytest.raises(SampleWeightError, match="weighted preprocessing"):
        validate_weighted_steps([{"transformer": "StandardScaler"}])


def test_zero_total_actual_training_split_fails() -> None:
    """Positive raw totals do not permit a zero-total actual training subset."""
    with pytest.raises(SampleWeightError, match="positive finite total"):
        DataSplitter(shuffle=False).split(_frame(), sample_weight=[0] * 32 + [1] * 8)


def test_direct_stateful_transformer_rejects_custom_weighted_step() -> None:
    """Direct transformers must not bypass the weighted pipeline's safety policy."""

    class ReorderingApplier(StandardScalerApplier):
        """Represent a subclass that reverses positions without changing row count."""

        def apply(self, df, params):
            """Reverse the incoming frame."""
            return df.iloc[::-1]

    transformer = StatefulTransformer(StandardScalerCalculator(), ReorderingApplier(), "custom")
    data = SplitDataset(train=_frame(), test=_frame(), train_sample_weight=np.ones(40))
    with pytest.raises(SampleWeightError, match="weighted preprocessing"):
        transformer.fit_transform(data, {"columns": ["row"]})


def test_stateful_estimator_passes_weights_only_when_active(monkeypatch) -> None:
    """Existing custom fit signatures remain valid while weighted fits get the vector."""
    estimator = StatefulEstimator(LinearRegressionCalculator(), LinearRegressionApplier(), "model")
    original = estimator.calculator.fit
    seen = []

    def record(X, y, config, **kwargs):
        """Record the actual fit boundary while preserving its implementation."""
        seen.append(kwargs.copy())
        return original(X, y, config, **kwargs)

    monkeypatch.setattr(estimator.calculator, "fit", record)
    data = SplitDataset(train=_frame(), test=_frame())
    estimator.fit_predict(data, "target", {})
    data.train_sample_weight = np.arange(40) + 1
    estimator.fit_predict(data, "target", {})
    assert "sample_weight" not in seen[0]
    np.testing.assert_array_equal(seen[1]["sample_weight"], data.train_sample_weight)


def test_stateful_cross_validation_passes_stored_weights(monkeypatch) -> None:
    """Ordinary CV receives the same training slot as direct fitting."""
    estimator = StatefulEstimator(LinearRegressionCalculator(), LinearRegressionApplier(), "model")
    seen = []

    def record(**kwargs):
        """Capture the public CV boundary without testing the fold implementation twice."""
        seen.append(kwargs)
        return {"recorded": True}

    monkeypatch.setattr("skyulf.modeling.base.perform_cross_validation", record)
    data = SplitDataset(train=_frame(), test=_frame(), train_sample_weight=np.arange(40) + 1)
    result = estimator.cross_validate(data, "target", {})
    assert result == {"recorded": True}
    np.testing.assert_array_equal(seen[0]["sample_weight"], data.train_sample_weight)


@pytest.mark.parametrize("polars", [False, True])
def test_tuned_temporal_pipeline_fits_with_aligned_weights(monkeypatch, polars: bool) -> None:
    """Every actual fit must use weights in the tuner's temporal training order."""
    frame = _frame().copy()
    frame["event"] = pd.date_range("2024-01-01", periods=40)
    frame["value"] = np.sin(frame["row"])
    frame = frame.iloc[np.random.default_rng(8).permutation(len(frame))]
    weights = frame["row"].to_numpy() + 1.0
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["value"]}}
            ],
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": "ridge_regression"},
                "strategy": "grid",
                "search_space": {"alpha": [1.0]},
                "metric": "r2",
                "cv_folds": 2,
                "cv_type": "time_series_split",
                "cv_time_column": "event",
                "cv_shuffle": False,
                "n_jobs": 1,
            },
        }
    )
    observed = []
    original = Ridge.fit

    def capture(model, X, y, sample_weight=None):
        """Observe every candidate and final estimator fit after temporal sorting."""
        values = np.asarray(X)
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, values[:, 0] + 1)
        observed.append(len(values))
        return original(model, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", capture)
    pipeline.fit(pl.from_pandas(frame) if polars else frame, "target", sample_weight=weights)
    assert observed[-1] == 40
    assert len(observed) == 3
    assert pipeline.model_estimator is not None
    assert not hasattr(pipeline.model_estimator.calculator, "refit_sample_weight_")


@pytest.mark.parametrize("polars", [False, True])
def test_raw_split_and_safe_transform_fit_preserves_weight_alignment(
    monkeypatch, polars: bool
) -> None:
    """A two-stage raw split followed by scaling must deliver aligned model weights."""
    frame = _frame().copy()
    frame["value"] = np.sin(frame["row"])
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "split",
                    "transformer": "TrainTestSplitter",
                    "params": {"validation_size": 0.2},
                },
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["value"]},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    assert pipeline.model_estimator is not None
    original = pipeline.model_estimator.calculator.fit
    seen = []

    def capture(X, y, config, **kwargs):
        """Assert positional identity at the actual model calculator boundary."""
        np.testing.assert_array_equal(kwargs["sample_weight"], np.asarray(X["row"]) + 1)
        seen.append(len(X))
        return original(X, y, config, **kwargs)

    monkeypatch.setattr(pipeline.model_estimator.calculator, "fit", capture)
    pipeline.fit(
        pl.from_pandas(frame) if polars else frame, "target", sample_weight=np.arange(40) + 1
    )
    assert seen == [24]


@pytest.mark.parametrize("presplit", [False, True])
def test_feature_target_separation_preserves_weighted_fit(presplit: bool) -> None:
    """Target separation must preserve raw or existing training-slot weights."""
    frame = _frame()
    weights = np.arange(40) + 1.0
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "target",
                    "transformer": "feature_target_split",
                    "params": {"target_column": "target"},
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    if presplit:
        pipeline.fit(SplitDataset(train=frame, test=frame, train_sample_weight=weights), "target")
    else:
        pipeline.fit(frame, "target", sample_weight=weights)
    expected = LinearRegression().fit(frame[["row"]], frame["target"], sample_weight=weights)
    np.testing.assert_allclose(pipeline.predict(frame[["row"]]), expected.predict(frame[["row"]]))
