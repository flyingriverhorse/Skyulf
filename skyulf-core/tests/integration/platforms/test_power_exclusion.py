"""Invalid Power values have aligned training and explicit saved-scoring outcomes."""

import logging
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

pl = pytest.importorskip("polars")

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline, scoring_output_schema
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.transformations.power import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
)


def _native(frame, engine):
    """Preserve the requested training engine without changing row positions."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _power(method="box-cox", standardize=True):
    """Use an explicit selected feature so target and metadata are never transformed."""
    return {
        "name": "power",
        "transformer": "PowerTransformer",
        "params": {"columns": ["x"], "method": method, "standardize": standardize},
    }


def _imputer():
    """Keep ordinary missing values scoreable after Power preserves them."""
    return {
        "name": "impute",
        "transformer": "SimpleImputer",
        "params": {"columns": ["x"], "strategy": "mean"},
    }


def _saved(tmp_path, engine, method, standardize, *, steps=None):
    """Fit and reload a real saved artifact using a model that rejects NaNs."""
    values = np.linspace(1.0, 9.0, 80)
    data = pd.DataFrame({"x": values, "target": (values > 5).astype(int)})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [*(steps or [_power(method, standardize)]), _imputer()],
            "modeling": {"type": "logistic_regression", "params": {}},
        }
    )
    pipeline.fit(
        SplitDataset(train=_native(data, engine), test=_native(data.head(0), engine)),
        target_column="target",
    )
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    return load_pipeline(path), path


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [False, True])
def test_saved_power_invalid_rows_are_explicit_and_partition_local(
    tmp_path, engine, method, standardize
):
    """One invalid row must not fail valid neighbours or change saved result ordering."""
    artifact, path = _saved(tmp_path, engine, method, standardize)
    bad = 0.0 if method == "box-cox" else np.inf
    frame = pd.DataFrame({"x": [2.0, bad, np.nan, 6.0, -np.inf]}, index=[4, 4, 1, 4, 1])
    actual = score_pipeline(frame, artifact)
    chunks = pd.concat([score_pipeline(frame.iloc[i : i + 1], artifact) for i in range(len(frame))])
    assert_frame_equal(actual, chunks)
    assert actual.index.equals(frame.index)
    assert actual.scoring_status.tolist() == [
        "predicted",
        "excluded",
        "predicted",
        "predicted",
        "excluded",
    ]
    assert actual.exclusion_reason.fillna("").tolist() == [
        "",
        "invalid:power",
        "",
        "",
        "invalid:power",
    ]
    assert actual.iloc[[1, 4]][["prediction", "probability_0", "probability_1"]].isna().all().all()
    assert tuple(actual.columns) == tuple(column.name for column in scoring_output_schema(artifact))
    holdout = pd.DataFrame({"x": [*np.linspace(1.1, 8.9, 20), bad, np.nan]})
    sealed = seal_partition_probe(artifact, holdout, path, enabled=True)
    if engine == "pandas":
        certificate = require_partition_safe_pipeline(load_pipeline(path))
        assert certificate.steps[0].action == "exclude_invalid"
    else:
        with pytest.raises(ValueError, match="fitted on pandas"):
            require_partition_safe_pipeline(load_pipeline(path))
    assert_frame_equal(score_pipeline(frame, sealed), actual)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_power_training_filters_targets_and_weights_by_position(engine, caplog):
    """Invalid training features must remove the same target and weight despite duplicate labels."""
    values = [1.0, 0.0, 2.0, np.nan, 4.0, np.inf, 8.0, *np.arange(9.0, 23.0)]
    frame = pd.DataFrame({"x": values}, index=[5] * len(values))
    target = np.arange(len(values)) + 10
    weights = np.arange(len(values), dtype=float) + 1
    engineer = FeatureEngineer([_power()])
    with caplog.at_level(logging.WARNING):
        (features, labels), _ = engineer.fit_transform(
            (_native(frame, engine), target), sample_weight=weights
        )
    expected_positions = np.delete(np.arange(len(values)), [1, 5])
    assert len(features) == len(values) - 2
    np.testing.assert_array_equal(labels, target[expected_positions])
    np.testing.assert_array_equal(engineer.train_sample_weight_, weights[expected_positions])
    assert "excluded 2 invalid row" in caplog.text
    assert engineer.fitted_steps[0]["artifact"]["columns"] == ["x"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_all_invalid_power_fit_is_an_error(engine):
    """An invalid training population must not silently turn the selected transform into a no-op."""
    with pytest.raises(ValueError, match=r"100\.00%.*10%"):
        PowerTransformerCalculator().fit(
            _native(pd.DataFrame({"x": [0.0, -1.0, -np.inf, np.inf]}), engine),
            _power()["params"],
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_overflow_power_apply_removes_the_whole_row_and_target(engine):
    """A finite input whose learned transform overflows must not leave a partial feature row."""
    state = {
        "type": "power_transformer",
        "columns": ["x", "z"],
        "method": "yeo-johnson",
        "standardize": False,
        "lambdas": [100.0, 1.0],
        "scaler_params": {},
    }
    frame = _native(pd.DataFrame({"x": [-2.0, 1e308, -1.0], "z": [3.0, 4.0, 5.0]}), engine)
    output, target = PowerTransformerApplier().apply((frame, np.array([7, 8, 9])), state)
    assert len(output) == 2
    np.testing.assert_array_equal(target, [7, 9])
    assert np.isfinite(output.to_numpy()).all()


def test_empty_and_all_invalid_saved_power_skip_estimator(tmp_path, monkeypatch):
    """Requests with no valid features still return the declared typed result without model work."""
    artifact, _ = _saved(tmp_path, "pandas", "box-cox", True)

    def forbidden(*args, **kwargs):
        """Fail if the model receives a batch with zero surviving rows."""
        pytest.fail("Estimator called for an excluded request")

    monkeypatch.setattr(artifact.pipeline.model_estimator.applier, "predict", forbidden)
    frame = pd.DataFrame({"x": [0.0, -1.0, np.inf]})
    result = score_pipeline(frame, artifact)
    assert result.scoring_status.eq("excluded").all()
    assert result.prediction.isna().all()
    assert_frame_equal(score_pipeline(frame.head(0), artifact), result.head(0))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("power_first", [False, True])
def test_chained_filters_preserve_first_exclusion_reason(tmp_path, engine, power_first):
    """Later Power or outlier filters must not overwrite an earlier row's reason."""
    bounds = {
        "name": "bounds",
        "transformer": "ManualBounds",
        "params": {"bounds": {"x": {"lower": 0.0}}},
    }
    steps = [_power(), bounds] if power_first else [bounds, _power()]
    artifact, _ = _saved(tmp_path, engine, "box-cox", True, steps=steps)
    frame = pd.DataFrame({"x": [-1.0, np.nan, 2.0, 0.0, -np.inf]}, index=[2, 2, 1, 2, 1])
    actual = score_pipeline(frame, artifact)
    first = "invalid:power" if power_first else "outlier:bounds"
    # The fitted standardized Power output for x=2 may itself be below the later bound.
    assert actual.exclusion_reason.fillna("").iloc[np.array([0, 1, 3, 4])].tolist() == [
        first,
        "",
        "invalid:power",
        first,
    ]
    chunks = pd.concat([score_pipeline(frame.iloc[i : i + 1], artifact) for i in range(len(frame))])
    assert_frame_equal(actual, chunks)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_power_fit_statistics_only_use_valid_training_rows(engine):
    """Invalid neighbours cannot contaminate fitted lambdas or standardization."""
    valid = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0], "z": [2.0, 5.0, 7.0, 11.0]})
    valid = pd.concat([valid] * 5, ignore_index=True)
    mixed = pd.concat([valid, pd.DataFrame({"x": [0.0, np.nan, np.inf], "z": [5.0, 7.0, -1.0]})])
    params = {"columns": ["x", "z"], "method": "box-cox", "standardize": True}
    expected_frame = pd.concat([valid, pd.DataFrame({"x": [np.nan], "z": [7.0]})])
    expected = PowerTransformerCalculator().fit(_native(expected_frame, engine), params)
    actual = PowerTransformerCalculator().fit(_native(mixed, engine), params)
    assert actual == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_weighted_split_model_fit_sees_only_valid_power_rows(tmp_path, engine, caplog):
    """Actual model fitting receives the same surviving features, labels and weights."""
    values = np.linspace(1.0, 9.0, 80)
    raw = pd.DataFrame({"x": values, "target": (values > 5).astype(int)}, index=[3] * 80)
    raw.iloc[[3, 7, 55], 0] = [0.0, np.nan, np.inf]
    weights = np.arange(80, dtype=float) + 1
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [_power(), _imputer()],
            "modeling": {"type": "logistic_regression", "params": {}},
        }
    )
    with caplog.at_level(logging.WARNING):
        pipeline.fit(
            SplitDataset(
                train=_native(raw, engine),
                test=_native(raw.head(0), engine),
                train_sample_weight=weights,
            ),
            target_column="target",
        )
    expected_positions = np.delete(np.arange(80), [3, 55])
    np.testing.assert_array_equal(
        pipeline.feature_engineer.train_sample_weight_, weights[expected_positions]
    )
    assert "excluded 2 invalid row" in caplog.text
    save_pipeline(pipeline, tmp_path / "weighted")
    result = score_pipeline(
        pd.DataFrame({"x": [2.0, 0.0, 8.0]}), load_pipeline(tmp_path / "weighted")
    )
    assert result.scoring_status.tolist() == ["predicted", "excluded", "predicted"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_real_fit_rejects_overflowed_standardization_state(engine):
    """Finite extreme training inputs must not publish nonfinite saved scale parameters."""
    frame = _native(pd.DataFrame({"x": [1e300, 2e300, 4e300, 8e300]}), engine)
    with pytest.raises(ValueError, match="finite"):
        PowerTransformerCalculator().fit(frame, _power("yeo-johnson")["params"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [False, True])
def test_missing_power_values_fit_per_column_and_remain_available_to_imputer(
    engine, method, standardize
):
    """Missing features must retain their rows and other columns' fit observations."""
    from sklearn.preprocessing import PowerTransformer

    values = np.linspace(1.0, 9.0, 100)
    frame = pd.DataFrame({"x": values, "z": values**2}, index=[3] * 100)
    frame.iloc[:20, 0] = np.nan
    frame.iloc[20:40, 1] = np.nan
    config = {"columns": ["x", "z"], "method": method, "standardize": standardize}
    reference = PowerTransformer(method=method, standardize=standardize).fit(frame)
    expected = reference.transform(frame)
    state = PowerTransformerCalculator().fit(_native(frame, engine), config)
    np.testing.assert_allclose(state["lambdas"], reference.lambdas_, rtol=1e-12, atol=1e-12)
    if standardize:
        scaler = state["scaler_params"]
        assert isinstance(scaler, dict)
        np.testing.assert_allclose(scaler["mean"], reference._scaler.mean_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            scaler["scale"], reference._scaler.scale_, rtol=1e-12, atol=1e-12
        )
    actual, target = PowerTransformerApplier().apply(
        (_native(frame, engine), np.arange(100)), state
    )
    assert len(actual) == 100
    np.testing.assert_array_equal(target, np.arange(100))
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-12, atol=1e-12)
    assert np.array_equal(np.isnan(actual.to_numpy()), frame.isna().to_numpy())


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_power_then_imputer_keeps_missing_training_and_saved_scoring_rows(tmp_path, engine, method):
    """Saved scoring must impute ordinary missing values rather than report invalid exclusions."""
    values = np.linspace(1.0, 9.0, 100)
    raw = pd.DataFrame({"x": values, "target": (values > 5).astype(int)}, index=[7] * 100)
    raw.iloc[:20, 0] = np.nan
    imputer = _imputer()
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [_power(method), imputer],
            "modeling": {"type": "logistic_regression", "params": {}},
        }
    )
    metrics = pipeline.fit(
        SplitDataset(
            train=_native(raw, engine),
            test=_native(raw.head(0), engine),
            train_sample_weight=np.arange(100) + 1.0,
        ),
        target_column="target",
    )
    assert metrics["preprocessing"]["steps"]["0:power"]["rows_out"] == 100
    np.testing.assert_array_equal(
        pipeline.feature_engineer.train_sample_weight_, np.arange(100) + 1.0
    )
    path = tmp_path / "missing"
    save_pipeline(pipeline, path)
    artifact = load_pipeline(path)
    bad = 0.0 if method == "box-cox" else np.inf
    holdout = pd.DataFrame({"x": [*np.linspace(1.1, 8.9, 20), np.nan, bad]})
    artifact = seal_partition_probe(artifact, holdout, path, enabled=True)
    output = score_pipeline(holdout, artifact)
    assert output.scoring_status.tolist() == ["predicted"] * 21 + ["excluded"]
    assert pd.isna(output.exclusion_reason.iloc[20])
    assert output.exclusion_reason.iloc[21] == "invalid:power"
    assert np.isfinite(output.prediction.iloc[20])
    missing = pd.DataFrame({"x": [np.nan, np.nan]}, index=[4, 4])
    all_missing = score_pipeline(missing, artifact)
    assert all_missing.scoring_status.eq("predicted").all()
    assert all_missing.exclusion_reason.isna().all()
    assert all_missing.prediction.notna().all()
    chunks = pd.concat(
        [score_pipeline(holdout.iloc[i : i + 1], artifact) for i in range(len(holdout))]
    )
    assert_frame_equal(output, chunks)
    if engine == "pandas":
        assert (
            require_partition_safe_pipeline(load_pipeline(path)).steps[0].action
            == "exclude_invalid"
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_power_all_missing_selected_column_has_clear_fit_error(engine):
    """A column with no observed values cannot provide a fitted transform or standardizer."""
    with pytest.raises(ValueError, match="no observed training values.*x"):
        PowerTransformerCalculator().fit(
            _native(pd.DataFrame({"x": [np.nan, np.nan]}), engine), _power()["params"]
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target_kind", ["array", "list", "series"])
def test_missing_and_invalid_power_keep_boolean_targets_positionally(engine, target_kind):
    """Missing rows keep their labels while true invalid rows drop matching boolean labels."""
    state = PowerTransformerCalculator().fit(
        pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]}), _power()["params"]
    )
    frame = pd.DataFrame({"x": [1.0, None, 0.0, 2.0, np.inf, 4.0]}, index=[5] * 6).astype("Float64")
    labels = [False, True, True, False, False, True]
    target = np.array(labels) if target_kind == "array" else labels
    if target_kind == "series":
        target = pl.Series(labels) if engine == "polars" else pd.Series(labels, index=frame.index)
    output, actual = PowerTransformerApplier().apply((_native(frame, engine), target), state)
    assert len(output) == 4
    assert np.isnan(output["x"].to_numpy()[1])
    np.testing.assert_array_equal(np.asarray(actual), [False, True, False, True])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [False, True])
def test_power_decimal_missing_and_domain_values_share_numeric_mask(engine, method, standardize):
    """Decimal source arrays must use the same numeric values for transformation and row decisions."""
    train = pd.DataFrame({"x": [Decimal("1.0"), Decimal("2.0"), Decimal("3.0"), Decimal("5.0")]})
    state = PowerTransformerCalculator().fit(
        _native(train, engine), _power(method, standardize)["params"]
    )
    frame = pd.DataFrame(
        {"x": [Decimal("2.0"), None, Decimal("0.0"), Decimal("-1.0"), Decimal("4.0")]},
        index=[4] * 5,
    )
    native = _native(frame, engine)
    actual, labels = PowerTransformerApplier().apply((native, np.arange(5)), state)
    numeric = pd.DataFrame({"x": [2.0, np.nan, 0.0, -1.0, 4.0]}, index=frame.index)
    expected, expected_labels = PowerTransformerApplier().apply((numeric, np.arange(5)), state)
    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy(), rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_array_equal(labels, expected_labels)
    assert len(actual) == (3 if method == "box-cox" else 5)
    assert np.isnan(actual["x"].to_numpy()[1])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("invalid_count", [5, 10, 15])
def test_power_training_invalid_share_guard_keeps_missing_and_weight_alignment(
    engine, method, invalid_count, caplog
):
    """Only invalid rows count toward the fixed limit; exactly ten percent remains trainable."""
    raw = pd.DataFrame({"x": np.linspace(1.0, 9.0, 100)}, index=[3] * 100)
    raw.iloc[:invalid_count, 0] = 0.0 if method == "box-cox" else np.inf
    raw.iloc[50:90, 0] = np.nan
    labels = np.arange(100) % 2 == 0
    weights = np.arange(100, dtype=float) + 1
    engineer = FeatureEngineer([_power(method), _imputer()])
    if invalid_count > 10:
        with pytest.raises(
            ValueError, match=r"PowerTransformer.*15/100.*15\.00%.*10%.*Yeo-Johnson.*repair"
        ):
            engineer.fit_transform((_native(raw, engine), labels), sample_weight=weights)
        return
    with caplog.at_level(logging.WARNING):
        (features, target), _ = engineer.fit_transform(
            (_native(raw, engine), labels), sample_weight=weights
        )
    assert len(features) == 100 - invalid_count
    np.testing.assert_array_equal(target, labels[invalid_count:])
    np.testing.assert_array_equal(engineer.train_sample_weight_, weights[invalid_count:])
    assert f"excluded {invalid_count} invalid row" in caplog.text
    assert np.isfinite(features.to_numpy()).all()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("overflow_count", [5, 10])
@pytest.mark.parametrize("overflow_stage", ["power", "standardization"])
def test_power_training_guard_counts_domain_and_fitted_output_overflow_once(
    engine, overflow_count, overflow_stage, monkeypatch
):
    """Training must gate the union of domain and fitted-transform failures over original rows."""
    from skyulf.preprocessing.transformations import power

    raw = pd.DataFrame({"x": np.linspace(1.0, 9.0, 100), "z": np.linspace(2.0, 10.0, 100)})
    raw.iloc[:5] = 0.0
    raw.iloc[5 : 5 + overflow_count] = 100.0
    raw.iloc[50:80, 0] = np.nan
    original_fit = power.PowerTransformer.fit

    def fit_with_extreme_lambdas(self, values):
        """Retain real fitting but inject learned lambdas that exercise genuine numeric overflow."""
        result = original_fit(self, values)
        result.lambdas_ = np.full(values.shape[1], 200.0 if overflow_stage == "power" else 1.0)
        return result

    monkeypatch.setattr(power.PowerTransformer, "fit", fit_with_extreme_lambdas)
    if overflow_stage == "standardization":
        original_scale_fit = power.StandardScaler.fit

        def fit_with_small_scale(self, values):
            """Exercise finite learned parameters whose normalization overflows selected rows."""
            result = original_scale_fit(self, values)
            result.mean_ = np.zeros(values.shape[1])
            result.scale_ = np.full(values.shape[1], 1e-307)
            return result

        monkeypatch.setattr(power.StandardScaler, "fit", fit_with_small_scale)
    params = {
        "columns": ["x", "z"],
        "method": "box-cox",
        "standardize": overflow_stage == "standardization",
    }
    if overflow_count == 10:
        with pytest.raises(ValueError, match=r"15/100.*15\.00%.*10%"):
            PowerTransformerCalculator().fit(_native(raw, engine), params)
        return
    state = PowerTransformerCalculator().fit(_native(raw, engine), params)
    output, target = PowerTransformerApplier().apply((_native(raw, engine), np.arange(100)), state)
    assert len(output) == 90
    np.testing.assert_array_equal(target, np.arange(10, 100))
