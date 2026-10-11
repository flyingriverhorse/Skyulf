"""Feature decisions survive target, numeric precision and runtime changes."""

import pickle
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.feature_selection import f_regression

from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)
from skyulf.preprocessing.feature_generation import _common as similarity
from skyulf.preprocessing.feature_selection.univariate import UnivariateSelectionCalculator
from skyulf.preprocessing.scaling.robust import RobustScalerApplier, RobustScalerCalculator
from skyulf.preprocessing.time_series.date_features import (
    DateFeaturesApplier,
    DateFeaturesCalculator,
)


def _native(frame, engine):
    """Select a public engine without changing the input values."""
    return pl.from_pandas(frame) if engine == "polars" else frame.copy(deep=True)


def _pandas(frame):
    """Normalize only the returned values for engine-independent assertions."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("config_mode", ["omitted", "Auto", "metadata"])
def test_univariate_default_selects_continuous_target_signal(engine, config_mode):
    """Automatic scores must not treat each continuous value as an independent class."""
    frame = pd.DataFrame({"signal": np.arange(8.0), "noise": [3, 1, 7, 4, 0, 6, 2, 5]})
    target = np.arange(8.0) / 3 + np.array([0.1, 0.12, 0.09, 0.1, 0.11, 0.1, 0.09, 0.12])
    config = {"columns": list(frame), "k": 1}
    if config_mode == "Auto":
        config["score_func"] = "Auto"
    if config_mode == "metadata":
        config = {**vars(UnivariateSelectionCalculator)["__node_meta__"].params, **config}
    artifact = UnivariateSelectionCalculator().fit((_native(frame, engine), target), config)
    assert artifact["selected_columns"] == ["signal"]
    expected = f_regression(frame, target)[0]
    np.testing.assert_allclose(list(artifact["feature_scores"].values()), expected)


@pytest.mark.parametrize("score_func", ["f_classif", "chi2", "mutual_info_classif"])
def test_explicit_classification_score_rejects_continuous_target(score_func):
    """An incompatible explicit scorer must fail before fitting arbitrary class partitions."""
    frame = pd.DataFrame({"signal": np.arange(16.0), "noise": np.arange(16.0)[::-1]})
    with pytest.raises(ValueError, match="continuous|regression|classification"):
        UnivariateSelectionCalculator().fit(
            (frame, np.arange(16.0) / 3), {"score_func": score_func, "k": 1}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_univariate_class_labels_keep_classification_default(engine):
    """Automatic target handling must retain genuine numeric class labels."""
    frame = pd.DataFrame({"signal": [0, 1, 0, 1, 0, 1], "noise": [2, 2, 3, 3, 4, 4]})
    artifact = UnivariateSelectionCalculator().fit(
        (_native(frame, engine), [0, 1, 0, 1, 0, 1]), {"k": 1, "score_func": "Auto"}
    )
    assert artifact["selected_columns"] == ["signal"]


@pytest.mark.parametrize(
    "with_centering,with_scaling", [(True, True), (True, False), (False, True)]
)
def test_robust_scaler_float32_uses_fitted_float64_arithmetic(with_centering, with_scaling):
    """The Polars transform must preserve the precision and dtype advertised by its schema."""
    frame = pd.DataFrame({"x": np.array([1.13, 2.31, 5.76, 9.47, 11.19], dtype=np.float32)})
    config = {"columns": ["x"], "with_centering": with_centering, "with_scaling": with_scaling}
    artifact = RobustScalerCalculator().fit(frame, config)
    expected = RobustScalerApplier().apply(frame, artifact)
    result = RobustScalerApplier().apply(
        pl.from_pandas(frame), pickle.loads(pickle.dumps(artifact))
    )
    assert result.schema["x"] == pl.Float64
    np.testing.assert_allclose(result["x"].to_numpy(), expected["x"], rtol=1e-14, atol=1e-14)
    assert frame["x"].dtype == np.float32


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "unit,multiplier", [("s", 1), ("ms", 1000), ("us", 1000000), ("ns", 1000000000)]
)
def test_epoch_units_survive_date_feature_artifact_replay(engine, unit, multiplier):
    """Both engines must interpret numeric epochs using the unit fixed when the node was fitted."""
    seconds = 1704197040  # 2024-01-02 12:04 UTC
    frame = pd.DataFrame({"when": pd.Series([seconds * multiplier, None], dtype="Int64")})
    native = _native(frame, engine)
    artifact = DateFeaturesCalculator().fit(
        native,
        {
            "columns": ["when"],
            "features": ["year", "month", "day", "hour", "minute"],
            "epoch_unit": unit,
        },
    )
    result = _pandas(DateFeaturesApplier().apply(native, pickle.loads(pickle.dumps(artifact))))
    assert artifact["epoch_unit"] == unit
    assert result.loc[
        0, ["when_year", "when_month", "when_day", "when_hour", "when_minute"]
    ].tolist() == [2024, 1, 2, 12, 4]
    assert (
        result.loc[1, ["when_year", "when_month", "when_day", "when_hour", "when_minute"]]
        .isna()
        .all()
    )
    assert native.columns.tolist() == ["when"] if engine == "pandas" else native.columns == ["when"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize(
    "values", [[1704197040, 1704283440], [None, None], []], ids=["epochs", "all_null", "empty"]
)
def test_numeric_date_features_require_explicit_epoch_unit(engine, dtype, values):
    """Numeric timestamp units must never depend on the execution engine's implicit defaults."""
    frame = _native(pd.DataFrame({"when": pd.Series(values, dtype=dtype)}), engine)
    with pytest.raises(ValueError, match="epoch_unit"):
        DateFeaturesCalculator().fit(frame, {"columns": ["when"]})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize(
    "values", [[1704197040], [None, None], []], ids=["epochs", "all_null", "empty"]
)
def test_legacy_numeric_date_artifact_requires_refit(engine, dtype, values):
    """A legacy artifact cannot silently choose a different numeric unit on another engine."""
    frame = _native(pd.DataFrame({"when": pd.Series(values, dtype=dtype)}), engine)
    with pytest.raises(ValueError, match="epoch_unit|refit"):
        DateFeaturesApplier().apply(frame, {"columns": ["when"], "features": ["hour"]})


@pytest.mark.parametrize("unit", ["seconds", "D", "", 1])
def test_date_features_reject_unknown_epoch_units(unit):
    """Misspelled units must fail before an artifact records ambiguous parsing behavior."""
    with pytest.raises(ValueError, match="epoch_unit"):
        DateFeaturesCalculator().fit(
            pd.DataFrame({"when": [1]}), {"columns": ["when"], "epoch_unit": unit}
        )


def _similarity_config():
    """Use a token order difference that distinguishes rapidfuzz from SequenceMatcher."""
    return {
        "operations": [
            {
                "operation_type": "similarity",
                "input_columns": ["a"],
                "secondary_columns": ["b"],
                "method": "token_sort_ratio",
                "output_column": "score",
            }
        ]
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_similarity_replays_fitted_fallback_when_rapidfuzz_later_appears(monkeypatch, engine):
    """Installing an optional backend after training must not change saved feature values."""
    frame = _native(pd.DataFrame({"a": ["red blue"], "b": ["blue red"]}), engine)
    monkeypatch.setattr(similarity, "_HAS_RAPIDFUZZ", False)
    artifact = FeatureGenerationCalculator().fit(frame, _similarity_config())
    monkeypatch.setattr(similarity, "_HAS_RAPIDFUZZ", True)
    result = _pandas(FeatureGenerationApplier().apply(frame, pickle.loads(pickle.dumps(artifact))))
    assert result["score"].tolist() == [100 * SequenceMatcher(None, "red blue", "blue red").ratio()]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_similarity_rejects_missing_fitted_rapidfuzz_backend(monkeypatch, engine):
    """Losing the trained similarity engine must be visible before any feature operation executes."""
    frame = _native(pd.DataFrame({"a": ["red blue"], "b": ["blue red"]}), engine)
    monkeypatch.setattr(similarity, "_HAS_RAPIDFUZZ", True)
    artifact = FeatureGenerationCalculator().fit(frame, _similarity_config())
    monkeypatch.setattr(similarity, "_HAS_RAPIDFUZZ", False)
    with pytest.raises(ImportError, match="rapidfuzz"):
        FeatureGenerationApplier().apply(frame, artifact)


def test_legacy_similarity_artifact_warns_about_unpinned_backend(caplog):
    """Legacy fallback behavior must be visible so callers can refit for stable inference."""
    frame = pd.DataFrame({"a": ["red blue"], "b": ["blue red"]})
    result = FeatureGenerationApplier().apply(frame, _similarity_config())
    assert "score" in result
    assert "refit" in caplog.text.lower()


def test_robust_scaler_disabled_transform_retains_polars_dtype():
    """Precision promotion must not change the existing no-operation dtype contract."""
    frame = pl.DataFrame({"x": pl.Series([1.25, 2.5], dtype=pl.Float32)})
    params = RobustScalerCalculator().fit(
        frame, {"columns": ["x"], "with_centering": False, "with_scaling": False}
    )
    result = RobustScalerApplier().apply(frame, params)
    assert result.equals(frame) and result.schema == frame.schema


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_numeric_epoch_fractional_negative_and_overflow_values(engine):
    """Explicit epoch parsing must preserve subsecond day boundaries and coerce invalid ranges."""
    frame = _native(pd.DataFrame({"when": [-0.5, 1704197040.5, 1e30, np.nan, np.inf]}), engine)
    artifact = DateFeaturesCalculator().fit(
        frame,
        {"columns": ["when"], "features": ["year", "day", "hour", "minute"], "epoch_unit": "s"},
    )
    result = _pandas(DateFeaturesApplier().apply(frame, artifact))
    assert result.loc[0, ["when_year", "when_day", "when_hour", "when_minute"]].tolist() == [
        1969,
        31,
        23,
        59,
    ]
    assert result.loc[1, ["when_year", "when_day", "when_hour", "when_minute"]].tolist() == [
        2024,
        2,
        12,
        4,
    ]
    assert result.loc[2:, ["when_year", "when_day", "when_hour", "when_minute"]].isna().all().all()


def test_int128_epoch_overflow_cannot_wrap_into_a_valid_date():
    """Integer unit multiplication must validate bounds before wider integers can wrap."""
    frame = pl.DataFrame({"when": pl.Series([2**119], dtype=pl.Int128)})
    artifact = DateFeaturesCalculator().fit(
        frame, {"columns": ["when"], "features": ["year"], "epoch_unit": "s"}
    )
    result = DateFeaturesApplier().apply(frame, artifact)
    assert result["when_year"].to_list() == [None]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype,invalid", [("int64", -(2**63)), ("uint64", 2**64 - 1)])
def test_invalid_epoch_neighbor_cannot_round_valid_integer_timestamp(engine, dtype, invalid):
    """Masking invalid integer epochs must not shift a valid nanosecond across New Year."""
    frame = _native(
        pd.DataFrame({"when": pd.Series([1577836799999999999, invalid], dtype=dtype)}), engine
    )
    artifact = DateFeaturesCalculator().fit(
        frame, {"columns": ["when"], "features": ["year", "month", "day"], "epoch_unit": "ns"}
    )
    result = _pandas(DateFeaturesApplier().apply(frame, artifact))
    assert result.loc[0, ["when_year", "when_month", "when_day"]].tolist() == [2019, 12, 31]
    assert result.loc[1, ["when_year", "when_month", "when_day"]].isna().all()
