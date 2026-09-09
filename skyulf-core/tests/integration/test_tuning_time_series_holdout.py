"""Time-series holdout tuning must share training's feature selection."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator
from skyulf.modeling.regression import RidgeRegressionCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("explicit_time_column", [False, True])
@pytest.mark.parametrize(
    "validation_kind", ["named", "array", "already_selected", "already_selected_array"]
)
def test_time_series_holdout_uses_training_features_without_reordering_targets(
    engine, explicit_time_column, validation_kind
):
    """Dropping the time key must preserve held-out X/y pairing and caller-owned data."""
    frame = pd.DataFrame(
        {"x": np.arange(30, dtype=float), "time": pd.date_range("2024-01-01", periods=30)}
    )
    if explicit_time_column:
        frame["time"] = np.arange(30)
    train: Any = frame.iloc[:18].iloc[::-1].copy()
    validation: Any = frame.iloc[18:24].iloc[::-1].copy()
    y_train = pd.Series(train["x"].to_numpy() * 2, name="target")
    y_validation = pd.Series(validation["x"].to_numpy() * 2, name="target")
    if engine == "polars":
        train, validation = pl.from_pandas(train), pl.from_pandas(validation)
        y_train, y_validation = pl.from_pandas(y_train), pl.from_pandas(y_validation)
    if validation_kind == "array":
        validation = validation.to_numpy()
    elif validation_kind.startswith("already_selected"):
        validation = validation[["x"]]
        if validation_kind == "already_selected_array":
            validation = validation.to_numpy()

    model, result = TuningCalculator(RidgeRegressionCalculator()).fit(
        train,
        y_train,
        TuningConfig(
            strategy="grid",
            metric="r2",
            search_space={"alpha": [0.0]},
            cv_type="time_series_split",
            cv_time_column="time" if explicit_time_column else None,
            n_jobs=1,
        ),
        validation_data=(validation, y_validation),
    )

    assert result.best_score == pytest.approx(1.0)
    assert model.n_features_in_ == 1
    np.testing.assert_allclose(model.predict(np.array([[25.0], [2.0]])), [50.0, 4.0])
    assert list(train.columns) == ["x", "time"]


@pytest.mark.parametrize("as_array", [False, True], ids=["named", "array"])
def test_time_series_holdout_preserves_prepared_validation_for_thresholds(as_array):
    """Equal raw and encoded widths must not cause a prepared feature to be removed."""
    frame = pd.DataFrame(
        {
            "time": np.arange(40),
            "value": np.arange(40, dtype=float),
            "city": ["low", "high"] * 20,
        }
    )
    target = pd.Series([0, 1] * 20, name="target")
    train, validation = frame.iloc[:24], frame.iloc[24:32]
    y_train, y_validation = target.iloc[:24], target.iloc[24:32]
    steps = [{"name": "encode", "transformer": "OneHotEncoder", "params": {"columns": ["city"]}}]
    eager = FeatureEngineerFoldAdapter(steps, target_column="target")
    eager.fit_transform(train.drop(columns=["time"]), y_train)
    prepared, prepared_y = eager.transform(validation.drop(columns=["time"]), y_validation)
    assert list(prepared.columns) == ["value", "city_high", "city_low"]
    assert prepared.shape[1] == train.shape[1] == 3

    model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        train,
        y_train,
        TuningConfig(
            strategy="grid",
            metric="accuracy",
            search_space={"C": [1.0]},
            cv_type="time_series_split",
            cv_time_column="time",
            tune_threshold=True,
            n_jobs=1,
        ),
        validation_data=(prepared.to_numpy() if as_array else prepared, prepared_y),
        preprocessing=FeatureEngineerFoldAdapter(steps, target_column="target"),
        validation_frames=(validation, y_validation),
    )

    assert result.best_score == pytest.approx(1.0)
    assert model.n_features_in_ == 3
    assert result.decision_thresholds is not None
    assert set(result.decision_thresholds) == {0, 1}
