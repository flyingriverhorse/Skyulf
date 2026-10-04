"""Direct tuning must share fit's chronological feature and row alignment."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.regression import RidgeRegressionCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter

STRATEGIES = ["grid", "random", "optuna", "halving_grid", "halving_random"]


def _config(strategy="grid", **overrides):
    """Use one full-budget candidate so every strategy scores the same folds."""
    if strategy == "optuna":
        pytest.importorskip("optuna.integration.sklearn")
    values: dict[str, Any] = {
        "strategy": strategy,
        "search_space": {"alpha": [2.0]},
        "strategy_params": {"min_resources": 48, "max_resources": 48, "pruner": "none"},
        "n_trials": 1,
        "cv_folds": 3,
        "cv_type": "time_series_split",
        "cv_time_column": "time",
        "metric": "mse",
        "n_jobs": 1,
    }
    return TuningConfig(**(values | overrides))


def _data(engine="pandas", datetime_time=False):
    """Shuffle the clock independently of the signal and retain positional targets."""
    ids = np.random.default_rng(13).permutation(48)
    time = pd.Timestamp("2024-01-01") + pd.to_timedelta(ids, unit="D") if datetime_time else ids
    X = pd.DataFrame({"signal": np.sin(ids), "time": time}, index=ids + 100)
    y = pd.Series(ids / 3 + np.sin(ids), name="target", index=ids + 200)
    if engine == "polars":
        X, y = pl.from_pandas(X), pl.from_pandas(y)
    return X, y, ids + 1.0


def _preprocessor():
    """Fit scaling anew on each fold, without using validation statistics."""
    return FeatureEngineerFoldAdapter(
        [{"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["signal"]}}],
        target_column="target",
    )


def _oracle_score(X, y, folds, weights=None, scale=False):
    """Compute an independent sklearn score from explicitly chosen chronological rows."""
    scores = []
    for train, test in folds:
        train_x, test_x = X[train], X[test]
        if scale:
            scaler = StandardScaler().fit(train_x)
            train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)
        model = Ridge(alpha=2.0).fit(
            train_x, y[train], sample_weight=None if weights is None else weights[train]
        )
        scores.append(-mean_squared_error(y[test], model.predict(test_x)))
    return float(np.mean(scores))


def _chronological_score(weighted=False, scale=False, splitter=None):
    """Construct the correct feature-only timeline without invoking Skyulf sorting."""
    ids = np.arange(48)
    X = np.sin(ids).reshape(-1, 1)
    y = ids / 3 + np.sin(ids)
    folds = (splitter or TimeSeriesSplit(3)).split(X)
    return _oracle_score(X, y, folds, ids + 1.0 if weighted else None, scale)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("datetime_time", [False, True], ids=["numeric", "auto_datetime"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scale", [False, True])
def test_direct_tune_matches_chronological_fold_oracle(
    strategy, engine, datetime_time, weighted, scale
):
    """Weights must not control whether time is removed or folds see chronological rows."""
    X, y, weights = _data(engine, datetime_time)
    original_x, original_y = X.to_numpy().copy(), y.to_numpy().copy()
    result = TuningCalculator(RidgeRegressionCalculator()).tune(
        X,
        y,
        _config(strategy, cv_time_column=None if datetime_time else "time"),
        preprocessing=_preprocessor() if scale else None,
        preprocessing_frames=(X, y) if scale else None,
        sample_weight=weights if weighted else None,
    )

    assert result.best_score == pytest.approx(_chronological_score(weighted, scale))
    assert result.best_params == {"alpha": 2.0}
    np.testing.assert_array_equal(X.to_numpy(), original_x)
    np.testing.assert_array_equal(y.to_numpy(), original_y)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scale", [False, True])
def test_fit_retains_its_existing_time_alignment(weighted, scale):
    """The fit-to-tune delegation must not sort or drop an already removed time key again."""
    X, y, weights = _data(datetime_time=True)
    model, result = TuningCalculator(RidgeRegressionCalculator()).fit(
        X,
        y,
        _config(),
        preprocessing=_preprocessor() if scale else None,
        sample_weight=weights if weighted else None,
    )

    assert model.n_features_in_ == 1
    assert result.excluded_feature_columns == ["time"]
    assert result.best_score == pytest.approx(_chronological_score(weighted, scale))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("validation_array", [False, True])
def test_direct_holdout_aligns_columns_without_reordering_validation(
    weighted, scale, validation_array
):
    """A descending holdout must retain feature-target pairs when its time column is removed."""
    X, y, weights = _data(datetime_time=True)
    ids = np.arange(59, 47, -1)
    validation = pd.DataFrame(
        {"signal": np.sin(ids), "time": pd.Timestamp("2024-01-01") + pd.to_timedelta(ids, unit="D")}
    )
    valid_y = pd.Series(ids / 3 + np.sin(ids), name="target")
    validation_data = (validation.to_numpy() if validation_array else validation, valid_y)
    if scale:
        eager = _preprocessor()
        eager.fit_transform(X.drop(columns="time"), y)
        prepared_x, prepared_y = eager.transform(validation.drop(columns="time"), valid_y)
        validation_data = (
            prepared_x.to_numpy() if validation_array else prepared_x,
            prepared_y,
        )
    result = TuningCalculator(RidgeRegressionCalculator()).tune(
        X,
        y,
        _config(),
        validation_data=validation_data,
        preprocessing=_preprocessor() if scale else None,
        preprocessing_frames=(X, y) if scale else None,
        validation_frames=(validation, valid_y) if scale else None,
        sample_weight=weights if weighted else None,
    )
    all_ids = np.concatenate([np.arange(48), ids])
    expected = _oracle_score(
        np.sin(all_ids).reshape(-1, 1),
        all_ids / 3 + np.sin(all_ids),
        [(np.arange(48), np.arange(48, 60))],
        all_ids + 1.0 if weighted else None,
        scale,
    )

    assert result.best_score == pytest.approx(expected)
    np.testing.assert_array_equal(validation["signal"], np.sin(ids))


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("weighted", [False, True])
def test_direct_explicit_policy_preserves_window_and_weight_alignment(strategy, weighted):
    """Gap and window policy must remain the sole owner of strict temporal splits."""
    X, y, weights = _data(datetime_time=True)
    result = TuningCalculator(RidgeRegressionCalculator()).tune(
        X,
        y,
        _config(strategy, cv_shuffle=False, cv_gap=2, cv_test_size=6, cv_max_train_size=18),
        sample_weight=weights if weighted else None,
    )

    splitter = TimeSeriesSplit(3, gap=2, test_size=6, max_train_size=18)
    assert result.best_score == pytest.approx(_chronological_score(weighted, splitter=splitter))


@pytest.mark.parametrize("weighted", [False, True])
def test_direct_explicit_policy_still_rejects_numeric_clock(weighted):
    """Legacy sorting must not weaken the explicit policy's timestamp validation."""
    X, y, weights = _data()
    with pytest.raises(ValueError, match="timestamps, not ambiguous numeric epoch"):
        TuningCalculator(RidgeRegressionCalculator()).tune(
            X,
            y,
            _config(cv_shuffle=False, cv_gap=1),
            sample_weight=weights if weighted else None,
        )


@pytest.mark.parametrize("weighted", [False, True])
def test_direct_history_preprocessing_keeps_policy_owned_clock(weighted):
    """Carry history needs its time key until fold-local features have been generated."""
    X, y, weights = _data(datetime_time=True)
    steps = [
        {
            "name": "history",
            "transformer": "RollingAggregate",
            "params": {
                "columns": ["signal"],
                "sort_by": "time",
                "history_mode": "carry",
                "window": 2,
                "aggregations": ["mean"],
            },
        }
    ]
    result = TuningCalculator(RidgeRegressionCalculator()).tune(
        X,
        y,
        _config(cv_shuffle=False),
        preprocessing=FeatureEngineerFoldAdapter(steps, target_column="target"),
        preprocessing_frames=(X, y),
        sample_weight=weights if weighted else None,
    )

    assert np.isfinite(result.best_score)


def test_direct_non_temporal_tuning_keeps_numeric_time_as_a_feature():
    """Selecting ordinary K-fold must not apply time-series ordering or feature exclusion."""
    X, y, _ = _data()
    result = TuningCalculator(RidgeRegressionCalculator()).tune(
        X, y, _config(cv_type="k_fold", cv_shuffle=False)
    )
    expected = _oracle_score(X.to_numpy(), y.to_numpy(), KFold(3).split(X))

    assert result.best_score == pytest.approx(expected)
