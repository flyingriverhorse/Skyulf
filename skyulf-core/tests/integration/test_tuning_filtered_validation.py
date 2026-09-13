"""OC-251: searchers score the validation rows and targets retained by preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.regression import RidgeRegressionCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


def _iqr_adapter():
    """Run the actual IQR node rather than infer retained rows from output length."""
    return FeatureEngineerFoldAdapter(
        [{"name": "filter", "transformer": "IQR", "params": {"columns": ["x"]}}],
        target_column="target",
    )


def _data():
    """Place rejected rows throughout the frame and retain duplicate, unordered indexes."""
    x = np.tile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], 10).astype(float)
    index = np.tile([4, 2, 2, 7, 3, 3, 1, 9, 5, 5, 0, 8], 10)
    return pd.DataFrame({"x": x}, index=index), pd.Series(2 * x + 1, index=index, name="target")


def _expected_score(X, y, holdout, alpha=1):
    """Independently score Ridge on the known inlier rows of each training/validation fold."""
    splits = (
        [(np.arange(len(X)), np.arange(len(X)))]
        if holdout
        else KFold(3, shuffle=True, random_state=42).split(X, y)
    )
    scores = []
    for train, valid in splits:
        train = train[X.iloc[train]["x"].to_numpy() <= 10]
        valid = valid[X.iloc[valid]["x"].to_numpy() <= 10]
        model = Ridge(alpha=alpha).fit(X.iloc[train], y.iloc[train])
        scores.append(model.score(X.iloc[valid], y.iloc[valid]))
    return float(np.mean(scores))


@pytest.mark.parametrize("holdout", [False, True])
@pytest.mark.parametrize(
    "strategy, pruner",
    [
        ("grid", "none"),
        ("random", "none"),
        ("halving_grid", "none"),
        ("halving_random", "none"),
        ("optuna", "none"),
        ("optuna", "median"),
    ],
)
def test_all_tuning_strategies_score_only_aligned_iqr_validation_rows(strategy, pruner, holdout):
    """Filtering validation rows must preserve the same score across all tuning strategies."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    X, y = _data()
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    adapter = _iqr_adapter()
    model, result = TuningCalculator(RidgeRegressionCalculator()).fit(
        X,
        y,
        config=TuningConfig(
            strategy=strategy,
            metric="r2",
            n_trials=1,
            search_space={"alpha": [1.0]},
            cv_folds=3,
            cv_random_state=42,
            strategy_params={"pruner": pruner},
        ),
        preprocessing=adapter,
        validation_data=(X.copy(), y.copy()) if holdout else None,
        validation_frames=(X.copy(), y.copy()) if holdout else None,
    )
    assert result.best_score == pytest.approx(_expected_score(X, y, holdout), abs=1e-12)
    assert result.best_params == {"alpha": 1.0}
    retained_X, retained_y = adapter.transform(X, y)
    assert len(retained_X) == len(retained_y) == 110
    assert model.predict(retained_X.to_numpy()).shape == (110,)
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
def test_halving_resource_rounds_and_parallel_workers_keep_validation_pairs(strategy):
    """Smaller intermediate validation samples must stay aligned through successive rounds."""
    X, y = _data()
    model, result = TuningCalculator(RidgeRegressionCalculator()).fit(
        X,
        y,
        config=TuningConfig(
            strategy=strategy,
            metric="r2",
            n_trials=4,
            n_jobs=2,
            search_space={"alpha": [0.1, 1.0, 10.0, 100.0]},
            cv_folds=3,
            strategy_params={"factor": 2, "min_resources": 30},
        ),
        preprocessing=_iqr_adapter(),
    )
    assert result.n_trials == 7  # Four, two, then one surviving candidate.
    assert result.best_params == {"alpha": 0.1}
    assert result.best_score == pytest.approx(_expected_score(X, y, False, alpha=0.1), abs=1e-12)
    assert np.isfinite(model.predict([[2.0], [5.0]])).all()
