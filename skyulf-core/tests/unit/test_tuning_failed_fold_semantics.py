"""What a tuning search must do when a candidate's folds error (OC-200/OC-205).

The strategies disagreed. The grid loop averaged whichever folds survived, so a
candidate that errored on one fold could win on the mean of the rest and nothing
in the result showed the failure; the halving searchers are built with
``error_score=np.nan``, so they reported ``nan`` as the winning score and the
caller went on to refit a model and log a completion. Both now read "not every
fold scored" as an ineligible candidate, and a search left with no eligible
candidate as the actionable ``All trials failed`` error the grid strategy already
raised when *every* fold failed.
"""

import math

import pandas as pd
import pytest

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.regression import KNeighborsRegressorCalculator, RidgeRegressionCalculator


def test_halving_grid_fails_like_grid_when_no_fold_can_be_scored():
    """Four rows over four folds leaves one validation row each, so every fold errors.

    Grid already raised ``All trials failed`` on this input. Halving returned a
    fitted ``Ridge`` with ``best_score=nan`` and a ``nan`` trial score, so the
    caller logged a completion and refit a model that no score had chosen.
    """
    X = pd.DataFrame({"x": range(4)})
    y = pd.Series(range(4), dtype=float)
    tuner = TuningCalculator(RidgeRegressionCalculator())
    space = {"alpha": [1.0]}

    with pytest.raises(ValueError, match="All trials failed"):
        tuner.fit(
            X,
            y,
            config=TuningConfig(strategy="grid", metric="r2", search_space=space, cv_folds=4),
        )

    with pytest.raises(ValueError, match="All trials failed"):
        tuner.fit(
            X,
            y,
            config=TuningConfig(
                strategy="halving_grid",
                metric="r2",
                search_space=space,
                cv_folds=4,
                strategy_params={"min_resources": 4},
            ),
        )


def test_grid_disqualifies_a_candidate_whose_folds_partly_failed():
    """A fold failure must cost the candidate the search, not just one fold.

    ``KNeighborsRegressor(n_neighbors=3)`` over five rows in two unshuffled folds
    errors on the fold whose training set holds two rows and scores 0.0 on the
    other. Averaging the survivor reported ``best_score=0.0`` and handed back a
    fitted model as though the cross-validation had been complete.
    """
    X = pd.DataFrame({"x": range(5)})
    y = pd.Series([0.0] * 5)
    tuner = TuningCalculator(KNeighborsRegressorCalculator())
    logs: list[str] = []

    with pytest.raises(ValueError, match="All trials failed"):
        tuner.fit(
            X,
            y,
            config=TuningConfig(
                strategy="grid",
                metric="mse",
                search_space={"n_neighbors": [3]},
                cv_folds=2,
                cv_shuffle=False,
            ),
            log_callback=logs.append,
        )

    assert any("disqualified: 1/2 CV folds failed" in msg for msg in logs)


def test_a_partly_failed_candidate_still_loses_to_a_healthy_one():
    """Disqualifying a candidate must not take the whole search down with it.

    The same five rows, with ``n_neighbors`` also offering 2 — a value that fits
    on both folds. The candidate that errored on one fold is disqualified and the
    healthy one wins on a real complete-CV score.
    """
    X = pd.DataFrame({"x": range(5)})
    y = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0])
    tuner = TuningCalculator(KNeighborsRegressorCalculator())

    _model, result = tuner.fit(
        X,
        y,
        config=TuningConfig(
            strategy="grid",
            metric="mse",
            search_space={"n_neighbors": [3, 2]},
            cv_folds=2,
            cv_shuffle=False,
        ),
    )

    assert result.best_params == {"n_neighbors": 2}
    assert math.isfinite(result.best_score)
    scores = {t["params"]["n_neighbors"]: t["score"] for t in result.trials}
    assert scores[3] == -float("inf")
