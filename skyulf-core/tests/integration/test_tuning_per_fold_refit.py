"""F-15: per-fold preprocessing refit in the tuning engine.

Tuning's internal CV had the same leak as plain CV: preprocessing was
fitted once on the full training split, so every candidate score was
optimistically biased. The ``preprocessing`` hook re-fits inside
each candidate fold for the ``grid``/``random`` strategies (the custom
loop). Strategies whose CV runs inside sklearn searchers (halving,
optuna) are rejected with an explicit diagnostic rather than silently
reporting leaky scores; the app falls back to pre-transformed scoring
for those with a logged warning.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator
from skyulf.modeling.fold_preprocessing import FoldPreprocessor
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


def _make_classification_xy(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    X_arr, y_arr = make_classification(
        n_samples=n, n_features=4, n_informative=3, n_redundant=1, random_state=seed
    )
    return (
        pd.DataFrame(X_arr, columns=pd.Index(["a", "b", "c", "d"])),
        pd.Series(y_arr, name="target"),
    )


class RecordingPreprocessor:
    """Records which rows each fit/transform sees; returns data unchanged."""

    def __init__(self) -> None:
        self.fit_rows: list[list[int]] = []
        self.transform_rows: list[list[int]] = []

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.fit_rows.append(list(X.index))
        return X, y

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.transform_rows.append(list(X.index))
        return X, y


def test_recording_preprocessor_satisfies_protocol() -> None:
    assert isinstance(RecordingPreprocessor(), FoldPreprocessor)


def test_tuning_grid_refits_preprocessing_per_candidate_fold() -> None:
    """2 candidates x 3 folds = 6 fold-fits, each excluding its held-out rows."""
    X, y = _make_classification_xy()
    recorder = RecordingPreprocessor()
    tuner = TuningCalculator(LogisticRegressionCalculator())

    model, result = tuner.fit(
        X,
        y,
        config=TuningConfig(
            strategy="grid",
            metric="accuracy",
            search_space={"C": [0.1, 1.0]},
            cv_folds=3,
        ),
        preprocessing=recorder,
    )

    assert result.n_trials == 2
    assert hasattr(model, "predict")
    # 2 candidates x 3 folds = 6 fold fit/transform pairs, plus one final
    # full-split refit fit (the serving artifact) with no paired transform.
    assert len(recorder.fit_rows) == 7
    assert len(recorder.transform_rows) == 6
    all_rows = set(range(len(X)))
    for fit_rows, val_rows in zip(recorder.fit_rows[:6], recorder.transform_rows, strict=True):
        assert set(fit_rows).isdisjoint(val_rows), "fold fit must not see held-out rows"
        assert set(fit_rows) | set(val_rows) == all_rows
    assert set(recorder.fit_rows[-1]) == all_rows  # final refit sees the full split


def test_tuning_woe_noise_target_stays_near_chance() -> None:
    """Leakage proof for tuning scores: refitting WOE per fold kills the
    noise-target memorization that inflates the leaky best_score."""
    rng = np.random.default_rng(42)
    n, n_categories = 400, 200
    X = pd.DataFrame({"city": [f"c{v}" for v in rng.integers(0, n_categories, size=n)]})
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    steps: list[dict[str, Any]] = [
        {
            "name": "woe_city",
            "transformer": "WOEEncoder",
            "params": {"columns": ["city"], "regularization": 0.5},
        }
    ]
    config = TuningConfig(
        strategy="random",
        metric="roc_auc",
        n_trials=1,
        search_space={"C": [1.0]},
        cv_folds=5,
    )

    def disc(auc: float) -> float:
        return max(auc, 1.0 - auc)

    # Leaky control: full-fit WOE applied to every row before tuning.
    from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator

    leaky_params = WOEEncoderCalculator().fit((X, y), steps[0]["params"])
    X_leaky, _y_leaky = WOEEncoderApplier().apply((X, y), dict(leaky_params))
    _model_leaky, leaky_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X_leaky, y, config=config
    )

    # F-15 fix: the same WOE step refit inside every candidate fold.
    adapter = FeatureEngineerFoldAdapter(steps, target_column="target")
    _model_refit, refit_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config=config, preprocessing=adapter
    )

    assert disc(leaky_result.best_score) > 0.75, (
        f"expected the leaky encoding to memorise labels, got {leaky_result.best_score}"
    )
    assert disc(refit_result.best_score) < 0.65, (
        f"per-fold refit tuning should sit near chance, got {refit_result.best_score}"
    )


def test_halving_strategy_with_preprocessing_is_rejected() -> None:
    """Halving searchers run CV inside sklearn; the hook cannot reach those
    folds, so refuse loudly instead of silently reporting leaky scores."""
    X, y = _make_classification_xy()
    tuner = TuningCalculator(LogisticRegressionCalculator())
    with pytest.raises(ValueError, match="Per-fold"):
        tuner.fit(
            X,
            y,
            config=TuningConfig(strategy="halving_grid", search_space={"C": [1.0]}),
            preprocessing=RecordingPreprocessor(),
        )


def test_optuna_strategy_with_preprocessing_is_rejected() -> None:
    """Same refusal for the optuna strategy — before any optuna machinery."""
    X, y = _make_classification_xy()
    tuner = TuningCalculator(LogisticRegressionCalculator())
    with pytest.raises(ValueError, match="Per-fold"):
        tuner.fit(
            X,
            y,
            config=TuningConfig(strategy="optuna", search_space={"C": [1.0]}),
            preprocessing=RecordingPreprocessor(),
        )


def test_validation_data_with_preprocessing_is_rejected() -> None:
    """Holdout (PredefinedSplit) tuning cannot be re-run fold-wise in v1."""
    X, y = _make_classification_xy()
    tuner = TuningCalculator(LogisticRegressionCalculator())
    with pytest.raises(ValueError, match="validation"):
        tuner.fit(
            X,
            y,
            config=TuningConfig(strategy="grid", search_space={"C": [1.0]}),
            validation_data=(X.iloc[:20], y.iloc[:20]),
            preprocessing=RecordingPreprocessor(),
        )
