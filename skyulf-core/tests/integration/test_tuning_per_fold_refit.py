"""F-15: per-fold preprocessing refit in the tuning engine.

Tuning's internal CV had the same leak as plain CV: preprocessing was
fitted once on the full training split, so every candidate score was
optimistically biased. The ``preprocessing`` hook re-fits inside each
candidate fold: for ``grid``/``random`` via the engine's own fold loop,
and for ``halving_*``/``optuna`` by wrapping preprocessing + model in an
sklearn ``Pipeline`` so the searcher's internal CV refits per fold.
Holdout tuning with ``validation_data`` is rejected with an explicit
diagnostic — there are no folds to refit around.
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


def _noise_woe_setup() -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    """Pure-noise target + memorising WOE step (shared leakage probe)."""
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
    return X, y, steps


def _disc(auc: float) -> float:
    return max(auc, 1.0 - auc)


def test_tuning_halving_grid_refits_woe_noise_near_chance() -> None:
    """halving_grid runs its CV inside the sklearn searcher; the Pipeline
    wrapper must refit WOE per fold there too, killing the noise-target leak."""
    from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator

    X, y, steps = _noise_woe_setup()
    config = TuningConfig(
        strategy="halving_grid", metric="roc_auc", search_space={"C": [1.0]}, cv_folds=5
    )

    leaky_params = WOEEncoderCalculator().fit((X, y), steps[0]["params"])
    X_leaky, _ = WOEEncoderApplier().apply((X, y), dict(leaky_params))
    _m, leaky_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X_leaky, y, config=config
    )

    adapter = FeatureEngineerFoldAdapter(steps, target_column="target")
    _m, refit_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config=config, preprocessing=adapter
    )

    assert _disc(leaky_result.best_score) > 0.75, (
        f"expected the leaky encoding to memorise labels, got {leaky_result.best_score}"
    )
    assert _disc(refit_result.best_score) < 0.65, (
        f"per-fold refit halving tuning should sit near chance, got {refit_result.best_score}"
    )
    assert refit_result.best_params == {"C": 1.0}, (
        "Pipeline param prefixes must be stripped from best_params"
    )


def test_tuning_optuna_refits_woe_noise_near_chance() -> None:
    """Same honesty proof for the optuna strategy's searcher-internal CV."""
    from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator

    X, y, steps = _noise_woe_setup()
    config = TuningConfig(
        strategy="optuna", metric="roc_auc", n_trials=2, search_space={"C": [1.0]}, cv_folds=5
    )

    leaky_params = WOEEncoderCalculator().fit((X, y), steps[0]["params"])
    X_leaky, _ = WOEEncoderApplier().apply((X, y), dict(leaky_params))
    _m, leaky_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X_leaky, y, config=config
    )

    adapter = FeatureEngineerFoldAdapter(steps, target_column="target")
    _m, refit_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config=config, preprocessing=adapter
    )

    assert _disc(leaky_result.best_score) > 0.75, (
        f"expected the leaky encoding to memorise labels, got {leaky_result.best_score}"
    )
    assert _disc(refit_result.best_score) < 0.65, (
        f"per-fold refit optuna tuning should sit near chance, got {refit_result.best_score}"
    )
    assert refit_result.best_params == {"C": 1.0}


def test_tuning_halving_random_refits_woe_noise_near_chance() -> None:
    """The halving_random variant gets the same honesty proof as halving_grid."""
    from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator

    X, y, steps = _noise_woe_setup()
    config = TuningConfig(
        strategy="halving_random", metric="roc_auc", search_space={"C": [1.0]}, cv_folds=5
    )

    leaky_params = WOEEncoderCalculator().fit((X, y), steps[0]["params"])
    X_leaky, _ = WOEEncoderApplier().apply((X, y), dict(leaky_params))
    _m, leaky_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X_leaky, y, config=config
    )

    adapter = FeatureEngineerFoldAdapter(steps, target_column="target")
    _m, refit_result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config=config, preprocessing=adapter
    )

    assert _disc(leaky_result.best_score) > 0.75
    assert _disc(refit_result.best_score) < 0.65, (
        f"per-fold refit halving_random tuning should sit near chance, "
        f"got {refit_result.best_score}"
    )


def test_halving_multi_candidate_results_carry_unprefixed_params() -> None:
    """Several candidates exercise successive-halving iterations; extracted
    best_params and per-trial params must keep the caller's original keys."""
    X, y = _make_classification_xy()
    config = TuningConfig(
        strategy="halving_grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        cv_folds=3,
    )

    _model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config=config, preprocessing=RecordingPreprocessor()
    )

    assert set(result.best_params) == {"C"}, result.best_params
    assert result.trials, "halving must report at least one trial"
    for trial in result.trials:
        assert set(trial["params"]) == {"C"}, trial["params"]


def test_halving_grid_refits_on_fold_train_rows_only() -> None:
    """Fold isolation inside the halving searcher: every refit sees only its
    fold's training rows, never the rows it is scored against."""
    X, y = _make_classification_xy()
    recorder = RecordingPreprocessor()

    TuningCalculator(LogisticRegressionCalculator()).fit(
        X,
        y,
        config=TuningConfig(strategy="halving_grid", search_space={"C": [1.0]}, cv_folds=3),
        preprocessing=recorder,
    )

    assert recorder.fit_rows, "the searcher's internal CV must trigger per-fold refits"
    all_rows = set(range(len(X)))
    # One extra fit with no paired transform: the final best-model refit on
    # the full split (the serving artifact).
    assert len(recorder.fit_rows) == len(recorder.transform_rows) + 1
    for fit_rows, val_rows in zip(recorder.fit_rows[:-1], recorder.transform_rows, strict=True):
        assert set(fit_rows).isdisjoint(val_rows), "fold fit must not see held-out rows"
        assert set(fit_rows) | set(val_rows) <= all_rows
    assert set(recorder.fit_rows[-1]) == all_rows


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
