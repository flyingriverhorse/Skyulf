"""F-15: per-fold preprocessing refit in cross-validation.

Preprocessing fitted once on the full training split leaks held-out rows
into the fitted statistics, biasing every CV score optimistically. The
``preprocessing`` parameter re-fits a :class:`FoldPreprocessor` inside
every fold: fit on the fold's training rows only, apply to the held-out
rows. The app always threads the adapter through; ``None`` means the
caller already transformed the data before splitting.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.modeling.fold_preprocessing import FoldPreprocessor
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


def _make_classification_xy(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Build a clean numeric classification DataFrame and Series."""
    X_arr, y_arr = make_classification(
        n_samples=n, n_features=4, n_informative=3, n_redundant=1, random_state=seed
    )
    X = pd.DataFrame(X_arr, columns=pd.Index(["a", "b", "c", "d"]))
    y = pd.Series(y_arr, name="target")
    return X, y


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


def test_preprocessing_refits_on_fold_train_rows_only() -> None:
    """One fit_transform + one transform per fold; fit rows exclude held-out rows."""
    X, y = _make_classification_xy()
    recorder = RecordingPreprocessor()

    result = perform_cross_validation(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        X,
        y,
        config={},
        n_folds=3,
        cv_type="k_fold",
        preprocessing=recorder,
    )

    assert len(result["folds"]) == 3
    assert len(recorder.fit_rows) == 3
    assert len(recorder.transform_rows) == 3
    all_rows = set(range(len(X)))
    for fit_rows, val_rows in zip(recorder.fit_rows, recorder.transform_rows, strict=True):
        assert set(fit_rows).isdisjoint(val_rows), "fold fit must not see held-out rows"
        assert set(fit_rows) | set(val_rows) == all_rows


def test_nested_cv_refits_preprocessing_per_inner_and_outer_fold() -> None:
    """Nested CV must refit inside both loops, never on a fold's own held-out rows."""
    X, y = _make_classification_xy()
    recorder = RecordingPreprocessor()

    result = perform_cross_validation(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        X,
        y,
        config={},
        n_folds=3,
        cv_type="nested_cv",
        preprocessing=recorder,
    )

    # 3 outer folds x (2 inner + 1 outer) fits, each paired with a transform.
    assert len(recorder.fit_rows) == 9
    assert len(recorder.transform_rows) == 9
    for fit_rows, val_rows in zip(recorder.fit_rows, recorder.transform_rows, strict=True):
        assert set(fit_rows).isdisjoint(val_rows)


def test_woe_noise_target_refit_cv_stays_near_chance() -> None:
    """End-to-end leakage proof: refitting WOE per fold kills the noise-target leak.

    With 200 categories over 400 pure-noise rows, a full-fit WOE encoding
    memorises each row's own label (disc > 0.75). Refitting inside each fold
    must bring CV's roc_auc back to chance, while the same data pre-encoded
    with the leaky full-fit artifact stays strongly "predictive".
    """
    from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator

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

    def disc(auc: float) -> float:
        return max(auc, 1.0 - auc)

    # Leaky control: full-fit WOE applied to every row before CV.
    leaky_params = WOEEncoderCalculator().fit((X, y), steps[0]["params"])
    X_leaky, _y_leaky = WOEEncoderApplier().apply((X, y), dict(leaky_params))
    leaky = perform_cross_validation(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        X_leaky,
        y,
        config={},
        n_folds=5,
        cv_type="k_fold",
    )
    auc_leaky = leaky["aggregated_metrics"]["roc_auc"]["mean"]

    # F-15 fix: the same WOE step, refit inside every fold via the adapter.
    adapter = FeatureEngineerFoldAdapter(steps, target_column="target")
    refit = perform_cross_validation(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        X,
        y,
        config={},
        n_folds=5,
        cv_type="k_fold",
        preprocessing=adapter,
    )
    auc_refit = refit["aggregated_metrics"]["roc_auc"]["mean"]

    assert disc(auc_leaky) > 0.75, (
        f"expected the leaky encoding to memorise labels, got {auc_leaky}"
    )
    assert disc(auc_refit) < 0.65, f"per-fold refit CV should sit near chance, got {auc_refit}"


def test_stateful_estimator_cross_validate_passes_preprocessing_through() -> None:
    """StatefulEstimator.cross_validate must thread the preprocessor into CV."""
    X, y = _make_classification_xy()
    dataset = SplitDataset(train=(X, y), test=(X.iloc[:10], y.iloc[:10]))
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="lr",
    )
    recorder = RecordingPreprocessor()

    result = estimator.cross_validate(
        dataset, "target", config={}, n_folds=3, preprocessing=recorder
    )

    assert len(result["folds"]) == 3
    assert len(recorder.fit_rows) == 3


def test_no_preprocessing_param_scores_pretransformed_data() -> None:
    """No preprocessing param: CV scores the caller's data as-is (no hook invoked)."""
    X, y = _make_classification_xy()
    result = perform_cross_validation(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        X,
        y,
        config={},
        n_folds=3,
        cv_type="k_fold",
    )
    assert len(result["folds"]) == 3
    assert "accuracy" in result["aggregated_metrics"]
