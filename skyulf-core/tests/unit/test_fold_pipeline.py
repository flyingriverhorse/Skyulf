"""Unit tests for ``FoldAwareModelStep`` — the fit-time meta-estimator that
gives ``halving_*``/``optuna`` tuning leakage-free per-fold refits, including
chains that change the row count or the target (F-15 follow-up)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep


class LabelEncodingAdapter:
    """Encodes a string target to ints (sorted-class order, like the real
    LabelEncoder node); ``transform`` encodes held-out rows the same way."""

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        classes = np.unique(np.asarray(y))
        self.mapping_ = {c: i for i, c in enumerate(classes)}
        return X, y.map(self.mapping_)

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y.map(self.mapping_) if y is not None else y


class SwapEncodingAdapter:
    """Encodes a numeric target with a non-trivial permutation (0 <-> 1).

    The original space stays numeric so scorers with a hardcoded
    ``pos_label=1`` (sklearn's default ``f1``) still run, while the label
    map is genuinely exercised.
    """

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y.map({0: 1, 1: 0})

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y.map({0: 1, 1: 0}) if y is not None else y


class RowDoublingAdapter:
    """SMOTE-style stub: fit_transform doubles the training rows."""

    def __init__(self) -> None:
        self.fit_input_rows: list[int] = []
        self.fit_output_rows: list[int] = []

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.fit_input_rows.append(len(X))
        X_out = pd.concat([X, X], ignore_index=True)
        y_out = pd.concat([pd.Series(np.asarray(y)), pd.Series(np.asarray(y))], ignore_index=True)
        self.fit_output_rows.append(len(X_out))
        return X_out, y_out

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y


class CountingAdapter:
    """Counts fit_transform calls; deep-copy isolation probe."""

    def __init__(self) -> None:
        self.fits = 0

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.fits += 1
        return X, y

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y


class IdentityAdapter:
    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y


def _xy_str(n: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) + 0.5})
    y = pd.Series(["neg" if i % 2 == 0 else "pos" for i in range(n)], name="Species")
    return X, y


def _xy_int(n: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) + 0.5})
    y = pd.Series([i % 2 for i in range(n)], name="target")
    return X, y


# ---------------------------------------------------------------------------
# Label-space round trip
# ---------------------------------------------------------------------------


def test_predict_returns_original_space_labels() -> None:
    """A target-encoding chain must not leak encoded labels into predictions:
    predict maps back to the original space scorers compare against."""
    X, y = _xy_str()
    step = FoldAwareModelStep(estimator=LogisticRegression(), preprocessor=LabelEncodingAdapter())

    step.fit(X, y)
    preds = step.predict(X)

    assert set(np.unique(preds)) <= {"neg", "pos"}
    assert step.label_map_ == {0: "neg", 1: "pos"}


def test_classes_and_proba_align_with_original_space() -> None:
    """``classes_`` lives in the original label space and ``predict_proba``
    columns align with it — what roc_auc-style scorers depend on."""
    X, y = _xy_str()
    step = FoldAwareModelStep(estimator=LogisticRegression(), preprocessor=LabelEncodingAdapter())
    step.fit(X, y)

    assert list(step.classes_) == ["neg", "pos"]
    proba = step.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # Column order follows classes_: the "pos" column scores the pos rows higher.
    pos_col = list(step.classes_).index("pos")
    assert proba[:5, pos_col].mean() < proba[5:10, pos_col].mean()


def test_untouched_target_builds_no_label_map() -> None:
    X, y = _xy_int()
    step = FoldAwareModelStep(estimator=LogisticRegression(), preprocessor=IdentityAdapter())
    step.fit(X, y)
    assert step.label_map_ is None
    assert list(step.classes_) == [0, 1]


# ---------------------------------------------------------------------------
# Row-changing chains
# ---------------------------------------------------------------------------


def test_row_changing_chain_trains_on_more_rows_and_predicts_all() -> None:
    """SMOTE-style chains shape the training rows only: fit sees more rows
    than the input fold, predict keeps every held-out row."""
    X, y = _xy_int()
    adapter = RowDoublingAdapter()
    step = FoldAwareModelStep(estimator=LogisticRegression(), preprocessor=adapter)

    step.fit(X, y)
    # fit works on a deep copy of the adapter (clone discipline); the worker
    # carries the recorded row counts.
    worker = step.preprocessor_
    assert worker.fit_input_rows == [len(X)]
    assert worker.fit_output_rows == [2 * len(X)]

    assert len(step.predict(X)) == len(X)


# ---------------------------------------------------------------------------
# Clone / parallel safety
# ---------------------------------------------------------------------------


def test_step_is_sklearn_cloneable() -> None:
    step = FoldAwareModelStep(estimator=LogisticRegression(), preprocessor=CountingAdapter())
    cloned = clone(step)
    assert cloned is not step
    assert isinstance(cloned, FoldAwareModelStep)


def test_two_clones_never_share_fitted_state() -> None:
    """Searcher clones share one constructor preprocessor/estimator; each fit
    must work on its own deep copies so ``n_jobs > 1`` candidates are safe."""
    X, y = _xy_int()
    original_adapter = CountingAdapter()
    original_estimator = LogisticRegression()
    step = FoldAwareModelStep(estimator=original_estimator, preprocessor=original_adapter)

    c1, c2 = clone(step), clone(step)
    c1.fit(X.iloc[:20], y.iloc[:20])
    c2.fit(X.iloc[10:], y.iloc[10:])

    assert original_adapter.fits == 0  # workers were deep copies
    assert c1.preprocessor_ is not c2.preprocessor_
    assert c1.model_ is not c2.model_
    assert not hasattr(original_estimator, "coef_")  # constructor model never fitted


class ColumnSpyAdapter:
    """Records the column names fit_transform receives."""

    def __init__(self) -> None:
        self.columns: list[Any] = []

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.columns.append(list(X.columns))
        return X, y

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y


def test_array_input_rebuilds_frames_from_column_contract() -> None:
    """If searcher slicing ever strips the frames, the step rebuilds named
    frames from the constructor's column contract — the preprocessor still
    sees real column names."""
    X, y = _xy_int()
    spy = ColumnSpyAdapter()
    step = FoldAwareModelStep(
        estimator=LogisticRegression(),
        preprocessor=spy,
        feature_names=tuple(map(str, X.columns)),
    )

    step.fit(X.to_numpy(), y.to_numpy())

    assert step.preprocessor_.columns == [["a", "b"]]
    assert len(step.predict(X.to_numpy())) == len(X)


# ---------------------------------------------------------------------------
# Regression passthrough
# ---------------------------------------------------------------------------


def test_regressor_has_no_label_map_or_classes() -> None:
    X = pd.DataFrame({"a": np.arange(30, dtype=float)})
    y = pd.Series(np.linspace(0, 1, 30), name="target")
    step = FoldAwareModelStep(estimator=LinearRegression(), preprocessor=IdentityAdapter())

    step.fit(X, y)
    assert step.label_map_ is None
    with pytest.raises(AttributeError):
        _ = step.classes_
    assert len(step.predict(X)) == len(X)


# ---------------------------------------------------------------------------
# Inside a real searcher: scorer alignment through f1 and roc_auc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scoring", "adapter_cls"),
    [
        # roc_auc aligns via classes_ and works on the original string labels.
        ("roc_auc", LabelEncodingAdapter),
        # f1's default pos_label=1 forbids non-numeric labels, so exercise the
        # label map with a numeric original space instead.
        ("f1", SwapEncodingAdapter),
    ],
)
def test_searcher_scorers_align_with_original_label_space(scoring: str, adapter_cls) -> None:
    """Through a real searcher's CV loop: predictions, ``classes_`` and proba
    columns stay aligned with the original labels the scorer sees."""
    X, y = _xy_str(n=60)
    if adapter_cls is SwapEncodingAdapter:
        y = y.map({"neg": 0, "pos": 1})
    pipe = Pipeline(
        [
            (
                "model",
                FoldAwareModelStep(estimator=LogisticRegression(), preprocessor=adapter_cls()),
            )
        ]
    )

    search = GridSearchCV(pipe, {"model__estimator__C": [0.1, 1.0]}, cv=3, scoring=scoring)
    search.fit(X, y)

    assert search.best_score_ > 0
    best = search.best_estimator_
    preds = best.predict(X)
    assert set(np.unique(preds)) <= set(np.unique(y))
    assert set(np.asarray(best.classes_).tolist()) == set(np.unique(y).tolist())
