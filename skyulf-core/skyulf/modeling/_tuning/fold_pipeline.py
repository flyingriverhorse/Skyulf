"""Fold-aware estimator giving searcher-backed tuning leakage-free folds (F-15).

``halving_*`` and ``optuna`` tuning strategies run their cross-validation
inside sklearn/optuna searchers, where the engine's per-fold hook cannot
reach. The searcher only requires ``fit(X, y)`` / ``predict(X)``, so this
module wraps preprocessing + model together in one fit-time meta-estimator:
the searcher's internal CV drives a true per-fold refit, and chains that
change the row count (SMOTE/resampling, row drops, outlier removal) or the
target (label encoding) run safely inside ``fit`` on the fold's training
rows only.

The tuning engine wraps the step as::

    Pipeline([("model", FoldAwareModelStep(estimator=base, preprocessor=adapter))])

and routes the search space through ``model__estimator__<param>``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils.metaestimators import available_if

from ...core.validation import validate_prediction_rows
from ...engines.sklearn_bridge import SklearnBridge
from .._class_weights import sample_weight_for_fit
from .._cv_weights import fit_preprocessor, prepare_weights
from .._sample_weights import SampleWeightError


class FatalSampleWeightError(BaseException):
    """Carry weight failures through sklearn's candidate-level Exception handler.

    This private search boundary signal is restored to SampleWeightError by the
    runner, including after joblib transports it from a parallel worker.
    """


def _fitted_model_has(attr: str) -> Callable[[Any], bool]:
    """Build the ``available_if`` predicate for one response method of the wrapped model.

    Reads the fitted copy when there is one and the constructor argument otherwise,
    so ``hasattr`` answers correctly both before ``fit`` — when the searcher clones
    the step — and after it, when a scorer asks. Letting the inner ``getattr`` raise
    is what keeps the resulting ``AttributeError`` naming the estimator that really
    lacks the method.
    """

    def _check(self: Any) -> bool:
        getattr(getattr(self, "model_", self.estimator), attr)
        return True

    return _check


class FoldAwareModelStep(BaseEstimator):
    """Fit-time meta-estimator owning preprocessing + model together.

    Both the preprocessor and the base estimator are deep-copied at fit
    time, so clones made by the searcher (one per candidate/fold) never
    share fitted state — safe even when the searcher parallelises
    candidates with ``n_jobs > 1``.

    When the preprocessor re-encodes the target (e.g. string labels to
    integers), predictions and ``classes_`` are mapped back to the original
    label space so scorers compare against the untouched ``y`` the searcher
    holds; ``predict_proba`` columns stay aligned with ``classes_``.

    ``class_weight`` carries nonnative weighting separately from estimator
    parameters. It is converted after preprocessing, using only the labels
    passed to this fold's fit. The preprocessor may be absent when only
    class-weight conversion is needed.
    """

    def __init__(
        self,
        estimator: Any = None,
        preprocessor: Any = None,
        feature_names: tuple[str, ...] | None = None,
        class_weight: Any = None,
        propagate_weight_errors: bool = False,
    ) -> None:
        """Store the wrap targets verbatim; nothing is copied or fitted here.

        Plain assignment with ``None`` defaults is what sklearn's
        ``clone``/``get_params`` machinery requires — deep-copying is
        deferred to ``fit`` so every searcher clone gets its own fitted
        state. ``feature_names`` records the column contract used to
        rebuild named frames when the searcher hands back plain arrays.
        ``class_weight`` is kept cloneable and evaluated only inside fit.
        """
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.class_weight = class_weight
        self.propagate_weight_errors = propagate_weight_errors

    def _ensure_frames(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Rebuild named pandas frames when slicing hands non-pandas input.

        Polars input converts through ``to_pandas`` so dtypes survive — the
        ``np.asarray`` fallback would collapse mixed-type frames to object
        dtype and silently disable numeric steps (e.g. SimpleImputer). The
        plain-array path restores column names from the contract captured at
        construction.
        """
        if not hasattr(X, "iloc"):
            if hasattr(X, "to_pandas"):
                X = X.to_pandas()
            else:
                columns = pd.Index(self.feature_names) if self.feature_names else None
                X = pd.DataFrame(np.asarray(X), columns=columns)
        if y is not None and not hasattr(y, "iloc"):
            if hasattr(y, "to_pandas"):
                y = y.to_pandas()
            else:
                y = pd.Series(np.asarray(y), index=X.index)
        return X, y

    @staticmethod
    def _build_label_map(
        y_orig: Any, y_t: Any, model: Any, preprocessor: Any = None
    ) -> dict[Any, Any] | None:
        """Map encoded labels back to the original space, when encoding happened.

        Prefer the fitted encoder artifacts, which survive resampling and
        temporal permutations. Generic adapters fall back to paired uniques
        only when that produces an unambiguous bijection.
        """
        if not is_classifier(model):
            return None
        orig = np.asarray(y_orig)
        enc = np.asarray(y_t)
        decode = getattr(preprocessor, "original_target_labels", None)
        if decode is not None:
            labels = np.unique(enc)
            decoded = decode(labels)
            if decoded is not None:
                return (
                    None
                    if np.array_equal(labels, decoded)
                    else dict(zip(labels.tolist(), np.asarray(decoded).tolist(), strict=True))
                )
        if orig.shape == enc.shape:
            if np.array_equal(orig, enc):
                return None
            return _paired_label_map(orig, enc)
        # Row-count-changing chains (resampling) make row-wise pairing
        # impossible; unchanged value spaces need no map.
        if set(np.unique(orig).tolist()) == set(np.unique(enc).tolist()):
            return None
        raise ValueError(
            "The preprocessing chain both changed the row count and re-encoded "
            "the target; the original label space cannot be reconstructed. "
            "Move target encoding out of the resampled chain."
        )

    def __sklearn_tags__(self):
        """Propagate the wrapped model's tags so sklearn sees its estimator type.

        A bare ``BaseEstimator`` tags as neither classifier nor regressor,
        so ``predict_proba``-based scorers would refuse this step inside a
        searcher; copying the model's estimator/classifier/regressor/target
        tags keeps the wrap invisible to sklearn's response-method
        machinery.
        """
        # Propagate the wrapped model's estimator type so sklearn's
        # response-method machinery (scorers, Pipeline delegation) sees a
        # classifier as a classifier — a bare BaseEstimator tags as neither
        # and predict_proba scorers would refuse it.
        tags = super().__sklearn_tags__()
        if self.estimator is not None and hasattr(self.estimator, "__sklearn_tags__"):
            model_tags = self.estimator.__sklearn_tags__()
            tags.estimator_type = model_tags.estimator_type
            tags.classifier_tags = model_tags.classifier_tags
            tags.regressor_tags = model_tags.regressor_tags
            tags.target_tags = model_tags.target_tags
        return tags

    def fit(self, X: Any, y: Any = None, *, sample_weight: Any = None) -> FoldAwareModelStep:
        """Fit preprocessing and model on this fold's training rows only.

        Preprocessor and estimator are deep-copied first, so clones made by
        the searcher never share fitted state across folds or parallel
        workers — this is what makes row-resampling and target-re-encoding
        chains leakage-free inside the searcher's own CV. A label map is
        built when the chain re-encoded ``y``, for ``predict`` to invert.
        """
        try:
            return self._fit(X, y, sample_weight=sample_weight)
        except SampleWeightError as exc:
            if self.propagate_weight_errors:
                raise FatalSampleWeightError(str(exc)) from exc
            raise

    def _fit(self, X: Any, y: Any, *, sample_weight: Any) -> FoldAwareModelStep:
        """Apply the fold's preprocessing and effective weights before fitting."""
        sample_weight = prepare_weights(sample_weight, len(X), self.preprocessor)
        if self.preprocessor is not None:
            X, y = self._ensure_frames(X, y)
        worker = copy.deepcopy(self.preprocessor)
        model = copy.deepcopy(self.estimator)
        X_t, y_t, sample_weight = fit_preprocessor(worker, X, y, sample_weight)
        SklearnBridge.validate_features(X_t)
        sample_weight = sample_weight_for_fit(model, self.class_weight, y_t, sample_weight)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_t, y_t, **fit_kwargs)
        self.preprocessor_ = worker
        self.model_ = model
        self.label_map_ = self._build_label_map(y, y_t, model, worker)
        return self

    def _transform_x(self, X: Any) -> Any:
        if self.preprocessor_ is None:
            SklearnBridge.validate_features(X)
            return X, None
        X, _y = self._ensure_frames(X, None)
        tracker = getattr(self.preprocessor_, "transform_tracking_order", None)
        if tracker is not None:
            X_t, positions = tracker(X)
        else:
            X_t, _y_t = self.preprocessor_.transform(X, None)
            positions = None
        SklearnBridge.validate_features(X_t)
        validate_prediction_rows(len(X), len(X_t), stage="Fold preprocessing")
        return X_t, positions

    def _response(self, method: str, X: Any) -> Any:
        """Restore model responses to caller order after local temporal preprocessing."""
        transformed, positions = self._transform_x(X)
        response = getattr(self.model_, method)(transformed)
        validate_prediction_rows(len(X), len(response), stage="Fold model response")
        return response if positions is None else np.asarray(response)[np.argsort(positions)]

    def predict(self, X: Any) -> Any:
        """Predict with the fitted model on X run through the fitted chain.

        Predictions made in an encoded label space are mapped back through
        the fit-time label map so the searcher's scorer compares against
        the untouched ``y`` it holds.
        """
        pred = self._response("predict", X)
        if self.label_map_ is not None:
            pred = pd.Series(np.asarray(pred)).map(self.label_map_).to_numpy()
        return pred

    @available_if(_fitted_model_has("predict_proba"))
    def predict_proba(self, X: Any) -> Any:
        """Align probability columns with canonical original-label class order."""
        probabilities = self._response("predict_proba", X)
        return np.asarray(probabilities)[:, np.argsort(self._mapped_classes())]

    @available_if(_fitted_model_has("decision_function"))
    def decision_function(self, X: Any) -> Any:
        """Orient margins toward the same original classes as probability scorers."""
        scores = self._response("decision_function", X)
        order = np.argsort(self._mapped_classes())
        if len(order) == 2:
            return -scores if order[0] != 0 else scores
        if getattr(self.model_, "decision_function_shape", None) == "ovo":
            return _reorder_pairwise_scores(np.asarray(scores), order)
        return np.asarray(scores)[:, order]

    @property
    def classes_(self) -> Any:
        """Canonical original-label order expected by sklearn probability scorers."""
        return np.sort(self._mapped_classes())

    def _mapped_classes(self) -> np.ndarray:
        """Decode classes while retaining the fitted model's response column order."""
        classes = self.model_.classes_
        if self.label_map_ is None:
            return classes
        return np.array([self.label_map_.get(c, c) for c in np.asarray(classes).tolist()])


def _reorder_pairwise_scores(scores: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Preserve SVC's one-versus-one class pairs and margin direction after decoding."""
    pairs = [(i, j) for i in range(len(order)) for j in range(i + 1, len(order))]
    result = []
    for i, j in pairs:
        left, right = order[i], order[j]
        column = pairs.index((min(left, right), max(left, right)))
        result.append(scores[:, column] if left < right else -scores[:, column])
    return np.column_stack(result)


def _paired_label_map(original: np.ndarray, encoded: np.ndarray) -> dict[Any, Any]:
    """Reject ambiguous generic adapters instead of silently choosing the last row's label."""
    pairs = pd.unique(pd.Series(list(zip(original.tolist(), encoded.tolist(), strict=True))))
    mapping = {code: label for label, code in pairs}
    if len(mapping) != len(pairs) or len(set(mapping.values())) != len(mapping):
        raise ValueError(
            "Ambiguous target mapping; preprocessing must expose original_target_labels."
        )
    return mapping
