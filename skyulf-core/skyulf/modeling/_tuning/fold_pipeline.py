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

import copy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier


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
    """

    def __init__(
        self,
        estimator: Any = None,
        preprocessor: Any = None,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        """Store the wrap targets verbatim; nothing is copied or fitted here.

        Plain assignment with ``None`` defaults is what sklearn's
        ``clone``/``get_params`` machinery requires — deep-copying is
        deferred to ``fit`` so every searcher clone gets its own fitted
        state. ``feature_names`` records the column contract used to
        rebuild named frames when the searcher hands back plain arrays.
        """
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.feature_names = feature_names

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
    def _build_label_map(y_orig: Any, y_t: Any, model: Any) -> dict[Any, Any] | None:
        """Map encoded labels back to the original space, when encoding happened.

        Built from paired uniques over the whole fold — full coverage, no
        sampling risk for rare classes. ``None`` for regressors and for
        targets the chain left untouched.
        """
        if not is_classifier(model):
            return None
        orig = np.asarray(y_orig)
        enc = np.asarray(y_t)
        if orig.shape == enc.shape:
            if np.array_equal(orig, enc):
                return None
            pairs = pd.unique(pd.Series(list(zip(orig.tolist(), enc.tolist(), strict=True))))
            return {encoded: original for original, encoded in pairs}
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

    def fit(self, X: Any, y: Any = None) -> "FoldAwareModelStep":
        """Fit preprocessing and model on this fold's training rows only.

        Preprocessor and estimator are deep-copied first, so clones made by
        the searcher never share fitted state across folds or parallel
        workers — this is what makes row-resampling and target-re-encoding
        chains leakage-free inside the searcher's own CV. A label map is
        built when the chain re-encoded ``y``, for ``predict`` to invert.
        """
        X, y = self._ensure_frames(X, y)
        worker = copy.deepcopy(self.preprocessor)
        model = copy.deepcopy(self.estimator)
        X_t, y_t = worker.fit_transform(X, y)
        model.fit(X_t, y_t)
        self.preprocessor_ = worker
        self.model_ = model
        self.label_map_ = self._build_label_map(y, y_t, model)
        return self

    def _transform_x(self, X: Any) -> Any:
        X, _y = self._ensure_frames(X, None)
        X_t, _y_t = self.preprocessor_.transform(X, None)
        return X_t

    def predict(self, X: Any) -> Any:
        """Predict with the fitted model on X run through the fitted chain.

        Predictions made in an encoded label space are mapped back through
        the fit-time label map so the searcher's scorer compares against
        the untouched ``y`` it holds.
        """
        pred = self.model_.predict(self._transform_x(X))
        if self.label_map_ is not None:
            pred = pd.Series(np.asarray(pred)).map(self.label_map_).to_numpy()
        return pred

    def predict_proba(self, X: Any) -> Any:
        """Class probabilities from the fitted model over the transformed X.

        Probabilities need no remapping: their columns already align with
        the mapped-back ``classes_`` property, so scorers see a consistent
        label space.
        """
        return self.model_.predict_proba(self._transform_x(X))

    @property
    def classes_(self) -> Any:
        """The base model's classes mapped back to the original label space."""
        classes = self.model_.classes_
        if self.label_map_ is None:
            return classes
        return np.array([self.label_map_.get(c, c) for c in np.asarray(classes).tolist()])
