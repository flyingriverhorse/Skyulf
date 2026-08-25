"""sklearn-compatible preprocessing step for searcher-internal CV (F-15).

``halving_*`` and ``optuna`` tuning strategies run their cross-validation
inside sklearn/optuna searchers, where the engine's per-fold hook cannot
reach. Wrapping preprocessing + model in a ``Pipeline`` lets the searcher's
own folds drive the refit: ``fit_transform`` sees each fold's training rows,
``transform`` sees the held-out rows — the same discipline the custom
grid/random loop applies.

Because a transformer step can only hand ``X`` to the next step (``y`` is
threaded through unchanged by the Pipeline), this wrap is only valid for
preprocessors that keep the rows and the target aligned. The tuning engine
refuses the wrap — falling back to pre-transformed scoring with an explicit
log — when the chain resamples, drops rows, or re-encodes the target: a
static ``changes_row_count`` flag is the cheap fast path, and a runtime
alignment probe (fit_transform on a small slice, failing closed) is the
authoritative check so future step types cannot drift past it.
"""

import copy
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin


class FoldPreprocessingStep(BaseEstimator, TransformerMixin):
    """Pipeline step that refits a ``FoldPreprocessor`` on every ``fit``.

    The preprocessor is deep-copied at fit time, so clones made by the
    searcher (one per candidate) never share fitted state — safe even when
    the searcher parallelises candidates with ``n_jobs > 1``.

    ``fit_transform`` returns the refit output directly instead of letting
    ``TransformerMixin`` re-transform the training rows: row-count-changing
    steps (resampling, row drops) must shape the data the model trains on,
    while ``transform`` keeps every held-out row (the F-18 discipline).
    """

    def __init__(self, preprocessor: Any) -> None:
        self.preprocessor = preprocessor

    def fit(self, X: Any, y: Any = None) -> "FoldPreprocessingStep":
        worker = copy.deepcopy(self.preprocessor)
        worker.fit_transform(X, y)
        self.preprocessor_ = worker
        return self

    def fit_transform(self, X: Any, y: Any = None, **_fit_params: Any) -> Any:
        worker = copy.deepcopy(self.preprocessor)
        transformed, _y = worker.fit_transform(X, y)
        self.preprocessor_ = worker
        return transformed

    def transform(self, X: Any) -> Any:
        transformed, _y = self.preprocessor_.transform(X, None)
        return transformed
