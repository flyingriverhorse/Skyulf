"""Per-fold preprocessing refit contract (F-15).

Cross-validation and hyperparameter tuning re-fit only the *model* per fold
today; any preprocessing fitted upstream on the full training split leaks
held-out rows into the fitted statistics. Callers that can re-run their
preprocessing per fold pass an object satisfying :class:`FoldPreprocessor`
to ``perform_cross_validation`` / the tuning engine, and CV fits it on each
fold's training rows only before scoring on the held-out rows.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FoldPreprocessor(Protocol):
    """Structural contract for per-fold preprocessing.

    Implementations must be re-fittable: ``fit_transform`` may be called
    repeatedly (once per fold), and each call must discard any state from
    the previous fold. ``transform`` applies the artifacts learned by the
    most recent ``fit_transform`` to held-out rows without learning from
    them. ``y`` is passed alongside ``X`` because target-aware steps
    (target encoders, label encoders fitted on the target, resampling)
    need it at fit time and may transform it at apply time.
    """

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Fit on this fold's training rows only; return transformed (X, y)."""
        ...

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Apply the fitted artifacts to held-out rows without refitting."""
        ...
