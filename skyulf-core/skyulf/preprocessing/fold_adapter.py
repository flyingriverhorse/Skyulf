"""Adapter exposing ``FeatureEngineer`` as a :class:`FoldPreprocessor` (F-15).

Cloning is free here by construction: ``FeatureEngineer`` rebuilds every
step's calculator/applier fresh from ``steps_config`` via the node registry
on each ``fit_transform``, so constructing a new engineer per fold *is* the
clone operation — no fitted state is ever copied or reset.
"""

from typing import Any

from ..registry import NodeRegistry
from .pipeline import FeatureEngineer

# Splitter steps already ran upstream of the CV/tuning boundary; re-running
# them inside a fold would re-split the fold itself.
SPLITTER_STEP_TYPES = frozenset({"TrainTestSplitter", "Split", "feature_target_split"})


class FeatureEngineerFoldAdapter:
    """Re-runs a preprocessing step chain inside each CV/tuning fold.

    Args:
        steps_config: The upstream Feature-Engineering node's step list
            (plain dicts, validated by ``validate_preprocessing_steps``).
            Splitter steps are filtered out automatically.
        target_column: Name of the target, kept to reject payloads where X
            still embeds it — target-aware steps (target encoders, resampling,
            label encoding of the target) receive ``y`` through the ``(X, y)``
            payload the same way they do downstream of a real splitter.
    """

    def __init__(self, steps_config: list[dict[str, Any]], target_column: str):
        self._steps_config = [
            step for step in steps_config if step.get("transformer") not in SPLITTER_STEP_TYPES
        ]
        self._target_column = target_column
        # Validate eagerly (unknown transformer names, bad params) so a
        # misconfigured chain fails at construction, not mid-fold.
        # validate_preprocessing_steps does not check registry membership,
        # so resolve each step's calculator explicitly.
        FeatureEngineer(self._steps_config)
        for step in self._steps_config:
            NodeRegistry.get_calculator(step["transformer"])
        self._engineer: FeatureEngineer | None = None

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self._validate_payload(X)
        engineer = FeatureEngineer(self._steps_config)
        transformed, _metrics = engineer.fit_transform((X, y))
        self._engineer = engineer
        return transformed

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        if self._engineer is None:
            raise RuntimeError("transform() called before fit_transform()")
        self._validate_payload(X)
        # At transform/inference time no step needs the target (target-aware
        # encoders use their fitted artifact; splitters/resampling are skipped).
        # Feed a bare frame when y is absent — a ``(X, None)`` tuple would trip
        # ``pack_pipeline_output``'s tuple-shape-lost diagnostic on every step.
        payload = (X, y) if y is not None else X
        transformed = self._engineer.transform(payload)
        # Some appliers return a bare frame instead of the ``(X, y)`` payload;
        # re-pair so callers always get the FoldPreprocessor protocol shape.
        if not (isinstance(transformed, tuple) and len(transformed) == 2):
            transformed = (transformed, y)
        return transformed

    def _validate_payload(self, X: Any) -> None:
        if hasattr(X, "columns") and self._target_column in X.columns:
            raise ValueError(f"target column '{self._target_column}' already present in X")
