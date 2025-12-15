"""Wrapper for Scikit-Learn models."""

import logging
from typing import Any, Dict, Optional, Type

import pandas as pd
from sklearn.base import BaseEstimator

from .base import BaseModelApplier, BaseModelCalculator

logger = logging.getLogger(__name__)


class SklearnCalculator(BaseModelCalculator):
    """Base calculator for Scikit-Learn models."""

    def __init__(
        self,
        model_class: Type[BaseEstimator],
        default_params: Dict[str, Any],
        problem_type: str,
    ):
        self.model_class = model_class
        self._default_params = default_params
        self._problem_type = problem_type

    @property
    def default_params(self) -> Dict[str, Any]:
        return self._default_params

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        progress_callback=None,
        log_callback=None,
        validation_data=None,
    ) -> Any:
        """Fit the Scikit-Learn model."""
        # 1. Merge Config with Defaults
        params = self.default_params.copy()
        if config:
            # We support two configuration structures:
            # 1. Nested: {'params': {'C': 1.0, ...}} - Preferred
            # 2. Flat: {'C': 1.0, 'type': '...', ...} - Legacy/Simple support

            # Check for explicit 'params' dictionary first
            overrides = config.get("params", {})

            # If 'params' key exists but is None or empty, check if there are other keys at top level
            # that might be params. But be careful not to mix them.
            # If config has 'params', we assume it's the source of truth.

            if not overrides and "params" not in config:
                # Fallback to flat config if 'params' key is completely missing
                reserved_keys = {
                    "type",
                    "target_column",
                    "node_id",
                    "step_type",
                    "inputs",
                }
                overrides = {
                    k: v
                    for k, v in config.items()
                    if k not in reserved_keys and not isinstance(v, dict)
                }

            if overrides:
                params.update(overrides)

        msg = f"Initializing {self.model_class.__name__} with params: {params}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)

        # 2. Instantiate Model
        # Filter params to only those accepted by the model class
        # This prevents errors if extra config is passed
        # (Though sklearn usually ignores extra kwargs in __init__ if **kwargs is present,
        # strict models might not)
        # For now, we assume the user/config provides valid params.

        # Handle special cases like 'random_state' if needed, but usually passed directly.

        model = self.model_class(**params)

        # 3. Fit
        model.fit(X, y)

        return model


class SklearnApplier(BaseModelApplier):
    """Base applier for Scikit-Learn models."""

    def predict(self, df: pd.DataFrame, model_artifact: Any) -> pd.Series:
        """Generate predictions."""
        # model_artifact is the fitted sklearn estimator
        return pd.Series(model_artifact.predict(df), index=df.index)

    def predict_proba(
        self, df: pd.DataFrame, model_artifact: Any
    ) -> Optional[pd.DataFrame]:
        """Generate prediction probabilities."""
        if hasattr(model_artifact, "predict_proba"):
            try:
                probas = model_artifact.predict_proba(df)
                # Handle binary vs multiclass
                # If binary, classes_ usually has 2 entries.
                classes = getattr(model_artifact, "classes_", None)
                if classes is None:
                    # Fallback if classes_ is missing (unlikely for sklearn classifiers)
                    return pd.DataFrame(probas, index=df.index)

                return pd.DataFrame(probas, columns=classes, index=df.index)
            except Exception:
                return None
        return None
