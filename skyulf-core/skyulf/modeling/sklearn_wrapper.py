"""Wrapper for Scikit-Learn models."""

import logging
from typing import Any, Dict, Optional, Type, Union

import pandas as pd
from sklearn.base import BaseEstimator

from ..engines import SkyulfDataFrame
from ..engines.sklearn_bridge import SklearnBridge
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
        X: Union[pd.DataFrame, SkyulfDataFrame],
        y: Union[pd.Series, Any],
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
        model = self.model_class(**params)

        # 3. Fit
        # Convert to Numpy using Bridge (handles Polars/Pandas/Wrappers)
        X_np, y_np = SklearnBridge.to_sklearn((X, y))
        
        model.fit(X_np, y_np)

        return model


class SklearnApplier(BaseModelApplier):
    """Base applier for Scikit-Learn models."""
    
    def predict(self, df: Union[pd.DataFrame, SkyulfDataFrame], model_artifact: Any) -> Any:
        # Convert to Numpy
        X_np, _ = SklearnBridge.to_sklearn(df)
        
        preds = model_artifact.predict(X_np)
        
        # Return as Pandas Series for consistency
        # If input was Pandas, try to preserve index
        index = None
        if hasattr(df, "index"):
            index = df.index
        elif hasattr(df, "to_pandas"):
             # If it's a wrapper or Polars, we might lose index unless we convert
             # For now, default index is acceptable for predictions
             pass
             
        return pd.Series(preds, index=index)

    def predict_proba(
        self, df: Union[pd.DataFrame, SkyulfDataFrame], model_artifact: Any
    ) -> Optional[Any]:
        if not hasattr(model_artifact, "predict_proba"):
            return None
            
        X_np, _ = SklearnBridge.to_sklearn(df)
        probs = model_artifact.predict_proba(X_np)
        
        # Return as DataFrame
        index = None
        if hasattr(df, "index"):
            index = df.index
            
        # Column names usually 0, 1, etc. or classes_
        columns = None
        if hasattr(model_artifact, "classes_"):
            columns = model_artifact.classes_
            
        return pd.DataFrame(probs, index=index, columns=columns)
