from typing import Any, Dict, Type, Optional
import pandas as pd
from sklearn.base import BaseEstimator
from .base import BaseModelCalculator, BaseModelApplier

class SklearnCalculator(BaseModelCalculator):
    def __init__(self, model_class: Type[BaseEstimator], default_params: Dict[str, Any], problem_type: str):
        self.model_class = model_class
        self.default_params = default_params
        self._problem_type = problem_type

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], progress_callback=None, validation_data=None) -> Any:
        # 1. Merge Config with Defaults
        params = self.default_params.copy()
        if config:
            # We support two configuration structures:
            # 1. Nested: {'params': {'C': 1.0, ...}} - Preferred
            # 2. Flat: {'C': 1.0, 'type': '...', ...} - Legacy/Simple support
            
            # Check for explicit 'params' dictionary first
            overrides = config.get('params', {})
            
            # If no explicit params found, try to extract from top-level config
            # while filtering out reserved pipeline keys
            if not overrides:
                reserved_keys = {'type', 'target_column', 'node_id'}
                overrides = {k: v for k, v in config.items() if k not in reserved_keys}
            
            params.update(overrides)

        # 2. Instantiate Model
        model = self.model_class(**params)
        
        # 3. Fit
        model.fit(X, y)
        
        return model

class SklearnApplier(BaseModelApplier):
    def predict(self, df: pd.DataFrame, model_artifact: Any) -> pd.Series:
        # model_artifact is the fitted sklearn estimator
        return pd.Series(model_artifact.predict(df), index=df.index)

    def predict_proba(self, df: pd.DataFrame, model_artifact: Any) -> Optional[pd.DataFrame]:
        if hasattr(model_artifact, "predict_proba"):
            try:
                probas = model_artifact.predict_proba(df)
                # Handle binary vs multiclass
                # If binary, classes_ usually has 2 entries.
                classes = model_artifact.classes_
                return pd.DataFrame(probas, columns=classes, index=df.index)
            except Exception:
                return None
        return None
