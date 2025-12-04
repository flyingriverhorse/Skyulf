from typing import Any, Dict, Type
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
            # Filter config to only include valid params for the model? 
            # Or assume config only contains model params?
            # Usually config might contain 'type', 'target_column' etc.
            # We should probably look for a 'params' key in config, or just use the whole config 
            # but exclude known non-param keys.
            # For now, let's assume config['params'] holds the model hyperparameters if structured that way,
            # or just mix them in.
            # The V1 registry had default_params.
            # Let's assume config passed here IS the model configuration.
            # We'll exclude keys that are definitely not for the model if we find any.
            
            # Better approach: The config passed to fit() is likely the node config.
            # It might look like {'type': 'logistic_regression', 'params': {'C': 1.0}, ...}
            # Or it might be flat: {'type': 'logistic_regression', 'C': 1.0}
            
            # Let's assume a 'params' key exists for explicit overrides, 
            # or we use the top level config but filter out 'type', 'target_column'.
            
            overrides = config.get('params', {})
            # If config is flat, we might want to support that too, but 'params' dict is cleaner.
            # Let's support both: explicit 'params' dict takes precedence.
            
            if not overrides:
                # Try to use top-level keys that are not reserved
                reserved = {'type', 'target_column', 'node_id'}
                overrides = {k: v for k, v in config.items() if k not in reserved}
            
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
