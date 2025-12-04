"""Hyperparameter Tuner implementation."""

from typing import Any, Dict, List, Optional, Callable
import logging
import numpy as np
import pandas as pd
# Explicitly enable experimental halving search cv
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    HalvingGridSearchCV, 
    HalvingRandomSearchCV,
    KFold, 
    StratifiedKFold
)

from ..base import BaseModelCalculator, BaseModelApplier
from .schemas import TuningConfig, TuningResult

logger = logging.getLogger(__name__)

# Try importing Optuna with robust fallback for integration packages
HAS_OPTUNA = False
OptunaSearchCV = None

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    pass

if HAS_OPTUNA:
    try:
        from optuna.integration import OptunaSearchCV as _OptunaSearchCV
        OptunaSearchCV = _OptunaSearchCV
    except ImportError:
        try:
            from optuna.integration.sklearn import OptunaSearchCV as _OptunaSearchCV
            OptunaSearchCV = _OptunaSearchCV
        except ImportError:
            try:
                from optuna_integration.sklearn import OptunaSearchCV as _OptunaSearchCV
                OptunaSearchCV = _OptunaSearchCV
            except ImportError:
                HAS_OPTUNA = False
                logger.warning("Optuna installed but OptunaSearchCV not found. Install 'optuna-integration'.")

from typing import Any, Dict, List, Optional, Callable

# ... imports ...

class TunerCalculator:
    def __init__(self, model_calculator: BaseModelCalculator):
        self.model_calculator = model_calculator

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any], 
        progress_callback: Optional[Callable[[int, int], None]] = None,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None
    ) -> Any:
        """
        Fits the tuner (runs tuning).
        Adapts the generic fit interface to the specific tune method.
        """
        # Convert config dict to TuningConfig
        if isinstance(config, TuningConfig):
            tuning_config = config
        else:
            # Extract valid keys for TuningConfig
            valid_keys = TuningConfig.__annotations__.keys()
            filtered_config = {k: v for k, v in config.items() if k in valid_keys}
            tuning_config = TuningConfig(**filtered_config)
            
        return self.tune(X, y, tuning_config, progress_callback=progress_callback, validation_data=validation_data)

    def tune(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: TuningConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None
    ) -> TuningResult:
        """
        Runs hyperparameter tuning.
        """
        # 1. Prepare Estimator
        # We need a base estimator. Since our Calculator wraps the class, 
        # we need to instantiate the underlying sklearn model with default params.
        # Assuming model_calculator is SklearnCalculator
        if not hasattr(self.model_calculator, "model_class"):
             raise ValueError("Tuner currently only supports SklearnCalculator")
        
        base_estimator = self.model_calculator.model_class(**self.model_calculator.default_params)
        
        # 2. Prepare Splitter
        # If validation data is provided, use PredefinedSplit to train on X and validate on validation_data
        # Otherwise use CV
        
        X_for_search = X
        y_for_search = y
        
        if validation_data is not None:
            from sklearn.model_selection import PredefinedSplit
            X_val, y_val = validation_data
            
            # Concatenate Train and Val
            X_for_search = pd.concat([X, X_val], axis=0)
            y_for_search = pd.concat([y, y_val], axis=0)
            
            # Create test_fold array: -1 for train, 0 for val
            # -1 means "never in test set" (so always in training set)
            # 0 means "in test set for fold 0"
            test_fold = np.concatenate([
                np.full(len(X), -1),
                np.full(len(X_val), 0)
            ])
            
            cv = PredefinedSplit(test_fold)
        else:
            if self.model_calculator.problem_type == "classification":
                cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            else:
                cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            
        # 3. Select Search Strategy
        searcher = None
        
        if config.strategy == "grid":
            # Note: GridSearchCV does not support granular progress callbacks easily.
            # We could wrap the CV iterator, but n_jobs=-1 makes it complex.
            # For now, we only support progress for Optuna or if we implement custom loop.
            searcher = GridSearchCV(
                estimator=base_estimator,
                param_grid=config.search_space,
                scoring=config.metric,
                cv=cv,
                n_jobs=-1,
                refit=False # We just want best params
            )
        elif config.strategy == "random":
            searcher = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=config.search_space,
                n_iter=config.n_trials,
                scoring=config.metric,
                cv=cv,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False
            )
        elif config.strategy == "halving_grid":
            searcher = HalvingGridSearchCV(
                estimator=base_estimator,
                param_grid=config.search_space,
                scoring=config.metric,
                cv=cv,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False
            )
        elif config.strategy == "halving_random":
            searcher = HalvingRandomSearchCV(
                estimator=base_estimator,
                param_distributions=config.search_space,
                n_candidates=config.n_trials,  # Map n_trials to n_candidates
                scoring=config.metric,
                cv=cv,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False
            )
        elif config.strategy == "optuna":
            if not HAS_OPTUNA:
                raise ImportError("Optuna is not installed. Please install 'optuna' and 'optuna-integration'.")
            
            # Convert search space to Optuna distributions
            distributions = {}
            for k, v in config.search_space.items():
                if isinstance(v, list):
                    distributions[k] = optuna.distributions.CategoricalDistribution(v)
                else:
                    distributions[k] = v

            # Optuna callbacks
            callbacks = []
            if progress_callback:
                def _optuna_callback(study, trial):
                    # Optuna doesn't know total trials upfront easily if not set, but we have config.n_trials
                    progress_callback(trial.number + 1, config.n_trials)
                callbacks.append(_optuna_callback)

            searcher = OptunaSearchCV(
                estimator=base_estimator,
                param_distributions=distributions,
                n_trials=config.n_trials,
                timeout=config.timeout,
                cv=cv,
                scoring=config.metric,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False,
                verbose=0,
                callbacks=callbacks
            )
        else:
            raise ValueError(f"Unknown tuning strategy: {config.strategy}")
            
        # 4. Run Search
        # Ensure numpy
        X_arr = X_for_search.to_numpy() if hasattr(X_for_search, "to_numpy") else X_for_search
        y_arr = y_for_search.to_numpy() if hasattr(y_for_search, "to_numpy") else y_for_search
        
        searcher.fit(X_arr, y_arr)
        
        # 5. Extract Results
        best_params = searcher.best_params_
        best_score = searcher.best_score_
        
        # Collect trials
        trials = []
        # Special handling for Optuna
        if config.strategy == "optuna" and hasattr(searcher, "study_"):
            for trial in searcher.study_.trials:
                # Only include completed trials
                if trial.state.name == "COMPLETE":
                    trials.append({
                        "params": trial.params,
                        "score": trial.value
                    })
        elif hasattr(searcher, "cv_results_"):
            results = searcher.cv_results_
            if "params" in results:
                n_candidates = len(results["params"])
                for i in range(n_candidates):
                    trials.append({
                        "params": results["params"][i],
                        "score": results["mean_test_score"][i]
                    })
                
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trials),
            trials=trials
        )

class TunerApplier(BaseModelApplier):
    """
    Applier for TunerCalculator.
    Since Tuning does not produce a predictive model (it produces params),
    this applier returns dummy predictions.
    """
    def predict(self, df: pd.DataFrame, model_artifact: Any) -> pd.Series:
        # Return empty predictions or NaNs
        return pd.Series(np.nan, index=df.index)

