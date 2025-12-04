"""Hyperparameter Tuner implementation."""

from typing import Any, Dict, List, Optional
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

from ..base import BaseModelCalculator
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

class TunerCalculator:
    def __init__(self, model_calculator: BaseModelCalculator):
        self.model_calculator = model_calculator

    def tune(self, X: pd.DataFrame, y: pd.Series, config: TuningConfig) -> TuningResult:
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
        if self.model_calculator.problem_type == "classification":
            cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        else:
            cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            
        # 3. Select Search Strategy
        searcher = None
        
        if config.strategy == "grid":
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
            
            # OptunaSearchCV requires distributions, not lists.
            # We need to convert list search space to Optuna distributions if possible,
            # OR rely on OptunaSearchCV's ability to handle lists as Categorical.
            # OptunaSearchCV from integration handles lists as categorical choices.
            
            searcher = OptunaSearchCV(
                estimator=base_estimator,
                param_distributions=config.search_space,
                n_trials=config.n_trials,
                timeout=config.timeout,
                cv=cv,
                scoring=config.metric,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown tuning strategy: {config.strategy}")
            
        # 4. Run Search
        # Ensure numpy
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else X
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y
        
        searcher.fit(X_arr, y_arr)
        
        # 5. Extract Results
        best_params = searcher.best_params_
        best_score = searcher.best_score_
        
        # Collect trials
        trials = []
        if hasattr(searcher, "cv_results_"):
            results = searcher.cv_results_
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
