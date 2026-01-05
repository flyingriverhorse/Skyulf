"""Hyperparameter Tuner implementation."""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Explicitly enable experimental halving search cv
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    ParameterGrid,
    ParameterSampler,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)

from ..base import BaseModelApplier, BaseModelCalculator
from ...engines import SkyulfDataFrame, get_engine
from ...engines.sklearn_bridge import SklearnBridge
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
                logger.warning(
                    "Optuna installed but OptunaSearchCV not found. Install 'optuna-integration'."
                )


class TuningCalculator(BaseModelCalculator):
    """Calculator for hyperparameter tuning."""

    def __init__(self, model_calculator: BaseModelCalculator):
        self.model_calculator = model_calculator

    @property
    def problem_type(self) -> str:
        return self.model_calculator.problem_type

    def _clean_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively cleans the search space.
        - Converts "none" string to None.
        """
        cleaned: Dict[str, Any] = {}
        for k, v in search_space.items():
            if isinstance(v, list):
                cleaned[k] = [None if x == "none" else x for x in v]
            elif isinstance(v, dict):
                cleaned[k] = self._clean_search_space(v)
            else:
                cleaned[k] = None if v == "none" else v
        return cleaned

    def fit(
        self,
        X: Union[pd.DataFrame, SkyulfDataFrame],
        y: Union[pd.Series, Any],
        config: Dict[str, Any],
        progress_callback: Optional[
            Callable[[int, int, Optional[float], Optional[Dict]], None]
        ] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        validation_data: Optional[tuple[Union[pd.DataFrame, SkyulfDataFrame], Union[pd.Series, Any]]] = None,
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
            tuning_config = TuningConfig(**filtered_config)  # type: ignore

        # Convert data to Numpy for tuning
        X_np, y_np = SklearnBridge.to_sklearn((X, y))

        validation_data_np = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_np, y_val_np = SklearnBridge.to_sklearn((X_val, y_val))
            validation_data_np = (X_val_np, y_val_np)

        tuning_result = self.tune(
            X_np,
            y_np,
            tuning_config,
            progress_callback=progress_callback,
            log_callback=log_callback,
            validation_data=validation_data_np,
        )

        # Refit the best model on the full dataset
        best_params = tuning_result.best_params
        final_params = {**self.model_calculator.default_params, **best_params}

        # Ensure random_state is passed if available in config and not in params
        if "random_state" not in final_params and hasattr(
            tuning_config, "random_state"
        ):
            final_params["random_state"] = tuning_config.random_state

        if log_callback:
            log_callback(f"Refitting best model with params: {final_params}")

        # Mypy doesn't know that model_calculator has model_class because it's typed as BaseModelCalculator
        # We can cast it or ignore it.
        model_cls = getattr(self.model_calculator, "model_class", None)
        if not model_cls:
            raise ValueError("Model calculator does not have a model_class attribute")

        model = model_cls(**final_params)
        model.fit(X_np, y_np)

        return (model, tuning_result)

    def tune(  # noqa: C901
        self,
        X: Any,
        y: Any,
        config: TuningConfig,
        progress_callback: Optional[
            Callable[[int, int, Optional[float], Optional[Dict]], None]
        ] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        validation_data: Optional[tuple[Any, Any]] = None,
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

        base_estimator = self.model_calculator.model_class(
            **self.model_calculator.default_params
        )

        # 2. Prepare Splitter
        # If validation data is provided, use PredefinedSplit to train on X and validate on validation_data
        # Otherwise use CV

        X_for_search = X
        y_for_search = y

        if validation_data is not None:
            from sklearn.model_selection import PredefinedSplit

            X_val, y_val = validation_data

            # Concatenate Train and Val (Numpy arrays)
            X_for_search = np.concatenate([X, X_val], axis=0)
            y_for_search = np.concatenate([y, y_val], axis=0)

            # Create test_fold array: -1 for train, 0 for val
            # -1 means "never in test set" (so always in training set)
            # 0 means "in test set for fold 0"
            test_fold = np.concatenate([np.full(len(X), -1), np.full(len(X_val), 0)])

            cv = PredefinedSplit(test_fold)
        else:
            if not config.cv_enabled:
                # Single split validation (20% holdout)
                cv = ShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=config.random_state
                )
            elif config.cv_type == "time_series_split":
                cv = TimeSeriesSplit(n_splits=config.cv_folds)
            elif config.cv_type == "shuffle_split":
                cv = ShuffleSplit(
                    n_splits=config.cv_folds,
                    test_size=0.2,
                    random_state=config.random_state,
                )
            elif (
                config.cv_type == "stratified_k_fold"
                and self.model_calculator.problem_type == "classification"
            ):
                cv = StratifiedKFold(
                    n_splits=config.cv_folds,
                    shuffle=True,
                    random_state=config.random_state,
                )
            else:
                # Default to KFold (also fallback for stratified if regression)
                cv = KFold(
                    n_splits=config.cv_folds,
                    shuffle=True,
                    random_state=config.random_state,
                )

        # 3. Select Search Strategy
        searcher = None

        # Handle multiclass metrics and map user-friendly names
        metric = config.metric

        # Map common user-friendly metrics to sklearn scoring strings
        metric_map = {
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        }

        if metric in metric_map:
            metric = metric_map[metric]

        if self.model_calculator.problem_type == "classification":
            # Check if target is multiclass
            is_multiclass = False
            if isinstance(y, pd.Series):
                is_multiclass = y.nunique() > 2
            elif isinstance(y, np.ndarray):
                is_multiclass = len(np.unique(y)) > 2

            # If multiclass and metric is binary-default, switch to weighted
            # Note: We check against the mapped names now (e.g. "f1", "precision")
            if is_multiclass and metric in ["f1", "precision", "recall", "roc_auc"]:
                metric = f"{metric}_weighted"
                # roc_auc needs special handling (ovr/ovo) usually, but weighted often works for simple cases
                if (
                    config.metric == "roc_auc"
                ):  # Check original config metric name just in case
                    metric = "roc_auc_ovr_weighted"

        if config.strategy in ["grid", "random"]:
            # Use custom loop to support progress and log callbacks
            if log_callback:
                log_callback(
                    f"Starting {config.strategy} search with custom loop for detailed logging..."
                )

            # 1. Generate Candidates
            param_space = self._clean_search_space(config.search_space)
            candidates = []

            if config.strategy == "grid":
                candidates = list(ParameterGrid(param_space))
            else:
                # Random Search
                candidates = list(
                    ParameterSampler(
                        param_space,
                        n_iter=config.n_trials,
                        random_state=config.random_state,
                    )
                )

            total_candidates = len(candidates)
            if log_callback:
                log_callback(f"Total candidates to evaluate: {total_candidates}")

            trials: List[Dict[str, Any]] = []
            best_score = -float("inf")
            best_params = None

            # 2. Iterate Candidates
            for i, params in enumerate(candidates):
                if log_callback:
                    log_callback(
                        f"Evaluating Candidate {i + 1}/{total_candidates}: {params}"
                    )

                # Use custom cross-validation loop to enable per-fold logging and progress tracking.
                # We instantiate the model with the current candidate parameters and evaluate it
                # using the configured CV strategy.

                fold_scores = []

                # Ensure numpy
                X_arr = (
                    X_for_search.to_numpy()
                    if hasattr(X_for_search, "to_numpy")
                    else X_for_search
                )
                y_arr = (
                    y_for_search.to_numpy()
                    if hasattr(y_for_search, "to_numpy")
                    else y_for_search
                )

                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
                    # Split
                    X_train_fold = (
                        X_for_search.iloc[train_idx]
                        if hasattr(X_for_search, "iloc")
                        else X_for_search[train_idx]
                    )
                    y_train_fold = (
                        y_for_search.iloc[train_idx]
                        if hasattr(y_for_search, "iloc")
                        else y_for_search[train_idx]
                    )
                    X_val_fold = (
                        X_for_search.iloc[val_idx]
                        if hasattr(X_for_search, "iloc")
                        else X_for_search[val_idx]
                    )
                    y_val_fold = (
                        y_for_search.iloc[val_idx]
                        if hasattr(y_for_search, "iloc")
                        else y_for_search[val_idx]
                    )

                    # Instantiate and Fit
                    # Note: We must handle potential errors (e.g. incompatible params)
                    try:
                        model = self.model_calculator.model_class(
                            **{**self.model_calculator.default_params, **params}
                        )
                        model.fit(X_train_fold, y_train_fold)

                        # Score
                        from sklearn.metrics import get_scorer

                        scorer = get_scorer(metric)
                        score = scorer(model, X_val_fold, y_val_fold)
                        fold_scores.append(score)

                        if log_callback:
                            n_splits = cv.get_n_splits(X_arr, y_arr)
                            log_callback(
                                f"  [Candidate {i + 1}] CV Fold {fold_idx + 1}/{n_splits} Score: {score:.4f}"
                            )
                    except Exception as e:
                        if log_callback:
                            n_splits = cv.get_n_splits(X_arr, y_arr)
                            log_callback(
                                f"  [Candidate {i + 1}] CV Fold {fold_idx + 1}/{n_splits} Failed: {str(e)}"
                            )
                        fold_scores.append(-float("inf"))

                # Filter out failed folds for mean calculation if possible, or penalize
                valid_scores = [s for s in fold_scores if s != -float("inf")]
                if valid_scores:
                    mean_score = np.mean(valid_scores)
                else:
                    mean_score = -float("inf")

                if log_callback:
                    log_callback(f"Candidate {i + 1} Mean Score: {mean_score:.4f}")

                if progress_callback:
                    progress_callback(i + 1, total_candidates, mean_score, params)

                trials.append({"params": params, "score": mean_score})

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

            if log_callback:
                log_callback(f"Tuning Completed. Best Score: {best_score:.4f}")
                log_callback(f"Best Params: {best_params}")

            return TuningResult(
                best_params=best_params if best_params is not None else {},
                best_score=best_score,
                n_trials=total_candidates,
                trials=trials,
            )

        elif config.strategy == "halving_grid":
            searcher = HalvingGridSearchCV(
                estimator=base_estimator,
                param_grid=self._clean_search_space(config.search_space),
                scoring=metric,
                cv=cv,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False,
                error_score=np.nan,
            )
        elif config.strategy == "halving_random":
            searcher = HalvingRandomSearchCV(
                estimator=base_estimator,
                param_distributions=self._clean_search_space(config.search_space),
                n_candidates=config.n_trials,  # Map n_trials to n_candidates
                scoring=metric,
                cv=cv,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False,
                error_score=np.nan,
            )
        elif config.strategy == "optuna":
            if not HAS_OPTUNA:
                raise ImportError(
                    "Optuna is not installed. Please install 'optuna' and 'optuna-integration'."
                )

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
                    # trial.value is the score (or None if failed/pruned)
                    score = trial.value if trial.value is not None else None

                    if log_callback:
                        log_callback(
                            f"Optuna Trial {trial.number + 1} finished. Mean CV Score: {score}"
                        )

                    progress_callback(
                        trial.number + 1, config.n_trials, score, trial.params
                    )

                callbacks.append(_optuna_callback)

            searcher = OptunaSearchCV(
                estimator=base_estimator,
                param_distributions=distributions,
                n_trials=config.n_trials,
                timeout=config.timeout,
                cv=cv,  # type: ignore
                scoring=metric,
                n_jobs=-1,
                random_state=config.random_state,
                refit=False,
                verbose=0,
                callbacks=callbacks,
            )
        else:
            raise ValueError(f"Unknown tuning strategy: {config.strategy}")

        # 4. Run Search
        # Ensure numpy
        X_arr = (
            X_for_search.to_numpy()
            if hasattr(X_for_search, "to_numpy")
            else X_for_search
        )
        y_arr = (
            y_for_search.to_numpy()
            if hasattr(y_for_search, "to_numpy")
            else y_for_search
        )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Failed to report cross validation scores for TerminatorCallback",
                )
                searcher.fit(X_arr, y_arr)
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            error_msg = str(e)
            if "No trials are completed yet" in error_msg:
                raise ValueError(
                    "Hyperparameter tuning failed: No trials completed successfully. "
                    "This usually means the model failed to train with the provided hyperparameter combinations. "
                    "Please check your search space and data."
                ) from e

            if (
                "n_samples" in error_msg
                and "resample" in error_msg
                and "Got 0" in error_msg
            ):
                raise ValueError(
                    "Hyperparameter tuning with Halving strategy failed because the dataset is too small "
                    "for the configured halving parameters. Please try using 'Random Search' or 'Grid Search' instead, "
                    "or increase your dataset size."
                ) from e

            raise e

        # 5. Extract Results
        if not hasattr(searcher, "best_params_"):
            raise ValueError(
                "Hyperparameter tuning failed to find any valid combination of parameters. All trials likely failed."
            )

        best_params = searcher.best_params_
        best_score = searcher.best_score_

        # Collect trials
        trials = []
        # Special handling for Optuna
        if config.strategy == "optuna" and hasattr(searcher, "study_"):
            for trial in searcher.study_.trials:
                # Only include completed trials
                if trial.state.name == "COMPLETE":
                    trials.append({"params": trial.params, "score": trial.value})
        elif hasattr(searcher, "cv_results_"):
            results = searcher.cv_results_
            if "params" in results:
                n_candidates = len(results["params"])
                for i in range(n_candidates):
                    trials.append(
                        {
                            "params": results["params"][i],
                            "score": results["mean_test_score"][i],
                        }
                    )

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trials),
            trials=trials,
        )


class TuningApplier(BaseModelApplier):
    """
    Applier for TuningCalculator.
    Wraps the base model applier to provide predictions using the refitted best model.
    """

    def __init__(self, base_applier: BaseModelApplier):
        self.base_applier = base_applier

    def predict(self, df: pd.DataFrame, model_artifact: Any) -> pd.Series:
        # model_artifact is (fitted_model, tuning_result)
        if isinstance(model_artifact, tuple) and len(model_artifact) == 2:
            model, _ = model_artifact
            return self.base_applier.predict(df, model)
        # Fallback if artifact is just the result (legacy)
        return pd.Series(np.nan, index=df.index)

    def predict_proba(
        self, df: pd.DataFrame, model_artifact: Any
    ) -> Optional[pd.DataFrame]:
        if isinstance(model_artifact, tuple) and len(model_artifact) == 2:
            model, _ = model_artifact
            return self.base_applier.predict_proba(df, model)
        return None
