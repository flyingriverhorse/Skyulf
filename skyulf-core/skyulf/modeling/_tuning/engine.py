"""Hyperparameter Tuner implementation."""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, cast

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
from ...engines import SkyulfDataFrame
from ...engines.sklearn_bridge import SklearnBridge
from .schemas import TuningConfig, TuningResult

logger = logging.getLogger(__name__)

# Try importing Optuna with robust fallback for integration packages
HAS_OPTUNA = False
OptunaSearchCV: Any = None

try:
    import optuna  # ty: ignore[unresolved-import]

    HAS_OPTUNA = True
except ImportError:
    pass

if HAS_OPTUNA:
    try:
        from optuna.integration import (  # ty: ignore[unresolved-import]
            OptunaSearchCV as _OptunaSearchCV,
        )

        OptunaSearchCV = _OptunaSearchCV
    except ImportError:
        try:
            from optuna.integration.sklearn import (  # ty: ignore[unresolved-import]
                OptunaSearchCV as _OptunaSearchCV,
            )

            OptunaSearchCV = _OptunaSearchCV
        except ImportError:
            try:
                from optuna_integration.sklearn import (  # ty: ignore[unresolved-import]
                    OptunaSearchCV as _OptunaSearchCV,
                )

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
        validation_data: Optional[
            tuple[Union[pd.DataFrame, SkyulfDataFrame], Union[pd.Series, Any]]
        ] = None,
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

        # Convert data to Numpy for tuning
        X_np, y_np = SklearnBridge.to_sklearn((X, y))

        # --- VALIDATION: Check for NaNs/Inf in Data ---
        # Many tuning errors ("No trials completed") are actually due to dirty data causing instant failures.
        # We catch this early to give a clear message.
        if isinstance(X_np, np.ndarray) and np.issubdtype(X_np.dtype, np.number):
            if np.isnan(X_np).any():
                raise ValueError(
                    "Input features (X) contain NaN values. Please use an 'Imputer' node before this model."
                )
            if np.isinf(X_np).any():
                raise ValueError(
                    "Input features (X) contain Infinite values. Please scale or clean your data."
                )

        if isinstance(y_np, np.ndarray) and np.issubdtype(y_np.dtype, np.number):
            if np.isnan(y_np).any():
                raise ValueError(
                    "Target variable (y) contains NaN values. Please drop rows with missing targets or impute them."
                )
            if np.isinf(y_np).any():
                raise ValueError("Target variable (y) contains Infinite values.")
        # ----------------------------------------------

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
        if "random_state" not in final_params and hasattr(tuning_config, "random_state"):
            final_params["random_state"] = tuning_config.random_state

        if log_callback:
            log_callback(f"Refitting best model with params: {final_params}")

        # Mypy doesn't know that model_calculator has model_class because it's typed as BaseModelCalculator
        # We can cast it or ignore it.
        model_cls = getattr(self.model_calculator, "model_class", None)
        if not model_cls:
            raise ValueError("Model calculator does not have a model_class attribute")

        # Filter params to only include those accepted by the model_class constructor.
        # When the constructor accepts **kwargs (e.g. LightGBM, XGBoost), pass everything —
        # the simple `k in sig.parameters` check would otherwise silently strip params like
        # verbose=-1 / verbosity=-1 that are forwarded through **kwargs.
        import inspect

        sig = inspect.signature(model_cls)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_kwargs:
            valid_final_params = final_params
        else:
            valid_final_params = {k: v for k, v in final_params.items() if k in sig.parameters}

        model = model_cls(**valid_final_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
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

        # `model_class` only on SklearnCalculator; `Any` keeps call sites type-clean.
        model_class: Any = getattr(self.model_calculator, "model_class")

        base_estimator = model_class(**self.model_calculator.default_params)

        # 2. Prepare Splitter
        # If validation data is provided, use PredefinedSplit to train on X and validate on validation_data
        # Otherwise use CV

        # `Any` — reassigned to np.concatenate output below; keeps branches type-clean.
        X_for_search: Any = X
        y_for_search: Any = y

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
                cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=config.random_state)
            elif config.cv_type == "nested_cv":
                # Nested CV during tuning: use fewer inner folds for
                # candidate scoring. The outer evaluation loop runs
                # post-tuning in engine.py (as stratified_k_fold).
                inner_folds = min(3, config.cv_folds - 1) if config.cv_folds > 2 else 2
                if self.model_calculator.problem_type == "classification":
                    cv = StratifiedKFold(
                        n_splits=inner_folds,
                        shuffle=True,
                        random_state=config.random_state,
                    )
                else:
                    cv = KFold(
                        n_splits=inner_folds,
                        shuffle=True,
                        random_state=config.random_state,
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

        # --- VALIDATION: Metric Consistency Check ---
        # The schema defaults metric to "accuracy". If the user is doing Regression but "accuracy"
        # (or another classification metric) is selected, we raise a clear error instead of crashing deeply in sklearn.
        if self.model_calculator.problem_type == "regression":
            if metric in [
                "accuracy",
                "f1",
                "precision",
                "recall",
                "roc_auc",
                "f1_weighted",
                "balanced_accuracy",
                "log_loss",
                "matthews_corrcoef",
                "roc_auc_weighted",
                "roc_auc_ovr",
                "roc_auc_ovo",
                "roc_auc_ovr_weighted",
                "roc_auc_ovo_weighted",
                "pr_auc",
                "pr_auc_weighted",
                "g_score",
            ]:
                raise ValueError(
                    f"Configuration Error: You selected '{metric}' as the tuning metric, "
                    "but this is a Regression model. "
                    "Accuracy/F1/AUC are for Classification only. "
                    "Please open 'Advanced Settings' on this node and select a regression metric "
                    "(e.g., R2, RMSE, MAE)."
                )
        # -----------------------------------------------

        # Map common user-friendly metrics to sklearn scoring strings
        metric_map = {
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "explained_variance": "explained_variance",
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
            "f1_weighted": "f1_weighted",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
            "roc_auc_ovr": "roc_auc_ovr",
            "roc_auc_ovo": "roc_auc_ovo",
            "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
            "roc_auc_ovo_weighted": "roc_auc_ovo_weighted",
            "log_loss": "neg_log_loss",
            "matthews_corrcoef": "matthews_corrcoef",
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
                if config.metric == "roc_auc":  # Check original config metric name just in case
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
                    log_callback(f"Evaluating Candidate {i + 1}/{total_candidates}: {params}")

                # Use custom cross-validation loop to enable per-fold logging and progress tracking.
                # We instantiate the model with the current candidate parameters and evaluate it
                # using the configured CV strategy.

                fold_scores = []

                # Ensure numpy
                X_any = cast(Any, X_for_search)
                y_any = cast(Any, y_for_search)
                X_arr = X_any.to_numpy() if hasattr(X_any, "to_numpy") else X_any
                y_arr = y_any.to_numpy() if hasattr(y_any, "to_numpy") else y_any

                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
                    # Split
                    X_train_fold = (
                        X_any.iloc[train_idx] if hasattr(X_any, "iloc") else X_any[train_idx]
                    )
                    y_train_fold = (
                        y_any.iloc[train_idx] if hasattr(y_any, "iloc") else y_any[train_idx]
                    )
                    X_val_fold = X_any.iloc[val_idx] if hasattr(X_any, "iloc") else X_any[val_idx]
                    y_val_fold = y_any.iloc[val_idx] if hasattr(y_any, "iloc") else y_any[val_idx]

                    # Instantiate and Fit
                    # Note: We must handle potential errors (e.g. incompatible params)
                    try:
                        model = model_class(**{**self.model_calculator.default_params, **params})
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
                scoring_metric=metric,
            )

        elif config.strategy in ["halving_grid", "halving_random"]:
            strategy_params = getattr(config, "strategy_params", {})
            factor = strategy_params.get("factor", 3)
            resource = strategy_params.get("resource", "n_samples")
            min_resources = strategy_params.get("min_resources", "exhaust")

            # Halving search uses sklearn's internal scheduler and does NOT
            # expose per-trial callbacks (no equivalent of Optuna's callbacks=).
            # Emit a started log here so the Live Logs panel is never empty
            # while the search is running. Per-iteration progress is not
            # available without monkey-patching sklearn internals.
            if log_callback:
                space = self._clean_search_space(config.search_space)
                if config.strategy == "halving_grid":
                    grid_size = int(np.prod([len(v) for v in space.values()] or [0]))
                    log_callback(
                        f"Starting halving_grid search "
                        f"(grid_size={grid_size}, factor={factor}, "
                        f"resource={resource}, min_resources={min_resources}). "
                        f"sklearn HalvingGridSearchCV runs without per-trial callbacks; "
                        f"this may take a while."
                    )
                else:
                    log_callback(
                        f"Starting halving_random search "
                        f"(n_candidates={config.n_trials}, factor={factor}, "
                        f"resource={resource}, min_resources={min_resources}). "
                        f"sklearn HalvingRandomSearchCV runs without per-trial callbacks; "
                        f"this may take a while."
                    )

            if isinstance(min_resources, str) and min_resources.isdigit():
                min_resources = int(min_resources)

            if config.strategy == "halving_grid":
                searcher = HalvingGridSearchCV(
                    estimator=base_estimator,
                    param_grid=self._clean_search_space(config.search_space),
                    scoring=metric,
                    cv=cv,
                    n_jobs=-1,
                    random_state=config.random_state,
                    refit=False,
                    error_score=np.nan,
                    factor=factor,
                    resource=resource,
                    min_resources=min_resources,
                )
            else:
                searcher = HalvingRandomSearchCV(
                    estimator=base_estimator,
                    param_distributions=self._clean_search_space(config.search_space),
                    n_candidates=config.n_trials,
                    scoring=metric,
                    cv=cv,
                    n_jobs=-1,
                    random_state=config.random_state,
                    refit=False,
                    error_score=np.nan,
                    factor=factor,
                    resource=resource,
                    min_resources=min_resources,
                )
        elif config.strategy == "optuna":
            if not HAS_OPTUNA:
                raise ImportError(
                    "Optuna is not installed. Please install 'optuna' and 'optuna-integration'."
                )

            # Convert search space to Optuna distributions.
            # CMA-ES needs continuous distributions — numeric lists become
            # IntDistribution or FloatDistribution so CMA-ES samples the full
            # range instead of treating discrete values as categories.
            # String / bool / None lists remain CategoricalDistribution; CMA-ES
            # falls back to RandomSampler for those (unavoidable) but we suppress
            # the noisy warning via warn_independent_sampling=False.
            strategy_params = getattr(config, "strategy_params", {})
            use_cmaes = strategy_params.get("sampler", "tpe") == "cmaes"
            distributions = {}
            for k, v in config.search_space.items():
                if (
                    isinstance(v, list)
                    and use_cmaes
                    and v
                    and all(isinstance(x, (int, float)) for x in v)
                ):
                    lo, hi = min(v), max(v)
                    if all(isinstance(x, int) for x in v):
                        distributions[k] = optuna.distributions.IntDistribution(lo, hi)
                    else:
                        distributions[k] = optuna.distributions.FloatDistribution(
                            float(lo), float(hi)
                        )
                elif isinstance(v, list):
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

                    progress_callback(trial.number + 1, config.n_trials, score, trial.params)

                callbacks.append(_optuna_callback)

            # Sampler Selection
            sampler_name = strategy_params.get("sampler", "tpe")
            if sampler_name == "random":
                sampler = optuna.samplers.RandomSampler(seed=config.random_state)
            elif sampler_name == "cmaes":
                # Suppress the fallback warning for genuinely categorical params
                # (strings, booleans, None) — those can never be continuous and
                # the random fallback for them is expected behaviour.
                sampler = optuna.samplers.CmaEsSampler(
                    seed=config.random_state, warn_independent_sampling=False
                )
            else:
                sampler = optuna.samplers.TPESampler(seed=config.random_state)

            # Pruner Selection
            pruner_name = strategy_params.get("pruner", "median")
            if pruner_name == "hyperband":
                pruner = optuna.pruners.HyperbandPruner()
            elif pruner_name == "none":
                pruner = optuna.pruners.NopPruner()
            else:
                pruner = optuna.pruners.MedianPruner()

            study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

            searcher = OptunaSearchCV(
                estimator=base_estimator,
                param_distributions=distributions,
                n_trials=config.n_trials,
                timeout=config.timeout,
                cv=cv,
                scoring=metric,
                n_jobs=-1,
                refit=False,
                verbose=0,
                callbacks=callbacks,
                study=study,
            )
        else:
            raise ValueError(f"Unknown tuning strategy: {config.strategy}")

        # 4. Run Search
        # Ensure numpy
        X_any = cast(Any, X_for_search)
        y_any = cast(Any, y_for_search)
        X_arr = X_any.to_numpy() if hasattr(X_any, "to_numpy") else X_any
        y_arr = y_any.to_numpy() if hasattr(y_any, "to_numpy") else y_any

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Failed to report cross validation scores for TerminatorCallback",
                )
                # LightGBM 4.x sets feature_names_in_ even for numpy input; during
                # halving/optuna internal CV sklearn's validate_data emits this warning
                # on every fold's score() call. Suppress it here — the root cause is
                # already fixed in the LGBM calculator's fit() override.
                warnings.filterwarnings(
                    "ignore",
                    message=".*valid feature names.*",
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

            if "n_samples" in error_msg and "resample" in error_msg and "Got 0" in error_msg:
                raise ValueError(
                    "Hyperparameter tuning with Halving strategy failed because the dataset is too small "
                    "for the configured halving parameters. Please try using 'Random Search' or 'Grid Search' instead, "
                    "or increase your dataset size."
                ) from e

            raise e

        # 5. Extract Results
        try:
            # Accessing best_params_ raises ValueError if no trials completed successfully
            best_params = searcher.best_params_
            best_score = searcher.best_score_
        except ValueError as e:
            if "No trials are completed yet" in str(e):
                raise ValueError(
                    "Hyperparameter tuning failed: All trials failed. "
                    "This often happens if the model produces NaN scores "
                    "(e.g., due to unscaled data for linear models/SVMs, exploding gradients, "
                    "or mismatched parameters). "
                    "Try adding a 'Scale' node before this model or checking for NaN/Infinity in your data."
                ) from e
            raise e

        # Collect trials
        trials = []
        # Special handling for Optuna
        if config.strategy == "optuna" and hasattr(searcher, "study_"):
            for trial in cast(Any, searcher).study_.trials:
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

        # Final completion log for strategies that don't emit per-trial callbacks
        # (halving_grid / halving_random / optuna). The grid/random branch above
        # already logs completion inside its custom loop.
        if log_callback and config.strategy in [
            "halving_grid",
            "halving_random",
            "optuna",
        ]:
            log_callback(
                f"Tuning Completed ({config.strategy}). "
                f"Trials evaluated: {len(trials)}. Best Score: {best_score:.4f}"
            )
            log_callback(f"Best Params: {best_params}")

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trials),
            trials=trials,
            scoring_metric=metric,
        )


class TuningApplier(BaseModelApplier):
    """
    Applier for TuningCalculator.
    Wraps the base model applier to provide predictions using the refitted best model.
    """

    def __init__(self, base_applier: BaseModelApplier):
        self.base_applier = base_applier

    def predict(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame],
        model_artifact: Any,
    ) -> Union[pd.Series, Any]:
        # model_artifact is (fitted_model, tuning_result)
        if isinstance(model_artifact, tuple) and len(model_artifact) == 2:
            model, _ = model_artifact
            return self.base_applier.predict(df, model)
        # Fallback if artifact is just the result (legacy)
        return pd.Series(np.nan, index=df.index)

    def predict_proba(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame],
        model_artifact: Any,
    ) -> Optional[Union[pd.DataFrame, SkyulfDataFrame]]:
        if isinstance(model_artifact, tuple) and len(model_artifact) == 2:
            model, _ = model_artifact
            return self.base_applier.predict_proba(df, model)
        return None
