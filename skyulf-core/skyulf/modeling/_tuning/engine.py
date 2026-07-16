"""Hyperparameter Tuner implementation."""

import logging
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import parallel_backend

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

from ...engines import SkyulfDataFrame
from ...engines.sklearn_bridge import SklearnBridge
from ..base import BaseModelApplier, BaseModelCalculator
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

    def _clean_search_space(self, search_space: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively cleans the search space.
        - Converts "none" string to None.
        """
        cleaned: dict[str, Any] = {}
        for k, v in search_space.items():
            if isinstance(v, list):
                cleaned[k] = [None if x == "none" else x for x in v]
            elif isinstance(v, dict):
                cleaned[k] = self._clean_search_space(v)
            else:
                cleaned[k] = None if v == "none" else v
        return cleaned

    @staticmethod
    def _split_flat_and_nested_params(
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Splits ``params`` into flat constructor args and nested ``a__b`` keys."""
        flat = {k: v for k, v in params.items() if "__" not in str(k)}
        nested = {k: v for k, v in params.items() if "__" in str(k)}
        return flat, nested

    @staticmethod
    def _filter_params_to_signature(model_class: Any, flat: dict[str, Any]) -> dict[str, Any]:
        """Filters ``flat`` down to ``model_class``'s constructor params, unless it accepts ``**kwargs``."""
        import inspect

        sig = inspect.signature(model_class)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_kwargs:
            return flat
        return {k: v for k, v in flat.items() if k in sig.parameters}

    @staticmethod
    def _instantiate_model(model_class: Any, params: dict[str, Any]) -> Any:
        """Build an estimator, routing nested ``a__b`` keys through ``set_params``.

        Constructor args (no ``__``) are filtered to the model's signature
        (unless it accepts ``**kwargs``); nested keys — e.g. an ensemble's
        ``random_forest__n_estimators`` — are applied afterwards via
        ``set_params`` because sklearn estimators only accept them that way.
        """
        flat, nested = TuningCalculator._split_flat_and_nested_params(params)
        flat = TuningCalculator._filter_params_to_signature(model_class, flat)

        model = model_class(**flat)
        if nested:
            model.set_params(**nested)
        return model

    @staticmethod
    def _build_tuning_config(config: dict[str, Any] | TuningConfig) -> TuningConfig:
        """Convert a raw config dict (or an already-built TuningConfig) into a TuningConfig."""
        if isinstance(config, TuningConfig):
            return config
        # Extract valid keys for TuningConfig
        valid_keys = TuningConfig.__annotations__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return TuningConfig(**filtered_config)

    @staticmethod
    def _validate_no_nan_inf(
        arr: Any,
        nan_msg: str,
        inf_msg: str,
        object_nan_msg: str,
    ) -> None:
        """Raise ValueError if a numpy array contains NaN/Inf (numeric) or NaN (object dtype).

        Many tuning errors ("No trials completed") are actually due to dirty data causing
        instant failures. We catch this early to give a clear message. Object-dtype arrays
        (e.g. mixed dtypes or leftover categorical/string columns that were never encoded)
        are also scanned via pd.isna, since np.isnan/np.isinf raise on non-numeric dtypes.
        """
        if not isinstance(arr, np.ndarray):
            return
        if np.issubdtype(arr.dtype, np.number):
            if np.isnan(arr).any():
                raise ValueError(nan_msg)
            if np.isinf(arr).any():
                raise ValueError(inf_msg)
        elif arr.dtype == object and pd.isna(arr).any():
            raise ValueError(object_nan_msg)

    def _refit_best_model(
        self,
        tuning_result: TuningResult,
        tuning_config: TuningConfig,
        X_np: Any,
        y_np: Any,
        log_callback: Callable[[str], None] | None,
    ) -> Any:
        """Build and fit the final model on the full dataset using the tuned best params."""
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

        # Build the final model. ``_instantiate_model`` filters constructor args
        # to the signature (when there is no **kwargs) and routes nested
        # ``a__b`` keys — e.g. an ensemble's tuned base-model params — through
        # ``set_params`` so they are not silently dropped.
        model = self._instantiate_model(model_cls, final_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            model.fit(X_np, y_np)

        return model

    def fit(
        self,
        X: pd.DataFrame | SkyulfDataFrame,
        y: pd.Series | Any,
        config: dict[str, Any],
        progress_callback: Callable[[int, int, float | None, dict | None], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        validation_data: tuple[pd.DataFrame | SkyulfDataFrame, pd.Series | Any] | None = None,
    ) -> Any:
        """
        Fits the tuner (runs tuning).
        Adapts the generic fit interface to the specific tune method.
        """
        tuning_config = self._build_tuning_config(config)

        # For Time Series Split, sort data chronologically (and drop the time
        # column from features) before converting to numpy below - numpy has
        # no column names, so this must happen while X still carries them.
        # Mirrors the same fix already applied to perform_cross_validation();
        # without it, tuning with cv_type="time_series_split" silently leaks
        # the time column and evaluates folds out of chronological order.
        if tuning_config.cv_type == "time_series_split" and hasattr(X, "columns"):
            from ..cross_validation import _sort_by_time

            X, y = _sort_by_time(X, y, tuning_config.cv_time_column, log_callback, logger)

        # Convert data to Numpy for tuning
        X_np, y_np = SklearnBridge.to_sklearn((X, y))

        # --- VALIDATION: Check for NaNs/Inf in Data ---
        self._validate_no_nan_inf(
            X_np,
            "Input features (X) contain NaN values. Please use an 'Imputer' node before this model.",
            "Input features (X) contain Infinite values. Please scale or clean your data.",
            "Input features (X) contain missing/NaN values. Please use an 'Imputer' node before this model.",
        )
        self._validate_no_nan_inf(
            y_np,
            "Target variable (y) contains NaN values. Please drop rows with missing targets or impute them.",
            "Target variable (y) contains Infinite values.",
            "Target variable (y) contains missing/NaN values. Please drop rows with missing targets or impute them.",
        )
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
        model = self._refit_best_model(tuning_result, tuning_config, X_np, y_np, log_callback)

        return (model, tuning_result)

    def _build_cv_splitter(
        self,
        X: Any,
        y: Any,
        config: TuningConfig,
        validation_data: tuple[Any, Any] | None,
    ) -> tuple[Any, Any, Any]:
        """Builds the CV splitter (or ``PredefinedSplit``) plus the ``X``/``y`` to search over.

        When ``validation_data`` is provided, it is concatenated with ``X``/``y`` and a
        ``PredefinedSplit`` is used so the searcher trains on ``X`` and validates on it.
        Otherwise a CV splitter is chosen from ``config`` (holdout, nested CV inner folds,
        time series, shuffle, stratified, or plain K-fold).
        """
        if validation_data is not None:
            return self._build_predefined_split_cv(X, y, validation_data)

        return self._select_cv_by_type(config), X, y

    def _build_predefined_split_cv(
        self,
        X: Any,
        y: Any,
        validation_data: tuple[Any, Any],
    ) -> tuple[Any, Any, Any]:
        """Concatenates train/val data and builds a ``PredefinedSplit`` over it.

        The search treats ``X`` (train) as always-in-training-set (-1) and the concatenated
        ``validation_data`` as the single test fold (0), so the searcher trains on ``X`` and
        validates on ``validation_data``.
        """
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
        return cv, X_for_search, y_for_search

    @staticmethod
    def _build_holdout_cv(config: TuningConfig) -> Any:
        """Builds the single-split (20% holdout) CV used when ``cv_enabled`` is False."""
        return ShuffleSplit(n_splits=1, test_size=0.2, random_state=config.cv_random_state)

    @staticmethod
    def _build_shuffle_split_cv(config: TuningConfig) -> Any:
        """Builds a repeated shuffle-split CV splitter for ``cv_type == "shuffle_split"``."""
        return ShuffleSplit(
            n_splits=config.cv_folds,
            test_size=0.2,
            random_state=config.cv_random_state,
        )

    @staticmethod
    def _build_stratified_kfold_cv(config: TuningConfig) -> Any:
        """Builds a StratifiedKFold splitter for ``cv_type == "stratified_k_fold"``."""
        return StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=config.cv_shuffle,
            random_state=config.cv_random_state if config.cv_shuffle else None,
        )

    @staticmethod
    def _build_kfold_cv(config: TuningConfig) -> Any:
        """Builds the default plain KFold splitter (also the regression fallback for stratified)."""
        return KFold(
            n_splits=config.cv_folds,
            shuffle=config.cv_shuffle,
            random_state=config.cv_random_state if config.cv_shuffle else None,
        )

    def _select_cv_by_type(self, config: TuningConfig) -> Any:
        """Picks a CV splitter from ``config`` (holdout, nested CV inner folds, time series,
        shuffle, stratified, or plain K-fold), based on ``cv_enabled``/``cv_type``.
        """
        if not config.cv_enabled:
            # Single split validation (20% holdout)
            return self._build_holdout_cv(config)

        if config.cv_type == "nested_cv":
            # Nested CV during tuning: use fewer inner folds for
            # candidate scoring. The outer evaluation loop runs
            # post-tuning in engine.py (as stratified_k_fold).
            return self._build_nested_inner_cv(config)

        if config.cv_type == "time_series_split":
            return TimeSeriesSplit(n_splits=config.cv_folds)

        if config.cv_type == "shuffle_split":
            return self._build_shuffle_split_cv(config)

        if (
            config.cv_type == "stratified_k_fold"
            and self.model_calculator.problem_type == "classification"
        ):
            return self._build_stratified_kfold_cv(config)

        # Default to KFold (also fallback for stratified if regression)
        return self._build_kfold_cv(config)

    def _build_nested_inner_cv(self, config: TuningConfig) -> Any:
        """Builds the inner-fold CV splitter used for candidate scoring during nested CV tuning."""
        inner_folds = min(3, config.cv_folds - 1) if config.cv_folds > 2 else 2
        inner_cv_random_state = config.cv_random_state if config.cv_shuffle else None
        if self.model_calculator.problem_type == "classification":
            return StratifiedKFold(
                n_splits=inner_folds,
                shuffle=config.cv_shuffle,
                random_state=inner_cv_random_state,
            )
        return KFold(
            n_splits=inner_folds,
            shuffle=config.cv_shuffle,
            random_state=inner_cv_random_state,
        )

    _INVALID_REGRESSION_METRICS = frozenset(
        {
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
        }
    )

    _METRIC_ALIAS_MAP = {
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

    @classmethod
    def _validate_metric_for_problem_type(cls, problem_type: str, metric: str) -> None:
        """Raises a clear ``ValueError`` if a classification-only metric is used for regression."""
        if problem_type == "regression" and metric in cls._INVALID_REGRESSION_METRICS:
            raise ValueError(
                f"Configuration Error: You selected '{metric}' as the tuning metric, "
                "but this is a Regression model. "
                "Accuracy/F1/AUC are for Classification only. "
                "Please open 'Advanced Settings' on this node and select a regression metric "
                "(e.g., R2, RMSE, MAE)."
            )

    @staticmethod
    def _is_multiclass_target(y: Any) -> bool:
        """Returns whether ``y`` (a Series or ndarray) has more than 2 unique classes."""
        if isinstance(y, pd.Series):
            return y.nunique() > 2
        if isinstance(y, np.ndarray):
            return len(np.unique(y)) > 2
        return False

    @staticmethod
    def _weight_metric_for_multiclass(metric: str, original_metric: str) -> str:
        """Switches a binary-default metric to its weighted variant for multiclass targets."""
        weighted = f"{metric}_weighted"
        # roc_auc needs special handling (ovr/ovo) usually, but weighted often works for simple cases
        if original_metric == "roc_auc":  # Check original config metric name just in case
            return "roc_auc_ovr_weighted"
        return weighted

    def _resolve_metric(self, config: TuningConfig, y: Any) -> str:
        """Validates the metric against the problem type, maps friendly aliases to sklearn
        scoring strings, and switches binary-default metrics to weighted for multiclass targets.
        """
        metric = config.metric

        # --- VALIDATION: Metric Consistency Check ---
        # The schema defaults metric to "accuracy". If the user is doing Regression but "accuracy"
        # (or another classification metric) is selected, we raise a clear error instead of crashing deeply in sklearn.
        self._validate_metric_for_problem_type(self.model_calculator.problem_type, metric)
        # -----------------------------------------------

        # Map common user-friendly metrics to sklearn scoring strings
        if metric in self._METRIC_ALIAS_MAP:
            metric = self._METRIC_ALIAS_MAP[metric]

        if self.model_calculator.problem_type == "classification":
            # Check if target is multiclass
            is_multiclass = self._is_multiclass_target(y)

            # If multiclass and metric is binary-default, switch to weighted
            # Note: We check against the mapped names now (e.g. "f1", "precision")
            if is_multiclass and metric in ["f1", "precision", "recall", "roc_auc"]:
                metric = self._weight_metric_for_multiclass(metric, config.metric)

        return metric

    def _evaluate_candidate_cv(
        self,
        candidate_idx: int,
        params: dict[str, Any],
        model_class: Any,
        cv: Any,
        X_for_search: Any,
        y_for_search: Any,
        metric: str,
        log_callback: Callable[[str], None] | None,
    ) -> float:
        """Cross-validates one grid/random-search candidate and returns its mean fold score.

        Fold failures are logged and penalized with ``-inf`` instead of raised, so a single
        bad hyperparameter combination doesn't abort the whole search.
        """
        fold_scores = []

        # Ensure numpy
        X_any = cast(Any, X_for_search)
        y_any = cast(Any, y_for_search)
        X_arr = X_any.to_numpy() if hasattr(X_any, "to_numpy") else X_any
        y_arr = y_any.to_numpy() if hasattr(y_any, "to_numpy") else y_any

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
            score = self._fit_and_score_candidate_fold(
                candidate_idx=candidate_idx,
                fold_idx=fold_idx,
                params=params,
                model_class=model_class,
                cv=cv,
                X_any=X_any,
                y_any=y_any,
                X_arr=X_arr,
                y_arr=y_arr,
                train_idx=train_idx,
                val_idx=val_idx,
                metric=metric,
                log_callback=log_callback,
            )
            fold_scores.append(score)

        # Filter out failed folds for mean calculation if possible, or penalize
        valid_scores = [s for s in fold_scores if s != -float("inf")]
        return float(np.mean(valid_scores)) if valid_scores else -float("inf")

    def _fit_and_score_candidate_fold(
        self,
        candidate_idx: int,
        fold_idx: int,
        params: dict[str, Any],
        model_class: Any,
        cv: Any,
        X_any: Any,
        y_any: Any,
        X_arr: Any,
        y_arr: Any,
        train_idx: Any,
        val_idx: Any,
        metric: str,
        log_callback: Callable[[str], None] | None,
    ) -> float:
        """Fits one candidate on a single CV fold and returns its score, or ``-inf`` on failure.

        Errors (e.g. incompatible params) are caught and logged rather than raised, so a single
        bad fold doesn't abort the whole candidate evaluation.
        """
        # Split
        X_train_fold = X_any.iloc[train_idx] if hasattr(X_any, "iloc") else X_any[train_idx]
        y_train_fold = y_any.iloc[train_idx] if hasattr(y_any, "iloc") else y_any[train_idx]
        X_val_fold = X_any.iloc[val_idx] if hasattr(X_any, "iloc") else X_any[val_idx]
        y_val_fold = y_any.iloc[val_idx] if hasattr(y_any, "iloc") else y_any[val_idx]

        # Instantiate and Fit
        # Note: We must handle potential errors (e.g. incompatible params)
        try:
            model = self._instantiate_model(
                model_class,
                {**self.model_calculator.default_params, **params},
            )
            model.fit(X_train_fold, y_train_fold)

            # Score
            from sklearn.metrics import get_scorer

            scorer = get_scorer(metric)
            score = scorer(model, X_val_fold, y_val_fold)

            if log_callback:
                n_splits = cv.get_n_splits(X_arr, y_arr)
                log_callback(
                    f"  [Candidate {candidate_idx + 1}] CV Fold {fold_idx + 1}/{n_splits} Score: {score:.4f}"
                )
            return score
        except Exception as e:
            if log_callback:
                n_splits = cv.get_n_splits(X_arr, y_arr)
                log_callback(
                    f"  [Candidate {candidate_idx + 1}] CV Fold {fold_idx + 1}/{n_splits} Failed: {str(e)}"
                )
            return -float("inf")

    def _generate_search_candidates(self, config: TuningConfig) -> list[dict[str, Any]]:
        """Generates the list of hyperparameter candidates for grid or random search."""
        param_space = self._clean_search_space(config.search_space)
        if config.strategy == "grid":
            return list(ParameterGrid(param_space))
        # Random Search
        return list(
            ParameterSampler(
                param_space,
                n_iter=config.n_trials,
                random_state=config.random_state,
            )
        )

    def _evaluate_search_candidates(
        self,
        candidates: list[dict[str, Any]],
        X_for_search: Any,
        y_for_search: Any,
        model_class: Any,
        cv: Any,
        metric: str,
        progress_callback: Callable[[int, int, float | None, dict | None], None] | None,
        log_callback: Callable[[str], None] | None,
    ) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None]:
        """Evaluates every candidate via CV, emitting progress/log callbacks, and tracks the best.

        Returns the collected trials, the best score, and the best params (or ``None`` if all failed).
        """
        total_candidates = len(candidates)
        trials: list[dict[str, Any]] = []
        best_score = -float("inf")
        best_params = None

        for i, params in enumerate(candidates):
            if log_callback:
                log_callback(f"Evaluating Candidate {i + 1}/{total_candidates}: {params}")

            # Use custom cross-validation loop to enable per-fold logging and progress tracking.
            # We instantiate the model with the current candidate parameters and evaluate it
            # using the configured CV strategy.
            mean_score = self._evaluate_candidate_cv(
                i, params, model_class, cv, X_for_search, y_for_search, metric, log_callback
            )

            if log_callback:
                log_callback(f"Candidate {i + 1} Mean Score: {mean_score:.4f}")

            if progress_callback:
                progress_callback(i + 1, total_candidates, mean_score, params)

            trials.append({"params": params, "score": mean_score})

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        return trials, best_score, best_params

    def _run_grid_or_random_search(
        self,
        X_for_search: Any,
        y_for_search: Any,
        config: TuningConfig,
        model_class: Any,
        cv: Any,
        metric: str,
        progress_callback: Callable[[int, int, float | None, dict | None], None] | None,
        log_callback: Callable[[str], None] | None,
    ) -> TuningResult:
        """Runs a custom grid/random search loop (instead of sklearn's searchers) so
        per-candidate and per-fold progress/log callbacks can be emitted during tuning.
        """
        if log_callback:
            log_callback(
                f"Starting {config.strategy} search with custom loop for detailed logging..."
            )

        # 1. Generate Candidates
        candidates = self._generate_search_candidates(config)
        total_candidates = len(candidates)
        if log_callback:
            log_callback(f"Total candidates to evaluate: {total_candidates}")

        # 2. Iterate Candidates
        trials, best_score, best_params = self._evaluate_search_candidates(
            candidates,
            X_for_search,
            y_for_search,
            model_class,
            cv,
            metric,
            progress_callback,
            log_callback,
        )

        if log_callback:
            log_callback(f"Tuning Completed. Best Score: {best_score:.4f}")
            log_callback(f"Best Params: {best_params}")

        if best_params is None:
            raise ValueError(
                "Hyperparameter tuning failed: All trials failed. "
                "This usually means the model failed to train with the provided hyperparameter combinations. "
                "Please check your search space and data."
            )

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=total_candidates,
            trials=trials,
            scoring_metric=metric,
        )

    def _build_halving_searcher(
        self,
        config: TuningConfig,
        base_estimator: Any,
        cv: Any,
        metric: str,
        log_callback: Callable[[str], None] | None,
    ) -> Any:
        """Builds a HalvingGridSearchCV/HalvingRandomSearchCV searcher for the halving strategies."""
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
            return HalvingGridSearchCV(
                estimator=base_estimator,
                param_grid=self._clean_search_space(config.search_space),
                scoring=metric,
                cv=cv,
                n_jobs=config.n_jobs,
                random_state=config.random_state,
                refit=False,
                error_score=np.nan,
                factor=factor,
                resource=resource,
                min_resources=min_resources,
            )
        return HalvingRandomSearchCV(
            estimator=base_estimator,
            param_distributions=self._clean_search_space(config.search_space),
            n_candidates=config.n_trials,
            scoring=metric,
            cv=cv,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            refit=False,
            error_score=np.nan,
            factor=factor,
            resource=resource,
            min_resources=min_resources,
        )

    @staticmethod
    def _is_use_cmaes_numeric_list(v: Any, use_cmaes: bool) -> bool:
        """Returns whether ``v`` is a non-empty numeric list that should become a continuous range."""
        return (
            isinstance(v, list)
            and use_cmaes
            and bool(v)
            and all(isinstance(x, (int, float)) for x in v)
        )

    @staticmethod
    def _numeric_range_distribution(v: list) -> Any:
        """Builds an Optuna Int/FloatDistribution spanning the min/max of a numeric list."""
        lo, hi = min(v), max(v)
        if all(isinstance(x, int) for x in v):
            return optuna.distributions.IntDistribution(lo, hi)
        return optuna.distributions.FloatDistribution(float(lo), float(hi))

    @staticmethod
    def _distribution_for_value(k: str, v: Any, use_cmaes: bool) -> Any:
        """Builds the Optuna distribution for a single search-space entry.

        Numeric lists become continuous ``IntDistribution``/``FloatDistribution`` under
        CMA-ES (so it samples the full range); everything else stays categorical.
        """
        if TuningCalculator._is_use_cmaes_numeric_list(v, use_cmaes):
            return TuningCalculator._numeric_range_distribution(v)
        if isinstance(v, list):
            return optuna.distributions.CategoricalDistribution(v)
        return v

    @staticmethod
    def _build_optuna_distributions(
        search_space: dict[str, Any], use_cmaes: bool
    ) -> dict[str, Any]:
        """Converts a raw search space into Optuna distributions.

        Numeric lists become continuous ``IntDistribution``/``FloatDistribution`` under
        CMA-ES (so it samples the full range); everything else stays categorical.
        """
        return {
            k: TuningCalculator._distribution_for_value(k, v, use_cmaes)
            for k, v in search_space.items()
        }

    @staticmethod
    def _build_optuna_sampler(sampler_name: str, random_state: Any) -> Any:
        """Builds the Optuna sampler for the configured sampler name (random/cmaes/tpe)."""
        if sampler_name == "random":
            return optuna.samplers.RandomSampler(seed=random_state)
        if sampler_name == "cmaes":
            # Suppress the fallback warning for genuinely categorical params
            # (strings, booleans, None) — those can never be continuous and
            # the random fallback for them is expected behaviour.
            return optuna.samplers.CmaEsSampler(seed=random_state, warn_independent_sampling=False)
        return optuna.samplers.TPESampler(seed=random_state)

    @staticmethod
    def _build_optuna_pruner(pruner_name: str) -> Any:
        """Builds the Optuna pruner for the configured pruner name (hyperband/none/median)."""
        if pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        if pruner_name == "none":
            return optuna.pruners.NopPruner()
        return optuna.pruners.MedianPruner()

    def _build_optuna_searcher(
        self,
        config: TuningConfig,
        base_estimator: Any,
        cv: Any,
        metric: str,
        progress_callback: Callable[[int, int, float | None, dict | None], None] | None,
        log_callback: Callable[[str], None] | None,
    ) -> Any:
        """Builds an OptunaSearchCV searcher, wiring up distributions, sampler, pruner, and callbacks."""
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
        distributions = self._build_optuna_distributions(config.search_space, use_cmaes)

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
        sampler = self._build_optuna_sampler(sampler_name, config.random_state)

        # Pruner Selection
        pruner_name = strategy_params.get("pruner", "median")
        pruner = self._build_optuna_pruner(pruner_name)

        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

        return OptunaSearchCV(
            estimator=base_estimator,
            param_distributions=distributions,
            n_trials=config.n_trials,
            timeout=config.timeout,
            cv=cv,
            scoring=metric,
            n_jobs=config.n_jobs,
            refit=False,
            verbose=0,
            callbacks=callbacks,
            study=study,
        )

    @staticmethod
    def _to_numpy(data: Any) -> Any:
        """Converts a pandas object to a numpy array, leaving numpy arrays unchanged."""
        return data.to_numpy() if hasattr(data, "to_numpy") else data

    def _execute_search(self, searcher: Any, X_arr: Any, y_arr: Any, config: TuningConfig) -> None:
        """Fits the searcher, translating known sklearn/optuna failure messages into
        actionable ``ValueError``s and re-raising anything else unchanged.
        """
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
                if config.parallel_backend:
                    with parallel_backend(config.parallel_backend):
                        searcher.fit(X_arr, y_arr)
                else:
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

    @staticmethod
    def _extract_best_result(searcher: Any) -> tuple[Any, float]:
        """Reads ``best_params_``/``best_score_`` off a fitted searcher, translating the
        "no completed trials" ``ValueError`` into a clearer, actionable message.
        """
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
        return best_params, best_score

    @staticmethod
    def _collect_trials(searcher: Any, config: TuningConfig) -> list[dict[str, Any]]:
        """Extracts per-trial params/scores from a fitted searcher (Optuna study or cv_results_)."""
        trials: list[dict[str, Any]] = []
        # Special handling for Optuna
        if config.strategy == "optuna" and hasattr(searcher, "study_"):
            # Only include completed trials
            trials.extend(
                {"params": trial.params, "score": trial.value}
                for trial in cast(Any, searcher).study_.trials
                if trial.state.name == "COMPLETE"
            )
        elif hasattr(searcher, "cv_results_"):
            results = searcher.cv_results_
            if "params" in results:
                n_candidates = len(results["params"])
                trials.extend(
                    {
                        "params": results["params"][i],
                        "score": results["mean_test_score"][i],
                    }
                    for i in range(n_candidates)
                )
        return trials

    @staticmethod
    def _log_final_completion(
        log_callback: Callable[[str], None] | None,
        config: TuningConfig,
        trials: list[dict[str, Any]],
        best_score: float,
        best_params: Any,
    ) -> None:
        """Emits the completion log for searcher-based strategies that don't emit
        per-trial callbacks (halving_grid / halving_random / optuna).
        """
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

    def tune(
        self,
        X: Any,
        y: Any,
        config: TuningConfig,
        progress_callback: Callable[[int, int, float | None, dict | None], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        validation_data: tuple[Any, Any] | None = None,
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
        model_class: Any = self.model_calculator.model_class

        # ``default_params`` may carry structural args (e.g. an ensemble's
        # resolved ``estimators``); the instantiator filters/routes them safely.
        base_estimator = self._instantiate_model(model_class, self.model_calculator.default_params)

        # 2. Prepare Splitter
        # If validation data is provided, use PredefinedSplit to train on X and validate on validation_data
        # Otherwise use CV
        cv, X_for_search, y_for_search = self._build_cv_splitter(X, y, config, validation_data)

        # 3. Select Search Strategy
        # Handle multiclass metrics and map user-friendly names
        metric = self._resolve_metric(config, y)

        if config.strategy in ["grid", "random"]:
            # Use custom loop to support progress and log callbacks
            return self._run_grid_or_random_search(
                X_for_search,
                y_for_search,
                config,
                model_class,
                cv,
                metric,
                progress_callback,
                log_callback,
            )
        elif config.strategy in ["halving_grid", "halving_random"]:
            searcher = self._build_halving_searcher(
                config, base_estimator, cv, metric, log_callback
            )
        elif config.strategy == "optuna":
            searcher = self._build_optuna_searcher(
                config, base_estimator, cv, metric, progress_callback, log_callback
            )
        else:
            raise ValueError(f"Unknown tuning strategy: {config.strategy}")

        # 4. Run Search
        # Ensure numpy
        X_arr = self._to_numpy(X_for_search)
        y_arr = self._to_numpy(y_for_search)
        self._execute_search(searcher, X_arr, y_arr, config)

        # 5. Extract Results
        best_params, best_score = self._extract_best_result(searcher)

        # Collect trials
        trials = self._collect_trials(searcher, config)

        # Final completion log for strategies that don't emit per-trial callbacks
        # (halving_grid / halving_random / optuna). The grid/random branch above
        # already logs completion inside its custom loop.
        self._log_final_completion(log_callback, config, trials, best_score, best_params)

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
        df: pd.DataFrame | SkyulfDataFrame,
        model_artifact: Any,
    ) -> pd.Series | Any:
        # model_artifact is (fitted_model, tuning_result)
        if isinstance(model_artifact, tuple) and len(model_artifact) == 2:
            model, _ = model_artifact
            return self.base_applier.predict(df, model)
        # Fallback: artifact isn't the expected (model, tuning_result) tuple
        # (e.g. a plain fitted model, before it's wrapped by the tuner). Return
        # an all-null placeholder of the right length/engine instead of
        # crashing - `df.index` doesn't exist on a Polars DataFrame, so build
        # the placeholder in an engine-aware way.
        if hasattr(df, "index"):
            return pd.Series(np.nan, index=df.index)
        return pd.Series(np.full(len(df), np.nan))

    def predict_proba(
        self,
        df: pd.DataFrame | SkyulfDataFrame,
        model_artifact: Any,
    ) -> pd.DataFrame | SkyulfDataFrame | None:
        if isinstance(model_artifact, tuple) and len(model_artifact) == 2:
            model, _ = model_artifact
            return self.base_applier.predict_proba(df, model)
        return None
