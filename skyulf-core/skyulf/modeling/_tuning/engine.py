"""Hyperparameter Tuner implementation."""

import contextlib
import logging
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.exceptions import ConvergenceWarning

# Explicitly enable experimental halving search cv
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline

from ..._validation import raise_invalid_choice
from ...engines import SkyulfDataFrame
from ...engines.sklearn_bridge import SklearnBridge
from .._sklearn_compat import normalize_logistic_regression_params
from ..base import BaseModelApplier, BaseModelCalculator
from .fold_pipeline import FoldAwareModelStep
from .reporter import ConsoleTrialReporter
from .schemas import TuningConfig, TuningResult

if TYPE_CHECKING:
    from ..fold_preprocessing import FoldPreprocessor

logger = logging.getLogger(__name__)

# Optuna is an optional, heavyweight dependency only needed when a caller
# actually requests `strategy="optuna"` tuning. Resolution (including its
# multi-path sklearn-integration fallback chain) is deferred to
# `_ensure_optuna_loaded()`, called only from `_build_optuna_searcher`, so
# merely importing `skyulf`/`skyulf.modeling` never imports optuna or emits
# its "OptunaSearchCV not found" warning for users who never use this
# strategy.
HAS_OPTUNA = False
OptunaSearchCV: Any = None
optuna: Any = None
_optuna_load_attempted = False


def _ensure_optuna_loaded() -> bool:
    """Lazily import Optuna and resolve its sklearn-compatible OptunaSearchCV
    integration, memoizing the result so repeated tuning calls don't
    re-attempt the (multi-path fallback) import every time.

    Populates the module-level ``optuna``/``HAS_OPTUNA``/``OptunaSearchCV``
    globals on success, so the existing ``_build_optuna_distributions``/
    ``_build_optuna_sampler``/``_build_optuna_pruner`` helpers (which
    reference the bare ``optuna`` module name) keep working unchanged, since
    they're only ever called from ``_build_optuna_searcher`` after it has
    already called this function.
    """
    global HAS_OPTUNA, OptunaSearchCV, optuna, _optuna_load_attempted
    if _optuna_load_attempted:
        return HAS_OPTUNA
    _optuna_load_attempted = True

    try:
        import optuna as _optuna  # ty: ignore[unresolved-import]

        optuna = _optuna
        HAS_OPTUNA = True
    except ImportError:
        return HAS_OPTUNA

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
    return HAS_OPTUNA


class TuningCalculator(BaseModelCalculator):
    """Tune a plain ``BaseModelCalculator`` and refit it with the best parameters.

    ``fit`` accepts a ``TuningConfig`` or dict with ``strategy``, ``metric``,
    ``n_trials``, ``search_space`` (a mapping of parameter names to candidate
    lists), and optional cross-validation and ``strategy_params`` settings.
    ``grid`` and ``random`` use Skyulf's candidate loop; ``halving_grid`` and
    ``halving_random`` use sklearn searchers; ``optuna`` uses OptunaSearchCV
    and requires the Optuna integration package.

    Examples:
        >>> tuner = TuningCalculator(LogisticRegressionCalculator())
        >>> model, result = tuner.fit(X, y, {"strategy": "random", "search_space": {"C": [0.1, 1.0]}})
    """

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

        # LogisticRegression-only: sklearn >=1.8 deprecates the ``penalty``
        # constructor arg. The tuning engine builds estimators directly
        # (bypassing LogisticRegressionCalculator._resolve_fit_params), so a
        # ``penalty`` coming from the search space/best_params would otherwise
        # reach sklearn unnormalized and trigger the FutureWarning on every
        # fold fit and the final refit. Other models (e.g. SGDClassifier) also
        # have a ``penalty`` param with different, non-deprecated semantics,
        # so this must stay scoped to LogisticRegression specifically.
        if model_class is LogisticRegression:
            flat = normalize_logistic_regression_params(flat)

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
        *,
        allow_nan: bool = False,
    ) -> None:
        """Raise ValueError if a numpy array contains NaN/Inf (numeric) or NaN (object dtype).

        Many tuning errors ("No trials completed") are actually due to dirty data causing
        instant failures. We catch this early to give a clear message. Object-dtype arrays
        (e.g. mixed dtypes or leftover categorical/string columns that were never encoded)
        are also scanned via pd.isna, since np.isnan/np.isinf raise on non-numeric dtypes.

        ``allow_nan`` skips the NaN checks for models that handle missing values
        natively (XGBoost, LightGBM, HistGradientBoosting); Inf is still rejected.
        """
        if not isinstance(arr, np.ndarray):
            return
        if np.issubdtype(arr.dtype, np.number):
            if not allow_nan and np.isnan(arr).any():
                raise ValueError(nan_msg)
            if np.isinf(arr).any():
                raise ValueError(inf_msg)
        elif arr.dtype == object and not allow_nan and pd.isna(arr).any():
            raise ValueError(object_nan_msg)

    # Model classes that natively handle missing values in X; NaN must not be
    # rejected for these (y still is — no estimator accepts missing targets).
    _MISSING_NATIVE_MODEL_CLASSES = frozenset(
        {
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "HistGradientBoostingClassifier",
            "HistGradientBoostingRegressor",
        }
    )

    def _refit_best_model(
        self,
        tuning_result: TuningResult,
        tuning_config: TuningConfig,
        X_np: Any,
        y_np: Any,
        log_callback: Callable[[str], None] | None,
        iteration_callback: Callable[..., None] | None = None,
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
        # Boosting base calculators (XGBoost/LightGBM) attach an eval set +
        # iteration callback here so the final refit streams per-round
        # progress like any other boosting fit; every other model keeps a
        # plain fit.
        boosting_hook = getattr(self.model_calculator, "_boosting_fit_kwargs", None)
        extra_fit_kwargs = (
            boosting_hook(model, X_np, y_np, iteration_callback)
            if boosting_hook is not None
            else {}
        )
        detach_callbacks = extra_fit_kwargs.pop("_detach_callbacks", False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            model.fit(X_np, y_np, **extra_fit_kwargs)
        if detach_callbacks:
            model.callbacks = None
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                conv_msg = (
                    f"Final refit of {model_cls.__name__} with the best params did not "
                    f"fully converge: {w.message}"
                )
                logger.warning(conv_msg)
                if log_callback:
                    log_callback(conv_msg)
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        return model

    def fit(
        self,
        X: pd.DataFrame | SkyulfDataFrame,
        y: pd.Series | Any,
        config: dict[str, Any] | TuningConfig,
        progress_callback: Callable[[int, int, float | None, dict | None], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        validation_data: tuple[pd.DataFrame | SkyulfDataFrame, pd.Series | Any] | None = None,
        iteration_callback: Callable[..., None] | None = None,
        preprocessing: "FoldPreprocessor | None" = None,
        validation_frames: tuple[Any, Any] | None = None,
    ) -> Any:
        """
        Fits the tuner (runs tuning).
        Adapts the generic fit interface to the specific tune method.

        ``preprocessing`` (F-15): re-fits the given preprocessor on
        each candidate fold's training rows during search, so tuning scores
        stop leaking held-out rows into preprocessing statistics. The custom
        ``grid``/``random`` loop applies it directly; ``halving_*``/``optuna``
        get it via a fold-aware meta-estimator whose ``fit`` runs the chain
        inside every searcher-internal fold — safe even for chains that
        resample rows or re-encode the target. Holdout tuning
        (``validation_data`` set) refits the chain on the train rows only and
        scores candidates against the untouched validation split when
        ``validation_frames`` (the pre-transform validation payload) is
        supplied; frameless calls fall back to raw-payload scoring with an
        explicit log instead. When set, the final best-model refit runs the
        full split through the preprocessor once — the artifact serving uses.
        Polars payloads are accepted for both frame pairs — the engine and
        the fold-aware wrap convert them via ``to_pandas()`` with dtypes
        intact.
        """
        tuning_config = self._build_tuning_config(config)

        # Core-only console progress: the backend always supplies its own
        # callback (and never sets `progress`), so this only ever fires for
        # direct SDK use.
        reporter: ConsoleTrialReporter | None = None
        if progress_callback is None and tuning_config.progress:
            reporter = ConsoleTrialReporter()
            progress_callback = reporter

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
        # Models with native missing-value support (XGBoost, LightGBM,
        # HistGradientBoosting) accept NaN in X; forcing an Imputer on them
        # would wrongly block a legitimate configuration. y is always checked.
        model_cls = getattr(self.model_calculator, "model_class", None)
        x_allows_nan = (
            model_cls is not None
            and getattr(model_cls, "__name__", "") in self._MISSING_NATIVE_MODEL_CLASSES
        )
        # With per-fold refit, X is the pre-transform payload: NaN there is
        # legitimate when the pipeline carries an imputer, which re-runs on
        # each fold's training rows before the model ever sees the data.
        self._validate_no_nan_inf(
            X_np,
            "Input features (X) contain NaN values. Please use an 'Imputer' node before this model.",
            "Input features (X) contain Infinite values. Please scale or clean your data.",
            "Input features (X) contain missing/NaN values. Please use an 'Imputer' node before this model.",
            allow_nan=x_allows_nan or preprocessing is not None,
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

        # Wrap the whole candidate/fold search: sklearn's ConvergenceWarning
        # (raised via `warnings.warn`, not the `logging` module) would
        # otherwise only reach the server's stderr, once per fold/candidate,
        # and never surface to the user. Aggregate into a single summary log
        # line instead of one per fold (a full grid/random search can run
        # hundreds of fold fits) and re-emit any other warning category
        # unchanged so existing behavior for those is preserved.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tuning_result = self.tune(
                X_np,
                y_np,
                tuning_config,
                progress_callback=progress_callback,
                log_callback=log_callback,
                validation_data=validation_data_np,
                preprocessing=preprocessing,
                preprocessing_frames=(X, y) if preprocessing is not None else None,
                validation_frames=validation_frames,
            )
        convergence_count = 0
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                convergence_count += 1
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if convergence_count:
            conv_msg = (
                f"{convergence_count} candidate fit(s) during hyperparameter search did not "
                "fully converge (max_iter reached). Consider increasing max_iter, scaling "
                "features, or picking a different solver."
            )
            logger.warning(conv_msg)
            if log_callback:
                log_callback(conv_msg)

        # Refit the best model on the full dataset. With per-fold refit
        # enabled, the full dataset is the full-split frame run once through
        # the preprocessor — the same artifact the folds were scored against
        # and what serving will use for predictions.
        if preprocessing is not None:
            X_refit_frame, y_refit_frame = preprocessing.fit_transform(X, y)
            X_refit, y_refit = SklearnBridge.to_sklearn((X_refit_frame, y_refit_frame))
        else:
            X_refit, y_refit = X_np, y_np
        model = self._refit_best_model(
            tuning_result,
            tuning_config,
            X_refit,
            y_refit,
            log_callback,
            iteration_callback=iteration_callback,
        )

        if reporter is not None:
            reporter.finish(tuning_result)

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

        Numpy (frameless) variant — the frame-based per-fold-refit variant is
        ``_build_predefined_split_cv_frames``.

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

    def _build_predefined_split_cv_frames(
        self,
        preprocessing_frames: tuple[Any, Any],
        validation_frames: tuple[Any, Any],
    ) -> tuple[Any, Any, Any]:
        """Frame variant of ``_build_predefined_split_cv`` for per-fold refit.

        Concatenates the pre-transform train and validation frames (positional
        index reset so the mask aligns) and builds a ``PredefinedSplit`` where
        train rows are always in training (-1) and validation rows form the
        single scoring fold (0): the preprocessing chain refits on train rows
        only and candidates score against untouched validation rows.
        """
        from sklearn.model_selection import PredefinedSplit

        def _as_pandas(frame: Any) -> Any:
            return frame.to_pandas() if hasattr(frame, "to_pandas") else frame

        X_train, y_train = preprocessing_frames
        X_val, y_val = validation_frames

        X_for_search = pd.concat([_as_pandas(X_train), _as_pandas(X_val)], ignore_index=True)
        y_for_search = pd.concat([_as_pandas(y_train), _as_pandas(y_val)], ignore_index=True)

        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])

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
        preprocessing: "FoldPreprocessor | None" = None,
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
                preprocessing=preprocessing,
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
        preprocessing: "FoldPreprocessor | None" = None,
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
            # F-15: refit preprocessing inside the fold so its statistics
            # never see this fold's held-out rows (inside the try so a
            # preprocessing failure is contained like a model-fit failure).
            if preprocessing is not None:
                X_train_fold, y_train_fold = preprocessing.fit_transform(X_train_fold, y_train_fold)
                X_val_fold, y_val_fold = preprocessing.transform(X_val_fold, y_val_fold)

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
        preprocessing: "FoldPreprocessor | None" = None,
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
                i,
                params,
                model_class,
                cv,
                X_for_search,
                y_for_search,
                metric,
                log_callback,
                preprocessing,
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
        preprocessing: "FoldPreprocessor | None" = None,
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
            preprocessing,
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
        if not _ensure_optuna_loaded():
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

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"OptunaSearchCV is experimental.*",
                category=optuna.exceptions.ExperimentalWarning,
            )
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

    def _execute_search(
        self,
        searcher: Any,
        X_arr: Any,
        y_arr: Any,
        config: TuningConfig,
        log_callback: Callable[[str], None] | None = None,
    ) -> list[str]:
        """Fits the searcher, translating known sklearn/optuna failure messages into
        actionable ``ValueError``s and re-raising anything else unchanged.

        Returns the per-trial failure messages captured for optuna runs (one
        entry per failed trial). Optuna logs these at WARNING level on its own
        logger — without capturing them the only visible symptom is the generic
        "no trials completed" error, with the real cause stuck in stderr.
        """
        captured: list[str] = []
        optuna_logger: logging.Logger | None = None
        handler: logging.Handler | None = None
        if config.strategy == "optuna" and _ensure_optuna_loaded():

            class _TrialFailureHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    message = record.getMessage()
                    if "failed" in message:
                        captured.append(message)
                        if log_callback:
                            with contextlib.suppress(Exception):
                                log_callback(message)

            optuna_logger = logging.getLogger("optuna")
            handler = _TrialFailureHandler(level=logging.WARNING)
            optuna_logger.addHandler(handler)
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
        finally:
            if optuna_logger is not None and handler is not None:
                optuna_logger.removeHandler(handler)
        return captured

    @staticmethod
    def _extract_best_result(
        searcher: Any, first_trial_error: str | None = None
    ) -> tuple[Any, float]:
        """Reads ``best_params_``/``best_score_`` off a fitted searcher, translating the
        "no completed trials" ``ValueError`` into a clearer, actionable message that
        carries the first captured per-trial error when available.
        """
        try:
            # Accessing best_params_ raises ValueError if no trials completed successfully
            best_params = searcher.best_params_
            best_score = searcher.best_score_
        except ValueError as e:
            if "No trials are completed yet" in str(e):
                detail = f" First trial error: {first_trial_error}" if first_trial_error else ""
                raise ValueError(
                    "Hyperparameter tuning failed: All trials failed. "
                    "This often happens if the model produces NaN scores "
                    "(e.g., due to unscaled data for linear models/SVMs, exploding gradients, "
                    "or mismatched parameters). "
                    "Try adding a 'Scale' node before this model or checking for NaN/Infinity in your data."
                    + detail
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
    def _strip_model_prefix(params: Any) -> Any:
        """Removes the internal ``model__estimator__`` pipeline prefix from
        extracted params (see ``tune``'s wrapped Pipeline path) so callers see
        the original search-space keys."""
        if not isinstance(params, dict):
            return params
        return {
            key.removeprefix("model__").removeprefix("estimator__"): value
            for key, value in params.items()
        }

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
        preprocessing: "FoldPreprocessor | None" = None,
        preprocessing_frames: tuple[Any, Any] | None = None,
        validation_frames: tuple[Any, Any] | None = None,
    ) -> TuningResult:
        """
        Runs hyperparameter tuning.
        """
        # Holdout tuning with per-fold preprocessing refit: the train and
        # validation frames are concatenated (train rows masked -1 in a
        # PredefinedSplit) so every strategy refits the chain on train rows
        # only and scores candidates against the untouched validation split.
        holdout_refit = (
            preprocessing is not None
            and validation_data is not None
            and preprocessing_frames is not None
            and validation_frames is not None
        )
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

        # halving/optuna searchers run their CV internally, where the engine's
        # per-fold hook cannot reach; wrap preprocessing + model in one
        # fit-time meta-estimator so the searcher's own folds drive a true
        # per-fold refit (F-15). Preprocessing runs inside ``fit`` on the
        # fold's training rows, so chains that resample rows or re-encode
        # the target are safe here too.
        searcher_strategy = config.strategy in ("halving_grid", "halving_random", "optuna")
        wrapped = preprocessing is not None and searcher_strategy
        if wrapped and preprocessing_frames is None:
            # Numpy-only SDK call: the preprocessor needs named frames to
            # run, so the wrap cannot be built. Keep today's behaviour —
            # score the raw pre-transform payload — with an explicit log.
            wrapped = False
            if log_callback:
                log_callback(
                    "Per-fold preprocessing refit skipped for this tuning strategy: "
                    "no named frames are available to run the preprocessing chain "
                    "inside the searcher's folds. Scores are computed on the raw "
                    "pre-transform payload."
                )
        if wrapped and validation_data is not None and validation_frames is None:
            # Holdout tuning needs the validation rows as named frames too
            # (the searcher scores them in the original pre-transform space).
            # Without them the wrap would score misaligned rows, so fall back
            # to raw-payload scoring with an explicit log instead.
            wrapped = False
            if log_callback:
                log_callback(
                    "Per-fold preprocessing refit skipped for holdout tuning: "
                    "no named validation frames are available to score against. "
                    "Scores are computed on the raw pre-transform payload."
                )
        estimator: Any = base_estimator
        search_config = config
        if wrapped and preprocessing_frames is not None:
            frame_x = preprocessing_frames[0]
            feature_names = (
                tuple(map(str, frame_x.columns)) if hasattr(frame_x, "columns") else None
            )
            estimator = Pipeline(
                [
                    (
                        "model",
                        FoldAwareModelStep(
                            estimator=base_estimator,
                            preprocessor=preprocessing,
                            feature_names=feature_names,
                        ),
                    )
                ]
            )
            if log_callback:
                log_callback(
                    "Per-fold preprocessing refit runs inside the searcher via "
                    "the fold-aware estimator."
                )
            # Search-space keys must route through the pipeline to the base estimator.
            search_config = replace(
                config,
                search_space={
                    f"model__estimator__{key}": values
                    for key, values in (config.search_space or {}).items()
                },
            )

        # 2. Prepare Splitter
        # If validation data is provided, use PredefinedSplit to train on X and validate on validation_data
        # Otherwise use CV
        if holdout_refit and preprocessing_frames is not None and validation_frames is not None:
            cv, X_for_search, y_for_search = self._build_predefined_split_cv_frames(
                preprocessing_frames, validation_frames
            )
        else:
            cv, X_for_search, y_for_search = self._build_cv_splitter(X, y, config, validation_data)

        # 3. Select Search Strategy
        # Handle multiclass metrics and map user-friendly names
        metric = self._resolve_metric(config, y)

        if config.strategy in ["grid", "random"]:
            # Per-fold refit needs named frames (the adapter rebuilds a real
            # FeatureEngineer), not the numpy arrays used by the searchers.
            # Holdout mode is excluded: the splitter stage already produced
            # the concatenated train+validation frames the mask is aligned to.
            if (
                preprocessing is not None
                and preprocessing_frames is not None
                and validation_data is None
            ):
                X_for_search, y_for_search = preprocessing_frames
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
                preprocessing,
            )
        elif config.strategy in ["halving_grid", "halving_random"]:
            searcher = self._build_halving_searcher(
                search_config, estimator, cv, metric, log_callback
            )
        elif config.strategy == "optuna":
            searcher = self._build_optuna_searcher(
                search_config, estimator, cv, metric, progress_callback, log_callback
            )
        else:
            raise_invalid_choice(
                config.strategy,
                ("grid", "random", "halving_grid", "halving_random", "optuna"),
                "tuning strategy",
            )

        # 4. Run Search
        # The wrapped pipeline needs named frames (the adapter rebuilds a real
        # FeatureEngineer); numpy conversion would strip column names. In
        # holdout mode the splitter stage already produced the concatenated
        # train+validation frames the PredefinedSplit mask is aligned to.
        if wrapped:
            if validation_data is None and preprocessing_frames is not None:
                X_for_search, y_for_search = preprocessing_frames
            X_arr, y_arr = X_for_search, y_for_search
        else:
            X_arr = self._to_numpy(X_for_search)
            y_arr = self._to_numpy(y_for_search)
        trial_errors = self._execute_search(searcher, X_arr, y_arr, config, log_callback)

        # 5. Extract Results
        first_trial_error = trial_errors[0] if trial_errors else None
        best_params, best_score = self._extract_best_result(searcher, first_trial_error)

        # Collect trials
        trials = self._collect_trials(searcher, config)
        if wrapped:
            best_params = self._strip_model_prefix(best_params)
            trials = [
                {**trial, "params": self._strip_model_prefix(trial["params"])} for trial in trials
            ]

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
