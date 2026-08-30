"""Hyperparameter Tuner implementation.

Orchestrator module (F-18): holds ``TuningCalculator.fit``/``tune`` and
``TuningApplier``. The individual responsibilities live in sibling leaf
modules: ``params`` (search-space cleaning + estimator instantiation),
``splitters`` (CV splitter construction), ``metrics`` (metric validation +
scorer resolution), ``grid_random`` (the grid/random fold loop), ``refit``
(best-model refit + decision-threshold tuning), and ``strategies/``
(halving, optuna, and the shared searcher runner). The underscored methods
kept on ``TuningCalculator`` are one-line delegates preserved for the
existing test surface.
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from ..._validation import raise_invalid_choice
from ...engines import SkyulfDataFrame
from ...engines.sklearn_bridge import SklearnBridge
from .._evaluation.thresholds import apply_thresholds
from ..base import BaseModelApplier, BaseModelCalculator
from ..cross_validation import _sort_by_time
from . import splitters
from .fold_pipeline import FoldAwareModelStep
from .grid_random import fit_and_score_candidate_fold, run_grid_or_random_search
from .metrics import is_multiclass_target, resolve_metric, resolve_scorer
from .params import clean_search_space, instantiate_model, seed_params
from .refit import refit_best_model, resolve_threshold_metric, tune_decision_thresholds
from .reporter import ConsoleTrialReporter
from .schemas import TuningConfig, TuningResult
from .strategies import halving as _halving
from .strategies import optuna as _optuna_strategy
from .strategies import runner as _runner
from .strategies.optuna import _ensure_optuna_loaded, _optuna_state, _OptunaLoadState

if TYPE_CHECKING:
    from ..fold_preprocessing import FoldPreprocessor

logger = logging.getLogger(__name__)

# The optuna lazy-loader state lives in ``strategies/optuna.py``; the names
# below are re-exported so existing ``engine._optuna_state`` /
# ``engine._ensure_optuna_loaded`` access keeps working (F-14 compat).
__all__ = [
    "TuningApplier",
    "TuningCalculator",
    "_OptunaLoadState",
    "_ensure_optuna_loaded",
    "_optuna_state",
]


def __getattr__(name: str) -> Any:
    """Read-only views of the optuna load state under its legacy names (F-14).

    Reads the live module global so tests that replace this module's
    ``_optuna_state`` binding see their replacement through the views.
    """
    if name == "HAS_OPTUNA":
        return _optuna_state.has_optuna
    if name == "OptunaSearchCV":
        return _optuna_state.search_cv
    if name == "optuna":
        return _optuna_state.optuna_module
    if name == "_optuna_load_attempted":
        return _optuna_state.attempted
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

    # ------------------------------------------------------------------
    # Thin delegates to the split leaf modules (F-18). Kept so the pinned
    # internal surface — direct calls and monkeypatches in the test suite —
    # keeps resolving on ``TuningCalculator``.
    # ------------------------------------------------------------------

    def _clean_search_space(self, search_space: dict[str, Any]) -> dict[str, Any]:
        return clean_search_space(search_space)

    @staticmethod
    def _instantiate_model(model_class: Any, params: dict[str, Any]) -> Any:
        return instantiate_model(model_class, params)

    def _resolve_scorer(self, metric: str, y: Any) -> Any:
        return resolve_scorer(metric, y, getattr(self.model_calculator, "problem_type", None))

    @staticmethod
    def _is_multiclass_target(y: Any) -> bool:
        return is_multiclass_target(y)

    def _refit_best_model(
        self,
        tuning_result: TuningResult,
        tuning_config: TuningConfig,
        X_np: Any,
        y_np: Any,
        log_callback: Callable[[str], None] | None,
        iteration_callback: Callable[..., None] | None = None,
    ) -> Any:
        return refit_best_model(
            self.model_calculator,
            tuning_result,
            tuning_config,
            X_np,
            y_np,
            log_callback,
            iteration_callback,
        )

    def _resolve_threshold_metric(
        self,
        metric_name: str,
        log_callback: Callable[[str], None] | None,
        pos_label: Any = None,
    ) -> tuple[Callable[[Any, Any], float], str]:
        return resolve_threshold_metric(metric_name, log_callback, pos_label)

    def _tune_decision_thresholds(
        self,
        model: Any,
        tuning_result: TuningResult,
        tuning_config: TuningConfig,
        validation_data: tuple[Any, Any] | None,
        log_callback: Callable[[str], None] | None,
    ) -> None:
        tune_decision_thresholds(
            self.model_calculator,
            model,
            tuning_result,
            tuning_config,
            validation_data,
            log_callback,
        )

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
        fold_errors: list[str] | None = None,
        seed_params_overlay: dict[str, Any] | None = None,
    ) -> float:
        return fit_and_score_candidate_fold(
            candidate_idx,
            fold_idx,
            params,
            model_class,
            cv,
            X_any,
            y_any,
            X_arr,
            y_arr,
            train_idx,
            val_idx,
            metric,
            log_callback,
            preprocessing,
            fold_errors,
            seed_params_overlay,
            model_calculator=self.model_calculator,
        )

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
        return run_grid_or_random_search(
            X_for_search,
            y_for_search,
            config,
            model_class,
            cv,
            metric,
            progress_callback,
            log_callback,
            preprocessing,
            model_calculator=self.model_calculator,
        )

    @staticmethod
    def _collect_trials(searcher: Any, config: TuningConfig) -> list[dict[str, Any]]:
        return _runner.collect_trials(searcher, config)

    @staticmethod
    def _strip_model_prefix(params: Any) -> Any:
        return _runner.strip_model_prefix(params)

    # ------------------------------------------------------------------
    # Config + input validation (kept on the orchestrator).
    # ------------------------------------------------------------------

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
        model = refit_best_model(
            self.model_calculator,
            tuning_result,
            tuning_config,
            X_refit,
            y_refit,
            log_callback,
            iteration_callback=iteration_callback,
        )

        # F-13: optionally search a decision threshold for a binary
        # classifier on the validation split. Best-effort — a failure logs and
        # leaves predict() on the default decision rule.
        if tuning_config.tune_threshold:
            tune_decision_thresholds(
                self.model_calculator,
                model,
                tuning_result,
                tuning_config,
                validation_data,
                log_callback,
            )

        if reporter is not None:
            reporter.finish(tuning_result)

        return (model, tuning_result)

    @staticmethod
    def _to_numpy(data: Any) -> Any:
        """Converts a pandas object to a numpy array, leaving numpy arrays unchanged."""
        return data.to_numpy() if hasattr(data, "to_numpy") else data

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
        base_estimator = instantiate_model(
            model_class, {**self.model_calculator.default_params, **seed_params(config)}
        )

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
            cv, X_for_search, y_for_search = splitters.build_predefined_split_cv_frames(
                preprocessing_frames, validation_frames
            )
        else:
            cv, X_for_search, y_for_search = splitters.build_cv_splitter(
                X, y, config, validation_data, self.model_calculator.problem_type
            )

        # 3. Select Search Strategy
        # Handle multiclass metrics and map user-friendly names
        metric = resolve_metric(config, y, self.model_calculator.problem_type)

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
            return run_grid_or_random_search(
                X_for_search,
                y_for_search,
                config,
                model_class,
                cv,
                metric,
                progress_callback,
                log_callback,
                preprocessing,
                model_calculator=self.model_calculator,
            )
        elif config.strategy in ["halving_grid", "halving_random", "optuna"]:
            # Searchers score the payload handed to them below: the raw
            # pre-transform frames when wrapped (the fold-aware step encodes
            # inside fit but maps predictions back), otherwise the payload
            # labels. Resolve the scorer against that label space so binary
            # scorers get a valid pos_label even for string targets.
            scoring_y = (
                preprocessing_frames[1]
                if wrapped and preprocessing_frames is not None
                else y_for_search
            )
            scoring = resolve_scorer(
                metric, scoring_y, getattr(self.model_calculator, "problem_type", None)
            )
            if config.strategy in ["halving_grid", "halving_random"]:
                searcher = _halving.build_halving_searcher(
                    search_config, estimator, cv, scoring, log_callback
                )
            else:
                searcher = _optuna_strategy.build_optuna_searcher(
                    search_config, estimator, cv, scoring, progress_callback, log_callback
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
        trial_errors = _runner.execute_search(searcher, X_arr, y_arr, config, log_callback)

        # 5. Extract Results
        first_trial_error = trial_errors[0] if trial_errors else None
        best_params, best_score = _runner.extract_best_result(searcher, first_trial_error)

        # Collect trials
        trials = _runner.collect_trials(searcher, config)
        if wrapped:
            best_params = _runner.strip_model_prefix(best_params)
            trials = [
                {**trial, "params": _runner.strip_model_prefix(trial["params"])} for trial in trials
            ]

        # Final completion log for strategies that don't emit per-trial callbacks
        # (halving_grid / halving_random / optuna). The grid/random branch above
        # already logs completion inside its custom loop.
        _runner.log_final_completion(log_callback, config, trials, best_score, best_params)

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
            model, tuning_result = model_artifact
            thresholds = getattr(tuning_result, "decision_thresholds", None)
            if (
                thresholds is not None
                and hasattr(model, "predict_proba")
                and hasattr(model, "classes_")
            ):
                # F-13: apply the decision thresholds tuned on the validation
                # split instead of the model's default decision rule.
                X_np, _ = SklearnBridge.to_sklearn(df)
                y_proba = model.predict_proba(X_np)
                preds = apply_thresholds(y_proba, thresholds, classes=model.classes_)
                index = df.index if hasattr(df, "index") else None
                return pd.Series(preds, index=index)
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
