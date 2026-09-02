"""Wrapper for Scikit-Learn models."""

import inspect
import logging
import warnings
from typing import Any, cast

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_sample_weight

from ..engines import SkyulfDataFrame
from ..engines.sklearn_bridge import SklearnBridge
from ..types import DEFAULT_RANDOM_STATE
from .base import BaseModelApplier, BaseModelCalculator

logger = logging.getLogger(__name__)


class SklearnCalculator(BaseModelCalculator):
    """Base calculator for Scikit-Learn models."""

    def __init__(
        self,
        model_class: type[BaseEstimator],
        default_params: dict[str, Any],
        problem_type: str,
    ):
        # `Any` because sklearn stubs make BaseEstimator subclasses appear non-callable.
        self.model_class: Any = model_class
        self._default_params = default_params
        self._problem_type = problem_type

    @property
    def default_params(self) -> dict[str, Any]:
        return self._default_params

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(
        self,
        X: pd.DataFrame | SkyulfDataFrame,
        y: pd.Series | Any,
        config: dict[str, Any],
        progress_callback=None,
        log_callback=None,
        validation_data=None,
        iteration_callback=None,
    ) -> Any:
        """Fit the Scikit-Learn model."""
        # 1. Merge Config with Defaults
        params = self._resolve_fit_params(config)

        # A generic <select> UI element always submits its option value as a
        # string, so a "None" option (e.g. "no class weighting") arrives here
        # as the literal string "None", not Python None. Normalize that back
        # to None before anything below decides whether class weighting was
        # actually requested.
        if params.get("class_weight") in ("None", "none", ""):
            params["class_weight"] = None

        # Some estimators (e.g. XGBoost's sklearn wrapper) accept arbitrary
        # **kwargs in their constructor but have no built-in notion of class
        # weighting: a `class_weight` kwarg is silently stored and ignored at
        # fit time (no error — just a native warning). Detect that case up
        # front (by checking whether `class_weight` is an explicitly named
        # constructor parameter, not just swallowed by **kwargs) and, if the
        # value isn't None, translate it into a `sample_weight` array passed
        # to `.fit()` instead, so "balanced"/dict class weighting behaves the
        # same regardless of whether the underlying library supports it
        # natively.
        class_weight_to_apply = None
        if "class_weight" in params and not self._constructor_accepts_class_weight():
            class_weight_to_apply = params.pop("class_weight")

        msg = f"Initializing {self.model_class.__name__} with params: {params}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)

        # 2. Instantiate Model
        valid_params = self._filter_supported_params(params)
        model = self.model_class(**valid_params)

        # 3. Fit
        # Convert to Numpy using Bridge (handles Polars/Pandas/Wrappers)
        X_np, y_np = SklearnBridge.to_sklearn((X, y))

        sample_weight = None
        if class_weight_to_apply is not None:
            sample_weight = self._compute_sample_weight_for_fit(model, class_weight_to_apply, y_np)

        # sklearn's ConvergenceWarning (raised via `warnings.warn`, not the
        # `logging` module) would otherwise only reach the server's stderr
        # and never surface to the user — unlike the skyulf-core node
        # advisories already routed through `WarningCaptureHandler` via
        # `logger.warning(...)`. Capture everything sklearn emits during
        # `fit`, re-route ConvergenceWarning through this model's own
        # (``skyulf.*``-tree) logger so every sklearn-backed model gets the
        # same UI-visible treatment regardless of solver/estimator, and
        # re-emit any other warning category unchanged so existing
        # console/log behavior for those is preserved.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            boosting_kwargs = self._boosting_fit_kwargs(model, X_np, y_np, iteration_callback)
            detach_callbacks = boosting_kwargs.pop("_detach_callbacks", False)
            if sample_weight is not None:
                model.fit(X_np, y_np, sample_weight=sample_weight, **boosting_kwargs)
            else:
                model.fit(X_np, y_np, **boosting_kwargs)
        # Never leave live callback closures on the fitted model — artifacts
        # get pickled for storage/serving.
        if detach_callbacks:
            model.callbacks = None
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                conv_msg = f"{self.model_class.__name__} did not fully converge: {w.message}"
                logger.warning(conv_msg)
                if log_callback:
                    log_callback(conv_msg)
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        return model

    def _resolve_fit_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merges ``default_params`` with overrides from ``config``.

        Supports two configuration structures: a nested ``{'params': {...}}`` dict
        (preferred), or a flat legacy dict where non-reserved, non-dict keys are
        treated as params.
        """
        params = self.default_params.copy()
        if not config:
            return self._inject_default_seed(params)

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

        return self._inject_default_seed(params)

    def _inject_default_seed(self, params: dict[str, Any]) -> dict[str, Any]:
        """Single owner for seeding (finding F-21).

        Model defaults no longer carry their own ``random_state`` literals.
        If the caller didn't configure one, inject ``DEFAULT_RANDOM_STATE``
        here — but only when the wrapped estimator's constructor accepts it
        (named parameter or ``**kwargs``), so unsupported estimators don't
        get a dropped-param warning. An explicit user value (including
        ``None`` for "unseeded") always wins.
        """
        if "random_state" in params:
            return params
        sig = inspect.signature(self.model_class)
        accepts = (
            any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            or "random_state" in sig.parameters
        )
        if accepts:
            params["random_state"] = DEFAULT_RANDOM_STATE
        return params

    def _filter_supported_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Filters ``params`` down to those accepted by the model class constructor.

        Skips filtering when the constructor accepts ``**kwargs`` (e.g. XGBoost 2.x),
        since every named param would otherwise fail the membership check even though valid.
        """
        sig = inspect.signature(self.model_class)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if accepts_kwargs:
            return params

        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        dropped = set(params.keys()) - set(valid_params.keys())
        if dropped:
            logger.warning(
                f"Dropped parameters not supported by {self.model_class.__name__}: {dropped}"
            )
        return valid_params

    def _constructor_accepts_class_weight(self) -> bool:
        """True if the wrapped model's constructor explicitly declares a
        `class_weight` parameter (e.g. RandomForestClassifier, LGBMClassifier,
        LogisticRegression) — as opposed to merely accepting arbitrary
        **kwargs (e.g. XGBoost's sklearn wrapper) that silently swallow it."""
        sig = inspect.signature(self.model_class)
        return "class_weight" in sig.parameters

    def _compute_sample_weight_for_fit(self, model: Any, class_weight: Any, y_np: Any) -> Any:
        """Translate a `class_weight` value into a per-sample weight array for
        models with no native `class_weight` support, raising a clear error
        instead of silently no-op'ing if the model's `.fit()` doesn't accept
        `sample_weight` either."""
        fit_sig = inspect.signature(model.fit)
        if "sample_weight" not in fit_sig.parameters:
            raise ValueError(
                f"{self.model_class.__name__} does not support 'class_weight' natively "
                "and its fit() method does not accept 'sample_weight' either, so "
                "class weighting cannot be applied to this model."
            )
        return compute_sample_weight(class_weight, y_np)


class SklearnApplier(BaseModelApplier):
    """Base applier for Scikit-Learn models."""

    def predict(self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any) -> Any:
        # Convert to Numpy
        X_np, _ = SklearnBridge.to_sklearn(df)

        preds = model_artifact.predict(X_np)

        # Return as Pandas Series for consistency
        # If input was Pandas, try to preserve index
        index = None
        if hasattr(df, "index"):
            index = cast(pd.DataFrame, df).index
        elif hasattr(df, "to_pandas"):
            # If it's a wrapper or Polars, we might lose index unless we convert
            # For now, default index is acceptable for predictions
            pass

        return pd.Series(preds, index=index)

    def predict_proba(self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any) -> Any | None:
        if not hasattr(model_artifact, "predict_proba"):
            return None

        X_np, _ = SklearnBridge.to_sklearn(df)
        probs = model_artifact.predict_proba(X_np)

        # Return as DataFrame
        index = None
        if hasattr(df, "index"):
            index = cast(pd.DataFrame, df).index

        # Column names usually 0, 1, etc. or classes_. Coerce to native
        # Python types (str) so downstream JSON serialization of the
        # resulting DataFrame's columns doesn't choke on numpy scalar
        # types (e.g. np.int64), mirroring the class_names normalization
        # already done in modeling/_evaluation/classification.py.
        columns = None
        if hasattr(model_artifact, "classes_"):
            columns = pd.Index([str(c) for c in model_artifact.classes_])

        return pd.DataFrame(probs, index=index, columns=columns)
