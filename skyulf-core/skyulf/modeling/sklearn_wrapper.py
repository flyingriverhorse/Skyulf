"""Wrapper for Scikit-Learn models."""

import logging
import warnings
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

from ..engines import SkyulfDataFrame
from ..engines.sklearn_bridge import SklearnBridge
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
    ) -> Any:
        """Fit the Scikit-Learn model."""
        # 1. Merge Config with Defaults
        params = self._resolve_fit_params(config)

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
            model.fit(X_np, y_np)
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
            return params

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

        return params

    def _filter_supported_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Filters ``params`` down to those accepted by the model class constructor.

        Skips filtering when the constructor accepts ``**kwargs`` (e.g. XGBoost 2.x),
        since every named param would otherwise fail the membership check even though valid.
        """
        import inspect

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
            index = df.index
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
            index = df.index

        # Column names usually 0, 1, etc. or classes_. Coerce to native
        # Python types (str) so downstream JSON serialization of the
        # resulting DataFrame's columns doesn't choke on numpy scalar
        # types (e.g. np.int64), mirroring the class_names normalization
        # already done in modeling/_evaluation/classification.py.
        columns = None
        if hasattr(model_artifact, "classes_"):
            columns = [str(c) for c in model_artifact.classes_]

        return pd.DataFrame(probs, index=index, columns=columns)
