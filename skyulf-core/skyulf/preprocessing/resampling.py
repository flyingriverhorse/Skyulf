"""Resampling nodes (`Oversampling`, `Undersampling`).

Both Appliers route through :func:`apply_dual_engine`. Imblearn is purely
pandas/numpy-bound, so the Polars path round-trips through pandas (convert in,
run sampler, convert back) while keeping all engine handling out of class
bodies. Sampler construction is split per-method and per-family so each helper
stays at low CCN.
"""

import logging
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..core.capabilities import ExecutionCapability
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ._artifacts import OversamplingArtifact, UndersamplingArtifact
from ._schema import SkyulfSchema
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


SamplerBuilder = Callable[[str, dict[str, Any]], Any | None]


def _validate_resampling_state(raw: dict, kind: str, default_method: str) -> dict:
    """Inspect effective dispatch fields, retaining defaults and unused saved options."""
    if type(raw) is not dict or raw.get("type") != kind:
        raise ValueError("Unexpected local resampling state type.")
    if not isinstance(raw.get("method", default_method), str):
        raise ValueError("Fitted resampling method must be a string.")
    return raw


def _over_context_known(state: dict) -> bool:
    """Abstain for mixed row effects and custom estimator or strategy callbacks."""
    method = state.get("method", "smote")
    neighbors = {
        "random_over": (),
        "smote": ("k_neighbors",),
        "adasyn": ("k_neighbors",),
        "borderline_smote": ("k_neighbors", "m_neighbors"),
        "svm_smote": ("k_neighbors", "m_neighbors"),
        "kmeans_smote": ("k_neighbors",),
    }
    if method not in neighbors or callable(state.get("sampling_strategy")):
        return False
    if any(not isinstance(state.get(key, 5), Integral) for key in neighbors[method]):
        return False
    if method == "svm_smote" and state.get("svm_estimator") is not None:
        return False
    if method == "kmeans_smote":
        estimator = state.get("kmeans_estimator")
        return estimator is None or isinstance(estimator, Integral)
    return True


def _under_context_known(state: dict) -> bool:
    """Describe only built-in selection without callbacks or replacement duplicates."""
    method = state.get("method", "random_under_sampling")
    if method not in (
        "random_under_sampling",
        "nearmiss",
        "tomek_links",
        "edited_nearest_neighbours",
    ) or callable(state.get("sampling_strategy")):
        return False
    if method == "random_under_sampling":
        replacement = state.get("replacement", False)
        return isinstance(replacement, (bool, np.bool_)) and not replacement
    if method in ("nearmiss", "edited_nearest_neighbours"):
        return isinstance(state.get("n_neighbors", 3), Integral)
    return True


def _extract_y_polars(X: Any, y: Any, target_col: str | None) -> tuple[Any, Any]:
    """When ``y`` is missing, lift it out of the Polars frame using ``target_col``."""
    if y is not None:
        return X, y
    if target_col and target_col in X.columns:
        return X.drop(target_col), X.select(target_col).to_series()
    return X, None


def _extract_y_pandas(X: Any, y: Any, target_col: str | None) -> tuple[Any, Any]:
    """When ``y`` is missing, lift it out of the Pandas frame using ``target_col``."""
    if y is not None:
        return X, y
    if target_col and target_col in X.columns:
        return X.drop(columns=[target_col]), X[target_col]
    return X, None


def _to_pandas_y(y: Any) -> Any:
    """Best-effort conversion of ``y`` to a pandas Series."""
    if y is None:
        return None
    if hasattr(y, "to_pandas"):
        return y.to_pandas()
    return y


def _validate_numeric(X_pd: pd.DataFrame) -> None:
    """Reject non-numeric feature columns — imblearn requires all-numeric input."""
    non_numeric = X_pd.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(
            f"Resampling requires all features to be numeric. Found non-numeric columns: "
            f"{list(non_numeric)}. Please use an Encoder node "
            "(e.g., OneHotEncoder, OrdinalEncoder) before Resampling."
        )


def _finalize_resampled(
    X_res: Any, y_res: Any, columns: Any, fallback_name: str | None
) -> tuple[pd.DataFrame, pd.Series]:
    """Wrap raw imblearn output back into named DataFrame/Series."""
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=columns)
    if not isinstance(y_res, pd.Series):
        y_res = pd.Series(y_res, name=fallback_name)
    return X_res, y_res


def _validate_sample_indices(
    indices: Any, source_rows: int, feature_rows: int, target_rows: int
) -> np.ndarray:
    """Require a complete positional selection map before gathering original values."""
    positions = np.asarray(indices)
    if (
        positions.ndim != 1
        or positions.dtype.kind not in "iu"
        or len(positions) != feature_rows
        or len(positions) != target_rows
        or np.any(positions < 0)
        or np.any(positions >= source_rows)
    ):
        raise ValueError("Sampler returned invalid sample_indices_ for the resampled rows.")
    return positions


def _restore_selected_rows(
    sampler: Any, X: pd.DataFrame, y: Any, X_res: Any, y_res: Any
) -> tuple[Any, Any]:
    """Retain exact selected values while preserving the sampler's output indexes and names."""
    indices = getattr(sampler, "sample_indices_", None)
    if indices is None:
        return X_res, y_res
    positions = _validate_sample_indices(indices, len(X), len(X_res), len(y_res))
    feature_index = X_res.index if isinstance(X_res, pd.DataFrame) else pd.RangeIndex(len(X_res))
    target_index = y_res.index if isinstance(y_res, pd.Series) else pd.RangeIndex(len(y_res))
    selected_X = X.iloc[positions].set_axis(feature_index)
    selected_y = pd.Series(y).iloc[positions].set_axis(target_index)
    selected_y.name = getattr(y_res, "name", selected_y.name)
    return selected_X, selected_y


def _run_sampler(
    X_pd: pd.DataFrame,
    y_pd: Any,
    params: dict[str, Any],
    builder: SamplerBuilder,
    default_method: str,
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Run the sampler chosen by ``builder``; return ``None`` if no sampler matches."""
    _validate_numeric(X_pd)
    method = params.get("method", default_method)
    sampler = builder(method, params)
    if sampler is None:
        return None
    X_res, y_res = sampler.fit_resample(X_pd, y_pd)
    X_res, y_res = _restore_selected_rows(sampler, X_pd, y_pd, X_res, y_res)
    fallback_name = getattr(y_pd, "name", None) if y_pd is not None else params.get("target_column")
    return _finalize_resampled(X_res, y_res, X_pd.columns, fallback_name)


def _resample_polars(
    X: Any,
    y: Any,
    params: dict[str, Any],
    builder: SamplerBuilder,
    default_method: str,
) -> tuple[Any, Any]:
    """Polars apply path: convert → resample → convert back."""
    target_col = params.get("target_column")
    X, y = _extract_y_polars(X, y, target_col)
    if y is None:
        return X, y
    X_pd = X.to_pandas()
    y_pd = _to_pandas_y(y)
    out = _run_sampler(X_pd, y_pd, params, builder, default_method)
    if out is None:
        return X, y
    X_res, y_res = out
    return pl.from_pandas(X_res), pl.from_pandas(y_res)


def _resample_pandas(
    X: Any,
    y: Any,
    params: dict[str, Any],
    builder: SamplerBuilder,
    default_method: str,
) -> tuple[Any, Any]:
    """Pandas apply path: resample in place."""
    target_col = params.get("target_column")
    X, y = _extract_y_pandas(X, y, target_col)
    if y is None:
        return X, y
    out = _run_sampler(X, y, params, builder, default_method)
    if out is None:
        return X, y
    return out


# -----------------------------------------------------------------------------
# Oversampling
# -----------------------------------------------------------------------------


def _import_over_samplers() -> dict[str, Any]:
    """Lazy import of imblearn oversampling classes."""
    try:
        from imblearn.combine import (  # noqa: PLC0415 - optional preprocessing-imbalanced extra
            SMOTETomek,
        )
        from imblearn.over_sampling import (  # noqa: PLC0415 - optional preprocessing-imbalanced extra
            ADASYN,
            SMOTE,
            SVMSMOTE,
            BorderlineSMOTE,
            KMeansSMOTE,
            RandomOverSampler,
        )
    except ImportError as exc:
        logger.exception("imblearn is required for oversampling. `pip install imbalanced-learn`")
        raise ImportError(
            "imblearn is required for oversampling. `pip install imbalanced-learn`"
        ) from exc
    return {
        "random_over": RandomOverSampler,
        "smote": SMOTE,
        "adasyn": ADASYN,
        "borderline_smote": BorderlineSMOTE,
        "svm_smote": SVMSMOTE,
        "kmeans_smote": KMeansSMOTE,
        "smote_tomek": SMOTETomek,
    }


def _build_oversampler(method: str, params: dict[str, Any]) -> Any:
    """Construct an over-sampler by ``method`` name."""
    classes = _import_over_samplers()
    cls = classes.get(method)
    if cls is None:
        raise ValueError(
            f"Unsupported resampling method {method!r}. Supported oversampling methods: "
            f"{', '.join(sorted(classes))}."
        )

    strategy = params.get("sampling_strategy", "auto")
    random_state = params.get("random_state", 42)
    k_neighbors = params.get("k_neighbors", 5)

    if method == "random_over":
        return cls(sampling_strategy=strategy, random_state=random_state)
    if method == "smote":
        return cls(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors)
    if method == "adasyn":
        return cls(sampling_strategy=strategy, random_state=random_state, n_neighbors=k_neighbors)
    if method == "borderline_smote":
        return cls(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            m_neighbors=params.get("m_neighbors", 10),
            kind=params.get("kind", "borderline-1"),
        )
    if method == "svm_smote":
        return cls(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            m_neighbors=params.get("m_neighbors", 10),
            svm_estimator=params.get("svm_estimator"),
            out_step=params.get("out_step", 0.5),
        )
    if method == "kmeans_smote":
        return cls(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            kmeans_estimator=params.get("kmeans_estimator"),
            n_jobs=params.get("n_jobs", -1),
            cluster_balance_threshold=params.get("cluster_balance_threshold", 0.1),
            density_exponent=params.get("density_exponent", "auto"),
        )
    # smote_tomek
    smote = classes["smote"](
        sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors
    )
    return cls(
        sampling_strategy=strategy,
        random_state=random_state,
        smote=smote,
        n_jobs=params.get("n_jobs", -1),
    )


class OversamplingApplier(BaseApplier):
    """Grow the minority classes with the imblearn over-sampler named in the artifact.

    Sees the training split only: :mod:`.pipeline` excludes the resampling nodes
    from ``apply_on_test``/``apply_on_validation``, so synthetic rows can never
    reach a held-out split and inflate its metrics.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect the saved sampler configuration without fitting optional estimators."""
        return _validate_resampling_state(raw, "oversampling", "smote")

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Require the full class distribution; mixed SMOTETomek effects stay unknown."""
        if engine not in ("pandas", "polars"):
            return None
        OversamplingApplier.validate_inference_state(state)
        if not _over_context_known(state):
            return None
        return ExecutionCapability(engine, "apply", "local", "expand", "global")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Resample ``(X, y)`` with the configured over-sampler.

        ``y`` is lifted out of ``X`` via ``target_column`` when not supplied
        separately; with no usable target the data passes through untouched, as
        it does for a ``method`` no builder recognizes.

        Raises:
            ValueError: If any feature column is non-numeric, which imblearn
                cannot resample. Encode first.
        """
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {
                "polars": lambda Xi, yi, p: _resample_polars(
                    Xi, yi, p, _build_oversampler, "smote"
                ),
                "pandas": lambda Xi, yi, p: _resample_pandas(
                    Xi, yi, p, _build_oversampler, "smote"
                ),
            },
        )


@NodeRegistry.register("Oversampling", OversamplingApplier)
@node_meta(
    id="Oversampling",
    name="Oversampling",
    category="Preprocessing",
    description="Resample dataset to balance classes by oversampling minority class.",
    params={"method": "smote", "target_column": "target", "sampling_strategy": "auto"},
    learns_from_data=True,
)
class OversamplingCalculator(BaseCalculator):
    """Configure class balancing by oversampling the minority class."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return ``input_schema`` untouched."""
        # Resampling changes row counts only; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, _X: Any, _y: Any, config: dict[str, Any]) -> OversamplingArtifact:  # pylint: disable=arguments-differ
        """Snapshot the sampler settings from ``config`` into the artifact.

        Neither ``_X`` nor ``_y`` is read: the artifact is pure configuration,
        and the class imbalance it corrects is only observed when the applier
        runs the sampler on the training split.
        """
        return {
            "type": "oversampling",
            "synthetic_weight": config.get("synthetic_weight"),
            "method": config.get("method", "smote"),
            "target_column": config.get("target_column"),
            "sampling_strategy": config.get("sampling_strategy", "auto"),
            "random_state": config.get("random_state", 42),
            "k_neighbors": config.get("k_neighbors", 5),
            "m_neighbors": config.get("m_neighbors", 10),
            "kind": config.get("kind", "borderline-1"),
            "svm_estimator": config.get("svm_estimator"),
            "out_step": config.get("out_step", 0.5),
            "kmeans_estimator": config.get("kmeans_estimator"),
            "cluster_balance_threshold": config.get("cluster_balance_threshold", 0.1),
            "density_exponent": config.get("density_exponent", "auto"),
            "n_jobs": config.get("n_jobs", -1),
        }


# -----------------------------------------------------------------------------
# Undersampling
# -----------------------------------------------------------------------------


def _import_under_samplers() -> dict[str, Any]:
    """Lazy import of imblearn undersampling classes."""
    try:
        from imblearn.under_sampling import (  # noqa: PLC0415 - optional preprocessing-imbalanced extra
            EditedNearestNeighbours,
            NearMiss,
            RandomUnderSampler,
            TomekLinks,
        )
    except ImportError as exc:
        logger.exception("imblearn is required for undersampling. `pip install imbalanced-learn`")
        raise ImportError(
            "imblearn is required for undersampling. `pip install imbalanced-learn`"
        ) from exc
    return {
        "random_under_sampling": RandomUnderSampler,
        "nearmiss": NearMiss,
        "tomek_links": TomekLinks,
        "edited_nearest_neighbours": EditedNearestNeighbours,
    }


def _build_undersampler(method: str, params: dict[str, Any]) -> Any:
    """Construct an under-sampler by ``method`` name."""
    classes = _import_under_samplers()
    cls = classes.get(method)
    if cls is None:
        raise ValueError(
            f"Unsupported resampling method {method!r}. Supported undersampling methods: "
            f"{', '.join(sorted(classes))}."
        )

    strategy = params.get("sampling_strategy", "auto")

    if method == "random_under_sampling":
        return cls(
            sampling_strategy=strategy,
            random_state=params.get("random_state", 42),
            replacement=params.get("replacement", False),
        )
    if method == "nearmiss":
        return cls(
            sampling_strategy=strategy,
            version=params.get("version", 1),
            n_neighbors=params.get("n_neighbors", 3),
            n_jobs=params.get("n_jobs", -1),
        )
    if method == "tomek_links":
        return cls(sampling_strategy=strategy, n_jobs=params.get("n_jobs", -1))
    # edited_nearest_neighbours
    return cls(
        sampling_strategy=strategy,
        n_neighbors=params.get("n_neighbors", 3),
        kind_sel=params.get("kind_sel", "all"),
        n_jobs=params.get("n_jobs", -1),
    )


class UndersamplingApplier(BaseApplier):
    """Shrink the majority classes with the imblearn under-sampler named in the artifact.

    Sees the training split only: :mod:`.pipeline` excludes the resampling nodes
    from ``apply_on_test``/``apply_on_validation``. That guard matters more here
    than for oversampling, because these samplers delete real rows rather than
    synthesise new ones.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved selection settings without loading the optional sampler package."""
        return _validate_resampling_state(raw, "undersampling", "random_under_sampling")

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Require full class context; prediction still skips every sampling method."""
        if engine not in ("pandas", "polars"):
            return None
        UndersamplingApplier.validate_inference_state(state)
        if not _under_context_known(state):
            return None
        return ExecutionCapability(engine, "apply", "local", "filter", "global")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:
        """Resample ``(X, y)`` with the configured under-sampler.

        ``y`` is lifted out of ``X`` via ``target_column`` when not supplied
        separately; with no usable target the data passes through untouched, as
        it does for a ``method`` no builder recognizes.

        Raises:
            ValueError: If any feature column is non-numeric, which imblearn
                cannot resample. Encode first.
        """
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {
                "polars": lambda Xi, yi, p: _resample_polars(
                    Xi, yi, p, _build_undersampler, "random_under_sampling"
                ),
                "pandas": lambda Xi, yi, p: _resample_pandas(
                    Xi, yi, p, _build_undersampler, "random_under_sampling"
                ),
            },
        )


@NodeRegistry.register("Undersampling", UndersamplingApplier)
@node_meta(
    id="Undersampling",
    name="Undersampling",
    category="Preprocessing",
    description="Resample dataset to balance classes by undersampling majority class.",
    params={
        "method": "random_under_sampling",
        "target_column": "target",
        "sampling_strategy": "auto",
    },
    learns_from_data=True,
)
class UndersamplingCalculator(BaseCalculator):
    """Configure class balancing by undersampling the majority class."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return ``input_schema`` untouched."""
        # Resampling changes row counts only; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, _X: Any, _y: Any, config: dict[str, Any]) -> UndersamplingArtifact:  # pylint: disable=arguments-differ
        """Snapshot the sampler settings from ``config`` into the artifact.

        Neither ``_X`` nor ``_y`` is read: the artifact is pure configuration.
        Several of its keys are sampler-specific, so each method consumes only
        the subset its imblearn class accepts.
        """
        return {
            "type": "undersampling",
            "method": config.get("method", "random_under_sampling"),
            "target_column": config.get("target_column"),
            "sampling_strategy": config.get("sampling_strategy", "auto"),
            "random_state": config.get("random_state", 42),
            "replacement": config.get("replacement", False),
            "version": config.get("version", 1),  # For NearMiss
            "n_neighbors": config.get("n_neighbors", 3),  # For EditedNearestNeighbours
            "kind_sel": config.get("kind_sel", "all"),  # For EditedNearestNeighbours
            "n_jobs": config.get("n_jobs", -1),
        }
