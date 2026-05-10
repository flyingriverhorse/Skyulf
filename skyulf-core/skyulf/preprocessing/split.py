"""Train/test/val splitter and feature/target splitter nodes.

These nodes return a :class:`SplitDataset` (not the canonical ``(X, y)`` shape),
so they cannot use :func:`apply_dual_engine` — the dispatcher expects appliers
that return ``(X_out, y_out)``. Instead, we centralise the engine handling in
two small helpers:

* :func:`_to_pandas_remember_engine` converts to pandas and records whether a
  conversion happened.
* :func:`_back_to_engine` converts the result back to polars when needed.

This removes the inline ``if engine.name == EngineName.POLARS`` branches from
both :class:`DataSplitter` methods and :class:`FeatureTargetSplitApplier`.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union, cast

import pandas as pd
from sklearn.model_selection import train_test_split

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..data.dataset import SplitDataset
from .base import BaseApplier, BaseCalculator
from ._artifacts import FeatureTargetSplitArtifact, SplitArtifact
from ._schema import SkyulfSchema
from ..engines import EngineName, SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Engine-bridge helpers
# -----------------------------------------------------------------------------


def _to_pandas_remember_engine(data: Any) -> Tuple[Any, bool]:
    """Return ``(pandas_data, was_polars)``. Pass through pandas/None unchanged."""
    if data is None:
        return None, False
    engine = get_engine(data)
    if engine.name == EngineName.POLARS:
        return cast(Any, data).to_pandas(), True
    return data, False


def _back_to_engine(data: Any, was_polars: bool) -> Any:
    """Convert pandas back to polars when the original input was polars."""
    if data is None or not was_polars:
        return data
    if isinstance(data, (pd.DataFrame, pd.Series)):
        import polars as pl

        return pl.from_pandas(data)
    return data


def _safe_stratify(y: Any, label: str) -> Any:
    """Return ``y`` if every class has ≥ 2 members, else ``None`` with a warning."""
    if y is None:
        return None
    class_counts = cast(Any, y).value_counts()
    min_count = class_counts.min()
    if min_count < 2:
        logger.warning(
            "%s requested but the least populated class has only %s member(s). "
            "Stratification will be disabled.",
            label,
            min_count,
        )
        return None
    return y


# -----------------------------------------------------------------------------
# SplitApplier — entry point that picks tuple vs frame routing
# -----------------------------------------------------------------------------


def _build_splitter(params: Dict[str, Any]) -> "DataSplitter":
    """Construct a :class:`DataSplitter` from the node's params dict."""
    stratify = params.get("stratify", False)
    target_col = params.get("target_column")
    # Stratify requested without a target → use a sentinel so split_xy stratifies on y.
    stratify_col = target_col if stratify else None
    if stratify and not target_col:
        stratify_col = "__implicit_target__"

    return DataSplitter(
        test_size=params.get("test_size", 0.2),
        validation_size=params.get("validation_size", 0.0),
        random_state=params.get("random_state", 42),
        shuffle=params.get("shuffle", True),
        stratify_col=stratify_col,
    )


class SplitApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> SplitDataset:
        target_col = params.get("target_column")
        splitter = _build_splitter(params)

        if isinstance(df, tuple) and len(df) == 2:
            X, y = df
            return splitter.split_xy(cast(Any, X), y)

        # If a target column is configured and present, split features from target
        # so downstream nodes see real (X, y) tuples — not the placeholder
        # (df, None) shape that gets silently collapsed by transformers.
        if target_col and hasattr(df, "columns") and target_col in list(cast(Any, df).columns):
            frame = cast(Any, df)
            X = frame.drop(columns=[target_col])
            y = frame[target_col]
            return splitter.split_xy(X, y)

        return splitter.split(cast(Any, df))


@NodeRegistry.register("Split", SplitApplier)
@NodeRegistry.register("TrainTestSplitter", SplitApplier)
@node_meta(
    id="TrainTestSplitter",
    name="Train/Test Split",
    category="Data Operations",
    description="Split the dataset into training and testing sets.",
    params={
        "test_size": 0.2,
        "validation_size": 0.0,
        "random_state": 42,
        "shuffle": True,
        "stratify": False,
        "target_column": "target",
    },
)
class SplitCalculator(BaseCalculator):
    def fit(
        self, df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any], config: Dict[str, Any]
    ) -> SplitArtifact:
        # No learning from data; pass through known split params from config.
        # Constructed explicitly so the artifact shape matches SplitArtifact
        # rather than echoing arbitrary user keys back into the params dict.
        artifact: SplitArtifact = {"type": "split"}
        for key in (
            "test_size",
            "validation_size",
            "random_state",
            "shuffle",
            "stratify",
            "target_column",
        ):
            if key in config:
                artifact[key] = config[key]  # type: ignore[literal-required]
        return artifact

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: Dict[str, Any],
    ) -> Optional[SkyulfSchema]:
        # Split produces a SplitDataset whose train/test/val frames carry
        # the full input schema. Downstream consumers see the same columns.
        return input_schema


# -----------------------------------------------------------------------------
# DataSplitter — the workhorse
# -----------------------------------------------------------------------------


class DataSplitter:
    """Split a DataFrame (or X/y pair) into Train, Test, and optional Validation."""

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
        shuffle: bool = True,
        stratify_col: Optional[str] = None,
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify_col = stratify_col

    # ---- public API ---------------------------------------------------------

    def split_xy(
        self, X: Union[pd.DataFrame, SkyulfDataFrame], y: Union[pd.Series, Any]
    ) -> SplitDataset:
        X_pd, was_polars = _to_pandas_remember_engine(X)
        y_pd, _ = _to_pandas_remember_engine(y)

        stratify = _safe_stratify(y_pd, "Stratified split") if self.stratify_col else None

        X_tv, X_test, y_tv, y_test = train_test_split(
            X_pd,
            y_pd,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify,
        )

        validation, X_train, y_train = self._maybe_split_validation_xy(X_tv, y_tv)

        train = (_back_to_engine(X_train, was_polars), _back_to_engine(y_train, was_polars))
        test = (_back_to_engine(X_test, was_polars), _back_to_engine(y_test, was_polars))
        if validation is not None:
            validation = (
                _back_to_engine(validation[0], was_polars),
                _back_to_engine(validation[1], was_polars),
            )
        return SplitDataset(train=train, test=test, validation=validation)

    def split(self, df: Union[pd.DataFrame, SkyulfDataFrame]) -> SplitDataset:
        df_pd, was_polars = _to_pandas_remember_engine(df)
        stratify = self._frame_stratify(df_pd, label="Stratified split")

        train_val, test = train_test_split(
            df_pd,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify,
        )

        validation, train = self._maybe_split_validation_frame(train_val)
        return SplitDataset(
            train=_back_to_engine(train, was_polars),
            test=_back_to_engine(test, was_polars),
            validation=_back_to_engine(validation, was_polars),
        )

    # ---- private helpers ----------------------------------------------------

    def _frame_stratify(self, df_pd: Any, label: str) -> Any:
        """Pick + sanity-check the stratify column on a frame split."""
        if not (self.stratify_col and self.stratify_col in df_pd.columns):
            return None
        return _safe_stratify(df_pd[self.stratify_col], label)

    def _maybe_split_validation_xy(self, X_tv: Any, y_tv: Any) -> Tuple[Any, Any, Any]:
        """Carve a validation set off of (X_tv, y_tv); returns (val, X_train, y_train)."""
        if self.validation_size <= 0:
            return None, X_tv, y_tv

        relative_val_size = self.validation_size / (1 - self.test_size)
        stratify_val = (
            _safe_stratify(y_tv, "Stratified validation split") if self.stratify_col else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv,
            y_tv,
            test_size=relative_val_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_val,
        )
        return (X_val, y_val), X_train, y_train

    def _maybe_split_validation_frame(self, train_val: Any) -> Tuple[Any, Any]:
        """Carve a validation set off of ``train_val`` (frame mode); returns (val, train)."""
        if self.validation_size <= 0:
            return None, train_val

        relative_val_size = self.validation_size / (1 - self.test_size)
        stratify_val = self._frame_stratify(train_val, label="Stratified validation split")
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_val,
        )
        return val, train


# -----------------------------------------------------------------------------
# Feature/Target splitter
# -----------------------------------------------------------------------------


def _split_xy_one_polars(data: Any, target_col: str) -> Tuple[Any, Any]:
    import polars as pl

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    y = data.select(pl.col(target_col)).to_series()
    X = data.drop([target_col])
    return X, y


def _split_xy_one_pandas(data: Any, target_col: str) -> Tuple[Any, Any]:
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    y = data[target_col]
    X = data.drop(columns=[target_col])
    return X, y


def _split_xy_one(data: Any, target_col: str) -> Tuple[Any, Any]:
    """Engine-aware single-frame X/y split."""
    engine = get_engine(data)
    if engine.name == EngineName.POLARS:
        return _split_xy_one_polars(data, target_col)
    return _split_xy_one_pandas(data, target_col)


def _maybe_split_xy_member(data: Any, target_col: str) -> Tuple[Any, Any]:
    """Apply X/y split to one member of a SplitDataset (handles already-split tuples)."""
    if isinstance(data, tuple) and len(data) == 2:
        X, y = data
        if y is not None:
            return data  # Already split.
        if hasattr(X, "columns") and target_col in X.columns:
            return _split_xy_one(X, target_col)
        return data
    return _split_xy_one(data, target_col)


class FeatureTargetSplitApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, SplitDataset, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[Tuple[pd.DataFrame, pd.Series], SplitDataset]:
        target_col = params.get("target_column")
        if not target_col:
            raise ValueError("Target column must be specified for FeatureTargetSplitter")

        if isinstance(df, SplitDataset):
            train = _maybe_split_xy_member(df.train, target_col)
            test = _maybe_split_xy_member(df.test, target_col)
            validation = (
                _maybe_split_xy_member(df.validation, target_col)
                if df.validation is not None
                else None
            )
            return SplitDataset(train=train, test=test, validation=validation)

        if isinstance(df, tuple):
            return cast(Tuple[pd.DataFrame, pd.Series], df)

        return _split_xy_one(df, target_col)


@NodeRegistry.register("feature_target_split", FeatureTargetSplitApplier)
@node_meta(
    id="feature_target_split",
    name="Feature/Target Split",
    category="Data Operations",
    description="Split the dataset into features (X) and target (y).",
    params={"target_column": "target"},
)
class FeatureTargetSplitCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, SplitDataset, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> FeatureTargetSplitArtifact:
        # Only ``target_column`` is consumed downstream by the Applier.
        # Build a typed artifact rather than echoing the raw config dict.
        artifact: FeatureTargetSplitArtifact = {"type": "feature_target_split"}
        if "target_column" in config and config["target_column"] is not None:
            artifact["target_column"] = str(config["target_column"])
        return artifact

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: Dict[str, Any],
    ) -> Optional[SkyulfSchema]:
        # Output is (X, y). The target column lives in the y slot but is still
        # part of the dataset, so we keep it in the downstream schema. This
        # lets downstream column pickers (Encoder, Scaler, etc.) see and select
        # the target column. Runtime Appliers still receive the (X, y) tuple
        # via the dispatcher's ``unpack_pipeline_input`` and operate on X.
        return input_schema
