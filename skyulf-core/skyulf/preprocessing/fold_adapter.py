"""Adapters exposing preprocessing chains as :class:`FoldPreprocessor` (F-15).

Cloning is free here by construction: ``FeatureEngineer`` rebuilds every
step's calculator/applier fresh from ``steps_config`` via the node registry
on each ``fit_transform``, so constructing a new engineer per fold *is* the
clone operation — no fitted state is ever copied or reset.
"""

from typing import Any

import pandas as pd
import polars as pl

from ..registry import NodeRegistry
from .pipeline import FeatureEngineer

# Splitter steps already ran upstream of the CV/tuning boundary; re-running
# them inside a fold would re-split the fold itself.
SPLITTER_STEP_TYPES = frozenset({"TrainTestSplitter", "Split", "feature_target_split"})

# Steps whose fit_transform changes the row count (resampling, row drops,
# outlier removal). Inside a merged-branch fold they would desynchronise the
# column-wise branch merge (which requires equal row counts), so merged
# adapters reject them. The tuning engine's fold-aware estimator runs such
# chains inside ``fit`` on each fold's training rows, so single-branch
# tuning wraps accept them; ``changes_row_count`` documents the chain's
# nature for callers that still need to know.
ROW_COUNT_CHANGING_STEP_TYPES = frozenset(
    FeatureEngineer._ROW_DROPPING_TYPES
    | FeatureEngineer._RESAMPLING_TYPES
    | {"IQR", "ZScore", "Winsorize", "EllipticEnvelope"}
)

UNSAFE_BRANCH_STEP_TYPES = SPLITTER_STEP_TYPES | ROW_COUNT_CHANGING_STEP_TYPES


def _merge_branch_frames_columnwise(frames: list[pd.DataFrame], strategy: str) -> pd.DataFrame:
    """Merge equal-row-count branch frames column-wise per the engine's pure path.

    Mirror of the ownership-free path in
    ``MergeMixin._merge_frames_columnwise``: in fork-join graphs the merge's
    nearest-common-ancestor artifact is a SplitDataset, so ownership analysis
    is inert and the configured strategy decides every overlapping column.
    ``last_wins`` iterates inputs in order; ``first_wins`` iterates them
    reversed — column *position* is first-seen in either case, so train and
    test merges stay column-aligned.
    """
    indexed = list(enumerate(frames))
    ordered = indexed if strategy == "last_wins" else list(reversed(indexed))
    result_cols: dict[str, pd.Series] = {}
    for _idx, df in ordered:
        df_aligned = df.reset_index(drop=True)
        for col in df_aligned.columns:
            result_cols[col] = df_aligned[col]
    return pd.DataFrame(result_cols)


class MergedBranchFoldAdapter:
    """Re-runs fork-join preprocessing branches inside each CV/tuning fold.

    Built for graphs where a shared trunk ends in a splitter (fork point) and
    N parallel transformer branches fan back into one training node. Per fold
    it re-runs every branch step list on the fold-train payload and merges the
    branch frames exactly like the engine's pure-strategy column-wise merge,
    so the fold sees the same columns the full run produces — without any
    pre-fit statistics leaking from outside the fold.

    Args:
        branch_step_lists: One unfitted step list per branch, in the engine's
            merge input order.
        merge_strategy: ``"last_wins"`` (default engine behaviour) or
            ``"first_wins"`` — must match the training node's configured
            strategy so fold columns match the full-run merge.
        target_column: Target name, kept to reject payloads that still embed
            it; target-aware branch steps receive ``y`` through the payload.
        drop_columns: Columns removed upstream (e.g. by Drop Columns nodes);
            stripped again after each merge so a branch cannot resurrect them.
    """

    def __init__(
        self,
        branch_step_lists: list[list[dict[str, Any]]],
        merge_strategy: str,
        target_column: str,
        drop_columns: list[str] | tuple[str, ...] = (),
    ):
        if merge_strategy not in ("last_wins", "first_wins"):
            raise ValueError(f"unknown merge strategy '{merge_strategy}'")
        if not branch_step_lists:
            raise ValueError("at least one branch step list is required")
        for steps in branch_step_lists:
            if not steps:
                raise ValueError("branch step list must not be empty")
            FeatureEngineer(list(steps))
            for step in steps:
                transformer = step["transformer"]
                if transformer in UNSAFE_BRANCH_STEP_TYPES:
                    raise ValueError(
                        f"branch step '{transformer}' cannot run inside a fold: "
                        "it splits the data or changes row counts"
                    )
                NodeRegistry.get_calculator(transformer)
        self._branch_step_lists = [list(steps) for steps in branch_step_lists]
        self._merge_strategy = merge_strategy
        self._target_column = target_column
        self._drop_columns = list(drop_columns)
        self._engineers: list[FeatureEngineer] | None = None
        # Branch steps are screened against UNSAFE_BRANCH_STEP_TYPES, so the
        # merge keeps every row.
        self.changes_row_count = False

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self._validate_payload(X)
        engineers = [FeatureEngineer(list(steps)) for steps in self._branch_step_lists]
        frames, ys = self._run_branches(engineers, (X, y), fit=True)
        self._engineers = engineers
        return self._finalize(frames, ys)

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        if self._engineers is None:
            raise RuntimeError("transform() called before fit_transform()")
        self._validate_payload(X)
        payload = (X, y) if y is not None else X
        frames, ys = self._run_branches(self._engineers, payload, fit=False)
        return self._finalize(frames, ys)

    def _run_branches(
        self, engineers: list[FeatureEngineer], payload: Any, *, fit: bool
    ) -> tuple[list[pd.DataFrame], list[Any]]:
        frames: list[pd.DataFrame] = []
        ys: list[Any] = []
        input_y = payload[1] if isinstance(payload, tuple) else None
        for engineer in engineers:
            out = engineer.fit_transform(payload)[0] if fit else engineer.transform(payload)
            if isinstance(out, tuple) and len(out) == 2:
                frame, y_out = out
            else:
                # Appliers that return a bare frame pass the target through.
                frame, y_out = out, input_y
            if isinstance(frame, pl.DataFrame):
                frame = frame.to_pandas()
            frames.append(frame)
            ys.append(y_out)
        return frames, ys

    def _finalize(self, frames: list[pd.DataFrame], ys: list[Any]) -> tuple[Any, Any]:
        merged = _merge_branch_frames_columnwise(frames, self._merge_strategy)
        drop = [col for col in self._drop_columns if col in merged.columns]
        if drop:
            merged = merged.drop(columns=drop)
        # Mirrors the engine's SplitDataset merge: the first branch supplies
        # the target (branches are screened to never change row counts).
        return merged, ys[0]

    def _validate_payload(self, X: Any) -> None:
        if hasattr(X, "columns") and self._target_column in X.columns:
            raise ValueError(f"target column '{self._target_column}' already present in X")


class FeatureEngineerFoldAdapter:
    """Re-runs a preprocessing step chain inside each CV/tuning fold.

    Args:
        steps_config: The upstream Feature-Engineering node's step list
            (plain dicts, validated by ``validate_preprocessing_steps``).
            Splitter steps are filtered out automatically.
        target_column: Name of the target, kept to reject payloads where X
            still embeds it — target-aware steps (target encoders, resampling,
            label encoding of the target) receive ``y`` through the ``(X, y)``
            payload the same way they do downstream of a real splitter.
    """

    def __init__(self, steps_config: list[dict[str, Any]], target_column: str):
        self._steps_config = [
            step for step in steps_config if step.get("transformer") not in SPLITTER_STEP_TYPES
        ]
        self._target_column = target_column
        # True when any step reshapes the rows/target (resampling, row
        # drops); documents the chain's nature for callers that need it.
        self.changes_row_count = any(
            step.get("transformer") in ROW_COUNT_CHANGING_STEP_TYPES for step in self._steps_config
        )
        # Validate eagerly (unknown transformer names, bad params) so a
        # misconfigured chain fails at construction, not mid-fold.
        # validate_preprocessing_steps does not check registry membership,
        # so resolve each step's calculator explicitly.
        FeatureEngineer(self._steps_config)
        for step in self._steps_config:
            NodeRegistry.get_calculator(step["transformer"])
        self._engineer: FeatureEngineer | None = None

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self._validate_payload(X)
        engineer = FeatureEngineer(self._steps_config)
        transformed, _metrics = engineer.fit_transform((X, y))
        self._engineer = engineer
        return transformed

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        if self._engineer is None:
            raise RuntimeError("transform() called before fit_transform()")
        self._validate_payload(X)
        # At transform/inference time no step needs the target (target-aware
        # encoders use their fitted artifact; splitters/resampling are skipped).
        # Feed a bare frame when y is absent — a ``(X, None)`` tuple would trip
        # ``pack_pipeline_output``'s tuple-shape-lost diagnostic on every step.
        payload = (X, y) if y is not None else X
        transformed = self._engineer.transform(payload)
        # Some appliers return a bare frame instead of the ``(X, y)`` payload;
        # re-pair so callers always get the FoldPreprocessor protocol shape.
        if not (isinstance(transformed, tuple) and len(transformed) == 2):
            transformed = (transformed, y)
        return transformed

    def _validate_payload(self, X: Any) -> None:
        if hasattr(X, "columns") and self._target_column in X.columns:
            raise ValueError(f"target column '{self._target_column}' already present in X")


def frame_rows(frame: Any) -> int:
    """Row count for pandas/polars frames and numpy arrays (-1 if unknowable)."""
    height = getattr(frame, "height", None)
    if isinstance(height, int):
        return height
    try:
        return len(frame)
    except TypeError:
        return -1


class AuditedFoldPreprocessor:
    """Decorates a :class:`FoldPreprocessor` to record per-fold row counts.

    Every ``fit_transform``/``transform`` call records the number of input
    rows it received. The isolation invariant a leak-free run must satisfy is
    ``max(fit_rows) <= train_rows`` — a leaked fit would see the train split
    plus held-out rows. Exposed via :meth:`summary` so the app can log it and
    persist it in node metrics for post-hoc audit (findings 2026-08-26 §3/B).
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.fit_rows: list[int] = []
        self.transform_rows: list[int] = []
        self.changes_row_count = getattr(inner, "changes_row_count", False)

    @property
    def inner(self) -> Any:
        return self._inner

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.fit_rows.append(frame_rows(X))
        return self._inner.fit_transform(X, y)

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.transform_rows.append(frame_rows(X))
        return self._inner.transform(X, y)

    def summary(self, train_rows: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fit_calls": len(self.fit_rows),
            "max_fit_rows": max(self.fit_rows, default=0),
            "transform_calls": len(self.transform_rows),
        }
        if train_rows is not None:
            result["train_rows"] = train_rows
            result["isolation_ok"] = result["max_fit_rows"] <= train_rows
        return result
