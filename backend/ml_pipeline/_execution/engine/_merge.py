"""Multi-input merging helpers for :class:`PipelineEngine`.

Mixin slice — owns: per-node merge-strategy resolution, frame coercion,
column/row-wise frame merging, and SplitDataset/(X,y)-aware fan-in merging.

Relies on ``self._node_configs``, ``self._resolve_all_inputs``,
``self._ancestors_of``, ``self.merge_warnings``, and ``self.log`` from
:class:`PipelineEngine`.
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from skyulf.data.dataset import SplitDataset

from ..schemas import NodeConfig

logger = logging.getLogger(__name__)


class MergeMixin:
    # Type-only stubs so ty can resolve attributes/methods provided by
    # :class:`PipelineEngine` (or its sibling mixins). No runtime impact.
    _node_configs: dict[str, NodeConfig]
    merge_warnings: list[dict[str, Any]]
    log: Callable[[str], None]
    _resolve_all_inputs: Any
    _ancestors_of: Any
    """Frame coercion + multi-input merging split out of :class:`PipelineEngine`."""

    def _coerce_tuple_to_frame(self, payload: tuple, target_col: str) -> pd.DataFrame | None:
        """Coerce an ``(X, y)``-shaped tuple payload to a DataFrame, or ``None`` if empty/unusable."""
        if len(payload) < 1:
            return None
        first = payload[0]
        if not isinstance(first, pd.DataFrame):
            return None
        df = first.copy()
        if len(payload) == 2 and target_col:
            df[target_col] = payload[1]
        return df if not df.empty else None

    def _coerce_to_frame(self, payload: Any, target_col: str = "") -> pd.DataFrame | None:
        """Best-effort coercion of a single payload to a DataFrame.

        Returns ``None`` for empty / missing payloads (e.g. an empty test split)
        so callers can decide whether to skip them.
        """
        if payload is None:
            return None
        if isinstance(payload, pd.DataFrame):
            return payload if not payload.empty else None
        if isinstance(payload, tuple):
            return self._coerce_tuple_to_frame(payload, target_col)
        return None

    def _to_dataframe(self, artifact: Any, target_col: str = "") -> pd.DataFrame:
        """Normalize an artifact to a single DataFrame (train portion only).

        Kept for callers that explicitly want a flat frame. Multi-input merging
        should prefer :meth:`_merge_inputs`, which preserves SplitDataset shape
        when possible.
        """
        if isinstance(artifact, pd.DataFrame):
            return artifact
        if isinstance(artifact, SplitDataset):
            df = self._coerce_to_frame(artifact.train, target_col)
            if df is not None:
                return df
        df = self._coerce_to_frame(artifact, target_col)
        if df is not None:
            return df
        raise TypeError(
            f"Cannot convert artifact of type {type(artifact).__name__} to DataFrame. "
            "Only DataFrame, SplitDataset, and (X, y) tuples are supported."
        )

    def _get_merge_strategy(self, node_id: str) -> str:
        """Resolve per-node merge strategy from node params.

        Recognised values: ``last_wins`` (default), ``first_wins``. Anything
        else falls back to ``last_wins`` with a warning so a typo in the
        canvas config can't silently change semantics.
        """
        cfg = self._node_configs.get(node_id)
        if cfg is None:
            return "last_wins"
        strat = cfg.params.get("_merge_strategy", "last_wins")
        if strat not in ("last_wins", "first_wins"):
            self.log(
                f"Node {node_id}: unknown merge strategy '{strat}', falling back to 'last_wins'."
            )
            return "last_wins"
        return strat

    def _merge_frames_columnwise(
        self,
        frames: list[pd.DataFrame],
        node_id: str,
        strategy: str,
        prefix: str,
    ) -> pd.DataFrame:
        """Merge same-row-count frames column-wise, applying the merge strategy on overlap.

        Iterates frames in the order dictated by the strategy. The dict update
        semantics give us "later writes win", so iterating forward yields
        last_wins and reversed yields first_wins.
        """
        iter_frames = frames if strategy == "last_wins" else list(reversed(frames))
        result_cols: dict[str, pd.Series] = {}
        overwrites: list[str] = []
        new_only: list[str] = []
        for df in iter_frames:
            df_aligned = df.reset_index(drop=True)
            for col in df.columns:
                if col in result_cols:
                    overwrites.append(col)
                else:
                    new_only.append(col)
                result_cols[col] = df_aligned[col]
        merged = pd.DataFrame(result_cols)
        shape_log = " + ".join(str(df.shape) for df in frames)
        if overwrites:
            self.log(
                f"{prefix}: column-wise merge {shape_log} -> {merged.shape} "
                f"({strategy} overwrote {sorted(set(overwrites))})"
            )
        else:
            self.log(f"{prefix}: column-wise merge {shape_log} -> {merged.shape}")
        return merged

    def _merge_frames_rowwise(
        self,
        frames: list[pd.DataFrame],
        node_id: str,
        part_label: str,
        prefix: str,
        row_counts: list[int],
        col_sets: list[set[str]],
    ) -> pd.DataFrame:
        """Merge frames row-wise on their common columns, warning about any dropped ones."""
        common_cols = col_sets[0]
        for cs in col_sets[1:]:
            common_cols = common_cols & cs
        if not common_cols:
            raise ValueError(
                f"{prefix}: cannot row-merge inputs — no common columns. "
                f"Column sets: {[sorted(cs) for cs in col_sets]}"
            )
        if any(common_cols != cs for cs in col_sets):
            extras = sorted(set().union(*col_sets) - common_cols)
            self.log(f"{prefix}: row-merge dropping non-shared columns {extras}")
            # Surface dropped columns to the UI so users see what was lost
            # instead of having to dig through job logs.
            self.merge_warnings.append(
                {
                    "node_id": node_id,
                    "kind": "row_concat_drop",
                    "part": part_label or "rows",
                    "dropped_columns": extras,
                    "kept_columns": sorted(common_cols),
                    "message": (
                        f"Node '{node_id}': row-wise merge kept only the {len(common_cols)} "
                        f"shared columns; {len(extras)} column(s) present in some inputs but "
                        f"not all were dropped: {extras}."
                    ),
                }
            )
        merged = pd.concat(
            [df[sorted(common_cols)] for df in frames],
            axis=0,
            ignore_index=True,
        )
        self.log(
            f"{prefix}: row-wise merge "
            f"{' + '.join(str(rc) for rc in row_counts)} rows → {len(merged)} rows"
        )
        return merged

    def _merge_frames(
        self,
        frames: list[pd.DataFrame],
        node_id: str,
        part_label: str = "",
    ) -> pd.DataFrame:
        """Concatenate a list of DataFrames column-wise (preferred) or row-wise.

        ``part_label`` is only used in log messages (``"train"``, ``"test"``...)
        to make multi-split merges easier to follow in job logs.

        Column-overlap behaviour is governed by the per-node merge strategy
        (see :meth:`_get_merge_strategy`):

        * ``last_wins`` (default) — later inputs overwrite earlier ones on
          shared columns. Matches topological "downstream wins".
        * ``first_wins`` — earlier inputs are kept; later inputs only add
          new columns. Useful when an upstream branch is the source of truth.
        """
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]

        prefix = f"Node {node_id}"
        if part_label:
            prefix = f"{prefix} [{part_label}]"

        row_counts = [len(df) for df in frames]
        col_sets = [set(df.columns) for df in frames]
        same_rows = all(rc == row_counts[0] for rc in row_counts)
        strategy = self._get_merge_strategy(node_id)

        if same_rows:
            return self._merge_frames_columnwise(frames, node_id, strategy, prefix)

        return self._merge_frames_rowwise(frames, node_id, part_label, prefix, row_counts, col_sets)

    def _split_dataset_train_columns(self, art: SplitDataset) -> list[str]:
        """Best-effort extraction of column names from a ``SplitDataset``'s train slot."""
        if isinstance(art.train, pd.DataFrame):
            return list(art.train.columns)
        if isinstance(art.train, tuple):
            X = art.train[0]
            return list(X.columns) if hasattr(X, "columns") else []
        return []

    def _artifact_columns(self, art: Any) -> list[str]:
        """Best-effort extraction of column names from a single resolved artifact."""
        if isinstance(art, pd.DataFrame):
            return list(art.columns)
        if isinstance(art, SplitDataset):
            return self._split_dataset_train_columns(art)
        if isinstance(art, tuple) and len(art) == 2 and hasattr(art[0], "columns"):
            return list(art[0].columns)
        return []

    def _sibling_fan_in_overlap_columns(self, artifacts: list[Any]) -> list[str]:
        """Return column names that appear in 2+ artifacts (subject to last-wins)."""
        input_cols = [self._artifact_columns(art) for art in artifacts]

        seen: dict[str, int] = {}
        overlap: list[str] = []
        for cols in input_cols:
            for c in cols:
                seen[c] = seen.get(c, 0) + 1
        for c, cnt in seen.items():
            if cnt > 1:
                overlap.append(c)
        return overlap

    def _has_redundant_ancestor_edge(
        self, unique_inputs: list[str], ancestors_per_input: list[set[str]]
    ) -> bool:
        """Return True if any input is itself an ancestor of another input (redundant edge)."""
        return any(
            other in ancestors_per_input[i]
            for i, this in enumerate(unique_inputs)
            for j, other in enumerate(unique_inputs)
            if i != j
        )

    def _build_sibling_fan_in_advisory(
        self,
        node: NodeConfig,
        unique_inputs: list[str],
        shared: set[str],
        artifacts: list[Any],
    ) -> dict[str, Any]:
        """Build the sibling fan-in warning advisory payload for the UI banner."""
        # Compute concrete overlap columns + winner per artifact pair so
        # the UI banner can show "Tx overrides DMC on [Id, SepalLengthCm]"
        # instead of vague "last-wins on overlap".
        overlap = self._sibling_fan_in_overlap_columns(artifacts)

        strategy = self._get_merge_strategy(node.node_id)
        winner_id = unique_inputs[-1] if strategy == "last_wins" else unique_inputs[0]
        return {
            "node_id": node.node_id,
            "kind": "sibling_fan_in",
            "inputs": unique_inputs,
            "common_ancestors": sorted(shared),
            "overlap_columns": sorted(overlap),
            "winner_input": winner_id,
            "strategy": strategy,
            "message": (
                f"Node '{node.node_id}' merges {len(unique_inputs)} sibling "
                f"branches that share ancestor(s) {sorted(shared)}. "
                f"Columns are unioned; on overlap ({len(overlap)} column(s)) "
                f"the {strategy} input '{winner_id}' wins. If you wanted sequential "
                "application, chain the transformers linearly instead."
            ),
        }

    def _warn_sibling_fan_in(self, node: NodeConfig, artifacts: list[Any]) -> None:
        """Warn when a node fans in true sibling branches sharing a common ancestor.

        Only warns when inputs are TRUE siblings (no input is itself an
        ancestor of another). The "ancestor + its descendant" pattern (e.g.
        Splitter + Splitter→Scaler both feeding Encoder) is a redundant edge —
        the descendant supersedes the ancestor under last-wins, so the merge
        is harmless and we don't warn. We DO warn when two genuinely
        independent siblings off a shared ancestor get fanned in (the
        "Path A" UX trap), because the user likely meant a sequential chain.
        """
        unique_inputs = list(dict.fromkeys(node.inputs or []))
        if len(unique_inputs) <= 1:
            return

        ancestors_per_input = [self._ancestors_of(nid) for nid in unique_inputs]
        shared = set.intersection(*ancestors_per_input) if ancestors_per_input else set()

        # Skip when any input is an ancestor of another input (redundant edge).
        redundant_edge = self._has_redundant_ancestor_edge(unique_inputs, ancestors_per_input)

        if not shared or redundant_edge:
            return

        advisory = self._build_sibling_fan_in_advisory(node, unique_inputs, shared, artifacts)
        self.merge_warnings.append(advisory)
        self.log(f"WARN: {advisory['message']}")

    def _reject_model_inputs(self, node: NodeConfig, artifacts: list[Any]) -> None:
        """Raise if any resolved input is a Model object rather than data.

        Guards against obvious wiring mistakes (model object plugged into a
        data input).
        """
        # noqa: B905 -- `artifacts` may be shorter than `node.inputs` when duplicate
        # input edges are deduped in `_resolve_all_inputs`; `strict=True` would raise
        # in that legitimate case, so the length mismatch is intentional here.
        for input_id, art in zip(node.inputs, artifacts):  # noqa: B905
            if hasattr(art, "predict") or (hasattr(art, "fit") and not hasattr(art, "transform")):
                raise ValueError(
                    f"Node {node.node_id}: input from '{input_id}' is a Model object "
                    f"(type: {type(art).__name__}). Nodes expect data, not models. "
                    f"Did you connect a training/tuning output directly?"
                )

    def _merge_xy_tuples(self, node: NodeConfig, artifacts: list[Any]) -> Any:
        """Merge inputs that are all ``(X, y)`` tuples, preserving tuple shape.

        Merges X column-wise; reuses y from the first edge (duplicate edges
        to the same source share the same y).
        """
        x_frames = [a[0] for a in artifacts if isinstance(a[0], pd.DataFrame)]
        if not x_frames:
            raise ValueError(
                f"Node {node.node_id}: cannot merge (X, y) tuples - X parts are not DataFrames."
            )
        merged_x = self._merge_frames(x_frames, node.node_id, "X")
        return (merged_x, artifacts[0][1])

    def _merge_split_dataset_xy_part(
        self, node: NodeConfig, part_label: str, non_empty: list[Any]
    ) -> Any:
        """Merge (X, y) tuples for one SplitDataset slot, keeping y from the first branch.

        All branches produced (X, y) tuples → merge X columns, keep y from the
        first branch (all branches descend from the same Splitter, so y is
        identical).
        """
        x_frames: list[pd.DataFrame] = []
        for p in non_empty:
            x = p[0]
            if isinstance(x, pd.DataFrame) and not x.empty:
                x_frames.append(x)
        if not x_frames:
            return None
        merged_x = self._merge_frames(x_frames, node.node_id, part_label)
        return (merged_x, non_empty[0][1])

    def _merge_split_dataset_frame_part(
        self, node: NodeConfig, part_label: str, non_empty: list[Any], target_col: str
    ) -> pd.DataFrame | None:
        """Flatten mixed or pure-DataFrame parts and merge them as frames."""
        frames = [
            df for df in (self._coerce_to_frame(p, target_col) for p in non_empty) if df is not None
        ]
        if not frames:
            return None
        return self._merge_frames(frames, node.node_id, part_label)

    def _merge_split_dataset_part(
        self, node: NodeConfig, part_label: str, parts: list[Any], target_col: str
    ) -> Any:
        """Merge one SplitDataset slot (train/test/validation) across branches.

        Preserves ``(X, y)`` tuple shape when every branch produced a tuple —
        this keeps downstream X/y tabs and training contracts intact. Falls
        back to flat-DataFrame merging otherwise.
        """
        non_empty = [p for p in parts if p is not None]
        if not non_empty:
            return None
        if all(isinstance(p, tuple) and len(p) == 2 for p in non_empty):
            return self._merge_split_dataset_xy_part(node, part_label, non_empty)
        # Mixed or pure DataFrame parts → flatten and merge as frames.
        return self._merge_split_dataset_frame_part(node, part_label, non_empty, target_col)

    def _merge_split_datasets(
        self, node: NodeConfig, artifacts: list[Any], target_col: str
    ) -> SplitDataset:
        """Merge train/test/validation independently across all-SplitDataset inputs."""
        split_artifacts: list[SplitDataset] = [a for a in artifacts if isinstance(a, SplitDataset)]

        merged_train = self._merge_split_dataset_part(
            node, "train", [sd.train for sd in split_artifacts], target_col
        )
        if merged_train is None:
            raise ValueError(
                f"Node {node.node_id}: all upstream SplitDataset inputs have empty train splits."
            )
        merged_test = self._merge_split_dataset_part(
            node, "test", [sd.test for sd in split_artifacts], target_col
        )
        merged_val = self._merge_split_dataset_part(
            node, "validation", [sd.validation for sd in split_artifacts], target_col
        )

        # Empty test defaults to an empty DataFrame for downstream consumers
        # that assume `.test` is iterable.
        if merged_test is None:
            merged_test = pd.DataFrame()

        return SplitDataset(
            train=cast(Any, merged_train),
            test=cast(Any, merged_test),
            validation=merged_val,
        )

    def _merge_fallback_frames(
        self, node: NodeConfig, artifacts: list[Any], target_col: str
    ) -> pd.DataFrame:
        """Flatten mixed/all-DataFrame inputs to DataFrames and merge them.

        Logs a warning when SplitDatasets are flattened so the loss of
        held-out splits is visible in job logs.
        """
        if any(isinstance(a, SplitDataset) for a in artifacts):
            self.log(
                f"Node {node.node_id}: mixed SplitDataset/DataFrame inputs — "
                "merging on train portions only; held-out splits are dropped."
            )

        dataframes: list[pd.DataFrame] = []
        for i, art in enumerate(artifacts):
            df = self._coerce_to_frame(art, target_col)
            if df is None:
                df = self._to_dataframe(art, target_col)
            if df.empty:
                raise ValueError(
                    f"Node {node.node_id}: input #{i} produced an empty DataFrame "
                    "(0 rows). Check upstream preprocessing branches."
                )
            dataframes.append(df)

        return self._merge_frames(dataframes, node.node_id)

    def _merge_inputs(self, node: NodeConfig, target_col: str = "") -> Any:
        """Resolve and merge all upstream inputs for a multi-input node.

        Behaviour:

        * Single input → returned as-is (preserves DataFrame / SplitDataset).
        * All inputs are :class:`SplitDataset` → merge ``train`` / ``test`` /
          ``validation`` independently and return a new ``SplitDataset``.
        * Mixed or all-DataFrame inputs → flatten to DataFrames and merge.
          A warning is logged when SplitDatasets are flattened so the loss of
          held-out splits is visible in job logs.
        """
        artifacts = self._resolve_all_inputs(node)
        if len(artifacts) == 1:
            return artifacts[0]

        self.log(f"Node {node.node_id}: merging {len(artifacts)} inputs")

        self._warn_sibling_fan_in(node, artifacts)
        self._reject_model_inputs(node, artifacts)

        all_splits = all(isinstance(a, SplitDataset) for a in artifacts)
        all_xy_tuples = all(isinstance(a, tuple) and len(a) == 2 for a in artifacts)
        if all_xy_tuples:
            return self._merge_xy_tuples(node, artifacts)

        if all_splits:
            return self._merge_split_datasets(node, artifacts, target_col)

        return self._merge_fallback_frames(node, artifacts, target_col)
