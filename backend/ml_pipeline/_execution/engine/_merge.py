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
import polars as pl

from backend.config import get_settings
from skyulf.data.dataset import SplitDataset

from ..graph_utils import _extract_columns
from ..schemas import NodeConfig

logger = logging.getLogger(__name__)


class MergeMixin:
    # Type-only stubs so ty can resolve attributes/methods provided by
    # :class:`PipelineEngine` (or its sibling mixins). No runtime impact.
    _node_configs: dict[str, NodeConfig]
    merge_warnings: list[dict[str, Any]]
    log: Callable[[str], None]
    _resolve_all_inputs: Any
    _merge_input_order: Any
    _ancestors_of: Any
    artifact_store: Any
    """Frame coercion + multi-input merging split out of :class:`PipelineEngine`."""

    def _nearest_common_ancestor_id(self, node_id: str) -> str | None:
        """Deepest ancestor shared by every input of ``node_id``, or ``None``.

        Used as the "before" snapshot when deciding which branch actually
        modified an overlapping column, so the merge can prefer the branch that
        did real work instead of whichever edge happens to be last.
        """
        cfg = self._node_configs.get(node_id)
        if cfg is None:
            return None
        inputs = self._merge_input_order(cfg)
        if len(inputs) < 2:
            return None
        shared = set.intersection(*(self._ancestors_of(nid) for nid in inputs))
        if not shared:
            return None
        return max(shared, key=lambda nid: len(self._ancestors_of(nid)))

    def _baseline_frame(self, node_id: str) -> pd.DataFrame | None:
        """Load the nearest common ancestor's output as a DataFrame, if obtainable."""
        ancestor_id = self._nearest_common_ancestor_id(node_id)
        if ancestor_id is None:
            return None
        try:
            baseline = self._coerce_to_frame(self.artifact_store.load(ancestor_id))
            return self._to_pandas_frame(baseline) if baseline is not None else None
        except Exception as exc:  # noqa: BLE001 - baseline is an optimisation, never fatal
            self.log(f"Node {node_id}: could not load merge baseline '{ancestor_id}': {exc}")
            return None

    @staticmethod
    def _column_changed(frame: pd.DataFrame, baseline: pd.DataFrame, col: str) -> bool:
        """True when ``frame`` holds different values for ``col`` than ``baseline``.

        A differing row count means the branch reshaped the data, which counts
        as a change because the values can no longer be compared position-wise.
        """
        if col not in baseline.columns:
            return True
        if len(frame) != len(baseline):
            return True
        return not frame[col].reset_index(drop=True).equals(baseline[col].reset_index(drop=True))

    @staticmethod
    def _modifiers_agree(frames: list[pd.DataFrame], changed_by: list[int], col: str) -> bool:
        """True when every branch that changed ``col`` produced the same values.

        Two branches independently deriving an identical column (e.g. two
        MissingIndicator steps emitting the same ``*_missing`` flags) discard
        nothing when merged, so they are not a conflict the user must resolve.
        """
        reference = frames[changed_by[0]][col]
        return all(frames[idx][col].equals(reference) for idx in changed_by[1:])

    def _column_modifiers(self, frames: list[pd.DataFrame], node_id: str) -> dict[str, list[int]]:
        """Map each column shared by 2+ frames to the indices of frames that changed it.

        Columns nobody changed, or that only one branch changed, are not real
        conflicts: the merge can pick the single meaningful version regardless
        of input order. Branches that changed a column but agree on the result
        collapse to a single owner for the same reason. Returns an empty map
        when no baseline is available, in which case callers fall back to the
        configured merge strategy.
        """
        baseline = self._baseline_frame(node_id)
        if baseline is None:
            return {}

        counts: dict[str, int] = {}
        for df in frames:
            for col in df.columns:
                counts[col] = counts.get(col, 0) + 1

        modifiers: dict[str, list[int]] = {}
        for col, count in counts.items():
            if count < 2:
                continue
            changed_by = [
                idx
                for idx, df in enumerate(frames)
                if col in df.columns and self._column_changed(df, baseline, col)
            ]
            if len(changed_by) > 1 and self._modifiers_agree(frames, changed_by, col):
                changed_by = changed_by[:1]
            modifiers[col] = changed_by
        return modifiers

    def _column_owners(self, frames: list[pd.DataFrame], node_id: str) -> dict[str, int]:
        """Frame index that should supply each unambiguously-owned overlapping column."""
        owners: dict[str, int] = {}
        for col, changed_by in self._column_modifiers(frames, node_id).items():
            if len(changed_by) == 1:
                owners[col] = changed_by[0]
        return owners

    def _coerce_tuple_to_frame(self, payload: tuple, target_col: str) -> Any | None:
        """Coerce an ``(X, y)``-shaped tuple payload to a DataFrame, or ``None`` if empty/unusable."""
        if len(payload) < 1:
            return None
        first = payload[0]
        if isinstance(first, pl.DataFrame):
            df = first.clone()
            if len(payload) == 2 and target_col:
                df = df.with_columns(pl.Series(name=target_col, values=payload[1]))
            return df if not df.is_empty() else None
        if not isinstance(first, pd.DataFrame):
            return None
        df = first.copy()
        if len(payload) == 2 and target_col:
            df[target_col] = payload[1]
        return df if not df.empty else None

    def _coerce_to_frame(self, payload: Any, target_col: str = "") -> Any | None:
        """Best-effort coercion of a single payload to a DataFrame.

        Returns ``None`` for empty / missing payloads (e.g. an empty test split)
        so callers can decide whether to skip them.
        """
        if payload is None:
            return None
        if isinstance(payload, pl.DataFrame):
            return payload if not payload.is_empty() else None
        if isinstance(payload, pd.DataFrame):
            return payload if not payload.empty else None
        if isinstance(payload, tuple):
            return self._coerce_tuple_to_frame(payload, target_col)
        return None

    def _to_dataframe(self, artifact: Any, target_col: str = "") -> Any:
        """Normalize an artifact to a single DataFrame (train portion only).

        Kept for callers that explicitly want a flat frame. Multi-input merging
        should prefer :meth:`_merge_inputs`, which preserves SplitDataset shape
        when possible.
        """
        if isinstance(artifact, (pd.DataFrame, pl.DataFrame)):
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
        """Merge same-row-count frames column-wise, resolving overlapping columns.

        A column carried unchanged by one branch and rewritten by another is not
        a conflict — the branch that actually modified it owns it, whatever the
        input order. The configured strategy only breaks ties between two or
        more branches that each changed the same column.
        """
        owners = self._column_owners(frames, node_id)

        result_cols: dict[str, pd.Series] = {}
        contested: list[str] = []
        # OC-157: walk the inputs in their own order under both strategies.
        # first_wins used to be implemented by reversing this loop, which also
        # reversed the output columns — result_cols' insertion order is the
        # merged frame's column order, so the tiebreak direction leaked into
        # the shape handed to positional consumers. Declining to overwrite an
        # already-claimed column gives the same ownership; re-assigning an
        # existing dict key keeps its position, so both strategies now emit
        # columns in first-appearance input order.
        for idx, df in enumerate(frames):
            df_aligned = df.reset_index(drop=True)
            for col in df.columns:
                owner = owners.get(col)
                if owner is not None and owner != idx:
                    continue
                claimed = col in result_cols
                if claimed and owner is None:
                    contested.append(col)
                if claimed and strategy == "first_wins":
                    continue
                result_cols[col] = df_aligned[col]

        merged = pd.DataFrame(result_cols)
        shape_log = " + ".join(str(df.shape) for df in frames)
        if contested:
            self.log(
                f"{prefix}: column-wise merge {shape_log} -> {merged.shape} "
                f"({strategy} broke ties on {sorted(set(contested))})"
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
        """Merge frames row-wise on their common columns, surfacing the switch to the UI."""
        common_cols = col_sets[0]
        for cs in col_sets[1:]:
            common_cols = common_cols & cs
        if not common_cols:
            raise ValueError(
                f"{prefix}: cannot row-merge inputs — no common columns. "
                f"Column sets: {[sorted(cs) for cs in col_sets]}"
            )

        # OC-153: reaching this method at all is worth telling the user about.
        # Wiring branches into a merge node expresses a feature union, but a
        # union is impossible once the branches describe different numbers of
        # rows, so the engine stacks them and returns a *taller* frame than
        # either input. That used to be reported only when the column sets also
        # differed, which left the identical-columns case — one branch filtered,
        # the other not — stacking duplicated rows with no UI signal at all.
        same_columns = all(cs == col_sets[0] for cs in col_sets)
        counts = " vs ".join(str(rc) for rc in row_counts)
        if same_columns:
            remedy = (
                " Identical columns with differing row counts usually means one branch "
                "filtered rows (outlier removal, deduplication, dropna), so the rows it "
                "kept now appear more than once and are reweighted in training — and can "
                "land on both sides of a downstream split. Move the filtering step after "
                "the merge to keep one row per observation."
            )
        else:
            remedy = (
                " This is expected when appending separate datasets, but a merge node "
                "cannot join branches that no longer describe the same observations."
            )
        self.merge_warnings.append(
            {
                "node_id": node_id,
                "kind": "row_count_mismatch",
                "part": part_label or "rows",
                "row_counts": list(row_counts),
                "message": (
                    f"Node '{node_id}': inputs have different row counts ({counts}), so "
                    f"they were stacked row-wise into {sum(row_counts)} rows instead of "
                    f"joined column-wise.{remedy}"
                ),
            }
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

    @staticmethod
    def _to_pandas_frame(df: Any) -> pd.DataFrame:
        """Convert a Polars frame to pandas for the pandas-only merge internals."""
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    def _merge_frames(
        self,
        frames: list[Any],
        node_id: str,
        part_label: str = "",
    ) -> Any:
        """Concatenate a list of DataFrames column-wise (preferred) or row-wise.

        Engine boundary: the merge semantics below are pandas-only, so Polars
        inputs are converted in, and the merged result is converted back to
        Polars when the inputs were Polars and the configured engine is Polars
        — downstream nodes keep receiving the engine's frame type.

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

        had_polars = any(isinstance(df, pl.DataFrame) for df in frames)
        frames = [self._to_pandas_frame(df) for df in frames]

        prefix = f"Node {node_id}"
        if part_label:
            prefix = f"{prefix} [{part_label}]"

        row_counts = [len(df) for df in frames]
        col_sets = [set(df.columns) for df in frames]
        same_rows = all(rc == row_counts[0] for rc in row_counts)
        strategy = self._get_merge_strategy(node_id)

        merged = (
            self._merge_frames_columnwise(frames, node_id, strategy, prefix)
            if same_rows
            else self._merge_frames_rowwise(
                frames, node_id, part_label, prefix, row_counts, col_sets
            )
        )
        if had_polars and get_settings().SKYULF_ENGINE == "polars":
            return pl.from_pandas(merged)
        return merged

    def _split_dataset_train_columns(self, art: SplitDataset) -> list[str]:
        """Best-effort extraction of column names from a ``SplitDataset``'s train slot."""
        if isinstance(art.train, (pd.DataFrame, pl.DataFrame)):
            return list(art.train.columns)
        if isinstance(art.train, tuple):
            X = art.train[0]
            return list(X.columns) if hasattr(X, "columns") else []
        return []

    def _artifact_columns(self, art: Any) -> list[str]:
        """Best-effort extraction of column names from a single resolved artifact."""
        if isinstance(art, (pd.DataFrame, pl.DataFrame)):
            return list(art.columns)
        if isinstance(art, SplitDataset):
            return self._split_dataset_train_columns(art)
        if isinstance(art, tuple) and len(art) == 2 and hasattr(art[0], "columns"):
            return list(art[0].columns)
        return []

    def _sibling_fan_in_overlap_columns(self, artifacts: list[Any], node_id: str) -> list[str]:
        """Return columns two or more branches actually modified — the real conflicts.

        A column merely carried through by one branch and rewritten by another
        has an unambiguous owner and needs no tiebreak, so it is not reported.
        Falls back to plain name overlap when no baseline is available.
        """
        frames = [f for f in (self._coerce_to_frame(art) for art in artifacts) if f is not None]
        frames = [self._to_pandas_frame(f) for f in frames]
        if len(frames) == len(artifacts):
            modifiers = self._column_modifiers(frames, node_id)
            if modifiers:
                return [col for col, changed_by in modifiers.items() if len(changed_by) > 1]

        seen: dict[str, int] = {}
        for art in artifacts:
            for c in self._artifact_columns(art):
                seen[c] = seen.get(c, 0) + 1
        return [c for c, cnt in seen.items() if cnt > 1]

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
    ) -> dict[str, Any] | None:
        """Build the sibling fan-in advisory, or ``None`` when no branch conflicts.

        Only genuine conflicts — a column two or more branches each rewrote —
        need a winner. Reporting one otherwise puts a "wins merge" label on the
        canvas for a merge where nothing was actually discarded.
        """
        overlap = self._sibling_fan_in_overlap_columns(artifacts, node.node_id)
        if not overlap:
            return None

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
                f"{len(overlap)} column(s) were modified by more than one branch; "
                f"the {strategy} input '{winner_id}' wins and the other branch's "
                "edits to those columns are discarded. Chain the transformers "
                "linearly instead if you wanted both applied."
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
        unique_inputs = self._merge_input_order(node)
        if len(unique_inputs) <= 1:
            return

        ancestors_per_input = [self._ancestors_of(nid) for nid in unique_inputs]
        shared = set.intersection(*ancestors_per_input) if ancestors_per_input else set()

        # Skip when any input is an ancestor of another input (redundant edge).
        redundant_edge = self._has_redundant_ancestor_edge(unique_inputs, ancestors_per_input)

        if not shared or redundant_edge:
            return

        advisory = self._build_sibling_fan_in_advisory(node, unique_inputs, shared, artifacts)
        if advisory is None:
            return
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
        for input_id, art in zip(self._merge_input_order(node), artifacts):  # noqa: B905
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
        x_frames = [a[0] for a in artifacts if isinstance(a[0], (pd.DataFrame, pl.DataFrame))]
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
        x_frames: list[Any] = []
        for p in non_empty:
            x = p[0]
            if (
                isinstance(x, pl.DataFrame)
                and not x.is_empty()
                or isinstance(x, pd.DataFrame)
                and not x.empty
            ):
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
            merged_test = (
                pl.DataFrame() if get_settings().SKYULF_ENGINE == "polars" else pd.DataFrame()
            )

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

        dataframes: list[Any] = []
        for i, art in enumerate(artifacts):
            df = self._coerce_to_frame(art, target_col)
            if df is None:
                df = self._to_dataframe(art, target_col)
            if len(df) == 0:
                raise ValueError(
                    f"Node {node.node_id}: input #{i} produced an empty DataFrame "
                    "(0 rows). Check upstream preprocessing branches."
                )
            dataframes.append(df)

        return self._merge_frames(dataframes, node.node_id)

    def _upstream_dropped_columns(self, node: NodeConfig) -> list[str]:
        """Columns explicitly removed by any Drop Columns / feature-selection ancestor.

        Only statically declared columns are returned; threshold-based drops are
        data-dependent and can't be known from the graph alone.
        """
        dropped: list[str] = []
        for ancestor_id in self._ancestors_of(node.node_id):
            cfg = self._node_configs.get(ancestor_id)
            if cfg is None:
                continue
            dropped.extend(_extract_columns(cfg.step_type, cfg.params or {}))
        return sorted(set(dropped))

    def _strip_columns_from_frame(self, df: Any, columns: list[str]) -> Any:
        present = [c for c in columns if c in df.columns]
        if not present:
            return df
        if isinstance(df, pl.DataFrame):
            return df.drop(present)
        return df.drop(columns=present)

    def _strip_columns(self, payload: Any, columns: list[str]) -> Any:
        """Remove ``columns`` from any merge-result shape, leaving other shapes untouched."""
        if isinstance(payload, (pd.DataFrame, pl.DataFrame)):
            return self._strip_columns_from_frame(payload, columns)
        if isinstance(payload, SplitDataset):
            return SplitDataset(
                train=self._strip_columns(payload.train, columns),
                test=self._strip_columns(payload.test, columns),
                validation=self._strip_columns(payload.validation, columns),
            )
        if (
            isinstance(payload, tuple)
            and payload
            and isinstance(payload[0], (pd.DataFrame, pl.DataFrame))
        ):
            return (self._strip_columns_from_frame(payload[0], columns), *payload[1:])
        return payload

    def _enforce_upstream_drops(self, node: NodeConfig, merged: Any) -> Any:
        """Re-apply upstream drops so a sibling branch can't resurrect a dropped column.

        A fan-in merge unions columns, so a branch that bypassed a Drop Columns
        node silently reintroduces what the user asked to remove — and the model
        then trains on it. A drop is treated as authoritative for the whole
        subgraph below it, not just its own branch.
        """
        dropped = self._upstream_dropped_columns(node)
        if not dropped:
            return merged

        before = set(self._artifact_columns(merged))
        restored = [c for c in dropped if c in before]
        if not restored:
            return merged

        self.log(
            f"Node {node.node_id}: merge reintroduced upstream-dropped column(s) "
            f"{restored} from a sibling branch — dropping them again."
        )
        self.merge_warnings.append(
            {
                "node_id": node.node_id,
                "kind": "upstream_drop_reapplied",
                "inputs": self._merge_input_order(node),
                "dropped_columns": restored,
            }
        )
        return self._strip_columns(merged, restored)

    def _merge_inputs(self, node: NodeConfig, target_col: str = "") -> Any:
        """Resolve and merge all upstream inputs for a multi-input node.

        Behaviour:

        * Single input → returned as-is (preserves DataFrame / SplitDataset).
        * All inputs are :class:`SplitDataset` → merge ``train`` / ``test`` /
          ``validation`` independently and return a new ``SplitDataset``.
        * Mixed or all-DataFrame inputs → flatten to DataFrames and merge.
          A warning is logged when SplitDatasets are flattened so the loss of
          held-out splits is visible in job logs.

        Whatever the shape, columns removed by an upstream Drop Columns node are
        stripped again afterwards so a sibling branch cannot resurrect them.
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
            merged = self._merge_xy_tuples(node, artifacts)
        elif all_splits:
            merged = self._merge_split_datasets(node, artifacts, target_col)
        else:
            merged = self._merge_fallback_frames(node, artifacts, target_col)

        return self._enforce_upstream_drops(node, merged)
