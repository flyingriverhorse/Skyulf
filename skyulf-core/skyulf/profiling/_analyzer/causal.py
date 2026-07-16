"""Causal discovery via the PC algorithm (causal-learn)."""

import logging

import polars as pl

from ..schemas import CausalEdge, CausalGraph, CausalNode
from ._utils import _AnalyzerState

logger = logging.getLogger(__name__)


class CausalMixin(_AnalyzerState):
    """Causal-discovery helpers for :class:`EDAAnalyzer`."""

    def _select_target_correlated_columns(
        self, numeric_cols: list[str], primary_target: str
    ) -> list[str]:
        """Keep `primary_target` plus the 14 columns most correlated with it (by |corr|)."""
        corrs = []
        for col in numeric_cols:
            if col == primary_target:
                continue
            c = self.df.select(pl.corr(col, primary_target)).item()  # type: ignore[attr-defined]
            if c is not None:
                corrs.append((col, abs(c)))
        corrs.sort(key=lambda x: x[1], reverse=True)
        selected_cols = [x[0] for x in corrs[:14]]
        selected_cols.append(primary_target)
        return selected_cols

    def _select_highest_variance_columns(self, numeric_cols: list[str]) -> list[str]:
        """Keep the 15 highest-variance columns when no target hint is available."""
        variances = []
        for col in numeric_cols:
            var = self.df.select(pl.col(col).var()).item()  # type: ignore[attr-defined]
            if var is not None:
                variances.append((col, var))
        variances.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in variances[:15]]

    def _limit_columns_for_pc(self, numeric_cols: list[str]) -> list[str]:
        """Cap the column set to 15 columns since PC is O(2^p) in the number of variables.

        With a target-like column present we pick "target + top-14 by |corr|";
        otherwise we fall back to the 15 highest-variance columns.
        """
        if len(numeric_cols) <= 15:
            return numeric_cols

        # Heuristic: if any column name looks like a target, keep it + top-14 by |corr|.
        target_candidates = [
            c for c in numeric_cols if "target" in c.lower() or "label" in c.lower()
        ]
        primary_target = target_candidates[0] if target_candidates else None

        if primary_target:
            return self._select_target_correlated_columns(numeric_cols, primary_target)
        # No target hint → keep highest-variance columns.
        return self._select_highest_variance_columns(numeric_cols)

    @staticmethod
    def _causal_edge_from_endpoints(source: str, target: str, end_i: int, end_j: int) -> CausalEdge:
        """Build the `CausalEdge` for a known, non-absent PC endpoint combination."""
        edge_builders = {
            (-1, 1): lambda: CausalEdge(source=source, target=target, type="directed"),
            (1, -1): lambda: CausalEdge(source=target, target=source, type="directed"),
            (-1, -1): lambda: CausalEdge(source=source, target=target, type="undirected"),
            (1, 1): lambda: CausalEdge(source=source, target=target, type="bidirected"),
        }
        return edge_builders[(end_i, end_j)]()

    def _classify_causal_edge(
        self, source: str, target: str, end_i: int, end_j: int
    ) -> CausalEdge | None:
        """Classify a single (source, target) pair's edge type from its PC endpoints.

        causal-learn endpoint encoding: -1 = tail, 1 = arrowhead.
        """
        if end_j == 0 and end_i == 0:
            return None
        if (end_i, end_j) not in {(-1, 1), (1, -1), (-1, -1), (1, 1)}:
            return None
        return self._causal_edge_from_endpoints(source, target, end_i, end_j)

    def _build_causal_edges(self, adj_matrix, numeric_cols: list[str]) -> list[CausalEdge]:
        """Translate the causal-learn adjacency matrix into `CausalEdge` objects."""
        edges = []
        num_vars = len(numeric_cols)
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                edge = self._classify_causal_edge(
                    numeric_cols[i], numeric_cols[j], adj_matrix[j, i], adj_matrix[i, j]
                )
                if edge is not None:
                    edges.append(edge)
        return edges

    def _discover_causal_graph(self, numeric_cols: list[str]) -> CausalGraph | None:
        """Run the PC algorithm and return a directed/undirected/bidirected graph.

        PC complexity grows exponentially with #variables; we cap at 15. With
        a target column present we pick "target + top-14 by |corr|"; otherwise
        we fall back to the 15 highest-variance columns.
        """
        try:
            numeric_cols = self._limit_columns_for_pc(numeric_cols)

            # Cap rows for runtime budget; PC is O(n) in samples but O(2^p) in vars.
            limit = 5000
            df_numeric = self.df.select(numeric_cols).drop_nulls().head(limit)  # type: ignore[attr-defined]
            if df_numeric.height < 50:
                return None

            data = df_numeric.to_numpy()

            try:
                from causallearn.search.ConstraintBased.PC import (  # ty: ignore[unresolved-import]
                    pc,
                )
            except ImportError:
                return None

            cg = pc(data, alpha=0.05, indep_test="fisherz", show_progress=False)

            nodes = [CausalNode(id=col, label=col) for col in numeric_cols]
            edges = self._build_causal_edges(cg.G.graph, numeric_cols)

            return CausalGraph(nodes=nodes, edges=edges)

        except Exception as e:
            logger.error(f"Error in causal discovery: {e}")
            return None
