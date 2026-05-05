"""Causal discovery via the PC algorithm (causal-learn)."""

import logging
from typing import List, Optional

import polars as pl

from ..schemas import CausalEdge, CausalGraph, CausalNode
from ._utils import _AnalyzerState

logger = logging.getLogger(__name__)


class CausalMixin(_AnalyzerState):
    """Causal-discovery helpers for :class:`EDAAnalyzer`."""

    def _discover_causal_graph(self, numeric_cols: List[str]) -> Optional[CausalGraph]:
        """Run the PC algorithm and return a directed/undirected/bidirected graph.

        PC complexity grows exponentially with #variables; we cap at 15. With
        a target column present we pick "target + top-14 by |corr|"; otherwise
        we fall back to the 15 highest-variance columns.
        """
        try:
            if len(numeric_cols) > 15:
                # Heuristic: if any column name looks like a target, keep it + top-14 by |corr|.
                target_candidates = [
                    c for c in numeric_cols if "target" in c.lower() or "label" in c.lower()
                ]
                primary_target = target_candidates[0] if target_candidates else None

                if primary_target:
                    corrs = []
                    for col in numeric_cols:
                        if col == primary_target:
                            continue
                        c = self.df.select(  # type: ignore[attr-defined]
                            pl.corr(col, primary_target)
                        ).item()
                        if c is not None:
                            corrs.append((col, abs(c)))
                    corrs.sort(key=lambda x: x[1], reverse=True)
                    selected_cols = [x[0] for x in corrs[:14]]
                    selected_cols.append(primary_target)
                    numeric_cols = selected_cols
                else:
                    # No target hint → keep highest-variance columns.
                    variances = []
                    for col in numeric_cols:
                        var = self.df.select(pl.col(col).var()).item()  # type: ignore[attr-defined]
                        if var is not None:
                            variances.append((col, var))
                    variances.sort(key=lambda x: x[1], reverse=True)
                    numeric_cols = [x[0] for x in variances[:15]]

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
            edges = []

            adj_matrix = cg.G.graph
            num_vars = len(numeric_cols)

            # causal-learn endpoint encoding: -1 = tail, 1 = arrowhead.
            for i in range(num_vars):
                for j in range(i + 1, num_vars):
                    end_j = adj_matrix[i, j]
                    end_i = adj_matrix[j, i]

                    source = numeric_cols[i]
                    target = numeric_cols[j]

                    if end_j == 0 and end_i == 0:
                        continue
                    if end_i == -1 and end_j == 1:
                        edges.append(CausalEdge(source=source, target=target, type="directed"))
                    elif end_i == 1 and end_j == -1:
                        edges.append(CausalEdge(source=target, target=source, type="directed"))
                    elif end_i == -1 and end_j == -1:
                        edges.append(CausalEdge(source=source, target=target, type="undirected"))
                    elif end_i == 1 and end_j == 1:
                        edges.append(CausalEdge(source=source, target=target, type="bidirected"))

            return CausalGraph(nodes=nodes, edges=edges)

        except Exception as e:
            logger.error(f"Error in causal discovery: {e}")
            return None
