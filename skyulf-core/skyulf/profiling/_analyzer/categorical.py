"""Categorical column profiling helpers."""

from ..schemas import CategoricalStats
from ._utils import _AnalyzerState


class CategoricalMixin(_AnalyzerState):
    """Categorical helpers for :class:`EDAAnalyzer`."""

    def _analyze_categorical(self, col: str, row: dict, basic: dict) -> CategoricalStats:
        unique_count = basic.get(f"{col}__unique", 0)
        top_k_list = row.get(f"{col}__top_k", [])

        top_k = []
        if top_k_list is not None:
            for item in top_k_list:
                # Polars value_counts returns a struct: {col_name: value, count: c}.
                if isinstance(item, dict):
                    val_key = next((k for k in item if k != "count"), None)
                    if val_key is None:
                        continue
                    value = item[val_key]
                    if value is None:
                        # Polars' `value_counts` includes null as a real category.
                        # Drop it rather than stringify it: `str(None)` would put a
                        # literal "None" in top_k, indistinguishable from a genuine
                        # category value of that name. Null frequency is still
                        # reported by ColumnProfile.missing_count/missing_percentage.
                        continue
                    top_k.append({"value": str(value), "count": item["count"]})

        rare_count = row.get(f"{col}__rare", 0)
        return CategoricalStats(
            unique_count=unique_count, top_k=top_k, rare_labels_count=rare_count
        )
