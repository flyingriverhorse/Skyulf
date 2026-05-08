"""Schema/lineage primitive for the preprocessing layer (C7 starter).

A `SkyulfSchema` is a lightweight, pre-execution description of a
DataFrame's columns and dtypes. It powers two future use cases:

1. Canvas-side preview — show "this pipeline will produce columns X, Y, Z"
   without actually running the pipeline.
2. Compile-time validation — catch "downstream node needs column 'price'
   but it was dropped two steps ago" before submission.

Calculators opt in by overriding `BaseCalculator.infer_output_schema`.
The default returns `None`, meaning "I can't predict my output schema
from config alone" — callers fall back to runtime introspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SkyulfSchema:
    """Immutable schema description.

    Attributes:
        columns: Ordered list of column names.
        dtypes: Mapping of column name → string dtype label
            (engine-agnostic; e.g. ``"int64"``, ``"float64"``, ``"string"``,
            ``"category"``, ``"datetime"``, ``"bool"``, or ``"unknown"``).
            A column may be present in ``columns`` but absent from
            ``dtypes`` when its type is unknown.
    """

    columns: Tuple[str, ...]
    dtypes: Dict[str, str] = field(default_factory=dict)

    # ---- Constructors -----------------------------------------------------

    @classmethod
    def from_columns(
        cls, columns: Iterable[str], dtypes: Optional[Dict[str, str]] = None
    ) -> "SkyulfSchema":
        cols = tuple(columns)
        return cls(columns=cols, dtypes=dict(dtypes or {}))

    @classmethod
    def from_dataframe(cls, df: Any) -> "SkyulfSchema":
        """Best-effort schema extraction from a Pandas/Polars/Wrapper frame."""
        raw_cols = getattr(df, "columns", None)
        cols = list(raw_cols) if raw_cols is not None else []
        dtypes = _extract_pandas_dtypes(df)
        if not dtypes:
            dtypes = _extract_polars_dtypes(df)
        return cls(columns=tuple(cols), dtypes=dtypes)

    # ---- Mutations (return new instances) ---------------------------------

    def drop(self, names: Iterable[str]) -> "SkyulfSchema":
        drop_set = set(names)
        new_cols = tuple(c for c in self.columns if c not in drop_set)
        new_dtypes = {k: v for k, v in self.dtypes.items() if k not in drop_set}
        return replace(self, columns=new_cols, dtypes=new_dtypes)

    def add(self, name: str, dtype: str = "unknown") -> "SkyulfSchema":
        if name in self.columns:
            return self
        new_dtypes = dict(self.dtypes)
        new_dtypes[name] = dtype
        return replace(self, columns=self.columns + (name,), dtypes=new_dtypes)

    def rename(self, mapping: Dict[str, str]) -> "SkyulfSchema":
        new_cols = tuple(mapping.get(c, c) for c in self.columns)
        new_dtypes: Dict[str, str] = {}
        for k, v in self.dtypes.items():
            new_dtypes[mapping.get(k, k)] = v
        return replace(self, columns=new_cols, dtypes=new_dtypes)

    def with_dtype(self, name: str, dtype: str) -> "SkyulfSchema":
        if name not in self.columns:
            return self
        new_dtypes = dict(self.dtypes)
        new_dtypes[name] = dtype
        return replace(self, dtypes=new_dtypes)

    # ---- Queries ----------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.columns

    def column_list(self) -> List[str]:
        return list(self.columns)

    def __contains__(self, item: object) -> bool:
        return item in self.columns

    def __len__(self) -> int:
        return len(self.columns)


def _extract_pandas_dtypes(df: Any) -> Dict[str, str]:
    pd_dtypes = getattr(df, "dtypes", None)
    if pd_dtypes is None or not hasattr(pd_dtypes, "items"):
        return {}
    try:
        return {str(name): str(dt) for name, dt in pd_dtypes.items()}
    except Exception:  # noqa: BLE001 - best-effort schema infer
        return {}


def _extract_polars_dtypes(df: Any) -> Dict[str, str]:
    schema_attr = getattr(df, "schema", None)
    if schema_attr is None:
        return {}
    try:
        items = schema_attr.items() if hasattr(schema_attr, "items") else schema_attr
        return {str(name): str(dt) for name, dt in items}
    except Exception:  # noqa: BLE001
        return {}
