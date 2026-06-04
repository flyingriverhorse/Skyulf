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


class SchemaMismatchError(ValueError):
    """Raised when an actual frame schema violates an expected ``SkyulfSchema``.

    Carries structured details so callers can render a precise message instead
    of a generic ``KeyError`` deep inside a transformer:

    Attributes:
        missing: Expected columns absent from the actual frame.
        unexpected: Actual columns not present in the expected schema.
        dtype_mismatches: ``{column: (expected_dtype, actual_dtype)}`` for
            shared columns whose dtype labels differ (only when dtype checking
            is requested).
        order_mismatch: ``True`` when the shared columns appear in a different
            relative order than expected (only when order checking is requested).
    """

    def __init__(
        self,
        message: str,
        *,
        missing: Optional[List[str]] = None,
        unexpected: Optional[List[str]] = None,
        dtype_mismatches: Optional[Dict[str, Tuple[str, str]]] = None,
        order_mismatch: bool = False,
    ) -> None:
        super().__init__(message)
        self.missing = missing or []
        self.unexpected = unexpected or []
        self.dtype_mismatches = dtype_mismatches or {}
        self.order_mismatch = order_mismatch


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

    # ---- Contract validation ---------------------------------------------

    def assert_compatible(
        self,
        actual: "SkyulfSchema",
        *,
        check_dtypes: bool = False,
        check_order: bool = False,
        where: str = "input",
    ) -> None:
        """Validate that ``actual`` satisfies this (expected) schema.

        ``self`` is the expected schema (e.g. what an Applier was fitted on);
        ``actual`` is the schema observed at apply time. Raises
        :class:`SchemaMismatchError` describing every discrepancy. Presence of
        the expected columns is always checked; dtype and column-order checks
        are opt-in to keep the default contract permissive and non-breaking.

        Args:
            actual: The schema observed at runtime.
            check_dtypes: Also compare dtype labels for shared columns.
            check_order: Also require shared columns in the same relative order.
            where: Label used in the error message (e.g. ``"input"``).
        """
        missing, unexpected = _presence_diff(self, actual)
        dtype_mismatches = _dtype_mismatches(self, actual) if check_dtypes else {}
        order_mismatch = _check_order(self, actual, check_order, missing)

        if missing or unexpected or dtype_mismatches or order_mismatch:
            raise SchemaMismatchError(
                _format_mismatch(where, missing, unexpected, dtype_mismatches, order_mismatch),
                missing=missing,
                unexpected=unexpected,
                dtype_mismatches=dtype_mismatches,
                order_mismatch=order_mismatch,
            )


def _presence_diff(expected: "SkyulfSchema", actual: "SkyulfSchema") -> Tuple[List[str], List[str]]:
    """Return ``(missing, unexpected)`` column-name lists between two schemas."""
    actual_cols = set(actual.columns)
    expected_cols = set(expected.columns)
    missing = [c for c in expected.columns if c not in actual_cols]
    unexpected = [c for c in actual.columns if c not in expected_cols]
    return missing, unexpected


def _dtype_mismatches(
    expected: "SkyulfSchema", actual: "SkyulfSchema"
) -> Dict[str, Tuple[str, str]]:
    """Return ``{column: (expected_dtype, actual_dtype)}`` for shared columns."""
    actual_cols = set(actual.columns)
    out: Dict[str, Tuple[str, str]] = {}
    for col in expected.columns:
        if col in actual_cols and col in expected.dtypes and col in actual.dtypes:
            exp_dt, act_dt = expected.dtypes[col], actual.dtypes[col]
            if exp_dt != act_dt:
                out[col] = (exp_dt, act_dt)
    return out


def _check_order(
    expected: "SkyulfSchema", actual: "SkyulfSchema", check_order: bool, missing: List[str]
) -> bool:
    """Return ``True`` when shared columns appear in a different relative order.

    Skipped (returns ``False``) unless ``check_order`` is requested and all
    expected columns are present (order is meaningless with missing columns).
    """
    if not check_order or missing:
        return False
    actual_cols = set(actual.columns)
    expected_cols = set(expected.columns)
    shared_expected = [c for c in expected.columns if c in actual_cols]
    shared_actual = [c for c in actual.columns if c in expected_cols]
    return shared_expected != shared_actual


def _format_mismatch(
    where: str,
    missing: List[str],
    unexpected: List[str],
    dtype_mismatches: Dict[str, Tuple[str, str]],
    order_mismatch: bool,
) -> str:
    """Build a human-readable mismatch message from the collected diffs."""
    parts: List[str] = [f"Schema mismatch on {where}:"]
    if missing:
        parts.append(f" missing columns {missing}")
    if unexpected:
        parts.append(f" unexpected columns {unexpected}")
    if dtype_mismatches:
        parts.append(f" dtype mismatches {dtype_mismatches}")
    if order_mismatch:
        parts.append(" column order differs from expected")
    return "".join(parts)


def validate_schema(
    expected: SkyulfSchema,
    actual: Any,
    *,
    check_dtypes: bool = False,
    check_order: bool = False,
    where: str = "input",
) -> None:
    """Validate a live DataFrame against an ``expected`` schema.

    Thin convenience wrapper: builds a :class:`SkyulfSchema` from ``actual``
    (Pandas/Polars/wrapper frame) and delegates to
    :meth:`SkyulfSchema.assert_compatible`. Raises
    :class:`SchemaMismatchError` on any violation.
    """
    actual_schema = (
        actual if isinstance(actual, SkyulfSchema) else SkyulfSchema.from_dataframe(actual)
    )
    expected.assert_compatible(
        actual_schema,
        check_dtypes=check_dtypes,
        check_order=check_order,
        where=where,
    )


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
