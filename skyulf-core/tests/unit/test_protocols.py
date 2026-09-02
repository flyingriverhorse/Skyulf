"""Tests for the structural protocols in `skyulf.core.protocols`."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from skyulf.core.protocols import ApplierProtocol, CalculatorProtocol, PipelineStep
from skyulf.engines import PandasBackedFrame, PolarsBackedFrame, SkyulfDataFrame
from skyulf.preprocessing.base import BaseApplier, BaseCalculator, StatefulTransformer


class _DuckCalculator:
    """Plain duck-typed calculator — does NOT subclass `BaseCalculator`."""

    def fit(
        self, df: pd.DataFrame | SkyulfDataFrame | tuple, config: dict[str, Any]
    ) -> Mapping[str, Any]:
        assert isinstance(df, pd.DataFrame)
        return {"mean": df["x"].mean()}


class _DuckApplier:
    """Plain duck-typed applier — does NOT subclass `BaseApplier`."""

    def apply(self, df: pd.DataFrame | SkyulfDataFrame | tuple, params: dict[str, Any]) -> Any:
        assert isinstance(df, pd.DataFrame)
        out = df.copy()
        out["x"] = out["x"] - params["mean"]
        return out


class _NotAStep:
    """Has neither `fit` nor `apply` — should fail every protocol check."""


def test_duck_typed_calculator_satisfies_calculator_protocol() -> None:
    assert isinstance(_DuckCalculator(), CalculatorProtocol)


def test_duck_typed_applier_satisfies_applier_protocol() -> None:
    assert isinstance(_DuckApplier(), ApplierProtocol)


def test_abc_subclasses_still_satisfy_protocols() -> None:
    """Existing BaseCalculator/BaseApplier subclasses remain protocol-compatible."""

    class _AbcCalculator(BaseCalculator):
        def fit(
            self, df: pd.DataFrame | SkyulfDataFrame | tuple, config: dict[str, Any]
        ) -> Mapping[str, Any]:
            return {}

    class _AbcApplier(BaseApplier):
        def apply(self, df: pd.DataFrame | SkyulfDataFrame | tuple, params: dict[str, Any]) -> Any:
            return df

    assert isinstance(_AbcCalculator(), CalculatorProtocol)
    assert isinstance(_AbcApplier(), ApplierProtocol)


def test_non_conforming_object_fails_protocol_checks() -> None:
    obj = _NotAStep()
    assert not isinstance(obj, CalculatorProtocol)
    assert not isinstance(obj, ApplierProtocol)


def test_stateful_transformer_accepts_duck_typed_calculator_and_applier() -> None:
    """StatefulTransformer works with plain duck-typed objects (no ABC subclassing)."""
    transformer = StatefulTransformer(
        calculator=_DuckCalculator(), applier=_DuckApplier(), node_id="duck_step"
    )
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    params = transformer.calculator.fit(df, {})
    result = transformer.applier.apply(df, dict(params))

    assert params["mean"] == pytest.approx(2.0)
    assert result["x"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert isinstance(transformer, PipelineStep)


class _BaseFrameDuck:
    """Duck-typed object exposing exactly the base ``SkyulfDataFrame`` members.

    Mirrors the ``_DuckCalculator``/``_DuckApplier`` pattern above: a plain
    object with no engine import, so the protocol checks exercise the
    structural contract rather than a real wrapper's ``__getattr__``.
    """

    @property
    def columns(self) -> list[str]:
        return ["x"]

    @property
    def dtypes(self) -> Any:
        return {"x": "float64"}

    @property
    def shape(self) -> tuple[int, int]:
        return (3, 1)

    def __getitem__(self, key: Any) -> Any:
        return key

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def __len__(self) -> int:
        return 3

    def select(self, columns: list[str]) -> Any:
        return self

    def drop(self, columns: list[str]) -> Any:
        return self

    def with_column(self, name: str, values: Any) -> Any:
        return self

    def to_native(self) -> Any:
        return self

    def to_pandas(self) -> Any:
        return self

    def to_arrow(self) -> Any:
        return self

    def copy(self) -> Any:
        return self


class _PandasFrameDuck(_BaseFrameDuck):
    """Adds the pandas-only members (``.loc``/``.iloc``/``.select_dtypes``)."""

    @property
    def loc(self) -> Any:
        return self

    @property
    def iloc(self) -> Any:
        return self

    def select_dtypes(self, include: Any, exclude: Any = None) -> Any:
        return self


class _PolarsFrameDuck(_BaseFrameDuck):
    """Adds the polars-only members (``.with_columns``/``.filter``/``.to_polars``)."""

    def with_columns(self, *expressions: Any) -> Any:
        return self

    def filter(self, predicate: Any) -> Any:
        return self

    def to_polars(self) -> Any:
        return self


def test_pandas_frame_satisfies_pandas_backed_protocol_only() -> None:
    """A pandas-backed frame matches the pandas sub-protocol, not the polars one."""
    df = _PandasFrameDuck()

    assert isinstance(df, SkyulfDataFrame)
    assert isinstance(df, PandasBackedFrame)
    assert not isinstance(df, PolarsBackedFrame)


def test_polars_frame_satisfies_polars_backed_protocol_only() -> None:
    """A polars-backed frame matches the polars sub-protocol, not the pandas one."""
    df = _PolarsFrameDuck()

    assert isinstance(df, SkyulfDataFrame)
    assert isinstance(df, PolarsBackedFrame)
    assert not isinstance(df, PandasBackedFrame)
