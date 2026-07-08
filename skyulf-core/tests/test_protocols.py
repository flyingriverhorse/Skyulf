"""Tests for the structural protocols in `skyulf.core.protocols`."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from skyulf.core.protocols import ApplierProtocol, CalculatorProtocol, PipelineStep
from skyulf.engines import SkyulfDataFrame
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
