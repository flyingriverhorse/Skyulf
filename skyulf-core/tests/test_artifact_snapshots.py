"""Snapshot tests for fitted artifacts (syrupy).

`test_engine_parity.py` builds expectations by hand; these snapshots make any
change to an artifact's *shape* visible in PR review. Floats are rounded so the
snapshots stay stable across platforms — we are guarding structure, not the
last ULP (engine parity already guards numeric equality).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from skyulf.preprocessing.encoding import WOEEncoderCalculator
from skyulf.preprocessing.scaling import (
    MinMaxScalerCalculator,
    StandardScalerCalculator,
)


def _round(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats so snapshots are platform-stable."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, ndigits) for v in obj]
    return obj


def _fixed_numeric_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})


def _fixed_categorical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["x", "y", "x", "z", "y", "x", "z", "y"],
            "target": [1, 0, 1, 0, 1, 1, 0, 0],
        }
    )


def test_standard_scaler_artifact_snapshot(snapshot):
    params = StandardScalerCalculator().fit(_fixed_numeric_frame(), {"columns": ["a", "b"]})
    assert _round(params) == snapshot


def test_min_max_scaler_artifact_snapshot(snapshot):
    params = MinMaxScalerCalculator().fit(_fixed_numeric_frame(), {"columns": ["a", "b"]})
    assert _round(params) == snapshot


def test_woe_encoder_artifact_snapshot(snapshot):
    df = _fixed_categorical_frame()
    params = WOEEncoderCalculator().fit((df[["city"]], df["target"]), {"columns": ["city"]})
    assert _round(params) == snapshot
