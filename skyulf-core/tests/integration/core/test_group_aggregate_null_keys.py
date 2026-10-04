"""Pin missing-key parity at the public fitted feature-generation boundary."""

from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.engines import SkyulfPolarsWrapper
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)

_FRAME_KINDS = ["pandas", "polars", "pandas_wrapper", "polars_wrapper"]


def _frame(data: dict[str, list[Any]], frame_kind: str) -> Any:
    """Keep native Polars null and IEEE NaN keys distinct in the input fixture."""
    if frame_kind == "pandas":
        return pd.DataFrame(data)
    if frame_kind == "pandas_wrapper":
        return SkyulfPandasWrapper(pd.DataFrame(data))
    frame = pl.DataFrame(data)
    return SkyulfPolarsWrapper(frame) if frame_kind == "polars_wrapper" else frame


def _config(method: str) -> dict[str, Any]:
    """Use the single-group-column payload produced by the Canvas editor."""
    return {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": method,
                "input_columns": ["group"],
                "secondary_columns": ["value"],
                "output_column": "group_value",
            }
        ]
    }


@pytest.mark.parametrize("frame_kind", _FRAME_KINDS)
@pytest.mark.parametrize(
    ("method", "missing_group", "known_group"),
    [
        ("mean", 6.0, 20.0),
        ("sum", 12.0, 40.0),
        ("count", 2.0, 2.0),
        ("min", 4.0, 10.0),
        ("max", 8.0, 30.0),
        ("std", 2.8284271247461903, 14.142135623730951),
        ("median", 6.0, 20.0),
    ],
)
def test_null_and_nan_keys_share_training_aggregate(
    frame_kind: str, method: str, missing_group: float, known_group: float
) -> None:
    """Mixed null/NaN keys must share one learned group for training and held-out rows."""
    train = _frame(
        {
            "group": [1.0, None, float("nan"), None, 1.0],
            "value": [10.0, 4.0, 8.0, None, 30.0],
        },
        frame_kind,
    )
    params = FeatureGenerationCalculator().fit(train, _config(method))
    heldout = _frame(
        {
            "group": [float("nan"), None, 1.0, 9.0],
            "value": [10000.0, -10000.0, 999.0, 500.0],
        },
        frame_kind,
    )

    train_out = FeatureGenerationApplier().apply(train, params)
    heldout_out = FeatureGenerationApplier().apply(heldout, params)

    assert train_out["group_value"].to_list() == pytest.approx(
        [known_group, missing_group, missing_group, missing_group, known_group]
    )
    values = heldout_out["group_value"].to_list()
    assert values[:3] == pytest.approx([missing_group, missing_group, known_group])
    assert pd.isna(values[3])


@pytest.mark.parametrize("frame_kind", _FRAME_KINDS)
@pytest.mark.parametrize(("method", "known_group"), [("mean", 20.0), ("count", 2.0)])
def test_unseen_missing_keys_do_not_learn_from_heldout_rows(
    frame_kind: str, method: str, known_group: float
) -> None:
    """Missing keys absent from training must remain missing even with held-out values."""
    train = _frame({"group": [1.0, 1.0], "value": [10.0, 30.0]}, frame_kind)
    params = FeatureGenerationCalculator().fit(train, _config(method))
    heldout = _frame(
        {
            "group": [None, float("nan"), 9.0, 1.0],
            "value": [100.0, 200.0, 300.0, 400.0],
        },
        frame_kind,
    )

    out = FeatureGenerationApplier().apply(heldout, params)

    values = out["group_value"].to_list()
    assert all(pd.isna(value) for value in values[:3])
    assert values[3] == known_group


@pytest.mark.parametrize("frame_kind", _FRAME_KINDS)
def test_unfitted_public_aggregate_cannot_reach_legacy_batch_helpers(frame_kind: str) -> None:
    """Reject legacy artifacts before divergent batch helpers can aggregate inference rows."""
    frame = _frame({"group": [1.0, None], "value": [10.0, 20.0]}, frame_kind)

    with pytest.raises(ValueError, match="requires fitted statistics"):
        FeatureGenerationApplier().apply(frame, _config("mean"))
