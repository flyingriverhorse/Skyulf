"""Preserve literal text without losing genuine missing-value cleanup (OC-158)."""

import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from backend.data_ingestion.serialization import AsyncJSONSafeSerializer, JSONSafeSerializer

_LITERAL_VALUES = [
    "nan",
    "NaN",
    "NaT",
    "<NA>",
    "inf",
    "-inf",
    "infinity",
    "-infinity",
    "Nan",
    "NAN",
    "Inf",
    "nano",
    "Nancy",
    "naan",
    "",
]


@pytest.mark.parametrize("value", _LITERAL_VALUES)
async def test_literal_missing_tokens_survive_both_serializers(value):
    """Category labels must survive regardless of the chosen serializer."""
    payload = {value: [value, {"label": value}]}
    sync_result = JSONSafeSerializer.clean_for_json(payload)
    async_result = await AsyncJSONSafeSerializer.clean_for_json(payload)
    assert json.loads(json.dumps(sync_result, allow_nan=False)) == payload
    assert sync_result == async_result


@pytest.mark.parametrize("records_format", [True, False])
async def test_dataframe_text_survives_both_serializers(records_format):
    """Exporting a text column must not merge literal categories into nulls."""
    values = [*_LITERAL_VALUES, None]
    frame = pd.DataFrame({"label": pd.Series(values, dtype=object)})
    expected = [{"label": value} for value in values] if records_format else {"label": values}
    sync_result = JSONSafeSerializer.safe_dict_from_dataframe(frame, records_format)
    async_result = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(frame, records_format)
    assert json.loads(json.dumps(sync_result, allow_nan=False)) == expected
    assert sync_result == async_result


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.NA,
        pd.NaT,
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32("nan"),
        np.float32("inf"),
        np.float64("-inf"),
        Decimal("NaN"),
        np.datetime64("NaT"),
    ],
)
def test_sync_real_missing_values_remain_json_null(value):
    """Removing text matching must preserve cleanup of actual missing scalars."""
    result = JSONSafeSerializer.clean_for_json({"value": value})
    assert json.loads(json.dumps(result, allow_nan=False)) == {"value": None}


def test_sync_fallback_text_is_not_interpreted_as_missing():
    """An object's textual representation must not silently erase its value."""

    class Label:
        def __str__(self):
            """Represent a category that happens to resemble a missing token."""
            return "nan"

    assert JSONSafeSerializer.clean_for_json(Label()) == "nan"
