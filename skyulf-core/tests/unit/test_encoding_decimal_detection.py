"""Automatic categorical selection must not consume decimal numeric features."""

from decimal import Decimal

import pandas as pd
import polars as pl

from skyulf.preprocessing.encoding._common import detect_categorical_columns


def test_decimal_auto_detection_matches_native_polars():
    """Decimal object storage is numeric and must remain available to numeric preprocessing."""
    data = {"amount": [Decimal("1.20"), None, Decimal("2.35")], "category": ["a", None, "b"]}
    assert detect_categorical_columns(pd.DataFrame(data)) == ["category"]
    assert detect_categorical_columns(pl.DataFrame(data)) == ["category"]
