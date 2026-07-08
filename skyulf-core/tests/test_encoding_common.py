"""Tests for shared encoder helpers (skyulf.preprocessing.encoding._common)."""

import logging

import pandas as pd
import polars as pl
import pytest
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.encoding._common import (
    _exclude_target_column,
    _parse_categories_order,
    detect_categorical_columns,
)

_parse_categories_order_cases = TestCaseLoader(
    "preprocessing/encoding_common", group="parse_categories_order"
).load_with_ids()
_exclude_target_column_cases = TestCaseLoader(
    "preprocessing/encoding_common", group="exclude_target_column"
).load_with_ids()
_detect_categorical_columns_cases = TestCaseLoader(
    "preprocessing/encoding_common", group="detect_categorical_columns"
).load_with_ids()


class TestParseCategoriesOrder:
    """Scenarios loaded from
    ``tests/test_cases/preprocessing/encoding_common.json`` (group ``parse_categories_order``).
    """

    @pytest.mark.parametrize(
        _parse_categories_order_cases[0],
        _parse_categories_order_cases[1],
        ids=_parse_categories_order_cases[2],
    )
    def test_parse_categories_order(self, raw: object, n_cols: int, expected: object) -> None:
        assert _parse_categories_order(raw, n_cols) == expected


class TestExcludeTargetColumn:
    """Scenarios loaded from
    ``tests/test_cases/preprocessing/encoding_common.json`` (group ``exclude_target_column``).
    """

    @pytest.mark.parametrize(
        _exclude_target_column_cases[0],
        _exclude_target_column_cases[1],
        ids=_exclude_target_column_cases[2],
    )
    def test_exclude_target_column(
        self,
        columns: list[str],
        target_column: str | None,
        y_name: str | None,
        encoder_name: str,
        expected: list[str],
        expect_warning: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = {"target_column": target_column} if target_column else {}
        y = pd.Series([1, 0], name=y_name) if y_name else None
        with caplog.at_level(logging.WARNING):
            result = _exclude_target_column(columns, config, encoder_name, y=y)

        assert result == expected
        if expect_warning:
            assert any("Excluding target column" in rec.message for rec in caplog.records)


class TestDetectCategoricalColumns:
    """Scenarios loaded from
    ``tests/test_cases/preprocessing/encoding_common.json`` (group ``detect_categorical_columns``).
    """

    @pytest.mark.parametrize(
        _detect_categorical_columns_cases[0],
        _detect_categorical_columns_cases[1],
        ids=_detect_categorical_columns_cases[2],
    )
    def test_detect_categorical_columns(self, engine: str) -> None:
        data = {"city": ["a", "b"], "amount": [1, 2]}
        df = pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data)
        assert detect_categorical_columns(df) == ["city"]
