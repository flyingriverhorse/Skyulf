"""Unit tests for the AliasReplacement cleaning node.

Covers: alias-type resolution, custom-map normalisation, mapping selection,
single-value helper, Calculator.fit branches, Applier.apply (boolean /
country / custom), edge cases, and engine parity.
"""

from typing import Any, Dict, Optional

import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.cleaning._common import (
    ALIAS_PUNCTUATION_TABLE,
    COMMON_BOOLEAN_ALIASES,
    COUNTRY_ALIAS_MAP,
)
from skyulf.preprocessing.cleaning.alias import (
    AliasReplacementApplier,
    AliasReplacementCalculator,
    _normalize_alias_custom_map,
    _normalize_alias_pandas,
    _resolve_alias_mapping,
    _resolve_alias_type,
)

_alias_type_resolution_cases = TestCaseLoader(
    "preprocessing/cleaning_alias", group="type_resolution"
).load()
_normalize_custom_map_cases = TestCaseLoader(
    "preprocessing/cleaning_alias", group="normalize_custom_map"
).load()
_resolve_mapping_cases = TestCaseLoader(
    "preprocessing/cleaning_alias", group="resolve_mapping"
).load()
_normalize_pandas_value_cases = TestCaseLoader(
    "preprocessing/cleaning_alias", group="normalize_pandas_value"
).load()
_applier_uniform_cases = TestCaseLoader(
    "preprocessing/cleaning_alias", group="applier_uniform_normalisation"
).load()
_applier_value_list_cases = TestCaseLoader(
    "preprocessing/cleaning_alias", group="applier_value_lists"
).load()

# ---------------------------------------------------------------------------
# _resolve_alias_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_alias_type_resolution_cases)
def test_resolve_alias_type(config: Dict[str, Any], expected: str) -> None:
    """_resolve_alias_type must resolve legacy aliases and defaults correctly."""
    assert _resolve_alias_type(config) == expected


# ---------------------------------------------------------------------------
# _normalize_alias_custom_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_normalize_custom_map_cases)
def test_normalize_alias_custom_map(input_map: Dict[str, str], expected: Dict[str, str]) -> None:
    """_normalize_alias_custom_map must lowercase and strip punctuation from string keys."""
    assert _normalize_alias_custom_map(input_map) == expected


def test_normalize_custom_map_non_string_keys_unchanged() -> None:
    """Non-string keys (e.g. integers) must be preserved as-is."""
    result = _normalize_alias_custom_map({1: "one", 2: "two"})
    assert result[1] == "one"


# ---------------------------------------------------------------------------
# _resolve_alias_mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_resolve_mapping_cases)
def test_resolve_alias_mapping(
    alias_type: str, custom_map: Dict[str, str], expected_kind: str
) -> None:
    """_resolve_alias_mapping must select the canonical/custom map per alias_type."""
    result = _resolve_alias_mapping(alias_type, custom_map)
    expected = {
        "common_boolean_aliases": COMMON_BOOLEAN_ALIASES,
        "country_alias_map": COUNTRY_ALIAS_MAP,
        "custom_map": custom_map,
        "empty": {},
    }[expected_kind]
    assert result == expected


# ---------------------------------------------------------------------------
# _normalize_alias_pandas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_normalize_pandas_value_cases)
def test_normalize_alias_pandas(value: Any, expected: Any) -> None:
    """_normalize_alias_pandas must clean and resolve values against the mapping."""
    assert _normalize_alias_pandas(value, COMMON_BOOLEAN_ALIASES) == expected


# ---------------------------------------------------------------------------
# Calculator.fit
# ---------------------------------------------------------------------------


def test_calculator_fit_boolean_type() -> None:
    """fit must store alias_type='boolean' and columns in the artifact."""
    df = pd.DataFrame({"flag": ["yes", "no", "true"]})
    params = AliasReplacementCalculator().fit(df, {"columns": ["flag"], "alias_type": "boolean"})
    assert params["alias_type"] == "boolean"
    assert "flag" in params["columns"]


def test_calculator_fit_country_type() -> None:
    """fit must resolve and store alias_type='country'."""
    df = pd.DataFrame({"country": ["uk", "usa"]})
    params = AliasReplacementCalculator().fit(df, {"columns": ["country"], "alias_type": "country"})
    assert params["alias_type"] == "country"


def test_calculator_fit_custom_type_normalises_map() -> None:
    """fit must normalise custom_map keys (lowercase, no punctuation)."""
    df = pd.DataFrame({"label": ["Active", "Inactive"]})
    params = AliasReplacementCalculator().fit(
        df,
        {
            "columns": ["label"],
            "alias_type": "custom",
            "custom_map": {"Active": "active", "Inactive": "inactive"},
        },
    )
    assert params["alias_type"] == "custom"
    # Keys must be normalised
    custom_map = params["custom_map"]
    assert custom_map is not None
    assert "active" in custom_map


def test_calculator_fit_empty_columns_short_circuits() -> None:
    """Explicit empty columns list must short-circuit to {}."""
    df = pd.DataFrame({"flag": ["yes"]})
    params = AliasReplacementCalculator().fit(df, {"columns": []})
    assert params == {}


def test_calculator_fit_legacy_mode_key() -> None:
    """The legacy 'mode' config key must be accepted and resolved correctly."""
    df = pd.DataFrame({"flag": ["yes"]})
    params = AliasReplacementCalculator().fit(
        df, {"columns": ["flag"], "mode": "normalize_boolean"}
    )
    assert params["alias_type"] == "boolean"


def test_calculator_fit_custom_pairs_alias() -> None:
    """'custom_pairs' must be accepted as an alias for 'custom_map'."""
    df = pd.DataFrame({"label": ["Active"]})
    params = AliasReplacementCalculator().fit(
        df,
        {
            "columns": ["label"],
            "alias_type": "custom",
            "custom_pairs": {"Active": "active"},
        },
    )
    custom_map = params["custom_map"]
    assert custom_map is not None
    assert "active" in custom_map


# ---------------------------------------------------------------------------
# Applier.apply — pandas path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_applier_uniform_cases)
def test_applier_uniform_normalisation(
    col_name: str, values: list, alias_type: str, expected_value: str
) -> None:
    """Applier must normalise every equivalent alias value to the canonical form."""
    df = pd.DataFrame({col_name: values})
    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()
    params = calc.fit(df, {"columns": [col_name], "alias_type": alias_type})
    result = applier.apply(df, params)
    assert (result[col_name] == expected_value).all()


@pytest.mark.parametrize(*_applier_value_list_cases)
def test_applier_value_lists(
    values: list, alias_type: str, custom_map: Dict[str, str], expected: list
) -> None:
    """Applier must map/preserve each value in the list per the given alias config."""
    df = pd.DataFrame({"flag": values})
    params: Dict[str, Any] = {
        "columns": ["flag"],
        "alias_type": alias_type,
        "custom_map": custom_map,
    }
    result = AliasReplacementApplier().apply(df, params)
    assert list(result["flag"]) == expected


def test_applier_empty_dataframe() -> None:
    """Applying to an empty DataFrame must return an empty DataFrame."""
    df = pd.DataFrame({"flag": pd.Series([], dtype=str)})
    params: Dict[str, Any] = {
        "columns": ["flag"],
        "alias_type": "boolean",
        "custom_map": {},
    }
    result = AliasReplacementApplier().apply(df, params)
    assert result.shape == (0, 1)


def test_applier_no_valid_columns_is_noop() -> None:
    """Columns not present in the DataFrame must be silently skipped."""
    df = pd.DataFrame({"flag": ["yes"]})
    params: Dict[str, Any] = {
        "columns": ["nonexistent"],
        "alias_type": "boolean",
        "custom_map": {},
    }
    result = AliasReplacementApplier().apply(df, params)
    # Original column untouched
    assert result["flag"].iloc[0] == "yes"


def test_applier_empty_params_is_noop() -> None:
    """Empty params dict must return the DataFrame unchanged."""
    df = pd.DataFrame({"flag": ["yes"]})
    result = AliasReplacementApplier().apply(df, {})
    assert result["flag"].iloc[0] == "yes"


def test_polars_applier_no_valid_columns_is_noop() -> None:
    """Polars alias applier must short-circuit when no valid column is found."""
    df = pl.DataFrame({"flag": ["yes", "no"]})
    params: Dict[str, Any] = {
        "columns": ["nonexistent"],
        "alias_type": "boolean",
        "custom_map": {},
    }
    result = AliasReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert list(result["flag"]) == ["yes", "no"]


def test_infer_output_schema_returns_input_schema_unchanged() -> None:
    """AliasReplacement infer_output_schema must pass the schema through unchanged."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["flag"], {"flag": "string"})
    result = AliasReplacementCalculator().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# Engine parity (pandas vs polars Applier paths)
# ---------------------------------------------------------------------------


@st.composite
def _boolean_frame(draw: st.DrawFn, min_rows: int = 4, max_rows: int = 20) -> pd.DataFrame:
    """Generate a DataFrame with boolean-like string values."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    # Restricted to unambiguous aliases so parity differences can't hide
    values = draw(
        st.lists(
            st.sampled_from(["yes", "no", "true", "false", "1", "0", "maybe"]),
            min_size=n,
            max_size=n,
        )
    )
    return pd.DataFrame({"flag": values})


@settings(max_examples=25, deadline=None)
@given(df=_boolean_frame())
def test_alias_apply_engine_parity_boolean(df: pd.DataFrame) -> None:
    """Boolean alias replacement must produce identical results on pandas and polars."""
    params: Dict[str, Any] = {
        "columns": ["flag"],
        "alias_type": "boolean",
        "custom_map": {},
    }
    applier = AliasReplacementApplier()
    pd_result = applier.apply(df, params)
    pl_result = applier.apply(pl.from_pandas(df), params)
    if hasattr(pl_result, "to_pandas"):
        pl_result = pl_result.to_pandas()
    pd.testing.assert_frame_equal(
        pd_result.reset_index(drop=True),
        pl_result.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Real-shaped dataset integration
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has a ``city`` column with a missing value — verifying NaN passthrough
    in the alias replacement path when no matching alias is found.
    """

    def test_boolean_alias_on_city_with_nan_passes_through_unchanged(self) -> None:
        """Applying boolean alias replacement to a column that contains NaN and
        non-matching values must leave those values unchanged (NaN stays NaN,
        unrecognised strings are returned as-is).
        """
        df = load_sample_dataset("customers")
        calc = AliasReplacementCalculator()
        applier = AliasReplacementApplier()
        params = calc.fit(df, {"columns": ["city"], "alias_type": "boolean"})
        result = applier.apply(df, params)

        # City values like "New York", "Chicago" do not match any boolean alias
        # so they must be preserved verbatim.
        original_non_null = df.loc[df["city"].notna(), "city"]
        result_non_null = result.loc[df["city"].notna(), "city"]
        assert list(original_non_null) == list(result_non_null)
        # The NaN row must remain NaN after alias replacement.
        assert result.loc[df["city"].isna(), "city"].isna().all()
