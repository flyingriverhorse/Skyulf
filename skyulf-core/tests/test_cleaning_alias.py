"""Unit tests for the AliasReplacement cleaning node.

Covers: alias-type resolution, custom-map normalisation, mapping selection,
single-value helper, Calculator.fit branches, Applier.apply (boolean /
country / custom), edge cases, and engine parity.
"""

from typing import Any, Dict

import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset

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

# ---------------------------------------------------------------------------
# _resolve_alias_type
# ---------------------------------------------------------------------------


def test_resolve_alias_type_boolean() -> None:
    """'boolean' alias_type must pass through unchanged."""
    assert _resolve_alias_type({"alias_type": "boolean"}) == "boolean"


def test_resolve_alias_type_country() -> None:
    """'country' alias_type must pass through unchanged."""
    assert _resolve_alias_type({"alias_type": "country"}) == "country"


def test_resolve_alias_type_custom() -> None:
    """'custom' alias_type must pass through unchanged."""
    assert _resolve_alias_type({"alias_type": "custom"}) == "custom"


def test_resolve_alias_type_normalize_boolean_remaps_to_boolean() -> None:
    """Legacy 'normalize_boolean' value must be remapped to 'boolean'."""
    assert _resolve_alias_type({"alias_type": "normalize_boolean"}) == "boolean"


def test_resolve_alias_type_canonicalize_country_codes_remaps() -> None:
    """Legacy 'canonicalize_country_codes' must be remapped to 'country'."""
    assert _resolve_alias_type({"alias_type": "canonicalize_country_codes"}) == "country"


def test_resolve_alias_type_falls_back_to_mode() -> None:
    """When alias_type is missing the 'mode' key is used as fallback."""
    assert _resolve_alias_type({"mode": "boolean"}) == "boolean"


def test_resolve_alias_type_defaults_to_boolean() -> None:
    """When neither alias_type nor mode is provided the default is 'boolean'."""
    assert _resolve_alias_type({}) == "boolean"


# ---------------------------------------------------------------------------
# _normalize_alias_custom_map
# ---------------------------------------------------------------------------


def test_normalize_custom_map_lowercases_string_keys() -> None:
    """String keys must be lowercased during normalisation."""
    result = _normalize_alias_custom_map({"YES": "Yes", "NO": "No"})
    assert "yes" in result
    assert "no" in result


def test_normalize_custom_map_strips_punctuation_from_keys() -> None:
    """Punctuation must be stripped from string keys."""
    result = _normalize_alias_custom_map({"Yes!": "Yes"})
    assert "yes" in result


def test_normalize_custom_map_strips_spaces_from_keys() -> None:
    """Spaces must be removed from string keys after lowercasing."""
    result = _normalize_alias_custom_map({"United States": "USA"})
    assert "unitedstates" in result


def test_normalize_custom_map_non_string_keys_unchanged() -> None:
    """Non-string keys (e.g. integers) must be preserved as-is."""
    result = _normalize_alias_custom_map({1: "one", 2: "two"})
    assert result[1] == "one"


def test_normalize_custom_map_empty_returns_empty() -> None:
    """An empty input map must return an empty dict."""
    assert _normalize_alias_custom_map({}) == {}


# ---------------------------------------------------------------------------
# _resolve_alias_mapping
# ---------------------------------------------------------------------------


def test_resolve_mapping_boolean_returns_common_aliases() -> None:
    """boolean alias type must return the canonical COMMON_BOOLEAN_ALIASES dict."""
    result = _resolve_alias_mapping("boolean", {})
    assert result is COMMON_BOOLEAN_ALIASES


def test_resolve_mapping_country_returns_country_map() -> None:
    """country alias type must return the canonical COUNTRY_ALIAS_MAP dict."""
    result = _resolve_alias_mapping("country", {})
    assert result is COUNTRY_ALIAS_MAP


def test_resolve_mapping_custom_returns_provided_map() -> None:
    """custom alias type must return the caller-supplied map."""
    custom = {"foo": "bar"}
    result = _resolve_alias_mapping("custom", custom)
    assert result is custom


def test_resolve_mapping_unknown_returns_empty() -> None:
    """An unrecognised alias type must yield an empty dict."""
    result = _resolve_alias_mapping("does_not_exist", {})
    assert result == {}


# ---------------------------------------------------------------------------
# _normalize_alias_pandas
# ---------------------------------------------------------------------------


def test_normalize_alias_pandas_matches_key() -> None:
    """A known alias must be resolved to its canonical value."""
    mapping = COMMON_BOOLEAN_ALIASES
    # "Yes " → clean → "yes" → maps to "Yes"
    assert _normalize_alias_pandas("yes", mapping) == "Yes"


def test_normalize_alias_pandas_unknown_value_preserved() -> None:
    """A value with no matching alias must be returned unchanged."""
    assert _normalize_alias_pandas("maybe", COMMON_BOOLEAN_ALIASES) == "maybe"


def test_normalize_alias_pandas_non_string_passthrough() -> None:
    """Non-string values (int, float, None) must pass through without error."""
    assert _normalize_alias_pandas(42, COMMON_BOOLEAN_ALIASES) == 42
    assert _normalize_alias_pandas(None, COMMON_BOOLEAN_ALIASES) is None


def test_normalize_alias_pandas_strips_punctuation_and_spaces() -> None:
    """Punctuation and spaces in the value must be stripped before lookup."""
    # "Y.e.s" → lower → "y.e.s" → strip punct → "yes" → maps to "Yes"
    assert _normalize_alias_pandas("Y.e.s", COMMON_BOOLEAN_ALIASES) == "Yes"


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


def test_applier_boolean_normalises_yes_variants() -> None:
    """Boolean applier must map 'yes', 'Y', 'TRUE', '1', etc. to 'Yes'."""
    df = pd.DataFrame({"flag": ["yes", "Y", "TRUE", "1", "on", "True"]})
    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()
    params = calc.fit(df, {"columns": ["flag"], "alias_type": "boolean"})
    result = applier.apply(df, params)
    assert (result["flag"] == "Yes").all()


def test_applier_boolean_normalises_no_variants() -> None:
    """Boolean applier must map 'no', 'N', 'FALSE', '0', 'off', etc. to 'No'."""
    df = pd.DataFrame({"flag": ["no", "N", "FALSE", "0", "off", "f"]})
    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()
    params = calc.fit(df, {"columns": ["flag"], "alias_type": "boolean"})
    result = applier.apply(df, params)
    assert (result["flag"] == "No").all()


def test_applier_unknown_values_preserved() -> None:
    """Values that don't match any alias must be returned as-is."""
    df = pd.DataFrame({"flag": ["maybe", "perhaps", "dunno"]})
    params: Dict[str, Any] = {
        "columns": ["flag"],
        "alias_type": "boolean",
        "custom_map": {},
    }
    result = AliasReplacementApplier().apply(df, params)
    assert list(result["flag"]) == ["maybe", "perhaps", "dunno"]


def test_applier_country_normalises_uk_variants() -> None:
    """Country applier must map 'UK', 'England', etc. to 'United Kingdom'."""
    df = pd.DataFrame({"country": ["UK", "England", "United Kingdom", "uk"]})
    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()
    params = calc.fit(df, {"columns": ["country"], "alias_type": "country"})
    result = applier.apply(df, params)
    assert (result["country"] == "United Kingdom").all()


def test_applier_country_normalises_usa_variants() -> None:
    """Country applier must map 'USA', 'US', 'United States', etc. consistently."""
    df = pd.DataFrame({"country": ["USA", "US", "America", "States"]})
    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()
    params = calc.fit(df, {"columns": ["country"], "alias_type": "country"})
    result = applier.apply(df, params)
    assert (result["country"] == "USA").all()


def test_applier_custom_mapping() -> None:
    """Custom map must substitute only the mapped values."""
    df = pd.DataFrame({"status": ["Active", "Inactive", "Pending"]})
    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()
    params = calc.fit(
        df,
        {
            "columns": ["status"],
            "alias_type": "custom",
            "custom_map": {"Active": "active", "Inactive": "inactive"},
        },
    )
    result = applier.apply(df, params)
    assert result["status"].iloc[0] == "active"
    assert result["status"].iloc[1] == "inactive"
    # "Pending" has no mapping → preserved
    assert result["status"].iloc[2] == "Pending"


def test_applier_mixed_case_and_punctuation_normalised() -> None:
    """Values with mixed case and punctuation must still be matched after cleaning."""
    df = pd.DataFrame({"flag": ["Y.e.s", "N-O", "T.R.U.E."]})
    params: Dict[str, Any] = {
        "columns": ["flag"],
        "alias_type": "boolean",
        "custom_map": {},
    }
    result = AliasReplacementApplier().apply(df, params)
    assert result["flag"].iloc[0] == "Yes"
    assert result["flag"].iloc[1] == "No"
    assert result["flag"].iloc[2] == "Yes"


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
