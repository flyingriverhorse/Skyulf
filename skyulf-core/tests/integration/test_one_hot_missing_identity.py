"""Missing OneHot categories must remain distinct from user-supplied text."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import OneHotEncoder

from skyulf.preprocessing.encoding.one_hot import (
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
)

MISSING = "__mlops_missing__"
ESCAPE = "__mlops_literal__:"
ENGINE_PAIRS = [(fit, apply) for fit in ("pandas", "polars") for apply in ("pandas", "polars")]


def _frame(values, engine, categorical=False):
    """Preserve the caller's raw category values on either supported engine."""
    if engine == "pandas":
        return pd.DataFrame(
            {"color": pd.Series(values, dtype="category" if categorical else object)}
        )
    dtype = pl.Categorical if categorical else pl.String
    return pl.DataFrame({"color": pl.Series(values, dtype=dtype)})


@pytest.mark.parametrize("fit_engine,apply_engine", ENGINE_PAIRS)
@pytest.mark.parametrize("categorical", [False, True])
def test_literal_missing_and_escape_text_remain_distinct_after_pickle(
    fit_engine, apply_engine, categorical
):
    """Literal reserved text cannot share an indicator with missing data or another literal."""
    values = [None, MISSING, ESCAPE + MISSING, ESCAPE + ESCAPE + MISSING, "red"]
    train = _frame(values, fit_engine, categorical)
    artifact = OneHotEncoderCalculator().fit(
        train, {"columns": ["color"], "include_missing": True, "max_categories": None}
    )
    restored = pickle.loads(pickle.dumps(artifact))
    query = _frame(values, apply_engine, categorical)
    encoded = OneHotEncoderApplier().apply(query, restored)
    result = encoded.to_numpy()

    assert result.shape == (len(values), len(values))
    assert np.unique(result, axis=0).shape[0] == len(values)
    np.testing.assert_array_equal(result.sum(axis=1), np.ones(len(values)))
    assert "color_red" in encoded.columns
    assert "color_" + MISSING in encoded.columns
    assert artifact["missing_encoding_version"] == 1
    np.testing.assert_array_equal(result, OneHotEncoderApplier().apply(train, artifact).to_numpy())


@pytest.mark.parametrize("fit_engine,apply_engine", ENGINE_PAIRS)
@pytest.mark.parametrize("handle_unknown", ["ignore", "error"])
@pytest.mark.parametrize(
    "known,unknown", [(None, MISSING), (MISSING, None), (MISSING, ESCAPE + MISSING)]
)
def test_unseen_missing_or_literal_values_follow_unknown_policy(
    fit_engine, apply_engine, handle_unknown, known, unknown
):
    """A held-out reserved-looking value must not bypass the configured unknown-category policy."""
    artifact = OneHotEncoderCalculator().fit(
        _frame([known, "red"], fit_engine),
        {"columns": ["color"], "include_missing": True, "handle_unknown": handle_unknown},
    )
    artifact = pickle.loads(pickle.dumps(artifact))
    query = _frame([unknown], apply_engine)
    if handle_unknown == "error":
        with pytest.raises(ValueError, match="unknown categor"):
            OneHotEncoderApplier().apply(query, artifact)
    else:
        result = OneHotEncoderApplier().apply(query, artifact)
        np.testing.assert_array_equal(result.to_numpy(), np.zeros((1, 2)))


@pytest.mark.parametrize("fit_engine,apply_engine", ENGINE_PAIRS)
@pytest.mark.parametrize("include_missing", [False, True])
def test_ordinary_names_and_disabled_missing_encoding_stay_unchanged(
    fit_engine, apply_engine, include_missing
):
    """The policy must leave ordinary names and the explicit include_missing=False path intact."""
    values = ["red", None, "blue"] if include_missing else [MISSING, ESCAPE + MISSING, None]
    train = _frame(values, fit_engine)
    artifact = OneHotEncoderCalculator().fit(
        train, {"columns": ["color"], "include_missing": include_missing}
    )
    result = OneHotEncoderApplier().apply(_frame(values, apply_engine), artifact)
    expected_names = (
        ["color_" + MISSING, "color_blue", "color_red"]
        if include_missing
        else ["color_" + ESCAPE + MISSING, "color_" + MISSING, "color_None"]
    )

    assert list(result.columns) == expected_names
    assert np.unique(result.to_numpy(), axis=0).shape[0] == 3
    if not include_missing:
        assert "missing_encoding_version" not in artifact


@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_unversioned_artifact_retains_legacy_missing_replay(apply_engine):
    """Old fitted artifacts must replay their original mapping without a silent feature shift."""
    old_values = np.array([[MISSING], ["red"]], dtype=object)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int8, handle_unknown="error").fit(
        old_values
    )
    artifact = pickle.loads(
        pickle.dumps(
            {
                "type": "onehot",
                "columns": ["color"],
                "encoder_object": encoder,
                "feature_names": encoder.get_feature_names_out(["color"]).tolist(),
                "prefix_separator": "_",
                "drop_original": True,
                "include_missing": True,
            }
        )
    )
    result = OneHotEncoderApplier().apply(_frame([None, MISSING, "red"], apply_engine), artifact)

    np.testing.assert_array_equal(result.to_numpy(), [[1, 0], [1, 0], [0, 1]])


@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
@pytest.mark.parametrize("version", [None, True, 0, 2, "1"])
def test_unknown_missing_encoding_versions_are_rejected(apply_engine, version):
    """An unsupported artifact policy must fail instead of silently changing category identity."""
    frame = _frame(["red", None], apply_engine)
    artifact = OneHotEncoderCalculator().fit(frame, {"columns": ["color"], "include_missing": True})
    artifact["missing_encoding_version"] = version

    with pytest.raises(ValueError, match="missing encoding version"):
        OneHotEncoderApplier().apply(frame, artifact)


def test_polars_enum_literals_and_nulls_remain_distinct():
    """Escaping a fixed Enum must not turn new escape strings into nulls or invalid categories."""
    values = [MISSING, ESCAPE + MISSING, "red"]
    frame = pl.DataFrame({"color": pl.Series([*values, None], dtype=pl.Enum(values))})
    artifact = OneHotEncoderCalculator().fit(frame, {"columns": ["color"], "include_missing": True})
    result = OneHotEncoderApplier().apply(frame, pickle.loads(pickle.dumps(artifact)))

    assert result.shape == (4, 4)
    assert np.unique(result.to_numpy(), axis=0).shape[0] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_missing_policy_does_not_coerce_numeric_categories_without_missing_values(engine):
    """Literal escaping must preserve existing large integer category identity and names."""
    values = [2**53, 2**53 + 1]
    frame = (
        pd.DataFrame({"color": values}) if engine == "pandas" else pl.DataFrame({"color": values})
    )
    artifact = OneHotEncoderCalculator().fit(frame, {"columns": ["color"], "include_missing": True})
    result = OneHotEncoderApplier().apply(frame, artifact)

    assert list(result.columns) == [f"color_{value}" for value in values]
    np.testing.assert_array_equal(result.to_numpy(), np.eye(2))
