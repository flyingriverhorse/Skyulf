"""Behavioral regressions independently confirmed from the 0.9.1 review."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_missing_percentage_keeps_exact_threshold_boundary(engine):
    """Exactly seventy percent missing must satisfy an inclusive seventy percent limit."""
    frame = pd.DataFrame([[1.0] * 3 + [np.nan] * 7], columns=list("abcdefghij"))
    if engine == "polars":
        frame = pl.from_pandas(frame)
    engineer = FeatureEngineer(
        [{"name": "drop", "transformer": "DropMissingRows", "params": {"missing_threshold": 70}}]
    )

    result, _ = engineer.fit_transform(frame)

    assert len(result) == 1


@pytest.mark.parametrize("node", ["MissingIndicator", "DropMissingColumns"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_automatic_missing_column_selection_excludes_target(node, engine):
    """A declared target must not be deleted or used to generate a predictor automatically."""
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0], "target": [1.0, np.nan, 0.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    params = {"missing_threshold": 30} if node == "DropMissingColumns" else {}
    engineer = FeatureEngineer([{"name": "missing", "transformer": node, "params": params}])

    result, _ = engineer.fit_transform(frame, target_column="target")

    assert "target" in result.columns
    assert "target_missing" not in result.columns


@pytest.mark.parametrize("node", ["IQR", "ZScore"])
def test_polars_outliers_preserve_rows_when_fitted_columns_are_absent(node):
    """A scalar true mask must not reduce the target to one row when no bounds apply."""
    engineer = FeatureEngineer(
        [{"name": "outliers", "transformer": node, "params": {"columns": ["x"]}}]
    )
    engineer.fit_transform(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    features = pl.DataFrame({"other": [4.0, 5.0, 6.0]})
    target = pl.Series("target", [0, 1, 0])

    result, labels = engineer.transform((features, target))

    assert result.equals(features)
    assert labels.equals(target)


@pytest.mark.parametrize("node", ["IQR", "ZScore"])
def test_polars_outliers_coerce_numeric_strings_like_pandas(node):
    """Explicit numeric-string columns must use the same coercion during fit and replay."""
    frame = pd.DataFrame({"x": ["1", "2", "3", "N/A"]})
    config = [{"name": "outliers", "transformer": node, "params": {"columns": ["x"]}}]
    expected, _ = FeatureEngineer(config).fit_transform(frame)

    result, _ = FeatureEngineer(config).fit_transform(pl.from_pandas(frame))

    assert result["x"].to_list() == expected["x"].to_list()


@pytest.mark.parametrize("rule", ["negative", "zero", "custom_range"])
def test_polars_invalid_replacement_treats_string_as_literal(rule):
    """A replacement equal to another column's name must not copy that column's values."""
    frame = pl.DataFrame({"x": [-1.0, 0.0, 2.0], "other": [100.0, 200.0, 300.0]})
    engineer = FeatureEngineer(
        [
            {
                "name": "replace",
                "transformer": "InvalidValueReplacement",
                "params": {"columns": ["x"], "rule": rule, "replacement": "other", "min_value": 1},
            }
        ]
    )

    result, _ = engineer.fit_transform(frame)

    replaced_row = 1 if rule == "zero" else 0
    assert result["x"][replaced_row] == "other"


@pytest.mark.parametrize("method", ["add", "multiply", "ratio"])
def test_polars_generated_numeric_features_coerce_dirty_strings_like_pandas(method):
    """One unparsable numeric string must not silently remove the entire derived column."""
    frame = pd.DataFrame({"x": ["2", "N/A", "4"], "y": ["1", "2", "2"]})
    config = [
        {
            "name": "derive",
            "transformer": "FeatureGeneration",
            "params": {
                "operations": [
                    {
                        "method": method,
                        "operation_type": "ratio" if method == "ratio" else "arithmetic",
                        "input_columns": ["x"],
                        "secondary_columns": ["y"],
                        "output_column": "derived",
                    }
                ]
            },
        }
    ]
    expected, _ = FeatureEngineer(config).fit_transform(frame)

    actual, _ = FeatureEngineer(config).fit_transform(pl.from_pandas(frame))

    assert "derived" in actual.columns
    np.testing.assert_allclose(actual["derived"].to_numpy(), expected["derived"].to_numpy())


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node", ["KNNImputer", "IterativeImputer", "OrdinalEncoder", "TargetEncoder"]
)
@pytest.mark.parametrize("classes", [2, 3])
def test_empty_partition_preserves_transformed_schema(engine, node, classes):
    """Final refits with an empty held-out partition must retain the fitted output schema."""
    frame = pd.DataFrame({"x": np.tile([1, 2, 3], 10), "untouched": np.arange(30)})
    target = pd.Series(np.arange(30) % classes, name="target")
    if engine == "polars":
        frame, target = pl.from_pandas(frame), pl.from_pandas(target)
    engineer = FeatureEngineer(
        [{"name": "tested", "transformer": node, "params": {"columns": ["x"]}}]
    )
    engineer.fit_transform((frame, target))
    full = engineer.transform(frame)

    empty = engineer.transform(frame.head(0))

    assert len(empty) == 0
    assert list(empty.columns) == list(full.columns)
    assert list(empty.dtypes) == list(full.dtypes)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_zscore_fits_finite_values_without_poisoning_other_rows(engine):
    """One infinite training value must not cause every finite observation to be removed."""
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, np.inf, -np.inf, np.nan]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    engineer = FeatureEngineer(
        [{"name": "outliers", "transformer": "ZScore", "params": {"columns": ["x"]}}]
    )

    result, _ = engineer.fit_transform(frame)

    np.testing.assert_allclose(result["x"].to_numpy(), [0.0, 1.0, 2.0, np.nan], equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ordinal_error_policy_fits_known_categories_and_rejects_unknown(engine):
    """The documented strict unknown-category policy must allow fitting valid training data."""
    frame = pd.DataFrame({"x": ["a", "b", "a"]})
    unknown = pd.DataFrame({"x": ["c"]})
    if engine == "polars":
        frame, unknown = pl.from_pandas(frame), pl.from_pandas(unknown)
    engineer = FeatureEngineer(
        [
            {
                "name": "encode",
                "transformer": "OrdinalEncoder",
                "params": {"columns": ["x"], "handle_unknown": "error"},
            }
        ]
    )

    result, _ = engineer.fit_transform(frame)

    np.testing.assert_array_equal(result["x"].to_numpy(), [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="unknown categor"):
        engineer.transform(unknown)


@pytest.mark.parametrize(
    "dtype,values",
    [("Int64", [2, 2, None]), ("boolean", [True, True, None]), ("string", ["red", "red", None])],
)
def test_most_frequent_imputer_preserves_nullable_dtype(dtype, values):
    """Mode filling must retain existing nullable values and their semantic dtype."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    engineer = FeatureEngineer(
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["x"], "strategy": "most_frequent"},
            }
        ]
    )

    actual, _ = engineer.fit_transform(frame)

    pd.testing.assert_series_equal(actual["x"], pd.Series([values[0]] * 3, dtype=dtype, name="x"))


def test_constant_imputer_adds_missing_category():
    """A valid string constant must fill categorical nulls even outside fitted vocabulary."""
    frame = pd.DataFrame({"x": pd.Categorical(["red", None, "blue"])})
    engineer = FeatureEngineer(
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["x"], "strategy": "constant", "fill_value": "missing"},
            }
        ]
    )

    actual, _ = engineer.fit_transform(frame)

    assert actual["x"].tolist() == ["red", "missing", "blue"]
    assert isinstance(actual["x"].dtype, pd.CategoricalDtype)


def test_all_missing_categorical_mode_remains_missing():
    """An empty mode must leave nulls intact without adding NaN as a category."""
    frame = pd.DataFrame({"x": pd.Categorical([None, None], categories=["red"])})
    engineer = FeatureEngineer(
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["x"], "strategy": "most_frequent"},
            }
        ]
    )

    actual, _ = engineer.fit_transform(frame)

    pd.testing.assert_frame_equal(actual, frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_clip_values_preserves_large_integer_bounds(engine):
    """Integer clipping must not round values or bounds through a float64 intermediate."""
    bound = 2**53 + 1
    frame = pd.DataFrame({"x": [bound - 1, bound, bound + 2]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    engineer = FeatureEngineer(
        [
            {
                "name": "clip",
                "transformer": "ClipValues",
                "params": {"bounds": {"x": {"upper": bound}}},
            }
        ]
    )

    actual, _ = engineer.fit_transform(frame)

    assert actual["x"].to_list() == [bound - 1, bound, bound]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ordinal_keeps_explicit_order_when_another_column_is_absent(engine):
    """Filtering unavailable columns must keep each surviving column's configured order."""
    frame = pd.DataFrame({"x": ["small", "large", "medium"]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    engineer = FeatureEngineer(
        [
            {
                "name": "encode",
                "transformer": "OrdinalEncoder",
                "params": {
                    "columns": ["absent", "x"],
                    "categories_order": ["one,two", "small,medium,large"],
                },
            }
        ]
    )

    actual, _ = engineer.fit_transform(frame)

    assert actual["x"].to_list() == [0.0, 2.0, 1.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["SimpleTransformation", "GeneralTransformation"])
@pytest.mark.parametrize("dtype,values", [("int8", [20, 30]), ("int32", [50000, 60000])])
def test_square_avoids_integer_overflow(engine, node, dtype, values):
    """Squaring valid integers must not wrap into negative or unrelated feature values."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    engineer = FeatureEngineer(
        [
            {
                "name": "square",
                "transformer": node,
                "params": {"transformations": [{"column": "x", "method": "square"}]},
            }
        ]
    )

    actual, _ = engineer.fit_transform(frame)

    assert actual["x"].to_list() == [value**2 for value in values]


@pytest.mark.parametrize(
    "dtype,bounds",
    [
        ("int8", {"lower": -300, "upper": 300}),
        ("uint64", {"lower": -1, "upper": 2**64}),
    ],
)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_clip_ignores_bounds_outside_integer_dtype_range(engine, dtype, bounds):
    """Nonbinding bounds outside the dtype range must leave representable values unchanged."""
    values = [1, 2, 3] if dtype == "int8" else [2**63 + 1, 2**63 + 2, 2**63 + 3]
    frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    engineer = FeatureEngineer(
        [{"name": "clip", "transformer": "ClipValues", "params": {"bounds": {"x": bounds}}}]
    )

    actual, _ = engineer.fit_transform(frame)

    assert actual["x"].to_list() == values


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize(
    "bounds,expected",
    [
        ({"lower": 1.5}, [1.5, 5.0, 9.0]),
        ({"upper": 4.5}, [0.0, 4.5, 4.5]),
        ({"lower": 1.5, "upper": 1.75}, [1.5, 1.75, 1.75]),
    ],
)
def test_clip_fractional_bounds_promote_integers(engine, dtype, bounds, expected):
    """Fractional limits must be applied exactly, including nullable integer inputs."""
    frame = pd.DataFrame({"x": pd.Series([0, 5, 9], dtype=dtype)})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    engineer = FeatureEngineer(
        [{"name": "clip", "transformer": "ClipValues", "params": {"bounds": {"x": bounds}}}]
    )

    actual, _ = engineer.fit_transform(frame)

    assert actual["x"].to_list() == expected
    assert str(actual["x"].dtype).lower() == "float64"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_imputer_extends_inference_categories(engine):
    """Known-group and global fills may introduce training categories absent at inference."""
    train = pd.DataFrame(
        {"group": ["known", "known", "other"], "x": pd.Categorical(["b", "b", "c"])}
    )
    score = pd.DataFrame(
        {
            "group": ["known", "other", "unseen", None, "known"],
            "x": pd.Categorical([None, None, None, None, "a"], categories=["a"]),
        }
    )
    if engine == "polars":
        train, score = pl.from_pandas(train), pl.from_pandas(score)
    engineer = FeatureEngineer(
        [
            {
                "name": "impute",
                "transformer": "GroupImputer",
                "params": {"columns": ["x"], "group_by": "group", "strategy": "most_frequent"},
            }
        ]
    )
    engineer.fit_transform(train)

    actual = engineer.transform(score)

    assert actual["x"].to_list() == ["b", "c", "b", "b", "a"]
