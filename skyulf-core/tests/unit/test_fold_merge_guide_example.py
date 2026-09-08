"""Check the guide's merge example against core's real merge and fold adapter."""

import pandas as pd
import pytest

from skyulf.preprocessing.fold_adapter import (
    MergedBranchFoldAdapter,
    _merge_branch_frames_columnwise,
)


@pytest.mark.parametrize(
    ("strategy", "expected_age"),
    [("first_wins", [20, 40]), ("last_wins", [0.4, 0.8])],
)
def test_core_guide_example_keeps_both_distinct_features(strategy, expected_age):
    """The losing age version must not cause its branch's unique feature to disappear."""
    result = _merge_branch_frames_columnwise(
        [
            pd.DataFrame({"age": [20, 40], "income": [900, 1100]}),
            pd.DataFrame({"age": [0.4, 0.8], "city_code": [2, 3]}),
        ],
        strategy,
    )
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"age": expected_age, "income": [900, 1100], "city_code": [2, 3]}),
        check_like=True,
    )


@pytest.mark.parametrize(
    ("strategy", "expected_age"),
    [("first_wins", [20.0, float("nan")]), ("last_wins", [0.4, 0.8])],
)
def test_core_precedence_selects_whole_columns_not_nonmissing_cells(strategy, expected_age):
    """First-wins must preserve the selected column's missing value rather than blend rows."""
    result = _merge_branch_frames_columnwise(
        [
            pd.DataFrame({"age": [20.0, float("nan")], "income": [900, 1100]}),
            pd.DataFrame({"age": [0.4, 0.8], "city_code": [2, 3]}),
        ],
        strategy,
    )
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"age": expected_age, "income": [900, 1100], "city_code": [2, 3]}),
        check_like=True,
    )


@pytest.mark.parametrize(
    ("strategy", "expected_fit_age", "expected_transform_age"),
    [("first_wins", [20.0, 50.0], [100.0]), ("last_wins", [0.4, 1.0], [2.0])],
)
def test_real_fold_branches_keep_both_added_features(
    strategy, expected_fit_age, expected_transform_age
):
    """Fitted and held-out branch merges must retain both independently added indicators."""
    adapter = MergedBranchFoldAdapter(
        [
            [
                {
                    "name": "income_flag",
                    "transformer": "MissingIndicator",
                    "params": {"columns": ["income"]},
                }
            ],
            [
                {
                    "name": "age_scale",
                    "transformer": "MaxAbsScaler",
                    "params": {"columns": ["age"]},
                },
                {
                    "name": "city_flag",
                    "transformer": "MissingIndicator",
                    "params": {"columns": ["city_code"]},
                },
            ],
        ],
        merge_strategy=strategy,
        target_column="target",
    )
    target = pd.Series([0, 1], name="target")
    fitted, fitted_target = adapter.fit_transform(
        pd.DataFrame(
            {"age": [20.0, 50.0], "income": [900.0, float("nan")], "city_code": [float("nan"), 2.0]}
        ),
        target,
    )
    assert set(fitted.columns) == {
        "age",
        "income",
        "city_code",
        "income_missing",
        "city_code_missing",
    }
    assert fitted["age"].tolist() == pytest.approx(expected_fit_age)
    assert fitted["income_missing"].tolist() == [0, 1]
    assert fitted["city_code_missing"].tolist() == [1, 0]
    pd.testing.assert_series_equal(fitted_target, target)

    held_out_target = pd.Series([1], name="target")
    transformed, transformed_target = adapter.transform(
        pd.DataFrame({"age": [100.0], "income": [700.0], "city_code": [4.0]}),
        held_out_target,
    )
    assert set(transformed.columns) == set(fitted.columns)
    assert transformed["age"].tolist() == pytest.approx(expected_transform_age)
    assert transformed["income_missing"].tolist() == [0]
    assert transformed["city_code_missing"].tolist() == [0]
    pd.testing.assert_series_equal(transformed_target, held_out_target)
