"""Actual row boundaries and legacy splitter aliases must agree."""

import pandas as pd
import pytest

from skyulf import SkyulfPipeline


@pytest.mark.parametrize("first", ["TrainTestSplitter", "Split"])
@pytest.mark.parametrize("second", ["TrainTestSplitter", "Split"])
def test_repeated_row_splitter_reuses_existing_partition(first, second):
    """A second splitter, including the legacy alias, must not refit or repartition."""
    data = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
    steps = [
        {
            "name": "first",
            "transformer": first,
            "params": {"test_size": 0.25, "random_state": 42},
        },
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
    ]
    baseline = SkyulfPipeline({"preprocessing": steps, "modeling": {}})
    repeated = SkyulfPipeline(
        {
            "preprocessing": [
                *steps,
                {"name": "second", "transformer": second, "params": {"test_size": 0.5}},
            ],
            "modeling": {},
        }
    )
    expected = baseline.get_fitted_split(data, target_column="target")
    actual = repeated.get_fitted_split(data, target_column="target")
    for expected_frame, actual_frame in zip(expected, actual, strict=True):
        if isinstance(expected_frame, pd.DataFrame):
            assert isinstance(actual_frame, pd.DataFrame)
            pd.testing.assert_frame_equal(expected_frame, actual_frame)
        else:
            assert isinstance(actual_frame, pd.Series)
            pd.testing.assert_series_equal(expected_frame, actual_frame)
    assert len(actual[0]) == 15 and len(actual[2]) == 5
