"""End-to-end target context for raw-frame text preprocessing."""

from copy import deepcopy

import pandas as pd
import pytest

from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset


@pytest.mark.parametrize(
    "transformer", ["count_vectorizer", "tfidf_vectorizer", "hashing_vectorizer"]
)
@pytest.mark.parametrize("selection", [None, ["body", "target"], ["target"]])
def test_raw_split_target_never_becomes_an_encoded_text_feature(transformer, selection):
    """Passing target context must make raw-frame and explicit feature selection agree."""
    train = pd.DataFrame(
        {
            "body": ["shared orchard", "shared garden", "quiet orchard", "quiet garden"],
            "target": ["secretalpha", "secretbeta", "secretalpha", "secretbeta"],
        }
    )
    test = pd.DataFrame({"body": ["shared orchard"], "target": ["unseenheldoutlabel"]})
    data = SplitDataset(train=train, test=test)

    def pipeline(columns):
        """Create the same real transformer with only its selection varied."""
        params = {"n_features": 8} if transformer == "hashing_vectorizer" else {}
        if columns is not None:
            params["columns"] = columns
        return SkyulfPipeline(
            {
                "preprocessing": [{"name": "text", "transformer": transformer, "params": params}],
                "modeling": {},
            }
        )

    expected_columns = ["body"] if selection == ["body", "target"] else []
    expected = pipeline(expected_columns).get_fitted_split(data, target_column="target")
    actual_pipeline = pipeline(selection)
    original_config = deepcopy(actual_pipeline.config)
    actual = actual_pipeline.get_fitted_split(data, target_column="target")
    actual_pipeline.fit(data, target_column="target")
    for expected_frame, actual_frame in zip(expected, actual, strict=True):
        if isinstance(expected_frame, pd.DataFrame):
            pd.testing.assert_frame_equal(expected_frame, actual_frame)
        else:
            pd.testing.assert_series_equal(expected_frame, actual_frame)
    assert actual_pipeline.config == original_config
