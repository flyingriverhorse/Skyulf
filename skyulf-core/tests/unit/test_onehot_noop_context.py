"""An explicit empty encoder selection binds identity replay to the saved recipe."""

import json
import pickle
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine):
    """Use duplicate row labels and columns whose native null/dtype identities must survive."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "category": ["a", None, "b", "a"]})
    frame.index = [7, 2, 2, 0]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _equal(actual, expected):
    """Require exact values, schemas and pandas labels rather than numeric tolerance."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("options", [{}, {"max_categories": None}, {"include_missing": True}])
def test_explicit_empty_onehot_replays_identity_without_mutation(engine, options, monkeypatch):
    """Authentic empty state must retain all rows, fields and targets across request boundaries."""
    frame = _frame(engine)
    before = deepcopy(frame)
    config = {"columns": [], **options}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(frame, config)
    assert state == {}
    saved = pickle.dumps((config, state))
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    target = np.arange(len(frame))
    full, output_target = applier.apply((frame, target), state)
    assert output_target is target
    _equal(full, frame)
    for positions in ([0], [1, 2], [3, 2, 1, 0], []):
        sample = frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]
        _equal(applier.apply(sample, state), sample)

    def forbidden(*args, **kwargs):
        """Context lookup must inspect intent without learning or transforming data."""
        raise AssertionError("Unexpected fit or apply")

    monkeypatch.setattr(NodeRegistry.get_calculator("OneHotEncoder"), "fit", forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier("OneHotEncoder"), "apply", forbidden)
    capability = get_inference_capability("OneHotEncoder", config, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert capability.row_effect == "preserve"
    _equal(frame, before)
    assert pickle.dumps((config, state)) == saved


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"columns": None},
        {"columns": ["category"]},
        {"columns": [], "_auto_columns": True},
        {"columns": [], "max_categories": True},
        {"columns": [], "max_categories": np.bool_(True)},
        {"columns": [], "max_categories": 3.0},
        {"columns": [], "max_categories": 0},
        {"columns": [], "max_categories": np.int64(-1)},
        {"columns": [], "drop_first": 1},
        {"columns": [], "include_missing": "yes"},
        {"columns": [], "drop_original": 0},
        {"columns": [], "prefix_separator": None},
        {"columns": [], "handle_unknown": "other"},
        {"columns": [], "extra": 1},
    ],
)
def test_empty_onehot_requires_explicit_valid_recipe(config):
    """Empty state cannot prove automatic selection or authorize malformed options."""
    assert get_inference_capability("OneHotEncoder", config, {}, engine="pandas") is None


@pytest.mark.parametrize("state", [{"columns": []}, {"type": "onehot"}, {"encoder_object": None}])
def test_onehot_partial_state_is_not_an_empty_artifact(state):
    """A truncated saved artifact must not acquire an identity execution promise."""
    assert (
        get_inference_capability("OneHotEncoder", {"columns": []}, state, engine="polars") is None
    )


def test_empty_onehot_still_requires_the_registered_applier():
    """An empty dictionary cannot lend the built-in row contract to overridden code."""
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    applier.apply = lambda *args: None
    assert (
        get_inference_capability(
            "OneHotEncoder", {"columns": []}, {}, engine="pandas", applier=applier
        )
        is None
    )


@pytest.mark.parametrize(
    "engine,options,admitted",
    [
        ("pandas", {"max_categories": None}, True),
        ("pandas", {}, False),
        ("pandas", {"max_categories": None, "include_missing": True}, False),
        ("polars", {"max_categories": None}, False),
    ],
)
def test_saved_noop_onehot_preserves_worker_boundary(
    tmp_path, engine, options, admitted, monkeypatch
):
    """Only the unchanged uncapped pandas worker subset may certify an explicit identity step."""
    frame = _frame(engine)
    if isinstance(frame, pd.DataFrame):
        frame = frame.drop(columns="category")
        training = frame.assign(target=range(4))
    else:
        frame = frame.drop("category")
        training = frame.with_columns(pl.Series("target", range(4)))
    config = {"columns": [], **options}
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "empty", "transformer": "OneHotEncoder", "params": config}],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training[:0]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")

    def forbidden(*args, **kwargs):
        """Saved no-op replay never needs to create a new encoder."""
        raise AssertionError("Unexpected fit")

    calculator: Any = NodeRegistry.get_calculator("OneHotEncoder")
    monkeypatch.setattr(calculator, "fit", forbidden)
    artifact = load_pipeline(tmp_path / "model")
    records = artifact.pipeline.feature_engineer.fitted_steps
    before = artifact_digest(records)
    report = probe_fitted_preprocessing(artifact, frame, chunk_sizes=(1, 2))
    assert report["status"] == "passed"
    assert report["steps"][0]["context"] == "row"
    assert report["steps"][0]["state_validation"] == "node_owned"
    predictions = artifact.pipeline.predict(frame)
    singles = np.concatenate([artifact.pipeline.predict(frame[i : i + 1]) for i in range(4)])
    np.testing.assert_array_equal(predictions, singles)
    if admitted:
        certificate = require_partition_safe_pipeline(artifact)
        assert certificate.steps[0].node_type == "OneHotEncoder"
        params = dict(json.loads(certificate.steps[0].config_json)["items"])
        assert params["columns"] == {"kind": "list", "items": []}
        artifact.pipeline.config["preprocessing"][0]["params"]["drop_first"] = True
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(artifact)
    else:
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(artifact)
    assert artifact_digest(records) == before
