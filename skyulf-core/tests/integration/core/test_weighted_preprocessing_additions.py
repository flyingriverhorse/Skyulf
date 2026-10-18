"""New row-preserving owners retain positional weights through real saved pipelines."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.modeling._sample_weights import SampleWeightError
from skyulf.preprocessing._weight_policy import validate_weighted_steps
from skyulf.registry import NodeRegistry

NODES = ["FrequencyEncoder", "RareCategoryGrouping", "QuantileTransformer", "Normalizer"]


def _steps(node):
    """Use actual work for each new owner and encode the grouping node's string output."""
    if node in ("FrequencyEncoder", "RareCategoryGrouping"):
        steps = [{"name": "new", "transformer": node, "params": {"columns": ["cat"]}}]
        if node == "RareCategoryGrouping":
            steps.append(
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": {"columns": ["cat"], "max_categories": None},
                }
            )
        return steps
    return [
        {"name": "new", "transformer": node, "params": {"columns": ["x", "z"], "n_quantiles": 4}}
    ]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_weighted_new_owner_pipeline_preserves_positions_and_disk_predictions(
    node, engine, tmp_path, monkeypatch
):
    """Duplicate indexes and reordered rows must not change weights reaching real model fitting."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    rows = np.array([7, 2, 5, 0, 3, 6, 1, 4])
    train = pd.DataFrame(
        {"row": rows, "x": rows / 2 + 1, "z": np.cos(rows), "target": np.sin(rows) + rows % 3},
        index=[9] * len(rows),
    )
    if node in ("FrequencyEncoder", "RareCategoryGrouping"):
        train["cat"] = ["a", "a", "a", "b", "b", "c", "c", "rare"]
    weights = rows.astype(float) + 0.5
    original = LinearRegression.fit
    observed = []

    def actual_fit(model, X, y, sample_weight=None):
        """Inspect exact aligned arguments while still running sklearn's real weighted fit."""
        values = np.asarray(X)
        np.testing.assert_array_equal(values[:, 0], rows)
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, values[:, 0] + 0.5)
        np.testing.assert_allclose(y, train.target.to_numpy())
        observed.append(len(values))
        return original(model, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LinearRegression, "fit", actual_fit)
    native = pl.from_pandas(train) if engine == "polars" else train
    pipeline = SkyulfPipeline(
        {"preprocessing": _steps(node), "modeling": {"type": "linear_regression"}}
    )
    pipeline.fit(native, target_column="target", sample_weight=weights)
    features = train.drop(columns="target")
    expected = pipeline.predict(features)
    save_pipeline(pipeline, tmp_path / "weighted")
    restored = load_pipeline(tmp_path / "weighted")
    np.testing.assert_allclose(predict_pipeline(features, restored).prediction, expected)
    np.testing.assert_allclose(
        predict_pipeline(pl.from_pandas(features), restored).prediction, expected
    )
    np.testing.assert_array_equal(pipeline.feature_engineer.train_sample_weight_, weights)
    assert observed == [len(rows)]


@pytest.mark.parametrize("node", NODES)
def test_new_weight_owner_never_trusts_a_registry_replacement(node, monkeypatch):
    """Only the audited concrete pair can inherit the positional weight contract."""
    validate_weighted_steps([{"transformer": node}])
    replacement = type("UnreviewedReplacement", (NodeRegistry.get_applier(node),), {})
    monkeypatch.setitem(NodeRegistry._appliers, node, replacement)
    with pytest.raises(SampleWeightError, match="unsupported for weighted preprocessing"):
        validate_weighted_steps([{"transformer": node}])
