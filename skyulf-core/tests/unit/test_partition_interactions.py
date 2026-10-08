"""Saved interaction features must execute identically on batch and serving workers."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture
def interaction_artifact(tmp_path, request):
    """Persist a real raw-input pipeline whose estimator requires a generated product."""
    options = getattr(request, "param", {})
    frame = pd.DataFrame(
        {
            "income": [1.0, 3.0, 2.0, 7.0, 4.0, 9.0, 5.0, 6.0],
            "tenure": [2.0, 5.0, 3.0, 1.0, 4.0, 6.0, 2.0, 7.0],
        }
    )
    frame["target"] = frame.income * frame.tenure + 2
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["income", "tenure"], "strategy": "mean"},
                },
                {
                    "name": "products",
                    "transformer": "FeatureInteraction",
                    "params": {"columns": ["tenure", "income"], **options},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame.iloc[:6], test=frame.iloc[6:]), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "artifact")
    return load_local_pipeline(tmp_path / "artifact")


@pytest.mark.parametrize(
    "interaction_artifact",
    [{}, {"degree": 3, "interaction_only": False}, {"degree": 4, "include_bias": True}],
    indirect=True,
)
def test_raw_interactions_preserve_partition_reorder_null_and_empty_parity(interaction_artifact):
    """Workers must reuse the exact saved products without an upstream feature table."""
    artifact = interaction_artifact
    evidence = require_partition_safe_pipeline(artifact)
    raw = pd.DataFrame(
        {"income": [None, 15.0, 2.0, -1.0], "tenure": [3.0, None, 4.0, 5.0]},
        index=[90, 3, 7, 1],
    )
    whole = predict_local_pipeline(raw, artifact)
    split = pd.concat([predict_local_pipeline(raw.iloc[[i]], artifact) for i in range(len(raw))])
    reordered = predict_local_pipeline(raw.iloc[[2, 0, 3, 1]], artifact)
    pd.testing.assert_frame_equal(whole, split)
    pd.testing.assert_frame_equal(whole.sort_index(), reordered.sort_index())
    engineer = artifact.pipeline.feature_engineer
    features = engineer.transform(raw, preserve_rows=True)
    empty = engineer.transform(raw.iloc[:0], preserve_rows=True)
    assert empty.columns.tolist() == features.columns.tolist()
    assert empty.dtypes.to_dict() == features.dtypes.to_dict()
    assert artifact.manifest.input_columns == ("income", "tenure")
    assert evidence.steps[1].node_type == "FeatureInteraction"
    assert np.isfinite(whole.prediction).all()
    assert require_partition_safe_pipeline(artifact) == evidence


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "unknown"),
        ("columns", ["income", "income"]),
        ("columns", ["tenure", "income"]),
        ("degree", True),
        ("degree", 5),
        ("include_bias", 1),
        ("interaction_only", 0),
        ("combinations", [["income", "income"]]),
        ("combinations", []),
        ("feature_names", ["substituted_feature"]),
        ("callback", "anything"),
    ],
)
def test_interaction_admission_rejects_tampered_state(interaction_artifact, field, value):
    """A familiar node name must not admit altered products or ignored artifact fields."""
    interaction_artifact.pipeline.feature_engineer.fitted_steps[1]["artifact"][field] = value
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(interaction_artifact)


@pytest.mark.parametrize("field,value", [("degree", 3), ("columns", ["income"]), ("extra", 1)])
def test_interaction_admission_binds_config_to_saved_products(interaction_artifact, field, value):
    """Worker admission must compare the recipe with the fitted combination list."""
    interaction_artifact.pipeline.feature_engineer.fitted_steps[1]["params"][field] = value
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(interaction_artifact)


def test_interactions_do_not_grant_native_spark_or_custom_applier_support(interaction_artifact):
    """Python batch admission must not authorize a different runtime or callback body."""
    require_partition_safe_pipeline(interaction_artifact)
    with pytest.raises(UnsupportedExecutionError):
        require_capability("FeatureInteraction", "apply", "spark", config={})
    changed = deepcopy(interaction_artifact)
    changed.pipeline.feature_engineer.fitted_steps[1]["applier"].apply = lambda *args: None
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(changed)
