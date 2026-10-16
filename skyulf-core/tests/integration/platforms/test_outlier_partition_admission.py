"""Outlier admission binds exclusion decisions to the actual saved fitted step."""

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference._partition_builtins import builtin_contract
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe
from skyulf.pipeline import SkyulfPipeline
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize(
    ("node", "params"),
    [
        ("IQR", {"columns": ["x"], "multiplier": 1.5}),
        ("ZScore", {"columns": ["x"], "threshold": 2.0}),
        ("EllipticEnvelope", {"columns": ["x"], "contamination": 0.1}),
        ("ManualBounds", {"bounds": {"x": {"lower": -3.0, "upper": 3.0}}}),
    ],
)
def test_saved_outlier_certificate_binds_exclusions(node, params, tmp_path):
    """Actual probes must admit row outcomes without silently skipping fitted thresholds."""
    values = np.random.default_rng(81).normal(size=160)
    data = pd.DataFrame({"x": values, "target": (values > 0).astype(int)})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "bounds", "transformer": node, "params": params}],
            "modeling": {"type": "decision_tree_classifier", "params": {"max_depth": 3}},
        }
    )
    pipeline.fit(SplitDataset(train=data.iloc[:120], test=data.iloc[120:]), target_column="target")
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    artifact = load_pipeline(path)
    with pytest.raises(ValueError):
        require_partition_safe_pipeline(artifact)
    sealed = seal_partition_probe(artifact, data.iloc[120:], path, enabled=True)
    certificate = require_partition_safe_pipeline(sealed)
    assert certificate.steps[0].action == "exclude_outliers"
    frame = pd.DataFrame({"x": [0.0, 100.0, -100.0]})
    prediction = score_pipeline(frame, sealed)
    assert prediction.scoring_status.tolist() == ["predicted", "excluded", "excluded"]
    assert prediction.exclusion_reason.iloc[1:].tolist() == ["outlier:bounds", "outlier:bounds"]
    assert prediction.prediction.iloc[1:].isna().all()
    assert require_partition_safe_pipeline(load_pipeline(path)) == certificate


@pytest.mark.parametrize("node", ["IQR", "ZScore", "EllipticEnvelope"])
def test_outlier_noop_state_keeps_its_row_preserving_admission(node):
    """An explicitly empty column selection must not require a filtering state."""
    frame = pd.DataFrame({"x": np.arange(32, dtype=float)})
    params = {"columns": []}
    state = NodeRegistry.get_calculator(node)().fit(frame, params)
    record = {
        "name": "bounds",
        "type": node,
        "params": params,
        "artifact": state,
        "applier": NodeRegistry.get_applier(node)(),
    }
    resolved, _ = builtin_contract(
        record, {"name": "bounds", "transformer": node, "params": params}
    )
    assert resolved == {}
