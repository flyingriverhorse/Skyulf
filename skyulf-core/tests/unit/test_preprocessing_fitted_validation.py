"""Remote admission must reuse validation owned by the actual preprocessing node."""

from copy import deepcopy

import pandas as pd
import pytest

from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize(
    "node,config",
    [
        ("SimpleImputer", {"columns": ["value"], "strategy": "mean"}),
        ("SimpleImputer", {"columns": ["group"], "strategy": "most_frequent"}),
        ("SimpleImputer", {"columns": ["value"], "strategy": "constant", "fill_value": 0}),
        ("StandardScaler", {"columns": ["value"]}),
        ("MinMaxScaler", {"columns": ["value"], "feature_range": [0, 2]}),
        ("ClipValues", {"bounds": {"value": {"lower": 0, "upper": 10}}}),
        ("GroupImputer", {"columns": ["value"], "group_by": "group", "strategy": "mean"}),
        ("OneHotEncoder", {"columns": ["group"], "max_categories": None}),
        ("FeatureInteraction", {"columns": ["other", "value"], "degree": 2}),
    ],
)
def test_preprocessing_owns_validation_without_refitting(node, config, monkeypatch):
    """Admission can inspect a saved artifact without another implementation or fit."""
    frame = pd.DataFrame(
        {
            "value": [1.0, 4.0, 2.0, 8.0],
            "other": [2.0, 3.0, 1.0, 4.0],
            "group": ["A", "B", "A", "B"],
        }
    )
    calculator = NodeRegistry.get_calculator(node)
    applier = NodeRegistry.get_applier(node)
    state = calculator().fit(frame, deepcopy(config))
    before = artifact_digest(state)

    def forbidden_fit(*args, **kwargs):
        """Refitting during validation would leak inference data into saved parameters."""
        raise AssertionError("Validation must never fit")

    monkeypatch.setattr(calculator, "fit", forbidden_fit)
    validate = getattr(applier, "validate_fitted_state", None)
    resolve = getattr(applier, "resolve_fitted_config", None)
    assert callable(validate) and callable(resolve)
    validated = validate(state)
    resolved = resolve(deepcopy(config), validated)
    transformed = applier().apply(frame, validated)
    assert len(transformed) == len(frame)
    assert artifact_digest(state) == before
    assert resolved == resolve(resolved, validated)


@pytest.mark.parametrize("node", ["SimpleImputer", "StandardScaler"])
def test_empty_portable_preprocessing_remains_admitted(node, tmp_path):
    """Moving validation must preserve explicitly empty no-op preprocessing steps."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.inference.partition_safety import require_partition_safe_pipeline
    from skyulf.pipeline import SkyulfPipeline

    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "empty", "transformer": node, "params": {"columns": []}}],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame.iloc[:3], test=frame.iloc[3:]), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "empty")
    evidence = require_partition_safe_pipeline(load_local_pipeline(tmp_path / "empty"))
    assert evidence.steps[0].node_type == node
    assert evidence.steps[0].action == "apply"
