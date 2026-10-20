"""Remote admission must reuse validation owned by the actual preprocessing node."""

import pickle
from copy import deepcopy

import pandas as pd
import pytest
from tests.integration.core.test_preprocessing_small_data_roundtrip import _RECIPES, _data

from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
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
    from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
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
    save_pipeline(pipeline, tmp_path / "empty")
    evidence = require_partition_safe_pipeline(load_pipeline(tmp_path / "empty"))
    assert evidence.steps[0].node_type == node
    assert evidence.steps[0].action == "apply"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node",
    sorted(
        set(_RECIPES)
        - {"ColumnFunction", "FittedFunction", "RowFilterFunction", "sentence_embedder"}
    ),
)
def test_canvas_labels_preserve_registered_fitted_context(node, engine):
    """Adding the serialized canvas label cannot revoke a real fitted node's admission."""
    if node == "H3Index":
        pytest.importorskip("h3")
    config = deepcopy(NodeRegistry.get_all_metadata()[node]["params"])
    config.update(deepcopy(_RECIPES[node]))
    frame, target = _data(node, engine)
    payload = frame if node == "feature_target_split" else (frame, target)
    state = NodeRegistry.get_calculator(node)().fit(payload, config)
    expected = get_inference_capability(node, config, state, engine=engine)
    labeled = config | {"_display_name": f"Canvas {node}"}
    restored = pickle.loads(pickle.dumps(state))
    actual = get_inference_capability(node, labeled, restored, engine=engine)
    assert actual == expected, f"{node}: canvas metadata revoked {expected}"


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_canvas_imputer_ignores_inactive_constant_after_reload(strategy, engine):
    """Saved canvas defaults must not block replay of learned nonconstant fill values."""
    frame, _ = _data("SimpleImputer", engine)
    config = {"columns": ["missing"], "strategy": strategy, "fill_value": 0}
    applier = NodeRegistry.get_applier("SimpleImputer")
    state = NodeRegistry.get_calculator("SimpleImputer")().fit(frame, config)
    restored = pickle.loads(pickle.dumps(state))
    validate = getattr(applier, "validate_fitted_state", None)
    resolve = getattr(applier, "resolve_fitted_config", None)
    assert callable(validate) and callable(resolve)
    resolved = resolve(config, validate(restored))
    assert resolved["fill_value"] is None
    capability = get_inference_capability("SimpleImputer", config, restored, engine=engine)
    assert capability is not None and capability.context == "row"
    assert config["fill_value"] == 0


@pytest.mark.parametrize(
    "node,config",
    [
        ("SimpleImputer", {"columns": ["x"], "strategy": "median"}),
        ("GroupImputer", {"columns": ["x"], "group_by": "cat", "strategy": "mean"}),
        ("MinMaxScaler", {"columns": ["x"]}),
        ("OneHotEncoder", {"columns": ["cat"], "max_categories": None}),
        ("ClipValues", {"bounds": {"x": {"lower": 0, "upper": 10}}}),
        ("FeatureInteraction", {"columns": ["x", "b"], "degree": 2}),
    ],
)
def test_canvas_label_does_not_hide_unknown_behavior_fields(node, config):
    """Only the known presentation key may bypass strict fitted configuration checks."""
    frame, _ = _data(node, "pandas")
    applier = NodeRegistry.get_applier(node)
    validate = getattr(applier, "validate_fitted_state", None)
    resolve = getattr(applier, "resolve_fitted_config", None)
    assert callable(validate) and callable(resolve)
    state = validate(NodeRegistry.get_calculator(node)().fit(frame, config))
    with pytest.raises(ValueError, match="Unexpected .*fields"):
        resolve(config | {"_display_name": "My node", "_unknown": True}, state)


def test_canvas_constant_imputer_still_rejects_changed_fill_value():
    """Ignoring inactive constants must never accept a changed active constant."""
    frame, _ = _data("SimpleImputer", "pandas")
    config = {"columns": ["missing"], "strategy": "constant", "fill_value": 0}
    applier = NodeRegistry.get_applier("SimpleImputer")
    validate = getattr(applier, "validate_fitted_state", None)
    resolve = getattr(applier, "resolve_fitted_config", None)
    assert callable(validate) and callable(resolve)
    state = validate(NodeRegistry.get_calculator("SimpleImputer")().fit(frame, config))
    with pytest.raises(ValueError, match="Configured constant disagrees"):
        resolve(config | {"_display_name": "Fill", "fill_value": 1}, state)
