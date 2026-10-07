"""Saved declarative arithmetic executes identically on complete and split model sets."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.model_set import ComponentReference, load_model_set, save_model_set
from skyulf.inference.model_set_partition_safety import require_partition_safe_model_set
from skyulf.inference.model_set_scoring import (
    compose_model_set_outputs,
    predict_model_set,
    validate_model_set_composition,
)
from skyulf.pipeline import SkyulfPipeline


def rules(operation="weighted_sum"):
    """Keep the declaration independent of executable source and caller input features."""
    return {
        "outputs": [
            {
                "name": "blend",
                "version": "1",
                "operation": operation,
                "params": {"inputs": ["a__prediction", "b__prediction"], "weights": [1.0, 3.0]},
                "columns": [{"name": "combined", "dtype": "float64"}],
                "required_components": ["a", "b"],
            }
        ]
    }


def components(dtype="float64"):
    """Expose only declared prediction columns when validating operations."""
    return tuple(
        SimpleNamespace(branch=name, output_schema=(ColumnSpec(name="prediction", dtype=dtype),))
        for name in ("a", "b")
    )


@pytest.mark.parametrize(
    "operation,expected", [("weighted_sum", [14.0, 24.0]), ("weighted_mean", [3.5, 6.0])]
)
def test_declarative_operations_use_only_named_prediction_columns(operation, expected):
    """Saved Python source must never execute for a purely declarative composition."""
    source = "raise AssertionError('declarative composition loaded source')"
    config = validate_model_set_composition(
        rules(operation), source, components(), (ColumnSpec(name="id", dtype="int64"),)
    )
    predictions = pd.DataFrame(
        {
            "a__prediction": [2.0, 3.0],
            "b__prediction": [4.0, 7.0],
            "a__scoring_status": ["predicted", "predicted"],
            "b__scoring_status": ["predicted", "predicted"],
            "a__exclusion_reason": [None, None],
            "b__exclusion_reason": [None, None],
        }
    )
    result = compose_model_set_outputs(pd.DataFrame({"id": [9, 3]}), predictions, config, source)
    assert result.combined.tolist() == expected
    assert result["blend__scoring_status"].tolist() == ["predicted", "predicted"]


@pytest.mark.parametrize(
    "change",
    [
        "unknown_operation",
        "function",
        "version",
        "param",
        "dtype",
        "columns",
        "unknown_input",
        "raw_input",
        "status_input",
        "repeated_input",
        "missing_dependency",
        "extra_dependency",
        "weight_string",
        "weight_bool",
        "weight_null",
        "weight_nan",
        "weight_inf",
        "weight_size",
        "weight_overflow",
        "zero_mean",
        "negative_mean",
    ],
)
def test_invalid_declarative_contracts_fail_before_execution(change):
    """Only complete typed arithmetic on numeric direct component outputs may be saved."""
    config = rules()
    rule = config["outputs"][0]
    if change == "unknown_operation":
        rule["operation"] = "batch_mean"
    elif change == "function":
        rule["function"] = "callback"
    elif change == "version":
        rule["version"] = "2"
    elif change == "param":
        rule["params"]["callback"] = "custom"
    elif change == "dtype":
        rule["columns"][0]["dtype"] = "int64"
    elif change == "columns":
        rule["columns"].append({"name": "other", "dtype": "float64"})
    elif change in {"unknown_input", "raw_input", "status_input", "repeated_input"}:
        rule["params"]["inputs"][0] = {
            "unknown_input": "z__prediction",
            "raw_input": "x",
            "status_input": "a__scoring_status",
            "repeated_input": "b__prediction",
        }[change]
    elif change == "missing_dependency":
        rule["required_components"] = ["a"]
    elif change == "extra_dependency":
        rule["params"]["inputs"] = ["a__prediction"]
        rule["params"]["weights"] = [1.0]
    elif change.startswith("weight_"):
        value = {
            "weight_string": "1",
            "weight_bool": True,
            "weight_null": None,
            "weight_nan": np.nan,
            "weight_inf": np.inf,
            "weight_size": [],
            "weight_overflow": 10**500,
        }[change]
        rule["params"]["weights"] = value if change == "weight_size" else [1.0, value]
    else:
        rule["operation"] = "weighted_mean"
        rule["params"]["weights"] = [0.0, 0.0] if change == "zero_mean" else [-1.0, 2.0]
    with pytest.raises(ValueError):
        validate_model_set_composition(
            config, "", components(), (ColumnSpec(name="id", dtype="int64"),)
        )


def test_string_predictions_cannot_be_coerced_to_numeric_composition():
    """Classification labels cannot silently become arithmetic inputs through parsing."""
    with pytest.raises(ValueError, match="numeric"):
        validate_model_set_composition(
            rules(), "", components("string"), (ColumnSpec(name="id", dtype="int64"),)
        )


@pytest.fixture
def saved_set(tmp_path, request):
    """Persist real tree components and the complete frozen composition declaration."""
    paths = {}
    data = pd.DataFrame(
        {"x": [-4.0, -2.0, 0.0, 1.0, 2.0, 4.0], "target": [-8.0, -4.0, 0.0, 2.0, 4.0, 8.0]}
    )
    for branch, node in (("a", "decision_tree_regressor"), ("b", "extra_trees_regressor")):
        params = {"random_state": 42}
        if branch == "b":
            params.update(n_estimators=3, n_jobs=1)
        pipeline = SkyulfPipeline(
            {
                "preprocessing": [
                    {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}
                ],
                "modeling": {"type": node, "params": params},
            }
        )
        pipeline.fit(SplitDataset(train=data, test=data.iloc[:0]), target_column="target")
        path = tmp_path / branch
        save_local_pipeline(pipeline, path)
        digest = load_local_pipeline(path).manifest.pipeline_sha256
        paths[branch] = (ComponentReference(name=branch, version="1", digest=digest), path)
    artifact = save_model_set(
        tmp_path / "set",
        paths,
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source="raise AssertionError('inactive source executed')",
        composition_config=rules(getattr(request, "param", "weighted_sum")),
    )
    return load_model_set(artifact.directory)


@pytest.mark.parametrize("saved_set", ["weighted_sum", "weighted_mean"], indirect=True)
def test_saved_composition_matches_reordered_split_empty_and_null_batches(saved_set):
    """A certified package must reuse trained fills and explicit arithmetic after reload."""
    certificate = require_partition_safe_model_set(saved_set)
    query = pd.DataFrame(
        {"id": [90, 2, 18, 7, 4], "x": [np.nan, -2.0, 0.0, 9.0, np.nan]}, index=[7, 7, 9, 1, 5]
    )
    whole = predict_model_set(query, saved_set)
    chunks = [query.iloc[:1], query.iloc[1:3], query.iloc[3:], query.iloc[:0]]
    split = pd.concat([predict_model_set(chunk, saved_set) for chunk in chunks])
    pd.testing.assert_frame_equal(whole, split)
    reordered = predict_model_set(query.iloc[[4, 2, 0, 3, 1]], saved_set)
    pd.testing.assert_frame_equal(
        whole.sort_values("id").reset_index(drop=True),
        reordered.sort_values("id").reset_index(drop=True),
    )
    expected = whole.a__prediction + 3 * whole.b__prediction
    if saved_set.manifest.composition_config["outputs"][0]["operation"] == "weighted_mean":
        expected /= 4
    np.testing.assert_allclose(whole.combined.to_numpy(dtype=float), expected.to_numpy(dtype=float))
    assert certificate["composition_contract"] == "row_local_operations_v1"
    assert require_partition_safe_model_set(saved_set) == certificate


def test_mutating_saved_composition_invalidates_package_evidence(saved_set):
    """Workers must reject modified declaration bytes even when the operation remains safe."""
    require_partition_safe_model_set(saved_set)
    path = saved_set.directory / "manifest.json"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("weighted_sum", "weighted_mean"), encoding="utf-8")
    with pytest.raises(ValueError):
        require_partition_safe_model_set(saved_set)


def test_composition_exclusions_remain_row_local():
    """An excluded dependency propagates an outcome without arithmetic on its missing value."""
    config = rules()
    predictions = pd.DataFrame(
        {
            "a__prediction": [None, 3.0],
            "b__prediction": [4.0, 7.0],
            "a__scoring_status": ["excluded", "predicted"],
            "b__scoring_status": ["predicted", "predicted"],
            "a__exclusion_reason": ["missing", None],
            "b__exclusion_reason": [None, None],
        }
    )
    result = compose_model_set_outputs(
        pd.DataFrame({"id": [1, 2]}), predictions, deepcopy(config), ""
    )
    assert result.combined.isna().tolist() == [True, False]
    assert result.combined.iloc[1] == 24.0
    assert result["blend__exclusion_reason"].iloc[0] == "a: missing"
