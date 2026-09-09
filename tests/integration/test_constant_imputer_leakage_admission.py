"""Expose held-out dependence in an admitted constant-imputation configuration."""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution._leakage_validation import (
    validate_no_preprocessing_before_split,
)
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.modeling.base import extract_xy
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.mark.parametrize("fill_value", [0, -1])
@pytest.mark.parametrize(
    "explicit_columns", [False, True], ids=["automatic-columns", "explicit-columns"]
)
@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
def test_admitted_constant_imputation_cannot_depend_on_held_out_values(
    tmp_path, fill_value, explicit_columns, mode
):
    """An operation exempted as fixed must give identical training features as held-out values change."""
    baseline = pd.DataFrame(
        {"row_id": np.arange(6), "x": np.full(6, np.nan), "target": np.arange(6) % 2}
    )
    split_params = {"target_column": "target", "test_size": 1 / 3, "random_state": 42}
    partition, _metrics = FeatureEngineer(
        [{"name": "split", "transformer": "TrainTestSplitter", "params": split_params}]
    ).fit_transform(baseline)
    held_out_ids = extract_xy(partition.test, "target")[0]["row_id"].tolist()
    expected_train_ids = set(extract_xy(partition.train, "target")[0]["row_id"])
    changed = baseline.copy()
    changed.loc[held_out_ids, "x"] = [5.0, 9.0]
    outputs = []
    for name, frame in [("all_missing", baseline), ("held_out_observed", changed)]:
        nodes = [
            NodeConfig("load", "data_loader"),
            NodeConfig(
                "impute",
                "SimpleImputer",
                inputs=["load"],
                params={
                    **({"columns": ["x"]} if explicit_columns else {}),
                    "strategy": "constant",
                    "fill_value": fill_value,
                },
            ),
            NodeConfig("split", "TrainTestSplitter", inputs=["impute"], params=split_params),
            NodeConfig("model", "training", inputs=["split"], params={"target_column": "target"}),
        ]
        verdict = validate_no_preprocessing_before_split(nodes, on_leakage=mode)
        assert verdict["status"] == "passed"
        assert any(item["node_id"] == "impute" for item in verdict["exempted"])
        store = LocalArtifactStore(str(tmp_path / name))
        engine = PipelineEngine(store, FileSystemCatalog())
        engine._on_leakage = mode
        engine._node_configs = {node.node_id: node for node in nodes}
        store.save("load", frame)
        engine._run_transformer(nodes[1])
        engine._run_transformer(nodes[2])
        actual_partition = store.load("split")
        training_x, _target = extract_xy(actual_partition.train, "target")
        test_x, _test_target = extract_xy(actual_partition.test, "target")
        assert set(training_x["row_id"]) == expected_train_ids
        assert set(test_x["row_id"]) == set(held_out_ids)
        assert set(training_x["row_id"]).isdisjoint(held_out_ids)
        np.testing.assert_array_equal(training_x["x"], np.full(len(expected_train_ids), fill_value))
        expected_test = frame.set_index("row_id").loc[sorted(held_out_ids), "x"].fillna(fill_value)
        np.testing.assert_array_equal(test_x.sort_values("row_id")["x"], expected_test)
        outputs.append(training_x.sort_values("row_id")["x"].to_numpy())
    np.testing.assert_array_equal(
        outputs[0],
        outputs[1],
        err_msg="The admitted fixed operation changes training features when only held-out values change",
    )
