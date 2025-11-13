import pandas as pd
import pytest

from core.feature_engineering.preprocessing.bucketing import _apply_binning_discretization
from core.feature_engineering.preprocessing.encoding.dummy_encoding import apply_dummy_encoding
from core.feature_engineering.nodes.feature_eng.feature_selection import apply_feature_selection
from core.feature_engineering.preprocessing.encoding.hash_encoding import apply_hash_encoding
from core.feature_engineering.nodes.feature_eng.imputation import apply_imputation_methods
from core.feature_engineering.preprocessing.encoding.label_encoding import apply_label_encoding
from core.feature_engineering.preprocessing.encoding.one_hot_encoding import apply_one_hot_encoding
from core.feature_engineering.preprocessing.encoding.ordinal_encoding import apply_ordinal_encoding
from core.feature_engineering.nodes.feature_eng.outliers_removal import _apply_outlier_removal
from core.feature_engineering.nodes.feature_eng.polynomial_features import apply_polynomial_features
from core.feature_engineering.nodes.feature_eng.scaling import _apply_scale_numeric_features
from core.feature_engineering.nodes.feature_eng.skewness import (
    SKEWNESS_METHODS,
    _apply_skewness_transformations,
)
from core.feature_engineering.preprocessing.encoding.target_encoding import apply_target_encoding
from core.feature_engineering.nodes.feature_eng.transformer_audit import apply_transformer_audit
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["a", "b", "a", "c", "b", "c"],
            "target": [0, 1, 0, 1, 0, 1],
            SPLIT_TYPE_COLUMN: ["train", "train", "validation", "test", "train", "validation"],
        }
    )


@pytest.fixture
def rich_frame(sample_frame: pd.DataFrame) -> pd.DataFrame:
    frame = sample_frame.copy()
    frame["num2"] = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    frame["num_with_missing"] = [1.0, None, 2.0, None, 3.0, 4.0]
    frame["skewed"] = [1.0, 1.2, 1.4, 1.6, 20.0, 30.0]
    frame["cat2"] = ["x", "x", "y", "z", "y", "z"]
    return frame


@pytest.fixture(autouse=True)
def reset_pipeline_store():
    store = get_pipeline_store()
    store.clear_all()
    yield
    store.clear_all()


def _list_transformers(pipeline_id: str):
    store = get_pipeline_store()
    return store.list_pipelines(pipeline_id=pipeline_id)


def test_scaling_stores_transformer(sample_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-scaling"
    node = {"id": "scaling-node", "data": {"config": {"columns": ["num"], "auto_detect": False}}}

    _apply_scale_numeric_features(sample_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "scaler" for record in records)


def test_label_encoding_stores_transformer(sample_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-label"
    node = {"id": "label-node", "data": {"config": {"columns": ["cat"], "drop_original": False}}}

    apply_label_encoding(sample_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(
        record["transformer_name"] == "label_encoder" and record["column_name"] == "cat"
        for record in records
    )


def test_dummy_encoding_stores_transformer(sample_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-dummy"
    node = {"id": "dummy-node", "data": {"config": {"columns": ["cat"], "drop_first": False}}}

    apply_dummy_encoding(sample_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(
        record["transformer_name"] == "dummy_encoder" and record["column_name"] == "cat"
        for record in records
    )


def test_target_encoding_stores_transformer(sample_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-target"
    node = {
        "id": "target-node",
        "data": {
            "config": {
                "columns": ["cat"],
                "target_column": "target",
                "drop_original": False,
                "encode_missing": False,
            }
        },
    }

    apply_target_encoding(sample_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(
        record["transformer_name"] == "target_encoder" and record["column_name"] == "cat"
        for record in records
    )


def test_one_hot_encoding_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-onehot"
    node = {
        "id": "onehot-node",
        "data": {
            "config": {
                "columns": ["cat2"],
                "drop_first": False,
                "drop_original": False,
                "include_missing": False,
            }
        },
    }

    apply_one_hot_encoding(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "one_hot_encoder" for record in records)


def test_hash_encoding_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-hash"
    node = {
        "id": "hash-node",
        "data": {
            "config": {
                "columns": ["cat"],
                "auto_detect": False,
                "drop_original": False,
                "encode_missing": False,
                "n_buckets": 8,
            }
        },
    }

    apply_hash_encoding(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "hash_encoder" for record in records)


def test_ordinal_encoding_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-ordinal"
    node = {
        "id": "ordinal-node",
        "data": {
            "config": {
                "columns": ["cat"],
                "drop_original": False,
                "encode_missing": False,
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,
            }
        },
    }

    apply_ordinal_encoding(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "ordinal_encoder" for record in records)


def test_binning_stores_metadata(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-binning"
    node = {
        "id": "bin-node",
        "data": {
            "config": {
                "columns": ["num2"],
                "strategy": "equal_width",
                "equal_width_bins": 3,
                "drop_original": False,
            }
        },
    }

    _apply_binning_discretization(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "binning_equal_width" for record in records)


def test_polynomial_features_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-poly"
    node = {
        "id": "poly-node",
        "data": {
            "config": {
                "columns": ["num", "num2"],
                "degree": 2,
                "auto_detect": False,
                "include_input_features": True,
            }
        },
    }

    apply_polynomial_features(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "polynomial_features" for record in records)


def test_feature_selection_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-feature-selection"
    node = {
        "id": "feature-selection-node",
        "data": {
            "config": {
                "columns": ["num", "num2", "skewed"],
                "method": "select_k_best",
                "score_func": "f_classif",
                "k": 2,
                "target_column": "target",
                "auto_detect": False,
            }
        },
    }

    apply_feature_selection(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "feature_selection" for record in records)


def test_imputation_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-impute"
    node = {
        "id": "impute-node",
        "data": {
            "config": {
                "strategies": [
                    {
                        "method": "mean",
                        "columns": ["num_with_missing"],
                    }
                ]
            }
        },
    }

    apply_imputation_methods(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(
        record["transformer_name"] == "imputer" and record["column_name"] == "num_with_missing"
        for record in records
    )


def test_outlier_removal_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-outliers"
    node = {
        "id": "outlier-node",
        "data": {
            "config": {
                "columns": ["skewed"],
                "default_method": "zscore",
            }
        },
    }

    _apply_outlier_removal(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "outlier_removal" for record in records)


def test_skewness_stores_transformer(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-skewness"
    node = {
        "id": "skew-node",
        "data": {
            "config": {
                "transformations": [
                    {
                        "column": "skewed",
                        "method": "log",
                    }
                ]
            }
        },
    }

    _apply_skewness_transformations(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    assert any(record["transformer_name"] == "skewness_transform" for record in records)


def test_skewness_box_cox_metadata(rich_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-skewness-box-cox"
    node = {
        "id": "skew-node",
        "data": {
            "config": {
                "transformations": [
                    {
                        "column": "skewed",
                        "method": "box_cox",
                    }
                ]
            }
        },
    }

    _apply_skewness_transformations(rich_frame.copy(), node, pipeline_id=pipeline_id)

    records = _list_transformers(pipeline_id)
    box_cox_records = [
        record
        for record in records
        if record["transformer_name"] == "skewness_transform" and record["column_name"] == "skewed"
    ]
    assert box_cox_records, "Expected skewness transformer metadata for box-cox method"

    metadata = box_cox_records[0].get("metadata") or {}
    assert metadata.get("method") == "box_cox"
    assert metadata.get("method_label") == SKEWNESS_METHODS["box_cox"]["label"]
    assert isinstance(metadata.get("lambdas"), list) and metadata["lambdas"], "Box-Cox lambda not stored"


def test_transformer_audit_reflects_pipeline_store(sample_frame: pd.DataFrame) -> None:
    pipeline_id = "pipe-audit"
    scaling_node = {"id": "scaling-node", "data": {"config": {"columns": ["num"]}}}
    label_node = {"id": "label-node", "data": {"config": {"columns": ["cat"], "drop_original": False}}}

    _apply_scale_numeric_features(sample_frame.copy(), scaling_node, pipeline_id=pipeline_id)
    apply_label_encoding(sample_frame.copy(), label_node, pipeline_id=pipeline_id)

    frame = sample_frame.copy()
    signal_frame, summary, signal = apply_transformer_audit(
        frame,
        {"id": "audit-node"},
        pipeline_id=pipeline_id,
        node_map={"scaling-node": {"label": "Scaling"}, "label-node": {"label": "Labeling"}},
    )

    assert signal_frame.equals(frame)
    assert signal.total_transformers == 2
    assert summary.startswith("Transformer audit:")
    transformer_names = {entry.transformer_name for entry in signal.transformers}
    assert {"scaler", "label_encoder"} <= transformer_names
