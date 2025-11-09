import pandas as pd

from core.feature_engineering.nodes.feature_eng.imputation import apply_imputation_methods
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store


def _build_node(config):
    return {"id": "impute-node", "data": {"config": config}}


def test_imputation_mean_strategy_fills_missing_values():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
    node = _build_node({"strategies": [{"method": "mean", "columns": ["a"]}]})

    result, summary, signal = apply_imputation_methods(frame, node)

    assert summary.startswith("Imputation methods: filled 1 cell")
    assert result["a"].tolist() == [1.0, 2.0, 3.0]
    assert signal.filled_cells == 1
    assert signal.method_usage == {"mean": 1}
    assert signal.applied_columns[0].column == "a"
    assert signal.applied_columns[0].method == "mean"


def test_imputation_knn_with_pipeline_store_records_metadata():
    store = get_pipeline_store()
    store.clear_all()

    frame = pd.DataFrame(
        {
            "a": [None, 2.0, 3.0, 4.0],
            "b": [0.0, 1.0, 2.0, 3.0],
            SPLIT_TYPE_COLUMN: ["train", "train", "validation", "test"],
        }
    )
    node = _build_node({"strategies": [{"method": "knn", "columns": ["a", "b"]}]})

    result, summary, signal = apply_imputation_methods(frame, node, pipeline_id="pipe-knn")

    assert summary.startswith("Imputation methods: filled")
    assert pd.notna(result.loc[0, "a"])
    assert signal.method_usage.get("knn") == 1
    assert any(entry.column == "a" for entry in signal.applied_columns)

    entries = store.list_transformers(pipeline_id="pipe-knn")
    assert entries, "Expected stored transformer metadata"
    metadata = entries[0]["metadata"]
    assert metadata.get("method") == "knn"
    assert metadata.get("method_label") == "KNN"

    store.clear_all()
