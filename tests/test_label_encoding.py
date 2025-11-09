import pandas as pd

from core.feature_engineering.nodes.feature_eng.label_encoding import apply_label_encoding
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store


def _build_node(config):
    return {"id": "label-node", "data": {"config": config}}


def test_label_encoding_creates_encoded_column():
    frame = pd.DataFrame(
        {
            "category": ["alpha", "beta", "alpha", None],
            "other": [1, 2, 3, 4],
        }
    )

    node = _build_node({
        "columns": ["category"],
        "auto_detect": False,
        "drop_original": False,
        "output_suffix": "_enc",
    })

    result, summary, signal = apply_label_encoding(frame, node)

    assert "Label encoding: encoded 1 column" in summary
    assert "category_enc" in result.columns
    encoded_column = signal.encoded_columns[0]
    assert encoded_column.source_column == "category"
    assert encoded_column.encoded_column == "category_enc"
    assert encoded_column.class_count == 2
    assert result["category_enc"].dtype == "Int64"
    assert result["category_enc"].isna().sum() == 1


def test_label_encoding_skips_non_categorical_columns():
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "flag": [True, False, True],
        }
    )

    node = _build_node({
        "columns": ["numeric", "flag"],
        "auto_detect": False,
    })

    result, summary, signal = apply_label_encoding(frame, node)

    assert result.equals(frame)
    assert summary.startswith("Label encoding: no columns encoded")
    assert len(signal.skipped_columns) == 2
    assert any("numeric" in entry for entry in signal.skipped_columns)
    assert any("flag" in entry for entry in signal.skipped_columns)


def test_label_encoding_stores_method_label_for_transformer_audit():
    store = get_pipeline_store()
    store.clear_all()

    frame = pd.DataFrame(
        {
            "text_col": ["a", "b", "c", "a"],
            SPLIT_TYPE_COLUMN: ["train", "train", "validation", "test"],
        }
    )

    node = _build_node({
        "columns": ["text_col"],
        "auto_detect": False,
        "drop_original": False,
    })

    apply_label_encoding(frame, node, pipeline_id="pipe-456")

    entries = store.list_transformers(pipeline_id="pipe-456")
    assert entries, "Expected transformer metadata to be stored"
    metadata = entries[0]["metadata"]
    assert metadata.get("method") == "label_encoding"
    label = metadata.get("method_label", "")
    assert label.startswith("Label Encoding (")

    store.clear_all()
