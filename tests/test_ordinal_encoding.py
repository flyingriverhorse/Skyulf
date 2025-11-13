import pandas as pd

from core.feature_engineering.preprocessing.encoding.ordinal_encoding import apply_ordinal_encoding
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store


def _build_node(config):
    return {"id": "ordinal-node", "data": {"config": config}}


def test_ordinal_encoding_creates_encoded_column():
    frame = pd.DataFrame(
        {
            "quality": ["low", "medium", "high", None],
            "value": [10, 20, 30, 40],
        }
    )

    node = _build_node({"columns": ["quality"], "output_suffix": "_ord"})

    result, summary, signal = apply_ordinal_encoding(frame, node)

    assert "encoded 1 column" in summary
    assert "quality_ord" in result.columns
    encoded_series = result["quality_ord"]
    assert encoded_series.dtype == "Int64"
    assert encoded_series.isna().sum() == 1

    encoded_column = signal.encoded_columns[0]
    assert encoded_column.source_column == "quality"
    assert encoded_column.encoded_column == "quality_ord"
    assert encoded_column.replaced_original is False
    assert encoded_column.category_count == 3
    assert encoded_column.encode_missing is False


def test_ordinal_encoding_skips_non_categorical_columns():
    frame = pd.DataFrame(
        {
            "numeric": [1.1, 2.2, 3.3],
            "boolean_flag": [True, False, True],
        }
    )

    node = _build_node({"columns": ["numeric", "boolean_flag"]})

    result, summary, signal = apply_ordinal_encoding(frame, node)

    assert result.equals(frame)
    assert summary.startswith("Ordinal encoding: no columns encoded")
    assert len(signal.skipped_columns) == 2
    assert any("numeric" in entry for entry in signal.skipped_columns)
    assert any("boolean_flag" in entry for entry in signal.skipped_columns)


def test_ordinal_encoding_stores_and_reuses_encoder_with_splits():
    store = get_pipeline_store()
    store.clear_all()

    train_frame = pd.DataFrame(
        {
            "category": ["bronze", "silver", "gold", "silver"],
            SPLIT_TYPE_COLUMN: ["train", "train", "validation", "test"],
        }
    )

    node = _build_node({"columns": ["category"], "drop_original": True})

    transformed_train, summary_train, signal_train = apply_ordinal_encoding(
        train_frame,
        node,
        pipeline_id="pipeline-ordinal",
    )

    assert "encoded 1 column" in summary_train
    assert "category" in transformed_train.columns
    assert transformed_train["category"].dtype == "Int64"
    assert signal_train.encoded_columns[0].replaced_original is True

    entries = store.list_transformers(pipeline_id="pipeline-ordinal")
    assert entries, "Expected transformer to be stored"
    stored = entries[0]
    metadata = stored["metadata"]
    assert metadata.get("encoded_column") == "category"
    assert metadata.get("drop_original") is True
    assert metadata.get("handle_unknown") == signal_train.handle_unknown
    assert metadata.get("method") == "ordinal_encoding"
    assert "Ordinal Encoding" in metadata.get("method_label", "")

    inference_frame = pd.DataFrame(
        {
            "category": ["bronze", "gold", "gold"],
            SPLIT_TYPE_COLUMN: ["test", "test", "validation"],
        }
    )

    transformed_inference, summary_inference, signal_inference = apply_ordinal_encoding(
        inference_frame,
        node,
        pipeline_id="pipeline-ordinal",
    )

    assert "encoded 1 column" in summary_inference
    assert "category" in transformed_inference.columns
    assert signal_inference.encoded_columns[0].replaced_original is True

    store.clear_all()
