import pandas as pd

from core.feature_engineering.preprocessing.encoding.hash_encoding import apply_hash_encoding
from core.feature_engineering.preprocessing.split import SPLIT_TYPE_COLUMN
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store


def _build_node(config):
    return {"id": "hash-node", "data": {"config": config}}


def test_hash_encoding_creates_hashed_column():
    frame = pd.DataFrame(
        {
            "text_col": ["alpha", "beta", "gamma", None],
            "other": [1, 2, 3, 4],
        }
    )
    node = _build_node({
        "columns": ["text_col"],
        "n_buckets": 8,
        "encode_missing": True,
        "drop_original": False,
        "output_suffix": "_hashed",
    })

    result, summary, signal = apply_hash_encoding(frame, node)

    assert "Hash encoding: encoded 1 column" in summary
    assert "text_col_hashed" in result.columns
    assert result["text_col_hashed"].dtype == "Int64"
    encoded_column = signal.encoded_columns[0]
    assert encoded_column.source_column == "text_col"
    assert encoded_column.output_column == "text_col_hashed"
    assert encoded_column.bucket_count == 8
    assert encoded_column.encoded_missing is True


def test_hash_encoding_skips_non_categorical_columns():
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

    result, summary, signal = apply_hash_encoding(frame, node)

    assert result.equals(frame)
    assert summary.startswith("Hash encoding: no columns encoded")
    assert len(signal.skipped_columns) == 2
    assert any("numeric" in entry for entry in signal.skipped_columns)
    assert any("flag" in entry for entry in signal.skipped_columns)


def test_hash_encoding_stores_method_label_for_transformer_audit():
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
        "n_buckets": 16,
        "encode_missing": False,
        "drop_original": False,
        "output_suffix": "_hash",
    })

    apply_hash_encoding(frame, node, pipeline_id="pipe-123")

    entries = store.list_transformers(pipeline_id="pipe-123")
    assert entries, "Expected transformer metadata to be stored"
    metadata = entries[0]["metadata"]
    assert metadata.get("method") == "hash_encoding"
    assert metadata.get("method_label", "").startswith("Hash Encoding (")

    store.clear_all()

