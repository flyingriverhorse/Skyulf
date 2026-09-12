"""Text features must never overwrite retained inputs or emit ambiguous columns."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.registry import EngineRegistry
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.vectorization import sentence_embedder
from skyulf.registry import NodeRegistry


class _EmbeddingModel:
    """Replace only the external model boundary with a two-dimensional encoder."""

    def get_embedding_dimension(self):
        """Provide a fixed output width without downloading pretrained weights."""
        return 2

    def encode(self, texts, **kwargs):
        """Produce finite per-row values so real frame attachment remains exercised."""
        return np.asarray([[len(text), len(text.split())] for text in texts], dtype=float)


@pytest.fixture(
    params=[
        ("tokenizer", "text__tokens", {"add_token_count": True}),
        ("count_vectorizer", "text__count__apple", {}),
        ("tfidf_vectorizer", "text__tfidf__apple", {}),
        ("hashing_vectorizer", "text__hash__0", {"n_features": 2}),
        ("sentence_embedder", "text__emb__0", {}),
    ],
    ids=["tokenizer", "count", "tfidf", "hashing", "embedding"],
)
def text_node(request, monkeypatch):
    """Run the same public schema contract through every text-feature consumer."""
    node, collision, config = request.param
    if node == "sentence_embedder":
        monkeypatch.setattr(sentence_embedder, "_load_model", lambda name: _EmbeddingModel())
    return node, collision, {"columns": ["text"], **config}


@pytest.fixture(params=["pandas", "polars", "polars_wrapped"])
def frame_kind(request):
    """Cover both engines and the public native-Polars wrapper contract."""
    return request.param


def _frame(data: dict[str, list[Any]], frame_kind: str) -> Any:
    """Keep a non-default pandas index to detect misaligned generated features."""
    frame = pd.DataFrame(data, index=pd.Index(range(31, 31 + len(next(iter(data.values()))))))
    if frame_kind == "pandas":
        return frame
    native = pl.from_pandas(frame)
    return EngineRegistry.wrap(native) if frame_kind == "polars_wrapped" else native


def _pandas(frame: Any) -> pd.DataFrame:
    """Expose values for the same assertions regardless of the execution engine."""
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


@pytest.mark.parametrize("drop_original", [False, True])
def test_fit_rejects_retained_output_name_collision(text_node, frame_kind, drop_original):
    """Full input schemas must be validated even when vocabulary fitting narrows text columns."""
    node, collision, config = text_node
    frame = _frame({"text": ["apple pear", "pear"], collision: [91, 92]}, frame_kind)
    original = _pandas(frame).copy(deep=True)

    with pytest.raises(ValueError, match=f"collid.*{collision}"):
        NodeRegistry.get_calculator(node)().fit(frame, {**config, "drop_original": drop_original})

    pd.testing.assert_frame_equal(_pandas(frame), original)


@pytest.mark.parametrize("drop_original", [False, True])
def test_apply_rejects_new_inference_column_collision(text_node, frame_kind, drop_original):
    """Previously unseen retained columns cannot overwrite or alias fitted output names."""
    node, collision, config = text_node
    artifact = NodeRegistry.get_calculator(node)().fit(
        _frame({"text": ["apple pear", "pear"]}, frame_kind),
        {**config, "drop_original": drop_original},
    )
    frame = _frame({"text": ["pear"], collision: [91]}, frame_kind)
    original = _pandas(frame).copy(deep=True)

    with pytest.raises(ValueError, match=f"collid.*{collision}"):
        NodeRegistry.get_applier(node)().apply(frame, artifact)

    pd.testing.assert_frame_equal(_pandas(frame), original)


@pytest.mark.parametrize("stage", ["fit", "apply"])
def test_token_count_column_is_also_protected(frame_kind, stage):
    """Enabling token counts must not silently overwrite an existing count column."""
    calculator = NodeRegistry.get_calculator("tokenizer")()
    config = {"columns": ["text"], "add_token_count": True}
    frame = _frame({"text": ["apple pear"], "text__token_count": [91]}, frame_kind)
    artifact = calculator.fit(_frame({"text": ["apple pear"]}, frame_kind), config)

    with pytest.raises(ValueError, match="collid.*text__token_count"):
        if stage == "fit":
            calculator.fit(frame, config)
        else:
            NodeRegistry.get_applier("tokenizer")().apply(frame, artifact)

    assert _pandas(frame)["text__token_count"].tolist() == [91]


def test_tokenizer_reuses_dropped_source_names_without_reading_generated_values(frame_kind):
    """Every selected source must be tokenized from its original values before originals drop."""
    frame = _frame(
        {"text": ["APPLE!", "PEAR!"], "text__tokens": ["BANANA melon", "CHERRY!"], "keep": [9, 8]},
        frame_kind,
    )
    artifact = NodeRegistry.get_calculator("tokenizer")().fit(
        frame,
        {"columns": ["text", "text__tokens"], "drop_original": True, "add_token_count": True},
    )
    result = NodeRegistry.get_applier("tokenizer")().apply(frame, artifact)
    values = _pandas(result)

    assert type(result) is type(frame)
    assert values.to_dict("list") == {
        "keep": [9, 8],
        "text__tokens": ["apple", "pear"],
        "text__token_count": [1, 1],
        "text__tokens__tokens": ["banana melon", "cherry"],
        "text__tokens__token_count": [2, 1],
    }
    if frame_kind == "pandas":
        pd.testing.assert_index_equal(values.index, frame.index)
    assert _pandas(frame)["text__tokens"].tolist() == ["BANANA melon", "CHERRY!"]


def test_tokenizer_keep_original_rejects_selected_source_collision(frame_kind):
    """A selected source name is reserved when the caller keeps its original values."""
    frame = _frame({"text": ["APPLE!"], "text__tokens": ["BANANA!"]}, frame_kind)

    with pytest.raises(ValueError, match="collid.*text__tokens"):
        NodeRegistry.get_calculator("tokenizer")().fit(
            frame, {"columns": ["text", "text__tokens"], "drop_original": False}
        )


def test_pipeline_rejects_inference_collision(text_node, frame_kind):
    """Fitted pipeline replay must enforce the same protection as direct applier calls."""
    node, collision, config = text_node
    pipeline = FeatureEngineer([{"name": "text features", "transformer": node, "params": config}])
    pipeline.fit_transform(_frame({"text": ["apple pear", "pear"]}, frame_kind))
    frame = _frame({"text": ["apple"], collision: [91]}, frame_kind)

    with pytest.raises(ValueError, match=f"collid.*{collision}"):
        pipeline.transform(frame)

    assert _pandas(frame)[collision].tolist() == [91]


def test_multi_source_fit_validates_joined_output_prefix(text_node, frame_kind):
    """Names derived from all selected columns must be checked against retained inputs."""
    node, collision, config = text_node
    prefix = "body" if node == "tokenizer" else "text_body"
    collision = collision.replace("text", prefix, 1)
    frame = _frame({"text": ["apple"], "body": ["pear apple"], collision: [91]}, frame_kind)

    with pytest.raises(ValueError, match=f"collid.*{collision}"):
        NodeRegistry.get_calculator(node)().fit(frame, {**config, "columns": ["text", "body"]})


def test_missing_sources_do_not_reserve_unemitted_output_names(text_node, frame_kind):
    """A missing source is a no-op even when an unrelated input resembles its output name."""
    node, collision, config = text_node
    artifact = NodeRegistry.get_calculator(node)().fit(
        _frame({"text": ["apple pear"]}, frame_kind), config
    )
    frame = _frame({collision: [91]}, frame_kind)
    result = NodeRegistry.get_applier(node)().apply(frame, artifact)

    assert type(result) is type(frame)
    pd.testing.assert_frame_equal(_pandas(result), _pandas(frame))


@pytest.mark.parametrize(
    "node, suffix", [("count_vectorizer", "count"), ("tfidf_vectorizer", "tfidf")]
)
def test_unseen_vocabulary_does_not_reserve_new_feature_names(node, suffix, frame_kind):
    """New inference words cannot change the fitted schema or collide with retained data."""
    artifact = NodeRegistry.get_calculator(node)().fit(
        _frame({"text": ["apple pear"]}, frame_kind), {"columns": ["text"], "drop_original": True}
    )
    frame = _frame({"text": ["cherry"], f"text__{suffix}__cherry": [91]}, frame_kind)
    result = _pandas(NodeRegistry.get_applier(node)().apply(frame, artifact))

    assert result.to_dict("list") == {
        f"text__{suffix}__cherry": [91],
        f"text__{suffix}__apple": [0],
        f"text__{suffix}__pear": [0],
    }


def test_ordinary_output_preserves_engine_tuple_index_and_retained_values(text_node, frame_kind):
    """Schema validation must keep the normal transform's row and wrapper contracts intact."""
    node, collision, config = text_node
    frame = _frame({"text": ["apple apple", "pear"], "keep": [91, 92]}, frame_kind)
    labels = pd.Series([1, 0], name="label")
    if frame_kind != "pandas":
        labels = pl.from_pandas(labels)
    artifact = NodeRegistry.get_calculator(node)().fit((frame, labels), config)
    result, result_labels = NodeRegistry.get_applier(node)().apply((frame, labels), artifact)
    values = _pandas(result)

    assert result_labels is labels
    assert type(result) is type(frame)
    assert values.columns.is_unique
    assert values["keep"].tolist() == [91, 92]
    assert values["text"].tolist() == ["apple apple", "pear"]
    if node == "tokenizer":
        assert values[collision].tolist() == ["apple apple", "pear"]
        assert values["text__token_count"].tolist() == [2, 1]
    elif node == "count_vectorizer":
        assert values[collision].tolist() == [2, 0]
    elif node == "tfidf_vectorizer":
        assert values[collision].tolist() == [1, 0]
    elif node == "sentence_embedder":
        assert values[["text__emb__0", "text__emb__1"]].values.tolist() == [[11, 2], [4, 1]]
    else:
        np.testing.assert_allclose(
            np.linalg.norm(values[["text__hash__0", "text__hash__1"]].to_numpy(), axis=1), [1, 1]
        )
    if frame_kind == "pandas":
        pd.testing.assert_index_equal(values.index, frame.index)
    assert collision in values.columns
