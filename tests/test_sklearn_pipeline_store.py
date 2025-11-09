from typing import Iterator

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from core.feature_engineering.nodes.feature_eng.transformer_audit import apply_transformer_audit
from core.feature_engineering.sklearn_pipeline_store import (
    SklearnPipelineStore,
    get_pipeline_store,
)


@pytest.fixture
def fresh_store() -> Iterator[SklearnPipelineStore]:
    store = SklearnPipelineStore()
    yield store
    store.clear_all()


@pytest.fixture
def global_store() -> Iterator[SklearnPipelineStore]:
    store = get_pipeline_store()
    store.clear_all()
    yield store
    store.clear_all()


def test_register_pipeline_clones_estimator(fresh_store: SklearnPipelineStore) -> None:
    scaler = StandardScaler()
    stored = fresh_store.register_pipeline("pipe", "node", "scaler", scaler)

    assert stored is not scaler

    retrieved = fresh_store.get_pipeline("pipe", "node", "scaler")
    assert retrieved is stored
    assert isinstance(retrieved, StandardScaler)
    assert retrieved.get_params() == scaler.get_params()


def test_metadata_and_activity_tracking(fresh_store: SklearnPipelineStore) -> None:
    scaler = StandardScaler()
    metadata = {"input_columns": np.array(["feature"])}

    fresh_store.register_pipeline(
        "pipe",
        "node",
        "scaler",
        scaler,
        column_name="feature",
        metadata=metadata,
    )

    train = np.array([[1.0], [2.0], [3.0]])
    fresh_store.fit_transform(
        "pipe",
        "node",
        "scaler",
        train,
        column_name="feature",
        split_name="train",
    )

    validation = np.array([[4.0], [5.0]])
    fresh_store.transform(
        "pipe",
        "node",
        "scaler",
        validation,
        column_name="feature",
        split_name="validation",
    )

    records = fresh_store.list_pipelines(pipeline_id="pipe")
    assert len(records) == 1

    record = records[0]
    assert record["metadata"]["input_columns"] == ["feature"]

    split_activity = record["split_activity"]
    assert split_activity["train"]["action"] == "fit_transform"
    assert split_activity["train"]["row_count"] == 3
    assert split_activity["validation"]["action"] == "transform"
    assert split_activity["validation"]["row_count"] == 2


def test_clear_pipeline(fresh_store: SklearnPipelineStore) -> None:
    scaler = StandardScaler()
    fresh_store.register_pipeline("pipe", "node", "scaler", scaler)

    assert fresh_store.list_pipelines()

    fresh_store.clear_pipeline("pipe")
    assert fresh_store.list_pipelines() == []


def test_transformer_audit_prefers_pipeline_store(global_store: SklearnPipelineStore) -> None:
    scaler = StandardScaler()
    global_store.register_pipeline(
        "pipeline-1",
        "node-1",
        "scaler",
        scaler,
        column_name="feature",
        metadata={"input_columns": ["feature"]},
    )
    global_store.record_split_activity(
        "pipeline-1",
        "node-1",
        "scaler",
        split_name="train",
        action="fit_transform",
        column_name="feature",
        row_count=3,
    )

    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    node = {"id": "node-1"}
    node_map = {"node-1": {"label": "Scaling"}}

    _, summary, signal = apply_transformer_audit(
        frame,
        node,
        pipeline_id="pipeline-1",
        node_map=node_map,
    )

    assert summary.startswith("Transformer audit:")
    assert signal.total_transformers == 1
    assert signal.transformers[0].source_node_label == "Scaling"
    actions = {entry.split: entry.action for entry in signal.transformers[0].split_activity}
    assert actions["train"] == "fit_transform"
    assert actions["test"] == "not_available"
    assert actions["validation"] == "not_available"
