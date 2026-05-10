"""Smoke test: ensure branched + single-branch notebook builders both succeed."""

from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_builders import _NodeIn, _PipelineIn

nodes = [
    _NodeIn(node_id="loader", step_type="data_loader", params={"path": "data.csv"}, inputs=[]),
    _NodeIn(node_id="scaler", step_type="StandardScaler", params={}, inputs=["loader"]),
    _NodeIn(
        node_id="ft1",
        step_type="feature_target_split",
        params={"target_column": "y"},
        inputs=["scaler"],
    ),
    _NodeIn(
        node_id="tts1", step_type="TrainTestSplitter", params={"test_size": 0.2}, inputs=["ft1"]
    ),
    _NodeIn(
        node_id="train_xgb",
        step_type="basic_training",
        params={"algorithm": "xgboost_classifier"},
        inputs=["tts1"],
    ),
    _NodeIn(
        node_id="ft2",
        step_type="feature_target_split",
        params={"target_column": "y"},
        inputs=["scaler"],
    ),
    _NodeIn(
        node_id="tts2", step_type="TrainTestSplitter", params={"test_size": 0.3}, inputs=["ft2"]
    ),
    _NodeIn(
        node_id="train_rf",
        step_type="advanced_tuning",
        params={"algorithm": "random_forest_classifier"},
        inputs=["tts2"],
    ),
]
cfg = _PipelineIn(nodes=nodes)

nb = ne._build_full_notebook(cfg, "test-id", "My Test Pipeline")
print(f"full multi-branch ok: {len(nb['cells'])} cells")
assert any("Branch A" in "".join(c.get("source", [])) for c in nb["cells"])
assert any("Branch B" in "".join(c.get("source", [])) for c in nb["cells"])

nb2 = ne._build_compact_notebook(cfg, "test-id", "My Test Pipeline")
print(f"compact multi-branch ok: {len(nb2['cells'])} cells")
assert any("skyulf_pipeline_A.pkl" in "".join(c.get("source", [])) for c in nb2["cells"])
assert any("skyulf_pipeline_B.pkl" in "".join(c.get("source", [])) for c in nb2["cells"])

single = _PipelineIn(nodes=nodes[:5])
nb3 = ne._build_full_notebook(single, "test-id", "My Test")
print(f"single full ok: {len(nb3['cells'])} cells")

nb4 = ne._build_compact_notebook(single, "test-id", "My Test")
print(f"single compact ok: {len(nb4['cells'])} cells")

# Verify DATA_PATH guidance is always present
for label, n in [
    ("full multi", nb),
    ("compact multi", nb2),
    ("full single", nb3),
    ("compact single", nb4),
]:
    assert any(
        "How to set `DATA_PATH`" in "".join(c.get("source", [])) for c in n["cells"]
    ), f"{label} missing DATA_PATH guidance"
print("All assertions passed.")
