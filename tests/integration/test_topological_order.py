"""OC-69: the engine and schema predictor must not trust ``config.nodes`` list order.

The canvas converter's BFS can emit an acyclic-but-misordered node list (a
merge node is enqueued when *any* parent is dequeued, not all). Before the
fix, such a list made ``predict_schemas`` silently return ``None`` for the
merge node and the engine fail with "Artifact not found" for the upstream
node that hadn't run yet.
"""

from pathlib import Path

import pandas as pd

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.graph_utils import topological_order
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType
from skyulf.preprocessing import SkyulfSchema


def _node(
    node_id: str, step_type: str, inputs: list[str], params: dict | None = None
) -> NodeConfig:
    return NodeConfig(node_id=node_id, step_type=step_type, params=params or {}, inputs=inputs)


def _misordered_diamond_nodes() -> list[NodeConfig]:
    """Diamond with unequal branch depths, listed so the merge precedes a3.

    loader -> a1 -> a2 -> a3 --\
                                  D (inputs=[a3, b])
    loader -> b --------------/
    """
    return [
        _node("loader", StepType.DATA_LOADER, []),
        _node("a1", "SimpleTransformation", ["loader"]),
        _node("b", "SimpleTransformation", ["loader"]),
        _node("a2", "SimpleTransformation", ["a1"]),
        _node("D", "MissingIndicator", ["a3", "b"], {"columns": ["a"]}),
        _node("a3", "SimpleTransformation", ["a2"]),
    ]


# ---------- topological_order unit tests ----------


def test_topological_order_sorts_misordered_diamond() -> None:
    nodes = _misordered_diamond_nodes()
    ordered = topological_order(nodes)

    assert sorted(n.node_id for n in ordered) == sorted(n.node_id for n in nodes)  # same set
    pos = {n.node_id: i for i, n in enumerate(ordered)}
    for node in ordered:
        for parent in node.inputs:
            assert pos[parent] < pos[node.node_id], f"{parent} must precede {node.node_id}"


def test_topological_order_preserves_already_sorted_list() -> None:
    nodes = [
        _node("loader", StepType.DATA_LOADER, []),
        _node("a1", "SimpleTransformation", ["loader"]),
        _node("a2", "SimpleTransformation", ["a1"]),
        _node("D", "MissingIndicator", ["a2"]),
    ]
    assert [n.node_id for n in topological_order(nodes)] == ["loader", "a1", "a2", "D"]


def test_topological_order_is_idempotent() -> None:
    nodes = _misordered_diamond_nodes()
    once = topological_order(nodes)
    twice = topological_order(once)
    assert [n.node_id for n in once] == [n.node_id for n in twice]


# ---------- predict_schemas regression ----------


def _seed_schema() -> SkyulfSchema:
    return SkyulfSchema.from_dataframe(
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": ["x", "y"]})
    )


def test_predict_schemas_handles_misordered_diamond() -> None:
    """The merge node must get a real schema, not a silent None."""
    config = PipelineConfig(pipeline_id="p", nodes=_misordered_diamond_nodes())

    predicted = predict_schemas(config, initial_schemas={"loader": _seed_schema()})

    assert predicted["D"] is not None
    assert predicted["a3"] is not None


# ---------- engine end-to-end regression ----------


def _make_csv(tmp_path: Path) -> str:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "c": ["x", "y", "x", "y", "x", "y"],
        }
    )
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    return str(csv)


def test_engine_runs_misordered_diamond(tmp_path: Path) -> None:
    """A misordered-but-acyclic pipeline must execute, not hit 'Artifact not found'."""
    csv = _make_csv(tmp_path)
    nodes = _misordered_diamond_nodes()
    nodes[0] = _node("loader", StepType.DATA_LOADER, [], {"path": csv})
    config = PipelineConfig(pipeline_id="misordered_diamond", nodes=nodes)

    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    result = engine.run(config)

    assert result.status == "success", result.node_results
    assert "D" in store.list_artifacts()
