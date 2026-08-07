"""Merge input ordering must match what the fan-in advisory reports.

Regression guard for the bug where ``_resolve_all_inputs`` sorted merge inputs
by their index in the ``nodes`` array while the advisory derived its winner from
the last entry in ``inputs``. Moving a node in the canvas could then silently
discard a branch the banner claimed had won.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType

SEPALS = ["SepalLengthCm", "SepalWidthCm"]


def _make_iris(tmp_path: Path) -> str:
    df = pd.DataFrame(
        {
            "Id": list(range(1, 151)),
            "SepalLengthCm": [5.1 + (i % 10) * 0.1 for i in range(150)],
            "SepalWidthCm": [3.0 + (i % 7) * 0.1 for i in range(150)],
            "PetalLengthCm": [1.4 + (i % 5) * 0.2 for i in range(150)],
            "PetalWidthCm": [0.2 + (i % 4) * 0.1 for i in range(150)],
            "Species": ["a", "b", "c"] * 50,
        }
    )
    csv = tmp_path / "iris.csv"
    df.to_csv(csv, index=False)
    return str(csv)


def _build_config(csv: str, *, transformation_first: bool) -> PipelineConfig:
    """Two sibling branches fan into one node; only the nodes-array order varies."""
    data = NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv})
    drop_missing = NodeConfig(
        node_id="drop_missing",
        step_type="DropMissingColumns",
        inputs=["data"],
        params={"columns": ["Id"], "missing_threshold": 0},
    )
    transformation = NodeConfig(
        node_id="transformation",
        step_type="SimpleTransformation",
        inputs=["data"],
        params={"transformations": [{"column": c, "method": "log"} for c in SEPALS]},
    )
    branches = (
        [transformation, drop_missing] if transformation_first else [drop_missing, transformation]
    )
    return PipelineConfig(
        pipeline_id="merge_input_order",
        nodes=[
            data,
            *branches,
            NodeConfig(
                node_id="missing_indicator",
                step_type="MissingIndicator",
                inputs=["drop_missing", "transformation"],
                params={"columns": ["PetalLengthCm"]},
            ),
        ],
    )


def _merged_frame(payload: Any) -> pd.DataFrame:
    return payload if isinstance(payload, pd.DataFrame) else payload.train


@pytest.mark.parametrize("transformation_first", [False, True])
def test_last_wins_follows_edge_order_not_nodes_array(
    tmp_path: Path, transformation_first: bool
) -> None:
    csv = _make_iris(tmp_path)
    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    result = engine.run(_build_config(csv, transformation_first=transformation_first))
    assert result.status == "success"

    advisory = next(w for w in result.merge_warnings if w.get("kind") == "sibling_fan_in")
    assert advisory["winner_input"] == "transformation"

    expected = np.log1p(pd.read_csv(csv)["SepalLengthCm"].to_numpy())
    actual = _merged_frame(store.load("missing_indicator"))["SepalLengthCm"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_upstream_drop_warning_carries_inputs(tmp_path: Path) -> None:
    """The banner needs `inputs`; without it the warning renders as an empty merge."""
    csv = _make_iris(tmp_path)
    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    result = engine.run(_build_config(csv, transformation_first=False))

    warning = next(w for w in result.merge_warnings if w.get("kind") == "upstream_drop_reapplied")
    assert warning["inputs"] == ["drop_missing", "transformation"]
    assert warning["dropped_columns"] == ["Id"]
