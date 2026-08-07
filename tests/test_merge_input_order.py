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
def test_transform_survives_regardless_of_nodes_array_order(
    tmp_path: Path, transformation_first: bool
) -> None:
    """Moving a node in the canvas must not change which branch's values survive."""
    csv = _make_iris(tmp_path)
    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    result = engine.run(_build_config(csv, transformation_first=transformation_first))
    assert result.status == "success"

    expected = np.log1p(pd.read_csv(csv)["SepalLengthCm"].to_numpy())
    actual = _merged_frame(store.load("missing_indicator"))["SepalLengthCm"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_unchanged_passthrough_column_does_not_beat_a_real_transform(tmp_path: Path) -> None:
    """A branch that merely carries a column through must not overwrite one that rewrote it."""
    csv = _make_iris(tmp_path)
    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    # first_wins would hand the sepal columns to the pass-through branch under
    # the old order-based rule; ownership must override that.
    cfg = _build_config(csv, transformation_first=True)
    merge_node = next(n for n in cfg.nodes if n.node_id == "missing_indicator")
    merge_node.params["_merge_strategy"] = "first_wins"

    result = engine.run(cfg)
    assert result.status == "success"

    expected = np.log1p(pd.read_csv(csv)["SepalLengthCm"].to_numpy())
    actual = _merged_frame(store.load("missing_indicator"))["SepalLengthCm"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_no_winner_advisory_when_branches_do_not_contest_a_column(tmp_path: Path) -> None:
    """No branch conflict means no winner, so the canvas shows no 'wins merge' label."""
    csv = _make_iris(tmp_path)
    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    result = engine.run(_build_config(csv, transformation_first=False))

    assert not [w for w in result.merge_warnings if w.get("kind") == "sibling_fan_in"]


def test_two_branches_editing_the_same_column_still_report_a_winner(tmp_path: Path) -> None:
    """A real contest keeps the advisory so the discarded edit stays visible."""
    csv = _make_iris(tmp_path)
    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    cfg = PipelineConfig(
        pipeline_id="contested_column",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="log_branch",
                step_type="SimpleTransformation",
                inputs=["data"],
                params={"transformations": [{"column": "SepalLengthCm", "method": "log"}]},
            ),
            NodeConfig(
                node_id="scale_branch",
                step_type="MinMaxScaler",
                inputs=["data"],
                params={"columns": ["SepalLengthCm"]},
            ),
            NodeConfig(
                node_id="merge_node",
                step_type="MissingIndicator",
                inputs=["log_branch", "scale_branch"],
                params={"columns": ["PetalLengthCm"]},
            ),
        ],
    )
    result = engine.run(cfg)
    assert result.status == "success"

    advisory = next(w for w in result.merge_warnings if w.get("kind") == "sibling_fan_in")
    assert advisory["overlap_columns"] == ["SepalLengthCm"]
    assert advisory["winner_input"] == "scale_branch"


def test_upstream_drop_warning_carries_inputs(tmp_path: Path) -> None:
    """The banner needs `inputs`; without it the warning renders as an empty merge."""
    csv = _make_iris(tmp_path)
    store = LocalArtifactStore(str(tmp_path / "art"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())

    result = engine.run(_build_config(csv, transformation_first=False))

    warning = next(w for w in result.merge_warnings if w.get("kind") == "upstream_drop_reapplied")
    assert warning["inputs"] == ["drop_missing", "transformation"]
    assert warning["dropped_columns"] == ["Id"]
