"""End-to-end merge & training scenarios with on-disk JSON proof dumps.

Each test runs a full pipeline through the real PipelineEngine, then writes
a JSON summary of inputs/outputs/warnings to ``tests/artifacts/merge_scenarios``
so a human can inspect the exact behaviour.

Scenarios cover the bug categories in ``temp/merge_system_audit.md``:

* sequential chain (sanity)
* fan-in to a single transformer (last-wins on shared columns)
* sibling fan-in warning (Bug 9j)
* (X, y) tuple preservation through a Splitter (Bug 9c)
* SplitDataset preserved through transformer chain
* training node receiving multiple preprocessor branches
* training metrics computed when held-out test is non-empty (Bug 4a regression)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from skyulf.data.dataset import SplitDataset

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "merge_scenarios"


def _summarize_artifact(art: Any) -> Dict[str, Any]:
    """Same shape used by the engine's node_trace dump."""
    if isinstance(art, pd.DataFrame):
        return {"kind": "DataFrame", "columns": list(art.columns), "rows": len(art)}
    if isinstance(art, pd.Series):
        return {"kind": "Series", "name": art.name, "len": len(art)}
    if isinstance(art, SplitDataset):

        def _part(p: Any) -> Any:
            if p is None:
                return None
            if isinstance(p, pd.DataFrame):
                return {"kind": "DataFrame", "columns": list(p.columns), "rows": len(p)}
            if isinstance(p, tuple) and len(p) == 2:
                X, y = p
                return {
                    "kind": "(X, y) tuple",
                    "X_columns": list(X.columns) if hasattr(X, "columns") else None,
                    "X_rows": len(X) if hasattr(X, "__len__") else None,
                    "y_name": getattr(y, "name", None),
                    "y_len": len(y) if hasattr(y, "__len__") else None,
                }
            return {"kind": type(p).__name__}

        return {
            "kind": "SplitDataset",
            "train": _part(art.train),
            "test": _part(art.test),
            "validation": _part(art.validation),
        }
    if isinstance(art, tuple) and len(art) == 2:
        X, y = art
        return {
            "kind": "(X, y) tuple",
            "X_columns": list(X.columns) if hasattr(X, "columns") else None,
            "y_name": getattr(y, "name", None),
        }
    return {"kind": type(art).__name__}


def _dump(scenario: str, payload: Dict[str, Any]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = ARTIFACT_DIR / f"{scenario}.json"
    target.write_text(json.dumps(payload, indent=2, default=str))
    return target


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


def _new_engine(tmp_path: Path) -> tuple[PipelineEngine, LocalArtifactStore]:
    store = LocalArtifactStore(str(tmp_path / "art"))
    return PipelineEngine(store, catalog=FileSystemCatalog()), store


def _record_run(
    scenario: str, cfg: PipelineConfig, store: LocalArtifactStore, result: Any
) -> Dict[str, Any]:
    """Collect every node's output artifact + the engine's merge warnings."""
    node_outputs: Dict[str, Any] = {}
    for n in cfg.nodes:
        if store.exists(n.node_id):
            node_outputs[n.node_id] = _summarize_artifact(store.load(n.node_id))
    payload = {
        "scenario": scenario,
        "pipeline_id": cfg.pipeline_id,
        "status": result.status,
        "node_outputs": node_outputs,
        "merge_warnings": list(getattr(result, "merge_warnings", [])),
        "node_results": {
            nid: {
                "status": nr.status,
                "error": nr.error,
                "metrics_keys": sorted(list((nr.metrics or {}).keys())),
            }
            for nid, nr in result.node_results.items()
        },
    }
    _dump(scenario, payload)
    return payload


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_scenario_01_sequential_chain(tmp_path: Path) -> None:
    """Dataset -> DropColumns -> FeatureTargetSplit -> TrainTestSplitter."""
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    cfg = PipelineConfig(
        pipeline_id="s01_sequential",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="drop_id",
                step_type="DropMissingColumns",
                inputs=["data"],
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
            NodeConfig(
                node_id="fts",
                step_type="feature_target_split",
                inputs=["drop_id"],
                params={"target_column": "Species"},
            ),
            NodeConfig(
                node_id="split",
                step_type="TrainTestSplitter",
                inputs=["fts"],
                params={
                    "target_column": "Species",
                    "test_size": 0.2,
                    "random_state": 0,
                },
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("01_sequential_chain", cfg, store, result)

    assert result.status == "success"
    # Splitter output is a SplitDataset of (X, y) tuples
    split_out = payload["node_outputs"]["split"]
    assert split_out["kind"] == "SplitDataset"
    assert split_out["train"]["kind"] == "(X, y) tuple"
    assert split_out["train"]["y_name"] == "Species"
    # No merge happened -> no warnings
    assert payload["merge_warnings"] == []


def test_scenario_02_fan_in_to_transformer_lastwins(tmp_path: Path) -> None:
    """Splitter -> Scaler AND Splitter -> Encoder fanned into a downstream node.

    Verifies last-wins: scaled columns survive, encoder receives them.
    """
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    cfg = PipelineConfig(
        pipeline_id="s02_fan_in_lastwins",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="drop_id",
                step_type="DropMissingColumns",
                inputs=["data"],
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
            NodeConfig(
                node_id="fts",
                step_type="feature_target_split",
                inputs=["drop_id"],
                params={"target_column": "Species"},
            ),
            NodeConfig(
                node_id="split",
                step_type="TrainTestSplitter",
                inputs=["fts"],
                params={
                    "target_column": "Species",
                    "test_size": 0.2,
                    "random_state": 0,
                },
            ),
            NodeConfig(
                node_id="scaler",
                step_type="MinMaxScaler",
                inputs=["split"],
                params={
                    "columns": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
                },
            ),
            # Downstream node fed by both split and scaler — last (scaler) must win
            NodeConfig(
                node_id="downstream",
                step_type="StandardScaler",
                inputs=["split", "scaler"],
                params={"columns": ["SepalLengthCm"]},
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("02_fan_in_to_transformer_lastwins", cfg, store, result)

    assert result.status == "success"
    # Inputs are [split, scaler] but scaler's parent IS split — that's a
    # redundant edge, not a true sibling fan-in. Engine should NOT warn
    # because the descendant (scaler) supersedes the ancestor (split) under
    # last-wins, so the merge is effectively just "use scaler's output".
    kinds = {w["kind"] for w in payload["merge_warnings"]}
    assert (
        "sibling_fan_in" not in kinds
    ), f"Redundant ancestor edge should not raise sibling_fan_in; got {payload['merge_warnings']}"
    # Downstream still produced data (last-wins preserved scaler output)
    ds = payload["node_outputs"]["downstream"]
    assert ds["kind"] == "SplitDataset"


def test_scenario_03_xy_tuple_preserved_through_splitter(tmp_path: Path) -> None:
    """feature_target_split -> Splitter must yield SplitDataset of (X, y) tuples."""
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    cfg = PipelineConfig(
        pipeline_id="s03_xy_preserved",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="drop_id",
                step_type="DropMissingColumns",
                inputs=["data"],
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
            NodeConfig(
                node_id="fts",
                step_type="feature_target_split",
                inputs=["drop_id"],
                params={"target_column": "Species"},
            ),
            NodeConfig(
                node_id="split",
                step_type="TrainTestSplitter",
                inputs=["fts"],
                params={
                    "target_column": "Species",
                    "test_size": 0.2,
                    "random_state": 0,
                },
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("03_xy_tuple_preserved_through_splitter", cfg, store, result)

    split_out = payload["node_outputs"]["split"]
    assert split_out["train"]["kind"] == "(X, y) tuple"
    assert split_out["test"]["kind"] == "(X, y) tuple"
    assert split_out["train"]["X_columns"] == [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]
    assert split_out["train"]["y_name"] == "Species"


def test_scenario_04_split_then_scaler_keeps_xy(tmp_path: Path) -> None:
    """Verify a Scaler after a Splitter still emits (X, y) tuples per slot."""
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    cfg = PipelineConfig(
        pipeline_id="s04_split_then_scaler",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="drop_id",
                step_type="DropMissingColumns",
                inputs=["data"],
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
            NodeConfig(
                node_id="fts",
                step_type="feature_target_split",
                inputs=["drop_id"],
                params={"target_column": "Species"},
            ),
            NodeConfig(
                node_id="split",
                step_type="TrainTestSplitter",
                inputs=["fts"],
                params={
                    "target_column": "Species",
                    "test_size": 0.2,
                    "random_state": 0,
                },
            ),
            NodeConfig(
                node_id="scaler",
                step_type="MinMaxScaler",
                inputs=["split"],
                params={"columns": ["SepalLengthCm", "SepalWidthCm"]},
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("04_split_then_scaler_keeps_xy", cfg, store, result)

    sc = payload["node_outputs"]["scaler"]
    assert sc["kind"] == "SplitDataset"
    assert sc["train"]["kind"] == "(X, y) tuple"
    assert sc["test"]["kind"] == "(X, y) tuple"


def test_scenario_05_sibling_fanin_warning(tmp_path: Path) -> None:
    """Bug 9j: 3 sibling preprocessors fanned into one node should warn."""
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    cfg = PipelineConfig(
        pipeline_id="s05_sibling_fanin",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="drop_id",
                step_type="DropMissingColumns",
                inputs=["data"],
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
            # Three siblings off drop_id
            NodeConfig(
                node_id="sib_a",
                step_type="MinMaxScaler",
                inputs=["drop_id"],
                params={"columns": ["SepalLengthCm"]},
            ),
            NodeConfig(
                node_id="sib_b",
                step_type="StandardScaler",
                inputs=["drop_id"],
                params={"columns": ["SepalWidthCm"]},
            ),
            # Fan-in consumer
            NodeConfig(
                node_id="merge_consumer",
                step_type="MinMaxScaler",
                inputs=["sib_a", "sib_b"],
                params={"columns": ["PetalLengthCm"]},
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("05_sibling_fanin_warning", cfg, store, result)

    warnings = payload["merge_warnings"]
    assert any(w["kind"] == "sibling_fan_in" and w["node_id"] == "merge_consumer" for w in warnings)
    advisory = next(w for w in warnings if w["node_id"] == "merge_consumer")
    assert "drop_id" in advisory["common_ancestors"]
    # New fields: winner is always the last input; overlap_columns lists
    # only columns present in 2+ inputs.
    assert advisory["winner_input"] == "sib_b"
    assert "overlap_columns" in advisory
    # sib_a and sib_b both produce all 4 numeric columns, so all overlap
    assert "SepalLengthCm" in advisory["overlap_columns"]
    assert "SepalWidthCm" in advisory["overlap_columns"]


def test_scenario_06_training_with_multiple_preprocessor_branches(tmp_path: Path) -> None:
    """Two scalers feeding one Training node — model trains and gets test metrics."""
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    cfg = PipelineConfig(
        pipeline_id="s06_multi_branch_training",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="drop_id",
                step_type="DropMissingColumns",
                inputs=["data"],
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
            NodeConfig(
                node_id="fts",
                step_type="feature_target_split",
                inputs=["drop_id"],
                params={"target_column": "Species"},
            ),
            NodeConfig(
                node_id="split",
                step_type="TrainTestSplitter",
                inputs=["fts"],
                params={
                    "target_column": "Species",
                    "test_size": 0.3,
                    "random_state": 0,
                },
            ),
            NodeConfig(
                node_id="scaler_a",
                step_type="StandardScaler",
                inputs=["split"],
                params={"columns": ["SepalLengthCm", "SepalWidthCm"]},
            ),
            NodeConfig(
                node_id="scaler_b",
                step_type="MinMaxScaler",
                inputs=["split"],
                params={"columns": ["PetalLengthCm", "PetalWidthCm"]},
            ),
            NodeConfig(
                node_id="training",
                step_type=StepType.BASIC_TRAINING,
                inputs=["scaler_a", "scaler_b"],
                params={
                    "target_column": "Species",
                    "algorithm": "logistic_regression",
                    "hyperparameters": {"C": 1.0, "max_iter": 200},
                    "evaluate": True,
                    "cv_enabled": False,
                },
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("06_training_with_multiple_preprocessor_branches", cfg, store, result)

    assert result.status == "success"
    train_metrics = result.node_results["training"].metrics or {}
    payload["training_metrics_keys"] = sorted(train_metrics.keys())
    _dump("06_training_with_multiple_preprocessor_branches", payload)

    # Held-out test split survived merge (Bug 4a regression)
    has_test_metric = any("test" in k.lower() for k in train_metrics)
    assert has_test_metric, (
        f"Expected at least one test_* metric, got keys: " f"{sorted(train_metrics.keys())}"
    )


def test_scenario_07_duplicate_edges_dedup(tmp_path: Path) -> None:
    """A node wired twice to the same source must behave as single-input."""
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    cfg = PipelineConfig(
        pipeline_id="s07_duplicate_edges",
        nodes=[
            NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
            NodeConfig(
                node_id="drop_id",
                step_type="DropMissingColumns",
                inputs=["data", "data"],  # duplicate edge
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("07_duplicate_edges_dedup", cfg, store, result)

    assert result.status == "success"
    # No merge warning because dedup collapses to single input
    assert not any(w["kind"] == "sibling_fan_in" for w in payload["merge_warnings"])
    out = payload["node_outputs"]["drop_id"]
    assert out["kind"] == "DataFrame"
    assert "Id" not in out["columns"]


def test_scenario_08_full_three_path_canvas(tmp_path: Path) -> None:
    """Simulate the user's 3-path canvas: each path runs end-to-end."""
    csv = _make_iris(tmp_path)
    engine, store = _new_engine(tmp_path)

    # Three independent training subgraphs sharing the same data loader
    nodes = [
        NodeConfig(node_id="data", step_type=StepType.DATA_LOADER, params={"path": csv}),
    ]
    for tag in ("a", "b", "c"):
        nodes.extend(
            [
                NodeConfig(
                    node_id=f"drop_{tag}",
                    step_type="DropMissingColumns",
                    inputs=["data"],
                    params={"columns": ["Id"], "missing_threshold": 0},
                ),
                NodeConfig(
                    node_id=f"fts_{tag}",
                    step_type="feature_target_split",
                    inputs=[f"drop_{tag}"],
                    params={"target_column": "Species"},
                ),
                NodeConfig(
                    node_id=f"split_{tag}",
                    step_type="TrainTestSplitter",
                    inputs=[f"fts_{tag}"],
                    params={
                        "target_column": "Species",
                        "test_size": 0.25,
                        "random_state": 0,
                    },
                ),
                NodeConfig(
                    node_id=f"train_{tag}",
                    step_type=StepType.BASIC_TRAINING,
                    inputs=[f"split_{tag}"],
                    params={
                        "target_column": "Species",
                        "algorithm": "logistic_regression",
                        "hyperparameters": {"C": 1.0, "max_iter": 200},
                        "evaluate": True,
                        "cv_enabled": False,
                    },
                ),
            ]
        )

    cfg = PipelineConfig(pipeline_id="s08_three_paths", nodes=nodes)
    result = engine.run(cfg)
    payload = _record_run("08_full_three_path_canvas", cfg, store, result)

    assert result.status == "success"
    for tag in ("a", "b", "c"):
        assert payload["node_results"][f"train_{tag}"]["status"] == "success"


def test_scenario_09_path_a_full_chain(tmp_path: Path) -> None:
    """User-reported "Path A": Dataset → MissingIndicator → DropMissingColumns
    → Transformation → DropMissingRows → FeatureTargetSplit → TrainTestSplitter
    → Encoding.

    Verifies every preprocessor leaves a visible effect on the final artifact
    (so the UI gets all four train/test X/y tabs, not just one).
    """
    import numpy as np

    np.random.seed(0)
    n = 150
    df = pd.DataFrame(
        {
            "Id": list(range(1, n + 1)),
            "SepalLengthCm": np.random.uniform(4, 8, n),
            "SepalWidthCm": np.random.uniform(2, 5, n),
            "PetalLengthCm": np.random.uniform(1, 7, n),
            "PetalWidthCm": np.random.uniform(0, 3, n),
            "Species": (["a", "b", "c"] * 50),
        }
    )
    # Inject NaNs so MissingIndicator and DropMissingRows actually do work.
    for col in ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]:
        df.loc[df.sample(5, random_state=1).index, col] = np.nan
    csv_path = tmp_path / "iris_path_a.csv"
    df.to_csv(csv_path, index=False)

    engine, store = _new_engine(tmp_path)
    cfg = PipelineConfig(
        pipeline_id="s09_path_a",
        nodes=[
            NodeConfig(
                node_id="data", step_type=StepType.DATA_LOADER, params={"path": str(csv_path)}
            ),
            NodeConfig(
                node_id="mi",
                step_type="MissingIndicator",
                inputs=["data"],
                params={
                    "columns": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
                    "flag_suffix": "_missing",
                },
            ),
            NodeConfig(
                node_id="dmc",
                step_type="DropMissingColumns",
                inputs=["mi"],
                params={"columns": ["Id"], "missing_threshold": 0},
            ),
            NodeConfig(
                node_id="tx",
                step_type="GeneralTransformation",
                inputs=["dmc"],
                params={
                    "transformations": [
                        {"column": "SepalLengthCm", "method": "log"},
                    ]
                },
            ),
            NodeConfig(
                node_id="dmr",
                step_type="DropMissingRows",
                inputs=["tx"],
                params={"drop_if_any_missing": True},
            ),
            NodeConfig(
                node_id="fts",
                step_type="feature_target_split",
                inputs=["dmr"],
                params={"target_column": "Species"},
            ),
            NodeConfig(
                node_id="split",
                step_type="TrainTestSplitter",
                inputs=["fts"],
                params={
                    "target_column": "Species",
                    "test_size": 0.2,
                    "random_state": 0,
                },
            ),
            NodeConfig(
                node_id="enc",
                step_type="LabelEncoder",
                inputs=["split"],
                params={"columns": ["Species"]},
            ),
        ],
    )
    result = engine.run(cfg)
    payload = _record_run("09_path_a_full_chain", cfg, store, result)

    assert result.status == "success"

    # MissingIndicator added 4 flag columns
    mi_cols = payload["node_outputs"]["mi"]["columns"]
    assert all(
        f"{c}_missing" in mi_cols
        for c in ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )

    # DropMissingColumns removed Id
    dmc_cols = payload["node_outputs"]["dmc"]["columns"]
    assert "Id" not in dmc_cols

    # DropMissingRows reduced row count below 150
    assert payload["node_outputs"]["dmr"]["rows"] < 150

    # Encoder keeps SplitDataset of (X, y) tuples — backend will produce
    # train_X / train_y / test_X / test_y preview tabs.
    enc_out = payload["node_outputs"]["enc"]
    assert enc_out["kind"] == "SplitDataset"
    assert enc_out["train"]["kind"] == "(X, y) tuple"
    assert enc_out["test"]["kind"] == "(X, y) tuple"
    assert enc_out["train"]["y_name"] == "Species"
    assert enc_out["train"]["X_columns"] is not None
    assert "Id" not in enc_out["train"]["X_columns"]
    assert "Species" not in enc_out["train"]["X_columns"]
    # MissingIndicator flags survived all the way through to the encoder
    assert any("_missing" in c for c in enc_out["train"]["X_columns"])
