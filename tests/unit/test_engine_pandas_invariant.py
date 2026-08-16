"""Regression tests for the pipeline execution engine's pandas-only invariant.

Background: an audit flagged `backend/ml_pipeline/_execution/engine/` as
"pandas-only despite Polars being the default engine", claiming SHAP
explanations and drift-detection baselines were silently disabled on Polars
runs. That turned out to be a false alarm — the engine's injected
`DataCatalog` implementations all read via pandas, and skyulf's
`apply_dual_engine` preserves the input engine type, so no Polars frame can
reach the engine. These tests pin down that reasoning:

1. `_run_data_loader` raises loudly if a catalog ever returns a non-pandas
   frame, instead of letting dozens of downstream `isinstance(x,
   pd.DataFrame)` checks silently no-op.
2. `_normalize_train_frame` accepts a single-column `pd.DataFrame` target
   (a legitimate `split_xy` output shape) instead of silently dropping the
   target column from the saved drift-reference frame.
"""

from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.ml_pipeline._execution.engine._artifacts import ArtifactsMixin
from backend.ml_pipeline._execution.engine._node_runners import NodeRunnersMixin


class _Loader(NodeRunnersMixin):
    """Minimal harness exposing `_run_data_loader` with stubbed collaborators."""

    def __init__(self, loaded: Any) -> None:
        self.catalog = MagicMock()
        self.catalog.load.return_value = loaded
        self.catalog.get_dataset_name.return_value = "ds"
        self.artifact_store = MagicMock()
        self.logs: list[str] = []

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def _pipeline_has_training_node(self) -> bool:
        return False


def _node() -> Any:
    node = MagicMock()
    node.node_id = "n1"
    node.params = {"dataset_id": "ds", "sample": False}
    return node


class _Artifacts(ArtifactsMixin):
    """Minimal harness exposing `_normalize_train_frame`."""


def test_data_loader_accepts_pandas() -> None:
    """A pandas frame from the catalog passes through and is stored."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    loader = _Loader(df)
    assert loader._run_data_loader(_node(), job_id="unknown") == "n1"
    store = cast(MagicMock, loader.artifact_store)
    store.save.assert_called_once_with("n1", df)


def test_data_loader_rejects_non_pandas_frame() -> None:
    """A non-pandas frame must fail loudly rather than silently degrading
    SHAP/drift downstream."""
    pl = pytest.importorskip("polars")
    loader = _Loader(pl.DataFrame({"a": [1, 2]}))
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        loader._run_data_loader(_node(), job_id="unknown")


def test_normalize_train_frame_reattaches_single_column_dataframe_target() -> None:
    """A single-column DataFrame `y` must be squeezed back into the reference
    frame, not silently dropped."""
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.DataFrame({"target": [0, 1, 0]})
    out = _Artifacts()._normalize_train_frame((X, y), target_col="target")
    assert out is not None
    assert "target" in out.columns
    assert out["target"].tolist() == [0, 1, 0]


def test_normalize_train_frame_reattaches_series_target() -> None:
    """The pre-existing Series path keeps working."""
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.Series([0, 1, 0], name="target")
    out = _Artifacts()._normalize_train_frame((X, y), target_col="target")
    assert out is not None
    assert out["target"].tolist() == [0, 1, 0]


def test_normalize_train_frame_ignores_multi_column_dataframe_target() -> None:
    """A multi-column `y` is not a target; X is returned without a bogus column."""
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.DataFrame({"a": [0, 1, 0], "b": [1, 1, 1]})
    out = _Artifacts()._normalize_train_frame((X, y), target_col="target")
    assert out is not None
    assert "target" not in out.columns
