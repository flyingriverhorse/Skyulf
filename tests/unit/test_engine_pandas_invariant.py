"""Regression tests for the pipeline execution engine's frame-type guard.

Background: an audit flagged `backend/ml_pipeline/_execution/engine/` as
"pandas-only despite Polars being the default engine". The engine originally
enforced a hard pandas-only invariant at the ingestion choke point. With the
backend Polars migration (Phase 2), the guard became engine-aware:

1. `_run_data_loader` accepts a Polars frame when `SKYULF_ENGINE=polars`
   (the platform default), keeps accepting pandas, and still raises loudly on
   an engine mismatch (e.g. a Polars frame while `SKYULF_ENGINE=pandas`) or
   on a non-frame object — so downstream `isinstance(x, pd.DataFrame)`
   assumptions can never silently no-op.
2. `_normalize_train_frame` accepts a single-column `pd.DataFrame` target
   (a legitimate `split_xy` output shape) instead of silently dropping the
   target column from the saved drift-reference frame.
"""

from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.config import get_settings
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


def test_data_loader_accepts_polars_when_engine_polars(monkeypatch) -> None:
    """With SKYULF_ENGINE=polars (the default), a Polars frame passes through."""
    pl = pytest.importorskip("polars")
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    frame = pl.DataFrame({"a": [1, 2]})
    loader = _Loader(frame)
    assert loader._run_data_loader(_node(), job_id="unknown") == "n1"
    store = cast(MagicMock, loader.artifact_store)
    store.save.assert_called_once_with("n1", frame)


def test_data_loader_rejects_polars_frame_when_engine_pandas(monkeypatch) -> None:
    """An engine mismatch must fail loudly rather than silently degrading
    SHAP/drift downstream."""
    pl = pytest.importorskip("polars")
    settings = get_settings()
    monkeypatch.setattr(settings, "SKYULF_ENGINE", "pandas", raising=False)
    loader = _Loader(pl.DataFrame({"a": [1, 2]}))
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        loader._run_data_loader(_node(), job_id="unknown")


def test_data_loader_rejects_non_frame_objects(monkeypatch) -> None:
    """Anything that is neither a pandas nor a Polars frame fails loudly."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    loader = _Loader([{"a": 1}])
    with pytest.raises(TypeError, match=r"requires a pandas or Polars DataFrame"):
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


def test_normalize_train_frame_ignores_multi_column_polars_dataframe_target() -> None:
    """A multi-column Polars `y` is not a target; X is returned without a
    bogus column — mirroring the pandas guard."""
    pl = pytest.importorskip("polars")
    X = pl.DataFrame({"f": [1, 2, 3]})
    y = pl.DataFrame({"a": [0, 1, 0], "b": [1, 1, 1]})
    out = _Artifacts()._normalize_train_frame((X, y), target_col="target")
    assert out is not None
    assert "target" not in out.columns
    assert list(out.columns) == ["f"]


def test_normalize_train_frame_returns_none_for_non_frame_tuple() -> None:
    """A (X, y) tuple whose X is neither a pandas nor a Polars frame cannot
    be normalized into a drift-reference frame; return None."""
    import numpy as np

    out = _Artifacts()._normalize_train_frame(
        (np.array([[1.0], [2.0]]), [0, 1]), target_col="target"
    )
    assert out is None


def test_normalize_train_frame_polars_skips_unrecognized_target_type() -> None:
    """A Polars X with a y of an unrecognized type is returned without a
    target column instead of raising or fabricating data."""
    pl = pytest.importorskip("polars")
    X = pl.DataFrame({"f": [1, 2, 3]})
    out = _Artifacts()._normalize_train_frame((X, "not-a-target"), target_col="target")
    assert out is not None
    assert list(out.columns) == ["f"]


def test_save_reference_data_skips_empty_polars_frame() -> None:
    """An empty training frame must not be persisted as a drift reference."""
    pl = pytest.importorskip("polars")

    class _Store(ArtifactsMixin):
        def __init__(self) -> None:
            self.artifact_store = MagicMock()
            self.dataset_name = "ds"
            self.logs: list[str] = []

        def log(self, msg: str) -> None:
            self.logs.append(msg)

    store = _Store()
    store._persist_reference_frame(pl.DataFrame(), job_id="job-empty")
    store.artifact_store.save.assert_not_called()


def test_data_preview_accepts_plain_polars_frame() -> None:
    """A bare Polars frame input must produce a fit_transform preview with a
    'full' data summary, not fall through with an unknown operation mode."""
    pl = pytest.importorskip("polars")

    class _Preview(NodeRunnersMixin):
        def __init__(self, data: Any) -> None:
            self._data = data
            self.artifact_store = MagicMock()
            self.executed_transformers: list = []

        def _get_input(self, node: Any) -> Any:
            return self._data

    runner = _Preview(pl.DataFrame({"a": [1, 2]}))
    node_id, info = runner._run_data_preview(_node())

    assert node_id == "n1"
    assert info["operation_mode"] == "fit_transform"
    assert info["data_summary"]["full"]["shape"] == (2, 1)


def test_data_preview_leaves_unknown_mode_for_non_frame_input() -> None:
    """An input shape that is neither SplitDataset nor a frame must not be
    mis-described; operation_mode stays unknown."""

    class _Preview(NodeRunnersMixin):
        def __init__(self, data: Any) -> None:
            self._data = data
            self.artifact_store = MagicMock()
            self.executed_transformers: list = []

        def _get_input(self, node: Any) -> Any:
            return self._data

    runner = _Preview((pd.DataFrame({"a": [1]}), pd.Series([0])))
    _, info = runner._run_data_preview(_node())

    assert info["operation_mode"] == "unknown"
    assert info["data_summary"] == {}


# ── F-31/F-32: polars frames must produce drift references + feature names ──


def test_normalize_train_frame_accepts_polars_frame() -> None:
    """F-31: a polars training frame must yield a reference frame, not None
    (None silently disables drift detection under the polars engine)."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"f": [1, 2, 3], "target": [0, 1, 0]})
    out = _Artifacts()._normalize_train_frame(df, target_col="target")
    assert out is not None
    assert isinstance(out, pl.DataFrame)
    assert list(out.columns) == ["f", "target"]


def test_normalize_train_frame_accepts_polars_split_dataset() -> None:
    pl = pytest.importorskip("polars")
    from skyulf.data.dataset import SplitDataset

    sd = SplitDataset(
        train=pl.DataFrame({"f": [1, 2], "target": [0, 1]}),
        test=pl.DataFrame(),
        validation=None,
    )
    out = _Artifacts()._normalize_train_frame(sd, target_col="target")
    assert out is not None
    assert isinstance(out, pl.DataFrame)


def test_normalize_train_frame_reattaches_target_for_polars_xy_tuple() -> None:
    pl = pytest.importorskip("polars")
    import numpy as np

    X = pl.DataFrame({"f": [1, 2, 3]})
    y = np.array([0, 1, 0])
    out = _Artifacts()._normalize_train_frame((X, y), target_col="target")
    assert out is not None
    assert "target" in out.columns
    assert out["target"].to_list() == [0, 1, 0]


def test_normalize_train_frame_reattaches_polars_single_column_dataframe_target() -> None:
    """A single-column Polars DataFrame `y` (a legitimate `split_xy` output)
    must be squeezed back into the reference frame, mirroring the pandas path."""
    pl = pytest.importorskip("polars")
    X = pl.DataFrame({"f": [1, 2, 3]})
    y = pl.DataFrame({"target": [0, 1, 0]})
    out = _Artifacts()._normalize_train_frame((X, y), target_col="target")
    assert out is not None
    assert "target" in out.columns
    assert out["target"].to_list() == [0, 1, 0]


def test_preview_slot_info_accepts_polars_frame() -> None:
    """A non-empty Polars test/validation slot must produce a preview, not None."""
    pl = pytest.importorskip("polars")
    runner = _Loader(pd.DataFrame())
    info = runner._preview_slot_info(pl.DataFrame({"a": [1, 2]}), "test")
    assert info is not None
    assert info["name"] == "test"
    assert info["shape"] == (2, 1)
    assert info["columns"] == ["a"]


def test_feature_names_for_importance_accepts_polars() -> None:
    """F-32: feature-importance resolution must not return [] for polars frames."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"f1": [1.0], "f2": [2.0], "target": [0]})
    assert _Artifacts()._feature_names_for_importance(df, "target") == ["f1", "f2"]


def test_save_reference_data_persists_polars_frame() -> None:
    """F-31 end-to-end: a polars training frame must reach the artifact store
    as the drift reference, not be dropped as 'unsupported data shape'."""
    pl = pytest.importorskip("polars")

    class _Store(ArtifactsMixin):
        def __init__(self) -> None:
            self.artifact_store = MagicMock()
            self.dataset_name = "ds"
            self.logs: list[str] = []

        def log(self, msg: str) -> None:
            self.logs.append(msg)

    store = _Store()
    df = pl.DataFrame({"f": [1, 2, 3], "target": [0, 1, 0]})
    store._save_reference_data(df, job_id="job-9", target_col="target")
    store.artifact_store.save.assert_called_once_with("reference_data_ds_job-9", df)
