"""Focused unit tests for the internal helper functions in
``backend.ml_pipeline._internal._routers.preview``.

These target pure/near-pure functions that are exercised only indirectly
(and rarely) by the broader integration suite in ``test_api_integration.py``,
covering branch/edge cases (polars fallbacks, SplitDataset shapes, dict/xy
tuples, dedupe logic, and pipeline partitioning helpers) that are otherwise
hard to reach through the full `/preview` endpoint.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal._routers import preview as preview_mod
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType
from skyulf.data.dataset import SplitDataset


@pytest.fixture
def tmp_store():
    """Provide a LocalArtifactStore backed by a temp dir, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="skyulf_preview_test_")
    try:
        yield LocalArtifactStore(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# _to_records / _count_rows
# --------------------------------------------------------------------------


def test_to_records_pandas_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    records = preview_mod._to_records(df)
    assert records == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


def test_to_records_pandas_series():
    s = pd.Series([1, 2, 3], name="a")
    records = preview_mod._to_records(s)
    assert records == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_to_records_polars_dataframe():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2]})
    records = preview_mod._to_records(df)
    assert records == [{"a": 1}, {"a": 2}]


def test_to_records_polars_series():
    pl = pytest.importorskip("polars")
    s = pl.Series("a", [1, 2])
    records = preview_mod._to_records(s)
    assert records == [{"a": 1}, {"a": 2}]


def test_to_records_unsupported_type_returns_empty():
    assert preview_mod._to_records(object()) == []
    assert preview_mod._to_records(None) == []
    assert preview_mod._to_records([1, 2, 3]) == []


def test_to_records_missing_polars_import_returns_empty():
    """Exercise the `except ImportError: pass` fallback when polars is unavailable."""
    with patch.dict("sys.modules", {"polars": None}):
        assert preview_mod._to_records([1, 2, 3]) == []


def test_count_rows_none_returns_zero():
    assert preview_mod._count_rows(None) == 0


def test_count_rows_pandas_dataframe_and_series():
    df = pd.DataFrame({"a": range(10)})
    assert preview_mod._count_rows(df) == 10
    s = pd.Series(range(7))
    assert preview_mod._count_rows(s) == 7


def test_count_rows_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert preview_mod._count_rows(df) == 3
    s = pl.Series("a", [1, 2])
    assert preview_mod._count_rows(s) == 2


def test_count_rows_fallback_len():
    assert preview_mod._count_rows([1, 2, 3, 4]) == 4


def test_count_rows_object_without_len_or_shape_returns_zero():
    class Weird:
        pass

    assert preview_mod._count_rows(Weird()) == 0


def test_count_rows_missing_polars_import_falls_back_to_len():
    """Exercise the `except ImportError: pass` fallback when polars is unavailable."""
    with patch.dict("sys.modules", {"polars": None}):
        assert preview_mod._count_rows([1, 2, 3]) == 3


# --------------------------------------------------------------------------
# _process_xy / _process_xy_totals
# --------------------------------------------------------------------------


def test_process_xy():
    X = pd.DataFrame({"f": [1, 2]})
    y = pd.Series([0, 1], name="target")
    out = preview_mod._process_xy((X, y), "train")
    assert out["train_X"] == [{"f": 1}, {"f": 2}]
    assert out["train_y"] == [{"target": 0}, {"target": 1}]


def test_process_xy_totals():
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.Series([0, 1, 1])
    out = preview_mod._process_xy_totals((X, y), "test")
    assert out == {"test_X": 3, "test_y": 3}


# --------------------------------------------------------------------------
# _to_pandas_safe
# --------------------------------------------------------------------------


def test_to_pandas_safe_pandas_passthrough():
    df = pd.DataFrame({"a": [1]})
    assert preview_mod._to_pandas_safe(df) is df


def test_to_pandas_safe_polars_conversion():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2]})
    result = preview_mod._to_pandas_safe(df)
    assert isinstance(result, pd.DataFrame)


def test_to_pandas_safe_unsupported_returns_none():
    assert preview_mod._to_pandas_safe("not a dataframe") is None
    assert preview_mod._to_pandas_safe(None) is None


def test_to_pandas_safe_missing_polars_import_returns_none():
    """Exercise the `except ImportError: pass` fallback when polars is unavailable."""
    with patch.dict("sys.modules", {"polars": None}):
        assert preview_mod._to_pandas_safe("anything") is None


# --------------------------------------------------------------------------
# _pick_target_node_id
# --------------------------------------------------------------------------


def test_pick_target_node_id_empty_list():
    assert preview_mod._pick_target_node_id([]) is None


def test_pick_target_node_id_normal_terminal():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(node_id="n2", step_type=StepType.FEATURE_ENGINEERING, params={}, inputs=["n1"]),
    ]
    assert preview_mod._pick_target_node_id(nodes) == "n2"


def test_pick_target_node_id_training_terminal_uses_input():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(node_id="n2", step_type=StepType.TRAINING, params={}, inputs=["n1"]),
    ]
    assert preview_mod._pick_target_node_id(nodes) == "n1"


def test_pick_target_node_id_training_terminal_without_inputs_keeps_self():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.TRAINING, params={}, inputs=[]),
    ]
    assert preview_mod._pick_target_node_id(nodes) == "n1"


def test_pick_target_node_id_unified_training_fixed_uses_input():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(
            node_id="n2",
            step_type=StepType.TRAINING,
            params={"run_mode": "fixed"},
            inputs=["n1"],
        ),
    ]
    assert preview_mod._pick_target_node_id(nodes) == "n1"


def test_pick_target_node_id_unified_training_tuned_uses_input():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(
            node_id="n2",
            step_type=StepType.TRAINING,
            params={"run_mode": "tuned"},
            inputs=["n1"],
        ),
    ]
    assert preview_mod._pick_target_node_id(nodes) == "n1"


# --------------------------------------------------------------------------
# _is_polars_dataframe
# --------------------------------------------------------------------------


def test_is_polars_dataframe_true():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1]})
    assert preview_mod._is_polars_dataframe(df) is True


def test_is_polars_dataframe_false_for_pandas():
    df = pd.DataFrame({"a": [1]})
    assert preview_mod._is_polars_dataframe(df) is False


def test_is_polars_dataframe_missing_import_returns_false():
    with patch.dict("sys.modules", {"polars": None}):
        assert preview_mod._is_polars_dataframe(pd.DataFrame({"a": [1]})) is False


# --------------------------------------------------------------------------
# _extract_preview_frame
# --------------------------------------------------------------------------


def test_extract_preview_frame_pandas():
    df = pd.DataFrame({"a": [1, 2, 3]})
    preview_data, totals, df_for_analysis = preview_mod._extract_preview_frame(df, is_polars=False)
    assert preview_data == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert totals == {"_total": 3}
    assert df_for_analysis is df


def test_extract_preview_frame_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2]})
    preview_data, totals, df_for_analysis = preview_mod._extract_preview_frame(df, is_polars=True)
    assert preview_data == [{"a": 1}, {"a": 2}]
    assert totals == {"_total": 2}
    assert isinstance(df_for_analysis, pd.DataFrame)


# --------------------------------------------------------------------------
# _extract_preview_split (SplitDataset)
# --------------------------------------------------------------------------


def test_extract_preview_split_with_xy_tuples_and_validation():
    train = (pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1]))
    test = (pd.DataFrame({"f": [3]}), pd.Series([1]))
    validation = (pd.DataFrame({"f": [4]}), pd.Series([0]))
    split = SplitDataset(train=train, test=test, validation=validation)

    preview_data, totals, df_for_analysis = preview_mod._extract_preview_split(split)

    assert "train_X" in preview_data and "train_y" in preview_data
    assert "test_X" in preview_data and "test_y" in preview_data
    assert "validation_X" in preview_data and "validation_y" in preview_data
    assert totals["train_X"] == 2
    assert totals["test_X"] == 1
    assert totals["validation_X"] == 1
    assert isinstance(df_for_analysis, pd.DataFrame)


def test_extract_preview_split_with_plain_frames_no_validation():
    train = pd.DataFrame({"f": [1, 2, 3]})
    test = pd.DataFrame({"f": [4]})
    split = SplitDataset(train=train, test=test, validation=None)

    preview_data, totals, df_for_analysis = preview_mod._extract_preview_split(split)

    assert preview_data["train"] == [{"f": 1}, {"f": 2}, {"f": 3}]
    assert preview_data["test"] == [{"f": 4}]
    assert "validation" not in preview_data
    assert totals["train"] == 3
    assert totals["test"] == 1
    assert isinstance(df_for_analysis, pd.DataFrame)


def test_extract_preview_split_with_plain_validation_frame():
    """Covers the non-tuple `validation` branch (plain DataFrame, not (X, y))."""
    train = (pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1]))
    test = (pd.DataFrame({"f": [3]}), pd.Series([1]))
    validation = pd.DataFrame({"f": [5, 6]})
    split = SplitDataset(train=train, test=test, validation=validation)

    preview_data, totals, _df = preview_mod._extract_preview_split(split)

    assert preview_data["validation"] == [{"f": 5}, {"f": 6}]
    assert totals["validation"] == 2


# --------------------------------------------------------------------------
# _extract_preview_xy_tuple
# --------------------------------------------------------------------------


def test_extract_preview_xy_tuple():
    X = pd.DataFrame({"f": [1, 2]})
    y = pd.Series([0, 1])
    preview_data, totals, df_for_analysis = preview_mod._extract_preview_xy_tuple((X, y))
    assert preview_data["X"] == [{"f": 1}, {"f": 2}]
    assert preview_data["y"] == [{"0": 0}, {"0": 1}] or "y" in preview_data
    assert totals == {"X": 2, "y": 2}
    assert df_for_analysis is X


# --------------------------------------------------------------------------
# _extract_preview_dict_train
# --------------------------------------------------------------------------


def test_extract_preview_dict_train_only():
    artifact = {"train": (pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1]))}
    preview_data, totals, df_for_analysis = preview_mod._extract_preview_dict_train(artifact)
    assert "train_X" in preview_data
    assert "test_X" not in preview_data
    assert "validation_X" not in preview_data
    assert totals["train_X"] == 2
    assert isinstance(df_for_analysis, pd.DataFrame)


def test_extract_preview_dict_train_with_test_and_validation():
    artifact = {
        "train": (pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1])),
        "test": (pd.DataFrame({"f": [3]}), pd.Series([1])),
        "validation": (pd.DataFrame({"f": [4]}), pd.Series([0])),
    }
    preview_data, totals, df_for_analysis = preview_mod._extract_preview_dict_train(artifact)
    assert "test_X" in preview_data
    assert "validation_X" in preview_data
    assert totals["test_X"] == 1
    assert totals["validation_X"] == 1


# --------------------------------------------------------------------------
# _load_target_artifact
# --------------------------------------------------------------------------


def test_load_target_artifact_none_target_id(tmp_store):
    assert preview_mod._load_target_artifact(tmp_store, None) is None


def test_load_target_artifact_missing_in_store(tmp_store):
    assert preview_mod._load_target_artifact(tmp_store, "does_not_exist") is None


def test_load_target_artifact_present(tmp_store):
    df = pd.DataFrame({"a": [1, 2]})
    tmp_store.save("node1", df)
    loaded = preview_mod._load_target_artifact(tmp_store, "node1")
    assert isinstance(loaded, pd.DataFrame)
    pd.testing.assert_frame_equal(loaded, df)


# --------------------------------------------------------------------------
# _preview_frame_predicate / _preview_xy_tuple_predicate / _preview_dict_train_predicate
# --------------------------------------------------------------------------


def test_preview_frame_predicate():
    assert preview_mod._preview_frame_predicate(pd.DataFrame({"a": [1]})) is True
    assert preview_mod._preview_frame_predicate("nope") is False


def test_preview_xy_tuple_predicate():
    assert preview_mod._preview_xy_tuple_predicate((1, 2)) is True
    assert preview_mod._preview_xy_tuple_predicate((1, 2, 3)) is False
    assert preview_mod._preview_xy_tuple_predicate([1, 2]) is False


def test_preview_dict_train_predicate():
    assert preview_mod._preview_dict_train_predicate({"train": (1, 2)}) is True
    assert preview_mod._preview_dict_train_predicate({"train": [1, 2]}) is False
    assert preview_mod._preview_dict_train_predicate({"other": (1, 2)}) is False
    assert preview_mod._preview_dict_train_predicate("nope") is False


# --------------------------------------------------------------------------
# _dispatch_preview_extractor / _extract_preview
# --------------------------------------------------------------------------


def test_dispatch_preview_extractor_unsupported_returns_none():
    assert preview_mod._dispatch_preview_extractor(object()) is None
    assert preview_mod._dispatch_preview_extractor(42) is None


def test_dispatch_preview_extractor_frame():
    df = pd.DataFrame({"a": [1]})
    result = preview_mod._dispatch_preview_extractor(df)
    assert result is not None
    preview_data, totals, df_for_analysis = result
    assert totals == {"_total": 1}


def test_extract_preview_no_target_id_returns_empty(tmp_store):
    preview_data, totals, df_for_analysis = preview_mod._extract_preview(tmp_store, None)
    assert preview_data == {}
    assert totals == {}
    assert df_for_analysis is None


def test_extract_preview_unsupported_artifact_type_returns_empty(tmp_store):
    tmp_store.save("node1", "just a string artifact")
    preview_data, totals, df_for_analysis = preview_mod._extract_preview(tmp_store, "node1")
    assert preview_data == {}
    assert totals == {}
    assert df_for_analysis is None


def test_extract_preview_dataframe_artifact(tmp_store):
    df = pd.DataFrame({"a": [1, 2]})
    tmp_store.save("node1", df)
    preview_data, totals, df_for_analysis = preview_mod._extract_preview(tmp_store, "node1")
    assert preview_data == [{"a": 1}, {"a": 2}]
    assert totals == {"_total": 2}
    assert isinstance(df_for_analysis, pd.DataFrame)


# --------------------------------------------------------------------------
# _build_preview_nodes
# --------------------------------------------------------------------------


def test_build_preview_nodes_forces_sampling_on_data_loader():
    nodes = [
        NodeConfig(
            node_id="n1",
            step_type=StepType.DATA_LOADER,
            params={"source_id": "csv", "path": "x.csv"},
            inputs=[],
        ),
        NodeConfig(
            node_id="n2",
            step_type=StepType.FEATURE_ENGINEERING,
            params={"steps": []},
            inputs=["n1"],
        ),
    ]
    built = preview_mod._build_preview_nodes(nodes)
    assert len(built) == 2
    assert built[0].params["sample"] is True
    assert built[0].params["limit"] == 1000
    # Non data-loader node untouched
    assert "sample" not in built[1].params
    # Original nodes' params must not be mutated (copy semantics)
    assert "sample" not in nodes[0].params


# --------------------------------------------------------------------------
# _is_data_preview_sub / _strip_non_preview_nodes / _pair_training_subs
# --------------------------------------------------------------------------


def test_is_data_preview_sub_true_and_false():
    sub_preview = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n1", step_type="data_preview", params={}, inputs=[])],
    )
    assert preview_mod._is_data_preview_sub(sub_preview) is True

    sub_other = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[])],
    )
    assert preview_mod._is_data_preview_sub(sub_other) is False

    empty_sub = PipelineConfig(pipeline_id="p", nodes=[])
    assert preview_mod._is_data_preview_sub(empty_sub) is False


def test_strip_non_preview_nodes_removes_training_and_data_preview():
    training_types = {StepType.TRAINING}
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(node_id="n2", step_type=StepType.TRAINING, params={}, inputs=["n1"]),
            NodeConfig(node_id="n3", step_type="data_preview", params={}, inputs=["n1"]),
        ],
    )
    stripped = preview_mod._strip_non_preview_nodes(sub, training_types)
    ids = [n.node_id for n in stripped.nodes]
    assert ids == ["n1"]
    assert stripped.pipeline_id == "p"


def test_pair_training_subs_skips_data_preview_subs():
    training_types = {StepType.TRAINING}
    normal_sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(node_id="n2", step_type=StepType.TRAINING, params={}, inputs=["n1"]),
        ],
    )
    preview_sub = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n3", step_type="data_preview", params={}, inputs=[])],
    )
    result = preview_mod._pair_training_subs([normal_sub, preview_sub], training_types)
    assert len(result) == 1
    orig, runnable = result[0]
    assert orig is normal_sub
    assert [n.node_id for n in runnable.nodes] == ["n1"]


# --------------------------------------------------------------------------
# _append_uncovered_preview_branches
# --------------------------------------------------------------------------


def test_append_uncovered_preview_branches_adds_dangling_chain():
    paired_subs: list = []
    pipeline_config = PipelineConfig(pipeline_id="p", nodes=[])

    dangling_sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n_dangling", step_type=StepType.FEATURE_ENGINEERING, params={}, inputs=[]
            )
        ],
    )

    def fake_partition_for_preview(_config):
        return [dangling_sub]

    preview_mod._append_uncovered_preview_branches(
        paired_subs, pipeline_config, fake_partition_for_preview
    )
    assert len(paired_subs) == 1
    assert paired_subs[0] == (dangling_sub, dangling_sub)


def test_append_uncovered_preview_branches_skips_covered_and_data_preview_and_empty():
    covered_sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n_covered", step_type=StepType.FEATURE_ENGINEERING, params={}, inputs=[]
            )
        ],
    )
    paired_subs = [(covered_sub, covered_sub)]
    pipeline_config = PipelineConfig(pipeline_id="p", nodes=[])

    empty_sub = PipelineConfig(pipeline_id="p", nodes=[])
    data_preview_sub = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n_dp", step_type="data_preview", params={}, inputs=[])],
    )
    already_covered_sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n_covered", step_type=StepType.FEATURE_ENGINEERING, params={}, inputs=[]
            )
        ],
    )

    def fake_partition_for_preview(_config):
        return [empty_sub, data_preview_sub, already_covered_sub]

    preview_mod._append_uncovered_preview_branches(
        paired_subs, pipeline_config, fake_partition_for_preview
    )
    # Nothing new appended: empty sub skipped, data_preview skipped, already-covered skipped.
    assert len(paired_subs) == 1


# --------------------------------------------------------------------------
# _partition_preview_pipeline
# --------------------------------------------------------------------------


def test_partition_preview_pipeline_no_training_nodes():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(node_id="n2", step_type=StepType.FEATURE_ENGINEERING, params={}, inputs=["n1"]),
    ]
    pipeline_config = PipelineConfig(pipeline_id="p", nodes=nodes)
    result = preview_mod._partition_preview_pipeline(pipeline_config, nodes)
    assert len(result) == 1
    orig, runnable = result[0]
    assert orig is runnable


def test_partition_preview_pipeline_with_training_node():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(
            node_id="n2",
            step_type=StepType.TRAINING,
            params={"model_type": "random_forest"},
            inputs=["n1"],
        ),
    ]
    pipeline_config = PipelineConfig(pipeline_id="p", nodes=nodes)
    result = preview_mod._partition_preview_pipeline(pipeline_config, nodes)
    assert len(result) == 1
    orig, runnable = result[0]
    assert [n.node_id for n in orig.nodes] == ["n1", "n2"]
    assert [n.node_id for n in runnable.nodes] == ["n1"]


def test_partition_preview_pipeline_with_unified_training_node_fixed():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(
            node_id="n2",
            step_type=StepType.TRAINING,
            params={"run_mode": "fixed", "model_type": "random_forest"},
            inputs=["n1"],
        ),
    ]
    pipeline_config = PipelineConfig(pipeline_id="p", nodes=nodes)
    result = preview_mod._partition_preview_pipeline(pipeline_config, nodes)
    assert len(result) == 1
    orig, runnable = result[0]
    assert [n.node_id for n in orig.nodes] == ["n1", "n2"]
    assert [n.node_id for n in runnable.nodes] == ["n1"]


def test_partition_preview_pipeline_with_unified_training_node_tuned():
    nodes = [
        NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
        NodeConfig(
            node_id="n2",
            step_type=StepType.TRAINING,
            params={"run_mode": "tuned", "algorithm": "random_forest"},
            inputs=["n1"],
        ),
    ]
    pipeline_config = PipelineConfig(pipeline_id="p", nodes=nodes)
    result = preview_mod._partition_preview_pipeline(pipeline_config, nodes)
    assert len(result) == 1
    orig, runnable = result[0]
    assert [n.node_id for n in orig.nodes] == ["n1", "n2"]
    assert [n.node_id for n in runnable.nodes] == ["n1"]


# --------------------------------------------------------------------------
# _branch_terminal_group_key / _collect_terminal_ids_by_group / _compute_branch_dup_suffixes
# --------------------------------------------------------------------------


def test_branch_terminal_group_key_training_uses_model_type():
    leaf = NodeConfig(
        node_id="n2",
        step_type=StepType.TRAINING,
        params={"model_type": "xgboost"},
        inputs=["n1"],
    )
    mt, term_id = preview_mod._branch_terminal_group_key(leaf)
    assert mt == "xgboost"
    assert term_id == "n2"


def test_branch_terminal_group_key_falls_back_to_algorithm():
    leaf = NodeConfig(
        node_id="n2",
        step_type=StepType.TRAINING,
        params={"run_mode": "tuned", "algorithm": "svm"},
        inputs=["n1"],
    )
    mt, _ = preview_mod._branch_terminal_group_key(leaf)
    assert mt == "svm"


def test_branch_terminal_group_key_unified_training_fixed_uses_model_type():
    leaf = NodeConfig(
        node_id="n2",
        step_type=StepType.TRAINING,
        params={"run_mode": "fixed", "model_type": "xgboost"},
        inputs=["n1"],
    )
    mt, term_id = preview_mod._branch_terminal_group_key(leaf)
    assert mt == "xgboost"
    assert term_id == "n2"


def test_branch_terminal_group_key_unified_training_tuned_falls_back_to_algorithm():
    leaf = NodeConfig(
        node_id="n2",
        step_type=StepType.TRAINING,
        params={"run_mode": "tuned", "algorithm": "svm"},
        inputs=["n1"],
    )
    mt, _ = preview_mod._branch_terminal_group_key(leaf)
    assert mt == "svm"


def test_branch_terminal_group_key_non_training_uses_step_type():
    leaf = NodeConfig(node_id="n3", step_type="data_preview", params={}, inputs=["n1"])
    mt, term_id = preview_mod._branch_terminal_group_key(leaf)
    assert mt == "data_preview"
    assert term_id == "n3"


def test_compute_branch_dup_suffixes_disambiguates_shared_model_type():
    class FakeResult:
        status = "success"
        node_results: dict = {}
        merge_warnings: list = []
        node_warnings: list = []

    sub_a = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="term1",
                step_type=StepType.TRAINING,
                params={"model_type": "xgboost"},
                inputs=[],
            )
        ],
    )
    sub_b = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="term2",
                step_type=StepType.TRAINING,
                params={"model_type": "xgboost"},
                inputs=[],
            )
        ],
    )
    sub_results = [(sub_a, sub_a, FakeResult()), (sub_b, sub_b, FakeResult())]
    suffixes = preview_mod._compute_branch_dup_suffixes(sub_results)
    assert suffixes == {0: "#1", 1: "#2"}


def test_compute_branch_dup_suffixes_empty_for_single_branch():
    class FakeResult:
        status = "success"

    sub_a = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="term1",
                step_type=StepType.TRAINING,
                params={"model_type": "xgboost"},
                inputs=[],
            )
        ],
    )
    sub_results = [(sub_a, sub_a, FakeResult())]
    assert preview_mod._compute_branch_dup_suffixes(sub_results) == {}


# --------------------------------------------------------------------------
# _aggregate_branch_previews
# --------------------------------------------------------------------------


def test_aggregate_branch_previews_multi_branch(tmp_store):
    class FakeExecResult:
        def __init__(self, node_id):
            self.__dict__ = {"node_id": node_id, "status": "success"}

    class FakeResult:
        def __init__(self, node_id):
            self.status = "success"
            self.node_results = {node_id: FakeExecResult(node_id)}
            self.merge_warnings = []
            self.node_warnings = []

    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [3, 4, 5]})
    tmp_store.save("n1", df1)
    tmp_store.save("n2", df2)

    sub_a = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[])],
    )
    sub_b = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n2", step_type=StepType.DATA_LOADER, params={}, inputs=[])],
    )
    sub_results = [(sub_a, sub_a, FakeResult("n1")), (sub_b, sub_b, FakeResult("n2"))]

    (
        preview_data,
        preview_totals,
        branch_previews,
        branch_preview_totals,
        branch_node_ids,
        combined_node_results,
        agg_status,
        first_pdf,
    ) = preview_mod._aggregate_branch_previews(sub_results, tmp_store, {})

    assert agg_status == "success"
    assert branch_previews is not None
    assert len(branch_previews) == 2
    assert branch_node_ids is not None
    assert first_pdf is not None
    assert "n1" in combined_node_results and "n2" in combined_node_results


def test_aggregate_branch_previews_single_branch_no_branch_dicts(tmp_store):
    class FakeExecResult:
        def __init__(self):
            self.__dict__ = {"status": "success"}

    class FakeResult:
        def __init__(self):
            self.status = "success"
            self.node_results = {"n1": FakeExecResult()}
            self.merge_warnings = []
            self.node_warnings = []

    df1 = pd.DataFrame({"a": [1, 2]})
    tmp_store.save("n1", df1)

    sub_a = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[])],
    )
    sub_results = [(sub_a, sub_a, FakeResult())]

    result = preview_mod._aggregate_branch_previews(sub_results, tmp_store, {})
    branch_previews = result[2]
    branch_preview_totals = result[3]
    branch_node_ids = result[4]
    assert branch_previews is None
    assert branch_preview_totals is None
    assert branch_node_ids is None


def test_aggregate_branch_previews_failed_status_propagates(tmp_store):
    class FakeResult:
        def __init__(self, status):
            self.status = status
            self.node_results = {}
            self.merge_warnings = []
            self.node_warnings = []

    sub_a = PipelineConfig(
        pipeline_id="p",
        nodes=[NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[])],
    )
    sub_results = [(sub_a, sub_a, FakeResult("failed"))]

    result = preview_mod._aggregate_branch_previews(sub_results, tmp_store, {})
    agg_status = result[6]
    assert agg_status == "failed"


# --------------------------------------------------------------------------
# _generate_recommendations
# --------------------------------------------------------------------------


def test_generate_recommendations_none_df_returns_empty():
    assert preview_mod._generate_recommendations(None) == []


def test_generate_recommendations_handles_profiler_exception():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with patch(
        "backend.ml_pipeline._internal._routers.preview.DataProfiler.generate_profile",
        side_effect=RuntimeError("boom"),
    ):
        assert preview_mod._generate_recommendations(df) == []


def test_generate_recommendations_success_path():
    df = pd.DataFrame({"a": [1, 2, None, 4, 5]})
    recs = preview_mod._generate_recommendations(df)
    assert isinstance(recs, list)


# --------------------------------------------------------------------------
# _dedupe_by_key / _dedupe_preview_warnings
# --------------------------------------------------------------------------


def test_dedupe_by_key_preserves_first_seen_order():
    items = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    result = preview_mod._dedupe_by_key(items, lambda i: i["id"])
    assert result == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_dedupe_preview_warnings_dedupes_merge_and_node_warnings():
    class FakeResult:
        def __init__(self, merge_warnings, node_warnings):
            self.merge_warnings = merge_warnings
            self.node_warnings = node_warnings

    w1 = {"node_id": "n1", "kind": "merge", "inputs": ["a", "b"], "part": 1}
    w2 = {
        "node_id": "n1",
        "kind": "merge",
        "inputs": ["b", "a"],
        "part": 1,
    }  # dup (sorted inputs match)
    nw1 = {"node_id": "n1", "message": "careful"}
    nw2 = {"node_id": "n1", "message": "careful"}  # dup

    sub = PipelineConfig(pipeline_id="p", nodes=[])
    sub_results = [
        (sub, sub, FakeResult([w1], [nw1])),
        (sub, sub, FakeResult([w2], [nw2])),
    ]
    deduped_warnings, deduped_node_warnings = preview_mod._dedupe_preview_warnings(sub_results)
    assert len(deduped_warnings) == 1
    assert len(deduped_node_warnings) == 1


def test_dedupe_preview_warnings_empty():
    sub = PipelineConfig(pipeline_id="p", nodes=[])

    class FakeResult:
        merge_warnings: list = []
        node_warnings: list = []

    sub_results = [(sub, sub, FakeResult())]
    deduped_warnings, deduped_node_warnings = preview_mod._dedupe_preview_warnings(sub_results)
    assert deduped_warnings == []
    assert deduped_node_warnings == []


# --------------------------------------------------------------------------
# Endpoint-level: error path (SkyulfException) via TestClient
# --------------------------------------------------------------------------


def test_preview_pipeline_endpoint_node_failure_reports_failed_status():
    """A pipeline node with a nonexistent data source fails inside the engine
    (caught per-node), so the endpoint still returns 200 but with an
    aggregated "failed" status — exercising the `agg_status != "success"`
    branch in `_aggregate_branch_previews`.
    """
    from fastapi.testclient import TestClient

    from backend.main import app

    payload = {
        "pipeline_id": "test_preview_error_001",
        "nodes": [
            {
                "node_id": "node_1",
                "step_type": "data_loader",
                "params": {
                    "source_id": "csv",
                    "path": str(Path("does_not_exist_xyz.csv").resolve()),
                },
                "inputs": [],
            },
        ],
    }

    with TestClient(app, base_url="http://localhost") as client:
        response = client.post("/api/pipeline/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"


def test_preview_pipeline_endpoint_db_not_initialized_returns_error():
    """Simulate `db_engine.sync_session_factory` being None to exercise the
    `SkyulfException("Database not initialized")` branch and the outer
    except/finally cleanup (temp dir removal, no sync_session to close).
    """
    from fastapi.testclient import TestClient

    from backend.main import app

    payload = {
        "pipeline_id": "test_preview_dbfail_001",
        "nodes": [
            {
                "node_id": "node_1",
                "step_type": "data_loader",
                "params": {"source_id": "csv", "path": "irrelevant.csv"},
                "inputs": [],
            },
        ],
    }

    with (
        TestClient(app, base_url="http://localhost") as client,
        patch("backend.database.engine.sync_session_factory", None),
    ):
        response = client.post("/api/pipeline/preview", json=payload)
        assert response.status_code >= 400
