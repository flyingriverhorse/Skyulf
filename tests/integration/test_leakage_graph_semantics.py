"""Graph-level leakage isolation at the backend gate and real execution boundary."""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution._leakage_validation import (
    validate_no_preprocessing_before_split,
)
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.engine._feature_eng import _step_learns_from_data
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore


def _node(node_id, step_type, inputs=(), **params):
    """Build a real execution configuration without depending on fixture JSON."""
    return NodeConfig(node_id=node_id, step_type=step_type, inputs=list(inputs), params=params)


def _step(transformer, **params):
    """Build one ordered operation within a composite feature-engineering node."""
    return {"name": transformer, "transformer": transformer, "params": params}


def _training(inputs, **params):
    """Use a small real classifier whose CV requires preprocessing isolation."""
    config = {
        "target_column": "target",
        "algorithm": "logistic_regression",
        "cv_enabled": True,
        "cv_folds": 2,
        "evaluate": False,
    }
    config.update(params)
    return _node("model", "training", inputs, **config)


def _run(tmp_path, middle, model_inputs, *, mode="raise"):
    """Execute real loaders, preprocessing, and model folds on a bounded local dataset."""
    rng = np.random.default_rng(71)
    count = 96
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=count),
            "city": [f"c{i % 8}" for i in range(count)],
            "target": rng.integers(0, 2, size=count),
        }
    )
    path = tmp_path / "graph.csv"
    frame.to_csv(path, index=False)
    logs = []
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog(), log_callback=logs.append)
    nodes = [_node("load", "data_loader", path=str(path)), *middle, _training(model_inputs)]
    result = engine.run(PipelineConfig("graph-semantics", nodes, metadata={"on_leakage": mode}))
    return result, logs, store


@pytest.mark.parametrize("split_inside", [True, False])
def test_composite_cannot_hide_learned_operations_before_row_split(split_inside):
    """Submission must inspect the operations inside a composite before allowing its graph."""
    steps = [_step("StandardScaler", columns=["x"])]
    if split_inside:
        steps.append(_step("TrainTestSplitter", target_column="target"))
    nodes = [
        _node("load", "data_loader"),
        _node("features", "feature_engineering", ["load"], steps=steps),
    ]
    if not split_inside:
        nodes.append(_node("split", "TrainTestSplitter", ["features"], target_column="target"))
    nodes.append(_training([nodes[-1].node_id]))
    with pytest.raises(ValueError, match="StandardScaler"):
        validate_no_preprocessing_before_split(nodes)


def test_composite_order_preserves_real_split_protection():
    """A split inside a composite protects its later learners and downstream nodes."""
    nodes = [
        _node("load", "data_loader"),
        _node(
            "features",
            "feature_engineering",
            ["load"],
            steps=[
                _step("DropMissingColumns", columns=["unused"]),
                _step("TrainTestSplitter", target_column="target"),
                _step("StandardScaler", columns=["x"]),
            ],
        ),
        _training(["features"]),
    ]
    verdict = validate_no_preprocessing_before_split(nodes)
    assert verdict["status"] == "passed"
    assert any(row["step_type"] == "StandardScaler" for row in verdict["checked"])


@pytest.mark.parametrize("candidate_target,allowed", [("sibling_target", False), ("target", True)])
def test_selected_branch_target_cannot_be_replaced_by_sibling(candidate_target, allowed):
    """A sibling's label must neither exempt feature encoding nor block actual target encoding."""
    nodes = [
        _node("other_load", "data_loader"),
        _node("other_model", "training", ["other_load"], target_column="sibling_target"),
        _node("load", "data_loader"),
        _node("encoder", "OrdinalEncoder", ["load"], columns=[candidate_target]),
        _node("split", "TrainTestSplitter", ["encoder"], target_column="target"),
        _training(["split"]),
    ]
    if allowed:
        assert (
            validate_no_preprocessing_before_split(nodes, target_node_id="model")["status"]
            == "passed"
        )
    else:
        with pytest.raises(ValueError, match="encoder"):
            validate_no_preprocessing_before_split(nodes, target_node_id="model")


def test_fold_classifier_matches_target_only_gate_exemption():
    """Target-only encoders allowed before splitting must not disable fold reconstruction."""
    assert not _step_learns_from_data(
        _step("OrdinalEncoder", columns=["target"]), target_column="target"
    )
    assert _step_learns_from_data(
        _step("OrdinalEncoder", columns=["feature"]), target_column="target"
    )


@pytest.mark.parametrize("row_split_first", [False, True])
def test_feature_target_split_does_not_replace_row_boundary(tmp_path, row_split_first):
    """Column separation after a learned transform must not make its fitted output raw CV input."""
    middle = []
    previous = "load"
    if row_split_first:
        middle.append(
            _node("split", "TrainTestSplitter", [previous], target_column="target", test_size=0.2)
        )
        previous = "split"
    middle.extend(
        [
            _node("encoder", "WOEEncoder", [previous], columns=["city"]),
            _node("column_split", "feature_target_split", ["encoder"], target_column="target"),
        ]
    )
    result, logs, _ = _run(tmp_path, middle, ["column_split"])
    assert result.status == "success", logs
    metrics = result.node_results["model"].metrics
    assert "fold_refit_fallback" not in metrics
    assert metrics["fold_refit_audit"]["fit_calls"] > 0


def test_repeated_row_splitter_keeps_first_effective_boundary(tmp_path):
    """A skipped repeated splitter must not turn earlier train-only learners into apparent leakage."""
    middle = [
        _node("split", "TrainTestSplitter", ["load"], target_column="target", test_size=0.2),
        _node("encoder", "WOEEncoder", ["split"], columns=["city"]),
        _node("again", "Split", ["encoder"], target_column="target", test_size=0.4),
    ]
    result, logs, _ = _run(tmp_path, middle, ["again"])
    assert result.status == "success", logs
    assert "fold_refit_fallback" not in result.node_results["model"].metrics
    assert result.node_results["model"].metrics["fold_refit_audit"]["fit_calls"] > 0


@pytest.mark.parametrize("composite", [False, True])
def test_stateless_branches_merging_at_splitter_refit_downstream_learner(tmp_path, composite):
    """The split artifact after a stateless merge supplies untouched rows for downstream CV fits."""
    middle = [
        _node("left", "ValueReplacement", ["load"], columns=["x"], mapping={"100": 10}),
        _node("right", "DropMissingColumns", ["load"], columns=["absent"]),
    ]
    if composite:
        middle.append(
            _node(
                "split_encode",
                "feature_engineering",
                ["left", "right"],
                steps=[
                    _step("TrainTestSplitter", target_column="target", test_size=0.2),
                    _step("WOEEncoder", columns=["city"]),
                ],
            )
        )
        model_input = "split_encode"
    else:
        middle.extend(
            [
                _node(
                    "split",
                    "TrainTestSplitter",
                    ["left", "right"],
                    target_column="target",
                    test_size=0.2,
                ),
                _node("encoder", "WOEEncoder", ["split"], columns=["city"]),
            ]
        )
        model_input = "encoder"
    result, logs, _ = _run(tmp_path, middle, [model_input])
    assert result.status == "success", logs
    metrics = result.node_results["model"].metrics
    assert "fold_refit_fallback" not in metrics
    assert metrics["fold_refit_audit"]["isolation_ok"] is True
    assert metrics["fold_refit_audit"]["fit_calls"] > 0


@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
def test_unsupported_learned_merge_requires_explicit_leakage_optout(tmp_path, mode):
    """Unsupported CV graphs may score pre-fitted preprocessing only after an explicit opt-out."""
    middle = [
        _node("split", "TrainTestSplitter", ["load"], target_column="target", test_size=0.2),
        _node("left", "WOEEncoder", ["split"], columns=["city"]),
        _node("right", "WOEEncoder", ["split"], columns=["city"], regularization=0.5),
        _node("merge", "feature_engineering", ["left", "right"], steps=[]),
    ]
    result, logs, _ = _run(tmp_path, middle, ["merge"], mode=mode)
    training = result.node_results["model"]
    if mode == "raise":
        assert result.status == "failed"
        assert "Per-fold preprocessing" in training.error
        assert "cv_accuracy_mean" not in training.metrics
    else:
        assert result.status == "success", logs
        assert training.metrics["fold_refit_fallback"] == "unsupported_graph"
        assert (any("Per-fold preprocessing refit skipped" in log for log in logs)) == (
            mode == "warn"
        )


@pytest.mark.parametrize("composite", [False, True])
def test_raw_text_runner_receives_execution_target(tmp_path, composite):
    """Automatic text cleaning must preserve the target configured on the model's own branch."""
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    store.save("load", pd.DataFrame({"text": ["HELLO", "WORLD"], "target": ["YES", "NO"]}))
    operation = _step("TextCleaning", operations=[{"op": "case", "mode": "lower"}])
    node = (
        _node("clean", "feature_engineering", ["load"], steps=[operation])
        if composite
        else _node("clean", "TextCleaning", ["load"], **operation["params"])
    )
    nodes = [_node("load", "data_loader"), node, _training(["clean"])]
    engine._node_configs = {item.node_id: item for item in nodes}
    if composite:
        engine._run_feature_engineering(node)
    else:
        engine._run_transformer(node)
    output = store.load("clean")
    assert output["target"].tolist() == ["YES", "NO"]
    assert output["text"].tolist() == ["hello", "world"]


@pytest.mark.parametrize(
    "failure", ["divergent_loaders", "fork_not_splitter", "row_changing_branch"]
)
def test_unsupported_fold_shapes_fail_before_scoring_by_default(tmp_path, failure):
    """Every unsupported learned topology needs an explicit opt-out, not just nested merges."""
    split = _step("TrainTestSplitter", target_column="target", test_size=0.2)
    encode = _step("WOEEncoder", columns=["city"])
    if failure == "divergent_loaders":
        middle = [
            _node("other_load", "data_loader", dataset_id=str(tmp_path / "graph.csv")),
            _node("left", "feature_engineering", ["load"], steps=[split, encode]),
            _node("right", "feature_engineering", ["other_load"], steps=[split, encode]),
        ]
        expected_reason = "do not share one data loader"
    elif failure == "fork_not_splitter":
        middle = [
            _node("trunk", "feature_engineering", ["load"], steps=[split, encode]),
            _node("left", "StandardScaler", ["trunk"], columns=["x"]),
            _node("right", "MinMaxScaler", ["trunk"], columns=["x"]),
        ]
        expected_reason = "must end with"
    else:
        middle = [
            _node("split", "TrainTestSplitter", ["load"], target_column="target"),
            _node("left", "WOEEncoder", ["split"], columns=["city"]),
            _node(
                "right",
                "feature_engineering",
                ["split"],
                steps=[
                    encode,
                    _step("DropMissingRows", columns=["x"]),
                ],
            ),
        ]
        expected_reason = "changes row counts"
    result, _logs, _store = _run(tmp_path, middle, ["left", "right"])
    model = result.node_results["model"]
    assert result.status == "failed"
    assert expected_reason in model.error
    assert "cv_mean_score" not in model.metrics


def test_corrupt_raw_fold_payload_fails_before_scoring_by_default(tmp_path, monkeypatch):
    """Artifact reconstruction errors cannot silently turn protected CV into pre-fitted scoring."""
    from backend.ml_pipeline._execution.engine._feature_eng import FeatureEngMixin

    def corrupt(output, target_col):
        """Simulate an unavailable raw payload without changing the stored processed data."""
        raise RuntimeError("corrupt raw payload")

    monkeypatch.setattr(FeatureEngMixin, "_split_train_payload", staticmethod(corrupt))
    result, _logs, _store = _run(
        tmp_path,
        [
            _node("split", "TrainTestSplitter", ["load"], target_column="target"),
            _node("encode", "WOEEncoder", ["split"], columns=["city"]),
        ],
        ["encode"],
    )
    model = result.node_results["model"]
    assert result.status == "failed"
    assert "payload reconstruction failed" in model.error
    assert "cv_mean_score" not in model.metrics


def test_shared_encoder_with_ambiguous_targets_has_no_target_only_exemption():
    """A shared encoder cannot borrow either sibling model's target when they disagree."""
    nodes = [
        _node("load", "data_loader"),
        _node("encode", "OrdinalEncoder", ["load"], columns=["target"]),
        _node("first", "TrainTestSplitter", ["encode"], target_column="target"),
        _node("second", "TrainTestSplitter", ["encode"], target_column="other_target"),
        _node("first_model", "training", ["first"], target_column="target"),
        _node("second_model", "training", ["second"], target_column="other_target"),
    ]
    with pytest.raises(ValueError, match="OrdinalEncoder"):
        validate_no_preprocessing_before_split(nodes)


def test_encoder_param_cannot_override_branch_target_for_gate_exemption():
    """A misleading local target hint cannot exempt actual feature encoding before a split."""
    nodes = [
        _node("load", "data_loader"),
        _node("encode", "OrdinalEncoder", ["load"], columns=["city"], target_column="city"),
        _node("split", "TrainTestSplitter", ["encode"], target_column="target"),
        _training(["split"]),
    ]
    with pytest.raises(ValueError, match="OrdinalEncoder"):
        validate_no_preprocessing_before_split(nodes)


def test_composite_learner_without_own_split_cannot_borrow_sibling_boundary():
    """Composite expansion must retain the missing-branch-boundary admission check."""
    nodes = [
        _node("load", "data_loader"),
        _node(
            "features",
            "feature_engineering",
            ["load"],
            steps=[
                _step("StandardScaler", columns=["x"]),
            ],
        ),
        _node("unrelated", "TrainTestSplitter", ["load"], target_column="target"),
        _training(["features"], cv_enabled=False),
    ]
    with pytest.raises(ValueError, match="no train/test splitter on its input branch"):
        validate_no_preprocessing_before_split(nodes, target_node_id="model")


@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
@pytest.mark.parametrize("split_after", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_mixed_raw_and_split_inputs_cannot_protect_downstream_fit(mode, split_after, nested):
    """A protected merge input cannot hide an unsplit path into a later learned operation."""
    nodes = [
        _node("load", "data_loader"),
        _node("protected", "TrainTestSplitter", ["load"], target_column="target"),
        _node("raw", "DropMissingColumns", ["load"], columns=["absent"]),
        _node("merge", "feature_engineering", ["protected", "raw"], steps=[]),
    ]
    if nested:
        nodes.append(
            _node(
                "learner",
                "feature_engineering",
                ["merge"],
                steps=[
                    _step("feature_engineering", steps=[_step("StandardScaler", columns=["x"])])
                ],
            )
        )
    else:
        nodes.append(_node("learner", "StandardScaler", ["merge"], columns=["x"]))
    if split_after:
        nodes.append(_node("later_split", "Split", ["learner"], target_column="target"))
    nodes.append(_training([nodes[-1].node_id], cv_enabled=False))

    if mode == "raise":
        with pytest.raises(ValueError, match="StandardScaler"):
            validate_no_preprocessing_before_split(nodes, on_leakage=mode)
    else:
        verdict = validate_no_preprocessing_before_split(nodes, on_leakage=mode)
        assert verdict["status"] == "warnings"
        assert any(
            item["step_type"] == "StandardScaler" and item["violation"]
            for item in verdict["checked"]
        )


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("learner_first", [False, True])
def test_nested_composite_preserves_operation_order_at_every_depth(depth, learner_first):
    """Recursive wrappers must neither hide a pre-split fit nor invalidate a protected fit."""
    split = _step("TrainTestSplitter", target_column="target")
    scaler = _step("StandardScaler", columns=["x"])
    steps = [scaler, split] if learner_first else [split, scaler]
    for _ in range(depth):
        steps = [_step("feature_engineering", steps=steps)]
    nodes = [
        _node("load", "data_loader"),
        _node("features", "feature_engineering", ["load"], steps=steps),
        _training(["features"]),
    ]

    if learner_first:
        with pytest.raises(ValueError, match="StandardScaler"):
            validate_no_preprocessing_before_split(nodes)
    else:
        verdict = validate_no_preprocessing_before_split(nodes)
        assert verdict["status"] == "passed"
        assert any(item["step_type"] == "StandardScaler" for item in verdict["checked"])


@pytest.mark.parametrize("columns,allowed", [(["target"], True), (["city"], False)])
def test_nested_composite_encoder_uses_its_downstream_target(columns, allowed):
    """Nested target-only exemptions must use the executing branch's actual target."""
    nodes = [
        _node("load", "data_loader"),
        _node(
            "features",
            "feature_engineering",
            ["load"],
            steps=[_step("feature_engineering", steps=[_step("OrdinalEncoder", columns=columns)])],
        ),
        _node("split", "TrainTestSplitter", ["features"], target_column="target"),
        _training(["split"]),
    ]

    if allowed:
        verdict = validate_no_preprocessing_before_split(nodes)
        assert any(item["step_type"] == "OrdinalEncoder" for item in verdict["exempted"])
    else:
        with pytest.raises(ValueError, match="OrdinalEncoder"):
            validate_no_preprocessing_before_split(nodes)
