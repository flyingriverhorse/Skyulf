"""Model admission requires observed exact partition parity and sealed fitted state."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference._partition_models import CONTRACTS
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe
from skyulf.pipeline import SkyulfPipeline


def _seal(*args, **kwargs) -> Any:
    """Keep adversarial mutation tests explicitly dynamic after the typed training API."""
    return seal_partition_probe(*args, **kwargs)


def _artifact(
    tmp_path, node="hist_gradient_boosting_classifier", params=None
) -> tuple[Any, pd.DataFrame, Path]:
    """Fit a real small classifier with an independent sixteen-row holdout."""
    frame = pd.DataFrame({"x": np.arange(64, dtype=float) % 8, "y": np.arange(64) % 2})
    pipeline: Any = SkyulfPipeline(
        {"preprocessing": [], "modeling": {"type": node, "params": params or {}}}
    )
    pipeline.fit(SplitDataset(train=frame.iloc[:48], test=frame.iloc[48:]), target_column="y")
    path = tmp_path / "model"
    save_pipeline(pipeline, path)
    return load_pipeline(path), frame.iloc[48:], path


def test_source_free_model_seals_and_reloads(tmp_path):
    """Ordinary model training must seal evidence without requiring project Python source."""
    artifact, holdout, path = _artifact(tmp_path)
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(artifact)
    sealed = _seal(artifact, holdout, path, enabled=True)
    evidence = sealed.pipeline.config["partition_probe"]["model"]
    assert evidence["class"] == sealed.manifest.model_class
    assert evidence["checks"][0] == {"name": "full", "status": "passed"}
    assert any(
        step.action == "model_probed" for step in require_partition_safe_pipeline(sealed).steps
    )
    rows = holdout[["x"]]
    pd.testing.assert_frame_equal(
        predict_pipeline(rows, sealed),
        pd.concat([predict_pipeline(rows.iloc[i : i + 1], sealed) for i in range(len(rows))]),
        check_exact=True,
    )
    assert require_partition_safe_pipeline(load_pipeline(path)) == require_partition_safe_pipeline(
        sealed
    )


@pytest.mark.parametrize(
    "change", ["state", "checks", "class", "method", "module_method", "registration"]
)
def test_changed_model_evidence_rejected(tmp_path, monkeypatch, change):
    """A passing certificate cannot authorize changed state or executable identities."""
    from skyulf.registry import NodeRegistry

    artifact, holdout, path = _artifact(tmp_path)
    sealed = _seal(artifact, holdout, path, enabled=True)
    model = sealed.pipeline.model_estimator.model
    if change == "state":
        model._baseline_prediction += 1
    elif change == "checks":
        sealed.pipeline.config["partition_probe"]["model"]["checks"].pop()
    elif change == "class":
        sealed.pipeline.config["partition_probe"]["model"]["class"] = "forged.Class"
    elif change == "method":
        model.predict = lambda values: np.zeros(len(values))
    elif change == "module_method":
        from types import ModuleType

        from skyulf.pipeline.seal import artifact_digest

        before = artifact_digest(model)
        model.predict = ModuleType("replacement")
        assert artifact_digest(model) == before
    else:
        monkeypatch.setitem(NodeRegistry._calculators, "hist_gradient_boosting_classifier", object)
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(sealed)


@pytest.mark.parametrize("node", sorted({value[0] for value in CONTRACTS.values()}))
def test_each_eligible_model_fitted_and_probed(tmp_path, node):
    """Every installed eligible model has actual fit/save/probe/load coverage."""
    params = {}
    if node.startswith(("voting_", "stacking_")):
        params = {"base_estimators": ["decision_tree", "hist_gradient_boosting"]}
        if node.startswith("stacking_"):
            params.update(final_estimator="decision_tree", cv=2)
    if node == "sgd_classifier":
        params = {"loss": "log_loss"}
    artifact, holdout, path = _artifact(tmp_path, node, params)
    sealed = _seal(artifact, holdout, path, enabled=True)
    assert require_partition_safe_pipeline(sealed).steps[-1].action == "model_probed"


@pytest.mark.parametrize("subclass", [False, True])
def test_ensemble_rejects_substituted_member(tmp_path, subclass):
    """Sealing must reject an unknown child before trusting the enclosing sklearn class."""
    from sklearn.dummy import DummyClassifier

    artifact, holdout, path = _artifact(
        tmp_path, "voting_classifier", {"base_estimators": ["decision_tree"]}
    )
    sealed = _seal(artifact, holdout, path, enabled=True)

    class CustomChild(DummyClassifier):
        """Represent project-defined inference code under an ordinary ensemble."""

    sealed.pipeline.model_estimator.model.estimators_[0] = (
        CustomChild() if subclass else DummyClassifier()
    )
    with pytest.raises(UnsupportedExecutionError, match="Unaccepted nested"):
        require_partition_safe_pipeline(sealed)


def test_probe_rejects_too_few_rows(tmp_path):
    """Singleton checks need multiple independent chunks to provide evidence."""
    artifact, holdout, path = _artifact(tmp_path)
    with pytest.raises(ValueError, match="16..256"):
        _seal(artifact, holdout.head(15), path, enabled=True)


@pytest.mark.parametrize(
    "node,params", [("sgd_classifier", {"loss": "hinge"}), ("svc", {"probability": False})]
)
def test_models_without_probabilities_keep_label_only_contract(tmp_path, node, params):
    """SGD hinge and SVC without probability support must not invent probabilities."""
    artifact, holdout, path = _artifact(tmp_path, node, params)
    sealed = _seal(artifact, holdout, path, enabled=True)
    assert not sealed.manifest.classification_probabilities
    assert list(predict_pipeline(holdout[["x"]], sealed)) == ["prediction"]


@pytest.mark.parametrize("pipeline_thresholds", [False, True])
def test_tuned_probed_model_threshold_paths(tmp_path, pipeline_thresholds):
    """The real tuner facade and both threshold policies must be sealed and replayable."""
    frame = pd.DataFrame({"x": np.arange(64, dtype=float) % 8, "y": np.arange(64) % 2})
    pipeline: Any = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": "hist_gradient_boosting_classifier"},
                "strategy": "grid",
                "metric": "accuracy",
                "search_space": {},
                "cv_folds": 2,
                "n_trials": 1,
                "n_jobs": 1,
            },
        }
    )
    pipeline.fit(SplitDataset(train=frame.iloc[:48], test=frame.iloc[48:]), target_column="y")
    pipeline.model_estimator.model[1].decision_thresholds = {0: 0.8, 1: 0.2}
    if pipeline_thresholds:
        pipeline._tuned_thresholds = {0: 0.2, 1: 0.8}
    path = tmp_path / "tuned"
    save_pipeline(pipeline, path, use_tuned_thresholds=pipeline_thresholds)
    sealed = _seal(load_pipeline(path), frame.iloc[48:], path, enabled=True)
    assert require_partition_safe_pipeline(sealed).steps[-1].action == "model_probed"
    rows = frame.iloc[48:][["x"]]
    pd.testing.assert_frame_equal(
        predict_pipeline(rows, sealed),
        pd.concat([predict_pipeline(rows.iloc[i : i + 1], sealed) for i in range(len(rows))]),
        check_exact=True,
    )


@pytest.mark.parametrize(
    "change", ["subclass", "applier", "rows", "duplicate_check", "check_value", "threshold"]
)
def test_additional_certificate_tampering(tmp_path, change):
    """Identity and complete ordered evidence remain mandatory after serialization."""
    artifact, holdout, path = _artifact(tmp_path)
    sealed = _seal(artifact, holdout, path, enabled=True)
    estimator = sealed.pipeline.model_estimator
    detail = sealed.pipeline.config["partition_probe"]["model"]
    if change == "subclass":
        estimator.model.__class__ = type("Subclass", (type(estimator.model),), {})
    elif change == "applier":
        estimator.applier.predict = lambda frame, model: np.zeros(len(frame))
    elif change == "rows":
        detail["rows"] = True
    elif change == "duplicate_check":
        detail["checks"][-1] = detail["checks"][0]
    elif change == "check_value":
        detail["checks"][-1] = None
    else:
        sealed.pipeline._tuned_thresholds = {0: 0.9, 1: 0.1}
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(sealed)


@pytest.mark.parametrize("node", ["calibrated_classifier", "stacking_classifier"])
def test_nested_inference_override_rejected(tmp_path, node):
    """Calibration internals and stacking final estimators cannot replace prediction methods."""
    params = (
        {"base_estimator": "decision_tree", "cv": 2}
        if node == "calibrated_classifier"
        else {"base_estimators": ["decision_tree"], "final_estimator": "decision_tree", "cv": 2}
    )
    artifact, holdout, path = _artifact(tmp_path, node, params)
    sealed = _seal(artifact, holdout, path, enabled=True)
    model = sealed.pipeline.model_estimator.model
    child = (
        model.calibrated_classifiers_[0].estimator
        if node == "calibrated_classifier"
        else model.final_estimator_
    )
    child.predict = lambda rows: np.zeros(len(rows))
    with pytest.raises(UnsupportedExecutionError, match="Overridden"):
        require_partition_safe_pipeline(sealed)


def test_admission_never_predicts(tmp_path, monkeypatch):
    """Loading and admission inspect certificates without executing the fitted estimator."""
    artifact, holdout, path = _artifact(tmp_path)
    sealed = _seal(artifact, holdout, path, enabled=True)

    def forbidden(*args, **kwargs):
        """Fail the test if admission evaluates predictions."""
        raise AssertionError("prediction executed during admission")

    monkeypatch.setattr(type(sealed.pipeline.model_estimator.model), "predict", forbidden)
    monkeypatch.setattr(type(sealed.pipeline.model_estimator.model), "predict_proba", forbidden)
    assert require_partition_safe_pipeline(load_pipeline(path)).steps[-1].action == "model_probed"


def test_exact_floating_partition_difference_fails_closed(tmp_path):
    """Numerically close BLAS batch differences must never be hidden by a tolerance."""
    rng = np.random.default_rng(147)
    frame = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    frame["y"] = frame.to_numpy() @ np.array([0.1, 0.3, 0.7, 0.9, 1.1])
    pipeline: Any = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "ridge_regression"}})
    pipeline.fit(SplitDataset(train=frame.iloc[:48], test=frame.iloc[48:]), target_column="y")
    path = tmp_path / "exact"
    save_pipeline(pipeline, path)
    with pytest.raises(ValueError, match="Model partition probe failed chunks:1: output_mismatch"):
        _seal(load_pipeline(path), frame.iloc[48:], path, enabled=True)


@pytest.mark.parametrize("node", ["hist_gradient_boosting_classifier", "ridge_regression", "svc"])
def test_saved_prediction_empty_request_skips_estimator(tmp_path, monkeypatch, node):
    """The serving boundary must implement the empty contract accepted by model evidence."""
    artifact, holdout, _ = _artifact(tmp_path, node)
    expected_columns = list(predict_pipeline(holdout[["x"]], artifact))

    def forbidden(*args, **kwargs):
        """Empty inference must never reach preprocessing or estimator prediction."""
        raise AssertionError("empty request reached estimator")

    monkeypatch.setattr(artifact.pipeline, "predict", forbidden)
    empty = predict_pipeline(holdout[["x"]].head(0), artifact)
    assert empty.empty and list(empty) == expected_columns
    assert empty["prediction"].dtype == (
        "float64" if artifact.manifest.task == "regression" else "int64"
    )


@pytest.mark.parametrize(
    "key,value", [("version", True), ("rows", True), ("chunk_sizes", [True, 2, 7])]
)
def test_probe_header_rejects_numeric_type_aliases(tmp_path, key, value):
    """Python numeric equality must not let malformed certificate headers through."""
    artifact, holdout, path = _artifact(tmp_path)
    sealed = _seal(artifact, holdout, path, enabled=True)
    sealed.pipeline.config["partition_probe"][key] = value
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(sealed)


@pytest.mark.parametrize("evidence", [None, 1, [], "model", {"model": {}}])
def test_reviewed_model_rejects_malformed_probe_at_model_boundary(tmp_path, evidence):
    """Malformed optional evidence must identify model admission without leaking type errors."""
    artifact, _, _ = _artifact(tmp_path, "random_forest_classifier", {"n_estimators": 2})
    artifact.pipeline.config["partition_probe"] = evidence
    with pytest.raises(UnsupportedExecutionError, match=r"model\.apply on pandas"):
        require_partition_safe_pipeline(artifact)


@pytest.mark.parametrize("container", [list, tuple, set, frozenset, dict])
def test_nested_callbacks_rejected_before_model_probe(tmp_path, container):
    """Container nesting must not hide executable callbacks from source-free admission."""
    from skyulf.inference._partition_models import inspect_model

    artifact, _, _ = _artifact(tmp_path)
    model = artifact.pipeline.model_estimator.model

    def callback(values):
        """Represent executable state concealed inside an ordinary container."""
        return values

    model.callbacks = {"after": callback} if container is dict else container([callback])
    with pytest.raises(ValueError, match="callbacks"):
        inspect_model(model)


@pytest.mark.parametrize("spark_output", [False, True])
@pytest.mark.parametrize("rows", [0, 16])
def test_string_label_empty_and_nonempty_transport(tmp_path, spark_output, rows):
    """Both empty StringDtype and nonempty object labels must obey the declared wire schema."""
    import json
    from types import SimpleNamespace

    pytest.importorskip("mlflow")
    import pyarrow as pa

    from skyulf.integrations.mlflow.models.pipeline_model import (
        SkyulfPipelinePythonModel,
        _signature,
    )
    from skyulf.integrations.mlflow.spark.spark_model import (
        partition_safety_certificate,
        runtime_source_digest,
    )

    frame = pd.DataFrame({"x": np.arange(64, dtype=float) % 8, "y": ["normal", "premium"] * 32})
    pipeline = SkyulfPipeline({"modeling": {"type": "hist_gradient_boosting_classifier"}})
    pipeline.fit(SplitDataset(train=frame.iloc[:48], test=frame.iloc[48:]), target_column="y")
    path = tmp_path / "labels"
    save_pipeline(pipeline, path)
    sealed = _seal(load_pipeline(path), frame.iloc[48:], path, enabled=True)
    model = SkyulfPipelinePythonModel(
        None, partition_safety_certificate(sealed), runtime_source_digest()
    )
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(path)}))
    result = model.predict(
        None, frame.iloc[48:][["x"]].head(rows), params={"spark_output": spark_output}
    )
    signature = _signature(sealed, spark_certified=True)
    assert signature.outputs.to_dict()[0]["type"] == "string"
    arrow = pa.Table.from_pandas(
        result,
        schema=pa.schema(
            [
                ("prediction", pa.string()),
                ("probability_0", pa.float64()),
                ("probability_1", pa.float64()),
            ]
        ),
        preserve_index=False,
    )
    records = json.loads(result.to_json(orient="records"))
    assert arrow.num_rows == len(records) == rows
    assert arrow["prediction"].to_pylist() == [row["prediction"] for row in records]
    assert all(row["prediction"] in {"normal", "premium"} for row in records)


@pytest.mark.parametrize("node", ["xgboost_classifier", "lgbm_classifier"])
def test_native_tree_swap_invalidates_sealed_model_evidence(tmp_path, node):
    """A different native tree payload must invalidate an otherwise identical certificate."""
    from sklearn.base import clone

    if node == "xgboost_classifier":
        pytest.importorskip("xgboost")
        params = {"n_estimators": 4, "max_depth": 2, "learning_rate": 1.0, "n_jobs": 1}
    else:
        pytest.importorskip("lightgbm")
        params = {
            "n_estimators": 4,
            "num_leaves": 4,
            "min_child_samples": 1,
            "verbosity": -1,
            "n_jobs": 1,
        }
    artifact, holdout, path = _artifact(tmp_path, node, params)
    sealed = _seal(artifact, holdout, path, enabled=True)
    model = sealed.pipeline.model_estimator.model
    rows = holdout[["x"]]
    before = predict_pipeline(rows, sealed)
    other = clone(model)
    training = pd.DataFrame({"x": np.arange(48, dtype=float) % 8})
    other.fit(training, (training["x"] >= 4).astype(int))
    if node == "xgboost_classifier":
        model.get_booster().load_model(other.get_booster().save_raw())
    else:
        model.booster_.model_from_string(other.booster_.model_to_string())
    assert not predict_pipeline(rows, sealed).equals(before)
    with pytest.raises(UnsupportedExecutionError, match="fitted state"):
        require_partition_safe_pipeline(sealed)
