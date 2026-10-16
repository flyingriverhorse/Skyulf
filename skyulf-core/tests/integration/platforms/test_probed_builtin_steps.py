"""Built-in steps require matching partition evidence before worker admission."""

import pickle
import runpy
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference._partition_builtins import (
    PROBED_BUILTINS,
    builtin_contract,
    fresh_builtin_record,
)
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.inference.probed_steps import REQUIRED_CHECKS
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry

_INELIGIBLE_NODES = {
    "LagFeatures",
    "RollingAggregate",
    "Oversampling",
    "Undersampling",
    "DataSnapshot",
    "DatasetProfile",
}


@pytest.fixture(scope="module")
def builtin_recipes():
    """Reuse the real native replay fixtures so each admission case performs observable work."""
    return runpy.run_path(
        str(Path(__file__).parents[1] / "core" / "test_preprocessing_small_data_roundtrip.py")
    )


def _fit_builtin(node, recipes):
    """Fit and serialize actual node state without refitting during any probe check."""
    frame, target = recipes["_data"](node, "pandas")
    config = deepcopy(NodeRegistry.get_all_metadata()[node]["params"])
    config.update(deepcopy(recipes["_RECIPES"][node]))
    if node == "Casting":
        # Nullable output keeps the same dtype when a later input cannot be parsed.
        config["target_type"] = "Int64"
    state = NodeRegistry.get_calculator(node)().fit((frame, target), config)
    record = {
        "name": "probe",
        "type": node,
        "artifact": pickle.loads(pickle.dumps(state)),
        "params": config,
        "applier": NodeRegistry.get_applier(node)(),
    }
    return record, {"name": "probe", "transformer": node, "params": config}, frame


@pytest.mark.parametrize("node", PROBED_BUILTINS)
def test_builtin_saved_state_partition_checks(node, builtin_recipes):
    """A real saved recipe passes the checks under the fixed numerical comparison policy."""
    if node == "H3Index":
        pytest.importorskip("h3")
    record, recipe, frame = _fit_builtin(node, builtin_recipes)
    builtin_contract(record, recipe)
    detail, _ = _probe_step(
        record,
        recipe,
        frame,
        "pandas",
        (1, 2, 7),
        (256, 8 * 1024 * 1024),
        active=True,
        project_sha=None,
        perturb=True,
        fresh=fresh_builtin_record,
    )
    checks = {item["name"]: item["status"] for item in detail["checks"]}
    assert checks == dict.fromkeys(REQUIRED_CHECKS, "passed"), detail


@pytest.mark.parametrize("node", sorted(_INELIGIBLE_NODES))
def test_known_global_or_filtering_nodes_have_no_probe_identity(node):
    """A different recipe cannot admit a node explicitly excluded from the reviewed scope."""
    from skyulf.inference._partition_builtins import builtin_applier

    assert builtin_applier(node) is None


def _alias_artifact(tmp_path, modeling=None):
    """Keep a real categorical transformation before the reviewed encoder and model."""
    frame = pd.DataFrame(
        {"x": np.arange(64, dtype=float), "cat": ["red", "blue"] * 32, "y": [1.0, 4.0] * 32}
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "alias",
                    "transformer": "AliasReplacement",
                    "params": {
                        "columns": ["cat"],
                        "alias_type": "custom",
                        "custom_map": {"red": "scarlet"},
                    },
                },
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": ["cat"],
                        "max_categories": None,
                        "include_missing": False,
                    },
                },
            ],
            "modeling": modeling
            or {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=frame.iloc[:48], test=frame.iloc[48:]), target_column="y")
    path = tmp_path / "model"
    save_pipeline(pipeline, path)
    return load_pipeline(path), frame.iloc[48:], path


def test_builtin_evidence_admits_saved_pipeline_without_project_source(tmp_path):
    """Built-in recipes need their actual saved-state probe rather than arbitrary project code."""
    artifact, holdout, path = _alias_artifact(tmp_path)
    with pytest.raises(UnsupportedExecutionError, match="AliasReplacement"):
        require_partition_safe_pipeline(artifact)
    sealed = seal_partition_probe(artifact, holdout, path, enabled=True)
    certificate = require_partition_safe_pipeline(sealed)
    assert certificate.steps[0].action == "apply_probed_builtin"
    assert sealed.manifest.project_source_sha256 is None
    sample = holdout.drop(columns="y")
    whole = predict_pipeline(sample, sealed)
    pieces = pd.concat(
        [predict_pipeline(sample.iloc[i : i + 1], sealed) for i in range(len(sample))]
    )
    pd.testing.assert_frame_equal(whole, pieces, check_exact=True)
    assert require_partition_safe_pipeline(load_pipeline(path)) == certificate


@pytest.fixture(scope="module")
def sealed_builtin_path(tmp_path_factory):
    """Keep a disk artifact whose independent reloads can be tampered with safely."""
    artifact, holdout, path = _alias_artifact(tmp_path_factory.mktemp("builtin-sealed"))
    seal_partition_probe(artifact, holdout, path, enabled=True)
    return path


@pytest.mark.parametrize(
    "tamper",
    [
        "state",
        "recipe",
        "missing_check",
        "failed_check",
        "duplicate",
        "source",
        "rows",
        "version_type",
        "chunk_type",
    ],
)
def test_builtin_evidence_rejects_tampering(sealed_builtin_path, tamper):
    """Saved evidence cannot authorize changed state, recipe, source or missing observations."""
    artifact = load_pipeline(sealed_builtin_path)
    evidence = artifact.pipeline.config.get("partition_probe", {})
    if tamper == "state":
        artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]["extra"] = 1
    elif tamper == "recipe":
        artifact.pipeline.feature_engineer.steps_config[0]["params"]["custom_map"] = {}
    elif tamper == "missing_check":
        evidence["builtins"][0]["checks"].pop()
    elif tamper == "failed_check":
        evidence["builtins"][0]["checks"][0]["status"] = "failed"
    elif tamper == "duplicate":
        evidence["builtins"].append(deepcopy(evidence["builtins"][0]))
    elif tamper == "source":
        evidence["project_source_sha256"] = "0" * 64
    elif tamper == "version_type":
        evidence["version"] = True
    elif tamper == "chunk_type":
        evidence["chunk_sizes"] = [True, 2, 7]
    else:
        evidence["rows"] = 15
    with pytest.raises(UnsupportedExecutionError, match="AliasReplacement"):
        require_partition_safe_pipeline(artifact)


def test_builtin_applier_subclass_is_not_admitted(sealed_builtin_path):
    """Passing base-class evidence must not authorize an overridden transformation."""
    artifact = load_pipeline(sealed_builtin_path)
    record = artifact.pipeline.feature_engineer.fitted_steps[0]

    class Replacement(type(record["applier"])):
        """Impersonate the exact built-in owner without modifying the registry."""

    record["applier"] = Replacement()
    with pytest.raises(UnsupportedExecutionError, match="AliasReplacement"):
        require_partition_safe_pipeline(artifact)


def test_endpoint_plan_requires_model_and_builtin_evidence(tmp_path):
    """Endpoint preparation rechecks the same model and preprocessing certificate as Spark."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.serving import PinnedEndpointSpec, build_pinned_endpoint
    from skyulf.integrations.mlflow.models.pipeline_model import _signature
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel
    from skyulf.integrations.mlflow.spark import spark_model

    artifact, holdout, path = _alias_artifact(
        tmp_path,
        modeling={"type": "hist_gradient_boosting_regressor", "params": {"max_iter": 3}},
    )
    artifact = seal_partition_probe(artifact, holdout, path, enabled=True)
    certificate = spark_model.partition_safety_certificate(artifact)
    assert [step["action"] for step in certificate["steps"]] == [
        "apply_probed_builtin",
        "apply",
        "model_probed",
    ]
    spec = PinnedEndpointSpec("g5-proof", "main.ml.model", "1", "main", "observability", "g5")
    resolved = ResolvedModel(
        spec.model_name, spec.model_version, spec.model_uri, None, artifact.manifest.pipeline_sha256
    )
    package = SimpleNamespace(
        metadata={
            spark_model.SAFETY_KEY: certificate,
            spark_model.SOURCE_KEY: spark_model.runtime_source_digest(),
            "skyulf_artifact_kind": "local_pipeline",
            "skyulf_execution_scope": "whole_frame_local",
            "local_pipeline_digest": artifact.manifest.pipeline_sha256,
        },
        signature=_signature(artifact, spark_certified=True),
    )
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    assert plan.input_columns == ("x", "cat")
    assert plan.output_schema == (("prediction", "float64"),)
    artifact.pipeline.config.get("partition_probe", {})["model"]["checks"].pop()
    with pytest.raises(UnsupportedExecutionError):
        build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)


@pytest.mark.parametrize("replacement", ["callable", "module"])
def test_sealed_builtin_rejects_nested_transform_override(tmp_path, replacement):
    """Semantic digests omit routine attributes, so admission must reject those overrides itself."""
    frame = pd.DataFrame(
        {"x": np.arange(64, dtype=float), "b": np.arange(64) % 7.0, "y": np.arange(64) * 2.0}
    )
    frame.loc[frame.index % 5 == 0, "x"] = np.nan
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "knn", "transformer": "KNNImputer", "params": {"columns": ["x", "b"]}}
            ],
            "modeling": {"type": "decision_tree_regressor", "params": {"random_state": 42}},
        }
    )
    pipeline.fit(SplitDataset(train=frame.iloc[:48], test=frame.iloc[48:]), target_column="y")
    path = tmp_path / "knn"
    save_pipeline(pipeline, path)
    artifact = seal_partition_probe(load_pipeline(path), frame.iloc[48:], path, enabled=True)
    require_partition_safe_pipeline(artifact)
    sample = frame.iloc[48:][["x", "b"]]
    before = predict_pipeline(sample, artifact)
    state = artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]
    digest = artifact_digest(state)
    state["imputer_object"].transform = (
        (lambda values: np.full(np.asarray(values).shape, 999.0))
        if replacement == "callable"
        else ModuleType("replacement")
    )
    assert artifact_digest(state) == digest
    if replacement == "callable":
        assert not before.equals(predict_pipeline(sample, artifact))
    else:
        with pytest.raises(TypeError):
            predict_pipeline(sample, artifact)
    with pytest.raises(UnsupportedExecutionError, match="Overridden"):
        require_partition_safe_pipeline(artifact)


@pytest.mark.parametrize("node", ["count_vectorizer", "tfidf_vectorizer", "hashing_vectorizer"])
def test_builtin_vectorizer_rejects_nested_method_override(node, builtin_recipes):
    """Text transformers need the same nested method protection as numeric imputers."""
    record, recipe, _ = _fit_builtin(node, builtin_recipes)
    builtin_contract(record, recipe)
    record["artifact"]["vectorizer_object"].transform = lambda values: values
    with pytest.raises(ValueError, match="Overridden"):
        builtin_contract(record, recipe)


def test_custom_builtin_and_model_evidence_share_one_saved_artifact(tmp_path):
    """Source-bound custom replay must remain intact when builtins and model probes are added."""
    from tests.integration.platforms.test_probed_custom_steps import HOLDOUT, _project

    from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow

    config = _project(tmp_path)["pipeline"]
    config["admit_probed_custom_steps"] = True
    config["preprocessing"].append(
        {"name": "robust", "transformer": "RobustScaler", "params": {"columns": ["x", "z"]}}
    )
    config["modeling"] = {"type": "hist_gradient_boosting_regressor", "params": {"max_iter": 3}}
    rows = pd.DataFrame({"x": np.arange(32, dtype=float), "target": np.arange(32) * 2.0})
    path = tmp_path / "mixed"
    artifact = fit_workflow(
        config,
        SplitDataset(train=rows, test=rows.head(0)),
        target_column="target",
        artifact_path=path,
        max_rows=100,
        max_bytes=1024 * 1024,
    )
    artifact = seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    assert artifact.manifest.project_source_sha256 is not None
    certificate = require_partition_safe_pipeline(load_pipeline(path))
    assert [step.action for step in certificate.steps] == [
        "apply_probed",
        "apply_probed_builtin",
        "model_probed",
    ]
    expected = predict_pipeline(HOLDOUT[["x"]], artifact)
    actual = pd.concat(
        [predict_pipeline(HOLDOUT[["x"]].iloc[i : i + 1], artifact) for i in range(20)]
    )
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def test_builtin_sealing_rejects_small_holdout(tmp_path):
    """Builtin-only evidence needs the same meaningful sample size as custom and model probes."""
    artifact, holdout, path = _alias_artifact(tmp_path)
    with pytest.raises(ValueError, match="16 to 256"):
        seal_partition_probe(artifact, holdout.head(15), path, enabled=True)
    assert "partition_probe" not in load_pipeline(path).pipeline.config


@pytest.mark.parametrize("category", ["items", "get", "keys", "values", "update"])
def test_builtin_mapping_keys_are_not_method_overrides(category):
    """User category names may match dictionary methods without becoming executable state."""
    frame = pd.DataFrame({"cat": [category, "other"]})
    params = {"columns": ["cat"], "alias_type": "custom", "custom_map": {category: "goods"}}
    state = NodeRegistry.get_calculator("AliasReplacement")().fit((frame, None), params)
    record = {
        "name": "alias",
        "type": "AliasReplacement",
        "artifact": state,
        "params": params,
        "applier": NodeRegistry.get_applier("AliasReplacement")(),
    }
    builtin_contract(record, {"name": "alias", "transformer": "AliasReplacement", "params": params})
    output = record["applier"].apply(frame, state)
    assert output["cat"].tolist() == ["goods", "other"]


@pytest.mark.parametrize(
    "field,value",
    [("version", 1), ("float_tolerance", None), ("float_tolerance", {"rtol": 1e-6, "atol": 1e-12})],
)
def test_old_or_changed_comparison_policy_is_rejected(sealed_builtin_path, field, value):
    """Every admission route must require the trusted versioned tolerance policy."""
    artifact = load_pipeline(sealed_builtin_path)
    evidence = artifact.pipeline.config.get("partition_probe", {})
    assert evidence["version"] == 2
    assert evidence["float_tolerance"] == {"rtol": 1e-9, "atol": 1e-12}
    evidence[field] = value
    with pytest.raises(UnsupportedExecutionError, match="comparison policy"):
        require_partition_safe_pipeline(artifact)
