"""Opt-in Spark admission of project row steps backed by sealed training probe evidence."""

import json
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from importlib import import_module
from typing import Any

import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.inference.probed_steps import REQUIRED_CHECKS, without_partition_probe
from skyulf.integrations.databricks.lifecycle.workflow import training_spec
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.projects.workflow_config import (
    preview_workflow_config,
    validate_workflow_config,
)
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.fitting import candidate as retraining
from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
    local_admission_probe,
    seal_partition_probe,
)
from skyulf.integrations.databricks.training.tuning.cv import CVSpec
from skyulf.pipeline import SkyulfPipeline

SOURCE = '''
import pandas as pd
from skyulf.preprocessing import column_step, fitted_step

MODE = "__MODE__"
CONTEXT = __CONTEXT__
KIND = "__KIND__"
ARMED = False
CACHE = {}


def learn(df, y):
    """Learn one training mean."""
    return {"mean": float(df["x"].mean())}


def apply_saved(df, state):
    """Apply the saved mean, or the selected batch-dependent defect."""
__APPLY__


def doubled(df):
    """Compute a stateless row-wise column, or the selected batch-dependent defect."""
__COLUMN__


def build_preprocessing(recipe="default"):
    """Expose one saved step through the project factory."""
    if KIND == "column":
        return [column_step("clip_duration", doubled, output="z", inference_context=CONTEXT)]
    steps = [
        fitted_step("clip_duration", learn, apply_saved, output="z", inference_context=CONTEXT)
    ]
    if KIND == "duplicate":
        steps.append(
            fitted_step("clip_duration", learn, apply_saved, output="w", inference_context=CONTEXT)
        )
    return steps
'''

# 20 distinct, unsorted values: more than the 16-row minimum the probe requires.
HOLDOUT = pd.DataFrame(
    {
        "x": [100.0 + ((7 * i) % 20) * 1.25 for i in range(20)],
        "target": [float(i) for i in range(20)],
    }
)


# Apply bodies per mode. ``getattr(series, "mean")()`` hides a batch statistic from the
# static lint on purpose, so the dynamic probe checks are tested on their own.
APPLY = {
    "row": 'return df["x"] - state["mean"]',
    "batch_mean": 'return df["x"] - getattr(df["x"], "mean")()',
    "lint_mean": 'return df["x"].fillna(df["x"].mean()) - state["mean"]',
    "empty_fail": (
        'if ARMED and df.empty:\n        raise ValueError("no rows")\n'
        '    return df["x"] - state["mean"]'
    ),
    "index": 'return df["x"] + df.index.to_series().astype(float) - state["mean"]',
    "memo": (
        'cache = globals()["CACHE"]\n'
        '    if ARMED and "mean" not in cache and not df.empty:\n'
        '        cache["mean"] = float(getattr(df["x"], "mean")())\n'
        '    return df["x"] - cache.get("mean", state["mean"])'
    ),
    "lint_global": (
        'if "mean" not in CACHE:\n        CACHE["mean"] = state["mean"]\n'
        '    return df["x"] - CACHE["mean"]'
    ),
    "empty_dtype": (
        'return pd.Series([value - state["mean"] for value in df["x"]], index=df.index)'
    ),
    # Batch maximum only matters above the sample's range: chunks agree, perturb does not.
    "threshold": 'return df["x"] - state["mean"] + (getattr(df["x"], "max")() > 124.9)',
}
COLUMN = {
    "row": 'return df["x"] * 2.0',
    "batch_mean": 'return df["x"] - getattr(df["x"], "mean")()',
}


def _arm(artifact):
    """Switch on scoring-only behaviour after training has fitted the step."""
    record = artifact.pipeline.feature_engineer.fitted_steps[0]
    vars(import_module(record["artifact"]["apply"].split(":")[0]))["ARMED"] = True


def _project(tmp_path, mode="row", context="row", kind="fitted", config=None):
    """Load a one-step project through the real project loader."""
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text("from .steps import build_preprocessing\n", encoding="utf-8")
    source = SOURCE.replace("__MODE__", mode).replace("__CONTEXT__", repr(context))
    source = source.replace("__KIND__", kind).replace("__APPLY__", "    " + APPLY[mode])
    source = source.replace("__COLUMN__", "    " + COLUMN.get(mode, COLUMN["row"]))
    (root / "steps.py").write_text(source + f"\n# Project: {tmp_path.name}\n", encoding="utf-8")
    base = config or {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    return load_project_workflow(base, root)


def _fitted(tmp_path, mode="row", context="row", flag=True, kind="fitted"):
    """Fit a project step through the real project loader and artifact format."""
    pipeline = _project(tmp_path, mode, context, kind)["pipeline"]
    if flag:
        pipeline["admit_probed_custom_steps"] = True
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    path = tmp_path / "artifact"
    artifact = fit_workflow(
        pipeline,
        SplitDataset(train=rows, test=rows[:0]),
        target_column="target",
        artifact_path=path,
        max_rows=10,
        max_bytes=10000,
    )
    return artifact, path


def _sealed(tmp_path, **kwargs):
    """Run the training-side probe exactly as fit_candidate does."""
    artifact, path = _fitted(tmp_path, **kwargs)
    return seal_partition_probe(artifact, HOLDOUT, path, enabled=kwargs.get("flag", True)), path


@pytest.mark.parametrize("kind", ["fitted", "column"])
def test_row_step_with_flag_is_admitted_and_worker_recomputes_certificate(tmp_path, kind):
    """An evidenced row step must be certified as apply_probed and recompute equal on a worker."""
    artifact, path = _sealed(tmp_path, kind=kind)
    probe = artifact.pipeline.config["partition_probe"]
    assert probe["version"] == 2 and probe["rows"] == 20 and probe["chunk_sizes"] == [1, 2, 7]
    assert probe["project_source_sha256"] == artifact.manifest.project_source_sha256
    assert [step["name"] for step in probe["steps"]] == ["clip_duration"]
    assert [check["name"] for check in probe["steps"][0]["checks"]] == ["lint", *REQUIRED_CHECKS]
    assert probe["lint_version"] == 1 and "perturb" in REQUIRED_CHECKS
    evidence = require_partition_safe_pipeline(artifact)
    assert [(step.name, step.action) for step in evidence.steps] == [
        ("clip_duration", "apply_probed")
    ]
    assert evidence.pipeline_sha256 == artifact.manifest.pipeline_sha256
    certificate = json.dumps(asdict(evidence))
    code = """
import json, sys
from dataclasses import asdict
from skyulf.inference.fitted_pipeline import load_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.integrations.mlflow.spark.spark_model import (
    runtime_source_digest,
    validate_worker_certificate,
)
artifact = load_pipeline(sys.argv[1])
validate_worker_certificate(artifact, json.loads(sys.argv[2]), runtime_source_digest())
print(json.dumps(asdict(require_partition_safe_pipeline(artifact))))
"""
    worker = subprocess.run(
        [sys.executable, "-c", code, str(path), certificate],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert worker.returncode == 0, worker.stderr
    assert json.loads(worker.stdout) == json.loads(certificate)
    query = HOLDOUT.loc[:, ["x"]]
    whole = predict_pipeline(query, artifact)
    split = pd.concat(
        [predict_pipeline(query.iloc[i : i + 1], artifact) for i in range(len(query))]
    )
    pd.testing.assert_frame_equal(whole, split)


@pytest.mark.parametrize("kind", ["fitted", "column"])
def test_batch_statistic_apply_fails_training_naming_step_and_check(tmp_path, kind):
    """A batch mean differs per Spark partition, so training must stop before registration."""
    artifact, path = _fitted(tmp_path, mode="batch_mean", kind=kind)
    with pytest.raises(
        ValueError,
        match=r"'clip_duration' changed results when scored in chunks "
        r"of 1 \(chunks:1: output_mismatch\); it must use only its saved",
    ):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    assert "partition_probe" not in load_pipeline(path).pipeline.config


def test_empty_batch_not_supported_fails_training(tmp_path):
    """Spark can send an empty batch, so a not_supported empty result is never a pass."""
    artifact, path = _fitted(tmp_path, mode="empty_fail")
    # Training fits on an empty test split; fail empty input only on scoring batches.
    _arm(artifact)
    with pytest.raises(ValueError, match=r"'clip_duration' failed on an empty batch \(empty: "):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_empty_batch_dtype_change_names_the_likely_cause(tmp_path):
    """A list-built result turns object-typed on empty input; the message must say so."""
    artifact, path = _fitted(tmp_path, mode="empty_dtype")
    with pytest.raises(
        ValueError,
        match=r"returned a different result \(often the dtype\) for an empty batch "
        r"\(empty: output_mismatch\)",
    ):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


@pytest.mark.parametrize("rows", [0, 1, 15])
def test_small_holdout_fails_training(tmp_path, rows):
    """Chunk checks degenerate on a few rows, so fewer than 16 must not produce evidence."""
    artifact, path = _fitted(tmp_path)
    with pytest.raises(ValueError, match=rf"at least 16 holdout rows .*the holdout has {rows}\."):
        seal_partition_probe(artifact, HOLDOUT.head(rows), path, enabled=True)


def test_index_dependent_apply_fails_training(tmp_path):
    """Spark batches get a fresh RangeIndex, so an apply that reads df.index must fail."""
    artifact, path = _fitted(tmp_path, mode="index")
    with pytest.raises(
        ValueError,
        match=r"changed results when the batch index was renumbered "
        r"\(index_shift: output_mismatch\)",
    ):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_module_level_cache_fails_training(tmp_path):
    """A memoized batch statistic in module state differs in a fresh worker process."""
    artifact, path = _fitted(tmp_path, mode="memo")
    _arm(artifact)
    with pytest.raises(
        ValueError,
        match=r"'clip_duration' gave different results from a freshly loaded copy of the "
        r"project code \(fresh_module: output_mismatch\)",
    ):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_duplicate_step_names_fail_training(tmp_path):
    """Evidence is keyed by step name and type, so duplicates would make it ambiguous."""
    artifact, path = _fitted(tmp_path, kind="duplicate")
    with pytest.raises(ValueError, match=r"'clip_duration' is used by more than one"):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_stale_staging_directory_is_replaced(tmp_path):
    """A staging folder left by an interrupted run must not block the next seal."""
    artifact, path = _fitted(tmp_path)
    (path.parent / (path.name + ".probed")).mkdir()
    sealed = seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    assert require_partition_safe_pipeline(sealed).steps[0].action == "apply_probed"


def test_probe_is_not_run_without_the_flag(tmp_path):
    """Projects that did not opt in keep their exact fitted payload."""
    artifact, path = _fitted(tmp_path, mode="batch_mean", flag=False)
    assert seal_partition_probe(artifact, HOLDOUT, path, enabled=False) is artifact


@pytest.mark.parametrize(
    ("kind", "node"), [("fitted", "FittedFunction"), ("column", "ColumnFunction")]
)
def test_row_step_without_flag_is_rejected(tmp_path, kind, node):
    """Without the project opt-in a custom step stays rejected even if it is row-wise."""
    artifact, _ = _sealed(tmp_path, flag=False, kind=kind)
    with pytest.raises(UnsupportedExecutionError, match="allow_tested_custom_steps: true") as err:
        require_partition_safe_pipeline(artifact)
    assert "clip_duration" in str(err.value) and node in str(err.value)


@pytest.mark.parametrize("context", ["group", "window", "global"])
def test_steps_declared_with_other_rows_are_rejected(tmp_path, context):
    """A declared group/window/global step needs other rows; evidence never admits it."""
    artifact, _ = _sealed(tmp_path, context=context)
    assert artifact.pipeline.config["partition_probe"]["steps"] == []
    with pytest.raises(UnsupportedExecutionError, match=f"inference_context='{context}'"):
        require_partition_safe_pipeline(artifact)


@pytest.mark.parametrize("kind", ["fitted", "column"])
def test_undeclared_step_is_inferred_row_local_when_lint_and_probe_pass(tmp_path, kind):
    """Passing lint and every probe check make the row declaration optional."""
    artifact, _ = _sealed(tmp_path, context=None, kind=kind)
    entry = artifact.pipeline.config["partition_probe"]["steps"][0]
    assert entry["checks"][0] == {"name": "lint", "status": "passed"}
    assert [check["name"] for check in entry["checks"][1:]] == list(REQUIRED_CHECKS)
    evidence = require_partition_safe_pipeline(artifact)
    assert [step.action for step in evidence.steps] == ["apply_probed"]


def test_lint_rejects_batch_statistic_with_location_hint_and_guide(tmp_path):
    """A batch statistic in apply() fails before the probe, naming file, line and call."""
    artifact, path = _fitted(tmp_path, mode="lint_mean")
    with pytest.raises(ValueError) as error:
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    message = str(error.value)
    assert re.search(
        r"Custom step 'clip_duration' reads a batch statistic \(steps\.py line \d+: "
        r"df\['x'\]\.mean\(\)\)\. Compute it in learn\(\) and read it from state",
        message.replace('"', "'"),
    ), message
    assert message.endswith("databricks_spark_scoring.html#when-the-probe-fails"), message


def test_lint_rejects_module_global_write(tmp_path):
    """A cache written into module globals differs per worker process, so lint rejects it."""
    artifact, path = _fitted(tmp_path, mode="lint_global")
    with pytest.raises(ValueError, match=r"'clip_duration' writes module state \(steps\.py line"):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_preview_runs_the_admission_lint_offline(tmp_path, workflow_config):
    """preview.py --action train shows the same lint message before any cloud job."""
    workflow_config.update(engine="pandas", inference_mode="spark")
    config = _project(tmp_path, mode="lint_mean", config=workflow_config)
    preview = preview_workflow_config(config, action="train")
    assert "Spark/endpoint admission lint" in preview
    assert "Custom step 'clip_duration' reads a batch statistic (steps.py line" in preview
    workflow_config["inference_mode"] = "local"
    assert "admission lint" not in preview_workflow_config(
        {**config, "inference_mode": "local"}, action="train"
    )


@pytest.mark.parametrize(("mode", "passed"), [("row", True), ("threshold", False)])
def test_local_probe_sample_fits_and_probes_offline(tmp_path, mode, passed):
    """smoke.py --probe reproduces the training probe on a laptop sample."""
    config = {
        **_project(tmp_path, mode, context=None),
        "input_columns": ["x"],
        "target_column": "target",
    }
    if passed:
        assert local_admission_probe(config, HOLDOUT).startswith(
            "Spark admission probe passed on 20 sample rows: clip_duration"
        )
    else:
        with pytest.raises(ValueError, match=r"\(perturb: output_mismatch\)"):
            local_admission_probe(config, HOLDOUT)


def _search_probe_setup(tmp_path, method):
    """Build a small real search with metadata deliberately excluded from model inputs."""
    config: dict[str, Any] = {
        **_project(tmp_path, context=None),
        "input_columns": ["x"],
        "target_column": "target",
        "cv_enabled": True,
        "cv_folds": 2,
        "cv_type": method,
        "cv_shuffle": method == "nested_cv",
        "cv_inner_folds": 2,
        "cv_nested_type": "stratified_k_fold" if method == "nested_cv" else "auto",
    }
    config["pipeline"]["modeling"] = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "decision_tree_classifier"},
        "strategy": "grid",
        "metric": "accuracy",
        "n_trials": 1,
        "search_space": {"max_depth": [2]},
        "tune_threshold": method == "nested_cv",
    }
    sample = pd.DataFrame(
        {"x": [float(i % 7) for i in range(80)], "target": [i % 2 for i in range(80)]}
    )
    if method == "time_series_split":
        config["event_column"] = "observed_at"
        sample["observed_at"] = pd.date_range("2025-01-01", periods=80, tz="UTC")
    if method == "group_k_fold":
        config["cv_group_column"] = "household"
        sample["household"] = [i // 4 for i in range(80)]
    return config, sample


@pytest.mark.parametrize("method", ["nested_cv", "time_series_split", "group_k_fold"])
def test_local_probe_preserves_workflow_search_folds(tmp_path, monkeypatch, method):
    """The local pre-check must bind workflow CV and retain split-only metadata like training."""
    from skyulf.integrations.databricks.scoring.batch import frame_batch

    config, sample = _search_probe_setup(tmp_path, method)
    fitted = []

    def capture_fit(*args, **kwargs):
        """Inspect the genuinely fitted artifact, without replacing its training behavior."""
        artifact = fit_workflow(*args, **kwargs)
        fitted.append(artifact)
        return artifact

    monkeypatch.setattr(frame_batch, "fit_workflow", capture_fit)
    original = deepcopy(config)
    result = local_admission_probe(config, sample)
    assert config == original
    assert fitted[0].pipeline.config["modeling"]["cv_type"] == method
    assert fitted[0].manifest.input_columns == ("x",)
    assert result == "Spark admission probe passed on 80 sample rows: clip_duration."


@pytest.fixture
def metadata_artifact(tmp_path):
    """Fit and seal an actual time-CV pipeline so guard tests inspect real saved state."""
    from skyulf.integrations.databricks.training.tuning.search import prepare_search_pipeline

    config, sample = _search_probe_setup(tmp_path, "time_series_split")
    pipeline = prepare_search_pipeline(
        config["pipeline"],
        CVSpec.from_workflow(config),
        target_column="target",
        event_column="observed_at",
    )
    pipeline["admit_probed_custom_steps"] = True
    path = tmp_path / "artifact"
    artifact = fit_workflow(
        pipeline,
        SplitDataset(train=sample, test=sample.head(0)),
        target_column="target",
        artifact_path=path,
        max_rows=100,
        max_bytes=1024 * 1024,
    )
    return seal_partition_probe(artifact, sample, path, enabled=True), sample


def test_time_cv_metadata_drop_is_admitted_and_preserves_partition_predictions(metadata_artifact):
    """The internal metadata drop is harmless for raw inference and must retain its evidence."""
    artifact, sample = metadata_artifact
    evidence = require_partition_safe_pipeline(artifact)
    assert evidence.steps[0].action == "drop_fit_metadata"
    raw = sample[["x"]]
    expected = predict_pipeline(raw, artifact)
    actual = pd.concat(
        [predict_pipeline(raw.iloc[i : i + 7], artifact) for i in range(0, len(raw), 7)]
    )
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "order",
        "state",
        "applier",
        "instance_state",
        "result",
        "feature",
    ],
)
def test_time_cv_metadata_drop_rejects_altered_evidence(metadata_artifact, mutation):
    """Recognizing the built-in drop must never admit arbitrary omissions or changed callbacks."""
    artifact, _ = metadata_artifact
    records = artifact.pipeline.feature_engineer.fitted_steps
    record = records[0]
    if mutation == "missing":
        records.pop(0)
    elif mutation == "duplicate":
        records.append(deepcopy(record))
    elif mutation == "order":
        records.reverse()
    elif mutation == "state":
        record["artifact"]["columns"] = ["x"]
    elif mutation == "applier":
        record["applier"] = records[1]["applier"]
    elif mutation == "instance_state":
        record["applier"].unexpected = True
    elif mutation == "result":
        artifact.pipeline.model_estimator.model[1].excluded_feature_columns = []
    else:
        record["artifact"]["columns"] = ["x"]
        artifact.pipeline.model_estimator.model[1].excluded_feature_columns = ["x"]
        artifact.pipeline.config["modeling"]["cv_time_column"] = "x"
    with pytest.raises(UnsupportedExecutionError, match="Tuning metadata removal"):
        require_partition_safe_pipeline(artifact)


def test_perturb_catches_batch_dependence_that_chunks_miss(tmp_path):
    """Changing one row must not move other rows, even when chunk results agree by luck."""
    artifact, path = _fitted(tmp_path, mode="threshold")
    with pytest.raises(
        ValueError,
        match=r"'clip_duration' changed other rows' results when one row's "
        r"inputs changed \(perturb: output_mismatch\)",
    ):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_perturb_runs_only_for_probed_steps_in_the_diagnostic_probe(tmp_path):
    """Built-in steps keep their diagnostic checks; perturb is an admission-only strategy."""
    artifact, _ = _fitted(tmp_path)
    sample = HOLDOUT.loc[:, ["x"]]
    plain = probe_fitted_preprocessing(artifact, sample)
    assert "perturb" not in [check["name"] for check in plain["steps"][0]["checks"]]
    probed = probe_fitted_preprocessing(artifact, sample, perturb_nodes=("FittedFunction",))
    assert {"name": "perturb", "status": "passed"} in probed["steps"][0]["checks"]


def test_built_in_certificate_shape_is_unchanged(tmp_path):
    """Certified packages compare certificates exactly, so built-in models keep their shape."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    pipeline.fit(SplitDataset(train=rows, test=rows[:0]), target_column="target")
    save_pipeline(pipeline, tmp_path / "plain")
    certificate = asdict(require_partition_safe_pipeline(load_pipeline(tmp_path / "plain")))
    assert list(certificate) == [
        "certificate_version",
        "artifact_version",
        "pipeline_sha256",
        "fitted_engine",
        "config_sha256",
        "state_sha256",
        "output_schema",
        "steps",
    ]
    assert certificate["certificate_version"] == 1
    assert certificate["output_schema"] == (("prediction", "float64"),)
    assert [(step["name"], step["action"]) for step in certificate["steps"]] == [("fill", "apply")]
    assert list(certificate["steps"][0]) == [
        "name",
        "node_type",
        "action",
        "config_json",
        "state_sha256",
    ]


def _drop_step(probe, artifact):
    """Remove the step's entry from otherwise valid evidence."""
    probe["steps"] = []


def _wrong_digest(probe, artifact):
    """Record a digest of different saved state."""
    probe["steps"][0]["state_sha256"] = "0" * 64


def _changed_state(probe, artifact):
    """Change the saved learned state after the probe ran."""
    artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]["state"]["mean"] = 0.0


def _wrong_source(probe, artifact):
    """Bind the evidence to different project source."""
    probe["project_source_sha256"] = "1" * 64


def _wrong_version(probe, artifact):
    """Use an evidence format admission does not understand."""
    probe["version"] = 999


def _fewer_chunks(probe, artifact):
    """Claim a weaker probe with fewer chunk sizes and checks."""
    probe["chunk_sizes"] = [7]
    probe["steps"][0]["checks"] = [c for c in probe["steps"][0]["checks"] if "chunks" not in c]


def _failed_check(probe, artifact):
    """Record a failed chunk check."""
    probe["steps"][0]["checks"][2]["status"] = "failed"


def _empty_not_supported(probe, artifact):
    """Record the empty check as not_supported."""
    probe["steps"][0]["checks"][-1]["status"] = "not_supported"


def _too_few_rows(probe, artifact):
    """Claim evidence from fewer rows than the chunk checks need."""
    probe["rows"] = 8


def _missing_evidence(probe, artifact):
    """Keep the flag but drop the sealed evidence entirely."""
    del artifact.pipeline.config["partition_probe"]


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_drop_step, "no entry for this step"),
        (_wrong_digest, "does not match this step's saved state"),
        (_changed_state, "does not match this step's saved state"),
        (_wrong_source, "other project source"),
        (_wrong_version, "comparison policy"),
        (_fewer_chunks, "evidence version"),
        (_failed_check, "does not show every check passed"),
        (_empty_not_supported, "does not show every check passed"),
        (_too_few_rows, "fewer than 16 holdout rows"),
        (_missing_evidence, "No training probe evidence"),
    ],
)
def test_tampered_or_missing_evidence_is_rejected(tmp_path, tamper, message):
    """Admission recomputes digests and rejects any evidence that does not match exactly."""
    artifact, _ = _sealed(tmp_path)
    artifact.pipeline.config = deepcopy(artifact.pipeline.config)
    tamper(artifact.pipeline.config.get("partition_probe"), artifact)
    with pytest.raises(UnsupportedExecutionError, match=message) as err:
        require_partition_safe_pipeline(artifact)
    assert "clip_duration" in str(err.value)


def test_effective_config_comparison_ignores_only_partition_probe():
    """Sealed evidence must not break config checks, which still compare every other key."""
    effective = {"preprocessing": [], "admit_probed_custom_steps": True}
    sealed = {**effective, "partition_probe": {"version": 1}}
    assert without_partition_probe(sealed) == effective
    assert without_partition_probe({**sealed, "admit_probed_custom_steps": False}) != effective


def _spec(**changes):
    """Use a pinned minimal random split."""
    spec = retraining.TrainingSpec(
        table="workspace.test.labels",
        version=4,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=2000,
        max_bytes=1048576,
    )
    return replace(spec, **changes)


def test_flag_keeps_dataset_identity_and_old_payloads():
    """The opt-in must not change pinned data identity or break older saved specs."""
    enabled = _spec(admit_probed_custom_steps=True)
    payload = retraining.training_spec_payload(enabled, "pandas")
    assert retraining.TrainingSpec.from_payload(payload) == enabled
    payload.pop("admit_probed_custom_steps")
    assert retraining.TrainingSpec.from_payload(payload) == _spec()
    assert enabled.dataset_id == _spec().dataset_id


@pytest.mark.parametrize("value", [None, 0, 1, "true", {}, []])
def test_flag_rejects_non_boolean_settings(workflow_config, value):
    """Config must never opt in through truthy strings or numbers."""
    workflow_config["admit_probed_custom_steps"] = value
    with pytest.raises(ValueError, match="admit_probed_custom_steps must be boolean"):
        validate_workflow_config(workflow_config, action="train")


def test_flag_reaches_spec_and_effective_pipeline_config(workflow_config):
    """The workflow setting must reach the spec and be sealed only when it is on."""
    workflow_config["admit_probed_custom_steps"] = True
    validate_workflow_config(workflow_config, action="train")
    assert training_spec(workflow_config).admit_probed_custom_steps is True

    step = {"name": "z", "transformer": "ColumnFunction", "params": {"function": "m:f"}}

    def effective(spec, steps):
        """Derive the effective pipeline config exactly as lifecycle initialization does."""
        return retraining.candidate_config(
            spec,
            {"preprocessing": steps, "modeling": {"type": "linear_regression"}},
            engine="pandas",
            cv=CVSpec(),
            metric="heldout_rmse",
            min_improvement=0.0,
            champion_version=None,
            quality_threshold=None,
            risk_category=None,
        )

    on = _spec(admit_probed_custom_steps=True)
    assert effective(on, [step])["admit_probed_custom_steps"] is True
    # Nothing to probe: the effective config of a project without custom code stays as before.
    assert "admit_probed_custom_steps" not in effective(on, [])
    assert "admit_probed_custom_steps" not in effective(_spec(), [step])


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, False),
        ({"inference_mode": "local"}, False),
        ({"inference_mode": "spark"}, True),
        ({"online_features": {"required_features": [], "freshness": {}}}, True),
        ({"inference_mode": "local", "admit_probed_custom_steps": True}, True),
    ],
)
def test_admission_probe_runs_automatically_for_spark_and_online_projects(
    workflow_config, settings, expected
):
    """Spark and online-feature projects probe custom code without a manual flag; local not."""
    from skyulf.integrations.databricks.lifecycle.workflow import probed_admission

    workflow_config.pop("inference_mode", None)
    workflow_config.update(settings)
    # online_features also needs a full feature lookup, so check the trigger on its own there.
    assert probed_admission(workflow_config) is expected
    if "online_features" not in settings:
        assert training_spec(workflow_config).admit_probed_custom_steps is expected


@pytest.mark.parametrize(
    "field,value",
    [("version", 1), ("float_tolerance", None), ("float_tolerance", {"rtol": 1e-6, "atol": 1e-12})],
)
def test_old_or_changed_comparison_policy_is_rejected(tmp_path, field, value):
    """Every admission route must require the trusted versioned tolerance policy."""
    artifact, _ = _sealed(tmp_path)
    assert artifact.pipeline.config["partition_probe"]["version"] == 2
    assert artifact.pipeline.config["partition_probe"]["float_tolerance"] == {
        "rtol": 1e-9,
        "atol": 1e-12,
    }
    artifact.pipeline.config["partition_probe"][field] = value
    with pytest.raises(UnsupportedExecutionError, match="comparison policy"):
        require_partition_safe_pipeline(artifact)
