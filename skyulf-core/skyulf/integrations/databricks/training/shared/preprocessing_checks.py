"""Optional saved-model diagnostics on bounded held-out inputs."""

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from .....data.dataset import SplitDataset
from .....inference._partition_builtins import PROBED_BUILTINS, fresh_builtin_record
from .....inference.fitted_pipeline import FittedPipelineArtifact, load_pipeline, save_pipeline
from .....inference.preprocessing_probe import probe_fitted_preprocessing
from .....inference.probed_builtins import builtin_evidence, has_builtin_steps
from .....inference.probed_models import has_probed_model, probe_model
from .....inference.probed_rules import (
    has_scoring_rules,
    lint_rules,
    probe_rules,
    rule_evidence,
)
from .....inference.probed_steps import (
    PROBE_KEY,
    PROBED_NODES,
    fresh_rebinder,
    has_function_steps,
    lint_steps,
    module_state,
    partition_probe_evidence,
    require_project_source,
    require_unchanged_module_state,
)
from .....inference.project_code import fresh_project_module
from ....databricks.shared._frames import frame_bytes
from ..tuning.cv import CVSpec
from ..tuning.search import prepare_search_pipeline


def seal_partition_probe(
    artifact: FittedPipelineArtifact,
    holdout: pd.DataFrame | pl.DataFrame,
    artifact_path: str | Path,
    *,
    enabled: bool,
) -> FittedPipelineArtifact:
    """Probe built-ins/models and lint/probe custom steps and rules, sealing evidence.

    Runs when Spark admission applies (``enabled``: Spark or online-feature
    projects, or ``admit_probed_custom_steps``) and the pipeline has a
    eligible built-in/model, function step or saved scoring rules. Uses the
    same first 256 holdout rows as the diagnostic probe and needs at least 16. A
    lint finding, a failing check, or a step that changes JSON values in its
    module's globals while probed raises before the model is logged, naming the
    step or rule, the check and a fix. The evidence lives in ``pipeline.config``
    so the payload checksum covers it; the artifact is re-saved and reloaded.
    """
    if not enabled or not (
        has_function_steps(artifact)
        or has_scoring_rules(artifact)
        or has_probed_model(artifact)
        or has_builtin_steps(artifact)
    ):
        return artifact
    if artifact.manifest.fitted_engine != "pandas":
        # Spark workers and endpoints admit only pandas-fitted models; nothing to seal.
        return artifact
    evidence = partition_admission_evidence(artifact, holdout)
    pipeline = artifact.pipeline
    pipeline.config = {**pipeline.config, PROBE_KEY: evidence}
    path = Path(artifact_path)
    staged = path.with_name(path.name + ".probed")
    if staged.exists():
        # Left behind by an interrupted earlier run; it never replaced the artifact.
        shutil.rmtree(staged)
    save_pipeline(pipeline, staged, use_tuned_thresholds=artifact.manifest.use_tuned_thresholds)
    shutil.rmtree(path)
    staged.replace(path)
    return load_pipeline(path)


def partition_admission_evidence(
    artifact: FittedPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame
) -> dict[str, Any]:
    """Probe eligible model state and source-bound steps/rules, or raise a guided error.

    Also used offline by ``smoke.py --probe`` on a locally fitted model.
    """
    if has_function_steps(artifact) or has_scoring_rules(artifact):
        evidence = _custom_admission_evidence(artifact, holdout)
    else:
        evidence = {
            "version": 1,
            "rows": min(len(holdout), 256),
            "chunk_sizes": [1, 2, 7],
            "steps": [],
        }
        if has_builtin_steps(artifact):
            report = _holdout_probe(
                artifact, holdout, perturb_nodes=PROBED_BUILTINS, fresh=fresh_builtin_record
            )
            evidence["project_source_sha256"] = artifact.manifest.project_source_sha256
            evidence["builtins"] = builtin_evidence(artifact, report)
    if has_probed_model(artifact):
        evidence["model"] = probe_model(artifact, holdout)
    return evidence


def _custom_admission_evidence(
    artifact: FittedPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame
) -> dict[str, Any]:
    """Preserve the source-bound version-one step and rule certificate contract."""
    source = require_project_source(artifact)
    lint_steps(artifact)
    lint_rules(artifact)
    before = module_state(artifact)
    with fresh_project_module(source) as fresh:
        old = "_skyulf_project_" + str(artifact.manifest.project_source_sha256)
        rebind = fresh_rebinder(old, fresh.__name__)
        report = _holdout_probe(
            artifact,
            holdout,
            perturb_nodes=(*PROBED_NODES, *PROBED_BUILTINS),
            fresh=lambda record: (
                fresh_builtin_record(record)
                if record["type"] in PROBED_BUILTINS
                else rebind(record)
            ),
        )
        evidence = partition_probe_evidence(artifact, report)
        if has_builtin_steps(artifact):
            evidence["builtins"] = builtin_evidence(artifact, report)
        require_unchanged_module_state(artifact, before)
        if has_scoring_rules(artifact):
            evidence["rules"] = rule_evidence(
                artifact, probe_rules(artifact, _inputs(artifact, holdout), fresh)
            )
    return evidence


def local_admission_probe(config: dict[str, Any], sample: pd.DataFrame) -> str:
    """Fit the resolved recipe on a local labeled sample and run the admission probe.

    For ``smoke.py --probe``: the same sample is the training rows and
    the probe holdout, so a failure appears on the laptop before a train job.
    It checks behavior on these rows only; training probes the real holdout.
    """
    from ...scoring.batch.frame_batch import fit_workflow  # noqa: PLC0415

    cv = CVSpec.from_workflow(config)
    pipeline = prepare_search_pipeline(
        config["pipeline"],
        cv,
        target_column=config["target_column"],
        event_column=config.get("event_column"),
    )
    columns = [*config["input_columns"], config["target_column"]]
    for key in ("cv_time_column", "cv_group_column"):
        column = pipeline.get("modeling", {}).get(key)
        if column and column not in columns:
            columns.append(column)
    rows = sample.loc[:, columns].reset_index(drop=True)
    with tempfile.TemporaryDirectory(prefix="skyulf-probe-sample-") as directory:
        artifact = fit_workflow(
            pipeline,
            SplitDataset(train=rows, test=rows.head(0)),
            target_column=config["target_column"],
            artifact_path=Path(directory) / "model",
            max_rows=max(len(rows), 1),
            max_bytes=max(frame_bytes(rows) * 4, 1024 * 1024),
        )
        evidence = partition_admission_evidence(artifact, rows)
    names = [step["name"] for step in evidence["steps"]]
    names += [step["name"] for step in evidence.get("builtins", [])]
    names += [rule["name"] for rule in evidence.get("rules", [])]
    if "model" in evidence:
        names.append(evidence["model"]["node_type"])
    return f"Spark admission probe passed on {len(rows)} sample rows: {', '.join(names) or 'none'}."


def log_partition_probe(run: Any, artifact: FittedPipelineArtifact) -> None:
    """Show which steps and rules passed the admission probe in the training report."""
    evidence = artifact.pipeline.config.get(PROBE_KEY)
    if evidence is not None:
        run.client.log_dict(run.run_id, evidence, "partition_probe.json")


def log_preprocessing_probe(
    run: Any,
    artifact: FittedPipelineArtifact,
    holdout: pd.DataFrame | pl.DataFrame,
    *,
    enabled: bool,
) -> None:
    """Log a redacted diagnostic without refitting or changing promotion policy.

    Inspect at most the first 256 holdout rows, with the existing probe's 8 MiB
    input/output limit. Callback failures are diagnostic results; MLflow write
    errors retain the surrounding training workflow's failure policy.
    """
    if not enabled:
        return
    report = _holdout_probe(artifact, holdout)
    run.client.log_dict(run.run_id, report, "preprocessing_probe.json")


def _inputs(
    artifact: FittedPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame
) -> pd.DataFrame | pl.DataFrame:
    """Select only the model's raw input columns from the holdout."""
    columns = list(artifact.manifest.input_columns)
    return holdout.select(columns) if isinstance(holdout, pl.DataFrame) else holdout.loc[:, columns]


def _sample(
    artifact: FittedPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame
) -> pd.DataFrame | pl.DataFrame:
    """Select the first 256 holdout rows and only the model's raw input columns."""
    return _inputs(artifact, holdout.head(256))


def _holdout_probe(
    artifact: FittedPipelineArtifact,
    holdout: pd.DataFrame | pl.DataFrame,
    *,
    perturb_nodes: tuple[str, ...] = (),
    fresh: Callable[[dict], dict] | None = None,
) -> dict[str, Any]:
    """Keep values and exception messages out of a failed or unavailable check."""
    report: dict[str, Any] = {
        "report_version": 1,
        "status": "not_run",
        "admission": "diagnostic_only",
        "sample_rows": min(len(holdout), 256),
        "fitted_engine": artifact.manifest.fitted_engine,
        "pipeline_sha256": artifact.manifest.pipeline_sha256,
        "project_source_sha256": artifact.manifest.project_source_sha256,
        "steps": [],
    }
    if not len(holdout):
        return {**report, "reason": "empty_holdout"}
    try:
        return probe_fitted_preprocessing(
            artifact, _sample(artifact, holdout), perturb_nodes=perturb_nodes, fresh=fresh
        )
    except Exception as exc:  # noqa: BLE001 - optional diagnostic must redact callback failures
        return {**report, "status": "failed", "error_type": type(exc).__name__}
