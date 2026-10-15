"""Fail-closed admission for trusted fitted pandas pipelines on independent workers.

Certificates describe inspected state, not a security boundary or a replacement
for payload checksum validation. Recompute them after loading on each worker.
Native Spark declarations alone never authorize pandas batch execution.
"""

import json
from dataclasses import dataclass, fields
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression

from ..core.capabilities import UnsupportedExecutionError, require_capability
from ..core.portable_state import _pack
from ..modeling._evaluation.thresholds import _class_threshold_array
from ..modeling._tuning.engine import TuningApplier
from ..modeling._tuning.schemas import TuningResult
from ..modeling.base import StatefulEstimator
from ..modeling.classification import LogisticRegressionApplier
from ..modeling.regression import LinearRegressionApplier
from ..pipeline import SkyulfPipeline
from ..pipeline.seal import artifact_digest
from ..preprocessing.drop_and_missing.deduplicate import DeduplicateApplier
from ..preprocessing.drop_and_missing.drop_rows import DropMissingRowsApplier
from ..preprocessing.function_steps import (
    ColumnFunctionApplier,
    ColumnFunctionCalculator,
    FittedFunctionApplier,
    FittedFunctionCalculator,
    RowFilterFunctionApplier,
)
from ..preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from ..preprocessing.pipeline import FeatureEngineer
from ..preprocessing.scaling.standard import StandardScalerApplier, StandardScalerCalculator
from ..registry import NodeRegistry
from . import _partition_nodes as batch_nodes
from . import _partition_trees as batch_trees
from ._fitted_contract import check_fitted_schemas, resolve_fitted_step
from ._tuning_metadata import tuning_metadata_columns
from .fitted_pipeline import FittedPipelineArtifact
from .pipeline_scoring import scoring_output_schema
from .probed_rules import require_probed_rules
from .probed_steps import function_state_sha256, require_probed_step

_APPLIERS = {
    "SimpleImputer": SimpleImputerApplier,
    "StandardScaler": StandardScalerApplier,
    **batch_nodes.APPLIERS,
}
_CALCULATORS = {
    "SimpleImputer": SimpleImputerCalculator,
    "StandardScaler": StandardScalerCalculator,
    **batch_nodes.CALCULATORS,
}
_SKIPS = {
    "Deduplicate": DeduplicateApplier,
    "DropMissingRows": DropMissingRowsApplier,
    "RowFilterFunction": RowFilterFunctionApplier,
}
# Project function steps: admitted only with sealed training probe evidence.
_PROBED = {"ColumnFunction": ColumnFunctionApplier, "FittedFunction": FittedFunctionApplier}
_PROBED_CALCULATORS = {
    "ColumnFunction": ColumnFunctionCalculator,
    "FittedFunction": FittedFunctionCalculator,
}
_UNRECORDED = {"TrainTestSplitter", "Split", "feature_target_split"}
_MODELS = {LinearRegression: LinearRegressionApplier, LogisticRegression: LogisticRegressionApplier}


@dataclass(frozen=True)
class PartitionStepEvidence:
    """Detached evidence of one reviewed apply, probed apply or existing inference skip."""

    name: str
    node_type: str
    action: str
    config_json: str
    state_sha256: str


@dataclass(frozen=True)
class PartitionSafetyEvidence:
    """Bind the reviewed execution chain to its loaded payload and output contract."""

    certificate_version: int
    artifact_version: int
    pipeline_sha256: str
    fitted_engine: str
    config_sha256: str
    state_sha256: str
    output_schema: tuple[tuple[str, str], ...]
    steps: tuple[PartitionStepEvidence, ...]


def _reject(node: str, reason: str) -> UnsupportedExecutionError:
    """Keep unsupported execution diagnostics structured and tied to the worker path."""
    return UnsupportedExecutionError(node, "apply", "pandas", reason)


def _check_pipeline(artifact: FittedPipelineArtifact) -> None:
    """Require reviewed orchestration, saved schemas and a pandas fit before inspection."""
    pipeline, manifest = artifact.pipeline, artifact.manifest
    if manifest.format_version != 2:
        raise _reject("pipeline", "Re-export artifact format version 2 before Spark scoring.")
    if manifest.fitted_engine != "pandas" or pipeline.fitted_engine != "pandas":
        raise _reject("pipeline", "Spark workers require an artifact fitted on pandas.")
    if (
        type(pipeline) is not SkyulfPipeline
        or type(pipeline.feature_engineer) is not FeatureEngineer
    ):
        raise _reject("pipeline", "Custom pipeline or feature orchestration is unsupported.")
    _check_schemas(artifact)
    _check_engine_context(pipeline.feature_engineer)
    for component in (pipeline, pipeline.feature_engineer):
        _check_instance_methods(component)


def _check_engine_context(engineer: FeatureEngineer) -> None:
    """Reject retained native-engine contexts that would override pandas worker data."""
    options = engineer.execution_options
    if options is not None and options.engine != "pandas":
        raise _reject("pipeline", "Fitted execution_options engine conflicts with pandas workers.")
    if engineer.frame_spec is not None or engineer._spark_fitted:
        raise _reject("pipeline", "Native Spark engine context is unsupported on pandas workers.")


def _check_instance_methods(component: Any) -> None:
    """Reject instance-level method replacements on otherwise exact built-in types."""
    if any(callable(value) for value in vars(component).values()):
        raise _reject("pipeline", "Overridden inference methods are unsupported.")


def _check_schemas(artifact: FittedPipelineArtifact) -> None:
    """Bind column order and dtype metadata to the fitted inference schemas."""
    try:
        check_fitted_schemas(artifact)
    except ValueError as exc:
        raise _reject("pipeline", str(exc)) from exc


def _check_model(artifact: FittedPipelineArtifact) -> Any:
    """Admit only reviewed deterministic estimators and their exact prediction wrappers."""
    estimator = artifact.pipeline.model_estimator
    if type(estimator) is not StatefulEstimator:
        raise _reject("model", "Custom or absent model estimator is unsupported.")
    _check_instance_methods(estimator)
    model, actual_applier, tuning_result = _model_parts(estimator)
    applier = (
        _MODELS.get(type(model))
        or batch_trees.tree_applier(model)
        or batch_nodes.xgboost_applier(model)
    )
    if applier is None or type(actual_applier) is not applier:
        raise _reject(
            "model",
            "Only reviewed exact linear, logistic, sklearn tree and XGBRegressor models are admitted.",
        )
    _check_instance_methods(model)
    if vars(actual_applier):
        raise _reject("model", "Overridden model applier is unsupported.")
    if tuning_result is not None:
        _check_tuning_result(tuning_result, model, tuning_metadata_columns(artifact))
    actual = f"{type(model).__module__}.{type(model).__qualname__}"
    if actual != artifact.manifest.model_class:
        raise _reject("model", "Manifest model class disagrees with fitted model.")
    return estimator.model


def _model_parts(estimator: StatefulEstimator) -> tuple[Any, Any, TuningResult | None]:
    """Recognize only the exact tuple/facade emitted by the built-in tuner."""
    if type(estimator.model) is not tuple:
        return estimator.model, estimator.applier, None
    if len(estimator.model) != 2 or type(estimator.applier) is not TuningApplier:
        raise _reject("model", "Malformed or unreviewed tuning wrapper.")
    if set(vars(estimator.applier)) != {"base_applier"}:
        raise _reject("model", "Custom tuning wrapper state is unsupported.")
    model, result = estimator.model
    if type(result) is not TuningResult:
        raise _reject("model", "An exact TuningResult is required.")
    return model, estimator.applier.base_applier, result


def _check_tuning_result(
    result: TuningResult, model: Any, metadata_columns: tuple[str, ...]
) -> None:
    """Validate fixed serving policy and exclude unreviewed metadata/callback behavior."""
    if set(vars(result)) != {field.name for field in fields(TuningResult)}:
        raise _reject("model", "Custom tuning result state is unsupported.")
    _check_instance_methods(result)
    if type(result.excluded_feature_columns) is not list or result.excluded_feature_columns != list(
        metadata_columns
    ):
        raise _reject(
            "model", "Tuned feature exclusions require separate partition-safety evidence."
        )
    thresholds = result.decision_thresholds
    if thresholds is None:
        return
    if (
        type(model) not in {LogisticRegression, *batch_trees.CLASSIFIERS}
        or type(thresholds) is not dict
    ):
        raise _reject(
            "model", "Tuned thresholds require a reviewed classifier class-weight mapping."
        )
    if any(type(value) not in (int, float) for value in thresholds.values()):
        raise _reject("model", "Tuned thresholds must contain numeric scalar values.")
    _class_threshold_array(thresholds, model.classes_)


def _step_configs(engineer: FeatureEngineer) -> list[dict[str, Any]]:
    """Account for split markers that training deliberately omits from fitted history."""
    return [dict(step) for step in engineer.steps_config if step["transformer"] not in _UNRECORDED]


def _check_step_identity(record: dict, config: dict) -> None:
    """Reject changed recipe ordering, unknown identities and instance overrides."""
    node = record["type"]
    if config["name"] != record["name"] or config["transformer"] != node:
        raise ValueError("Configured step disagrees with fitted name/type.")
    expected = _APPLIERS.get(node) or _SKIPS.get(node) or _PROBED.get(node)
    if expected is None or type(record["applier"]) is not expected:
        raise ValueError("Unknown node or unreviewed applier identity.")
    if NodeRegistry.get_applier(node) is not expected:
        raise ValueError("Unreviewed applier registration.")
    if vars(record["applier"]):
        raise ValueError("Custom applier instance state is unsupported.")
    if record["artifact"].get("history_mode") == "carry":
        raise ValueError("Carry history cannot execute on independent workers.")


def _inspect_step(
    record: dict, config: dict, artifact: FittedPipelineArtifact | None = None
) -> PartitionStepEvidence:
    """Validate one known apply body without invoking its fit, apply or callbacks."""
    node, name = record["type"], record["name"]
    _check_step_identity(record, config)
    if node in _PROBED:
        return _inspect_probed_step(record, config, artifact)
    if node in _SKIPS:
        return PartitionStepEvidence(
            name,
            node,
            "skip_preserve_rows",
            json.dumps(_pack(config.get("params", {})), sort_keys=True),
            artifact_digest(record["artifact"]).hex(),
        )
    if NodeRegistry.get_calculator(node) is not _CALCULATORS[node]:
        raise ValueError("Unreviewed calculator registration.")
    state, params, _ = resolve_fitted_step(record, config)
    require_capability(
        node,
        "apply",
        "pandas",
        config=params,
        execution_kind="python_batch",
        row_effect="preserve",
        context="row",
    )
    return PartitionStepEvidence(
        name, node, "apply", json.dumps(_pack(params), sort_keys=True), artifact_digest(state).hex()
    )


def _inspect_probed_step(
    record: dict, config: dict, artifact: FittedPipelineArtifact | None
) -> PartitionStepEvidence:
    """Admit a project function step only with the opt-in flag and matching probe evidence.

    Without the loaded artifact there is no sealed evidence, so the step is rejected.
    """
    node = record["type"]
    if NodeRegistry.get_calculator(node) is not _PROBED_CALCULATORS[node]:
        raise ValueError("Unreviewed calculator registration.")
    state, params, _ = resolve_fitted_step(record, config, require_portable=False)
    require_probed_step(
        artifact.pipeline.config if artifact is not None else {},
        artifact.manifest.project_source_sha256 if artifact is not None else None,
        record["name"],
        node,
        state,
    )
    return PartitionStepEvidence(
        record["name"],
        node,
        "apply_probed",
        json.dumps(_pack(params), sort_keys=True),
        function_state_sha256(state),
    )


def _steps(artifact: FittedPipelineArtifact) -> tuple[PartitionStepEvidence, ...]:
    """Require complete ordered fitted history before granting per-step evidence."""
    engineer = artifact.pipeline.feature_engineer
    configs = _step_configs(engineer)
    metadata = tuning_metadata_columns(artifact)
    fitted = engineer.fitted_steps[1:] if metadata else engineer.fitted_steps
    if len(configs) != len(fitted):
        raise _reject("pipeline", "Incomplete or unsupported fitted step history.")
    active = {id(step) for step in engineer._transform_steps(preserve_rows=True)}
    result = []
    if metadata:
        state = engineer.fitted_steps[0]["artifact"]
        result.append(
            PartitionStepEvidence(
                "tuning_time_columns",
                "TuningColumnDrop",
                "drop_fit_metadata",
                json.dumps(_pack(state), sort_keys=True),
                artifact_digest(state).hex(),
            )
        )
    for record, config in zip(fitted, configs, strict=True):
        try:
            skipped = record["type"] in _SKIPS
            if (id(record) in active) == skipped:
                raise ValueError(
                    "Effective preserve_rows chain disagrees with reviewed skip contract."
                )
            result.append(_inspect_step(record, config, artifact))
        except (KeyError, TypeError, ValueError) as exc:
            raise _reject(
                str(record.get("type", "unknown")), f"Step {record.get('name')!r}: {exc}"
            ) from exc
    return tuple(result)


def require_partition_safe_pipeline(artifact: FittedPipelineArtifact) -> PartitionSafetyEvidence:
    """Inspect a loaded trusted artifact and return immutable pandas-worker evidence.

    No Spark session, callback or model prediction is invoked. Loaders must first
    verify the payload digest; workers recompute this evidence and compare with
    the driver certificate. Captured project source alone is not executable
    inference behavior and does not grant or remove node admission. Project
    ``column_step``/``fitted_step`` steps (undeclared or declared ``"row"``) and
    saved scoring rules are admitted only with matching training lint and probe
    evidence sealed in the pipeline config (``probed_steps``, ``probed_rules``).
    """
    if type(artifact) is not FittedPipelineArtifact:
        raise TypeError("Expected a loaded FittedPipelineArtifact.")
    try:
        return _pipeline_evidence(artifact)
    except UnsupportedExecutionError:
        raise
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise _reject("pipeline", f"Malformed fitted inference state: {exc}") from exc


def _pipeline_evidence(artifact: FittedPipelineArtifact) -> PartitionSafetyEvidence:
    """Build evidence after admission while keeping malformed metadata fail-closed."""
    _check_pipeline(artifact)
    steps = (*_steps(artifact), *_rules(artifact))
    model = _check_model(artifact)
    try:
        configured = artifact.pipeline.config.get("preprocessing", [])
        if artifact_digest(configured) != artifact_digest(
            artifact.pipeline.feature_engineer.steps_config
        ):
            raise ValueError("Pipeline recipe disagrees with fitted engineer configuration.")
        config_digest = artifact_digest(artifact.pipeline.config).hex()
        state_digest = artifact_digest(
            (model, artifact.pipeline._tuned_thresholds, steps, artifact.manifest.model_dump())
        ).hex()
    except (TypeError, ValueError) as exc:
        raise _reject("pipeline", f"Unsupported configuration or fitted state: {exc}") from exc
    manifest = artifact.manifest
    return PartitionSafetyEvidence(
        1,
        manifest.format_version,
        manifest.pipeline_sha256,
        manifest.fitted_engine,
        config_digest,
        state_digest,
        # Equals the raw prediction schema unless saved scoring rules add their columns.
        tuple((column.name, column.dtype) for column in scoring_output_schema(artifact)),
        steps,
    )


def _rules(artifact: FittedPipelineArtifact) -> tuple[PartitionStepEvidence, ...]:
    """Admit saved scoring rules only with sealed probe evidence, recorded as rules_probed."""
    try:
        admitted = require_probed_rules(artifact)
    except ValueError as exc:
        raise _reject("project_scoring", str(exc)) from exc
    return tuple(
        PartitionStepEvidence(name, node, "rules_probed", config_json, digest)
        for name, node, config_json, digest in admitted
    )
