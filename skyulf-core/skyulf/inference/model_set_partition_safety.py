"""Admit reviewed model components and explicit row-local declarative composition."""

from dataclasses import asdict
from typing import Any

from .fitted_pipeline import load_pipeline
from .model_set import ModelSetArtifact, verify_model_set_files
from .model_set_scoring import model_set_schema, validate_model_set_composition


def require_partition_safe_model_set(artifact: ModelSetArtifact) -> dict[str, Any]:
    """Validate every component and the reviewed independent or arithmetic contract.

    Empty composition outputs execute no callback, even when inactive source is
    captured in the package. Arbitrary Python composition is never certified by
    a user-supplied flag, source comment, or function name.
    """
    from .partition_safety import require_partition_safe_pipeline  # noqa: PLC0415

    verify_model_set_files(artifact)
    manifest = artifact.manifest
    config = manifest.composition_config
    if manifest.format_version != 1 or any("operation" not in rule for rule in config["outputs"]):
        raise ValueError(
            "Spark model-set composition requires independent_components_v1 or row_local_operations_v1."
        )
    validate_model_set_composition(config, "", manifest.components, manifest.record_key_schema)
    if (
        model_set_schema(manifest.components, manifest.record_key_schema, config)
        != manifest.output_schema
    ):
        raise ValueError("Spark model-set composition schema disagrees with its manifest.")
    components = {}
    for component in manifest.components:
        local = load_pipeline(artifact.directory / "components" / component.branch)
        if local.manifest.pipeline_sha256 != component.reference.digest:
            raise ValueError("Spark model-set component digest differs from its reference.")
        components[component.branch] = asdict(require_partition_safe_pipeline(local))
    return {
        "certificate_version": 1,
        "model_set_sha256": manifest.set_sha256,
        "composition_contract": "row_local_operations_v1"
        if config["outputs"]
        else "independent_components_v1",
        "components": components,
        "output_schema": [(column.name, column.dtype) for column in manifest.output_schema],
    }
