"""Admit reviewed model components and row-local declarative or probed composition."""

from dataclasses import asdict
from typing import Any

from .fitted_pipeline import load_pipeline
from .model_set import ModelSetArtifact, verify_model_set_files
from .model_set_scoring import model_set_schema, validate_model_set_composition
from .probed_composition import require_probed_composition


def require_partition_safe_model_set(artifact: ModelSetArtifact) -> dict[str, Any]:
    """Validate every component and the independent, arithmetic or probed contract.

    Empty composition outputs execute no callback, even when inactive source is
    captured in the package. Python combined rules are certified only by sealed
    training lint and probe evidence that matches each saved rule, never by a
    user-supplied flag, source comment, or function name.
    """
    from .partition_safety import require_partition_safe_pipeline  # noqa: PLC0415

    verify_model_set_files(artifact)
    manifest = artifact.manifest
    config = manifest.composition_config
    if manifest.format_version != 1:
        raise ValueError("Spark model-set composition requires model-set format version 1.")
    probed = require_probed_composition(artifact)
    source = (artifact.directory / "composition.py").read_text(encoding="utf-8") if probed else ""
    validate_model_set_composition(config, source, manifest.components, manifest.record_key_schema)
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
    certificate = {
        "certificate_version": 1,
        "model_set_sha256": manifest.set_sha256,
        "composition_contract": _contract(config, probed),
        "components": components,
        "output_schema": [(column.name, column.dtype) for column in manifest.output_schema],
    }
    if probed:
        certificate["rules_probed"] = probed
    return certificate


def _contract(config: dict[str, Any], probed: list[str]) -> str:
    """Name the admitted composition: probed functions, declared operations or none."""
    if probed:
        return "probed_row_local_functions_v1"
    return "row_local_operations_v1" if config["outputs"] else "independent_components_v1"
