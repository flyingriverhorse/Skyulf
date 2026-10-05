"""Admit independent partition-safe model components without custom composition."""

from dataclasses import asdict
from typing import Any

from .local_pipeline import load_local_pipeline
from .model_set import ModelSetArtifact, verify_model_set_files


def require_partition_safe_model_set(artifact: ModelSetArtifact) -> dict[str, Any]:
    """Validate every component and the reviewed independent-components contract.

    Empty composition outputs execute no callback, even when inactive source is
    captured in the package. Arbitrary Python composition is never certified by
    a user-supplied flag, source comment, or function name.
    """
    from .partition_safety import require_partition_safe_pipeline  # noqa: PLC0415

    verify_model_set_files(artifact)
    manifest = artifact.manifest
    if manifest.format_version != 1 or manifest.composition_config != {"outputs": []}:
        raise ValueError("Spark model-set composition requires independent_components_v1.")
    components = {}
    for component in manifest.components:
        local = load_local_pipeline(artifact.directory / "components" / component.branch)
        if local.manifest.pipeline_sha256 != component.reference.digest:
            raise ValueError("Spark model-set component digest differs from its reference.")
        components[component.branch] = asdict(require_partition_safe_pipeline(local))
    return {
        "certificate_version": 1,
        "model_set_sha256": manifest.set_sha256,
        "composition_contract": "independent_components_v1",
        "components": components,
        "output_schema": [(column.name, column.dtype) for column in manifest.output_schema],
    }
