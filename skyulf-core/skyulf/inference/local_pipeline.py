"""Compatibility alias for :mod:`skyulf.inference.fitted_pipeline`."""

import sys
from typing import TYPE_CHECKING

from skyulf.inference import fitted_pipeline as _implementation

if TYPE_CHECKING:
    from skyulf.inference.fitted_pipeline import (
        _MAX_MANIFEST_BYTES as _MAX_MANIFEST_BYTES,
    )
    from skyulf.inference.fitted_pipeline import (
        _MAX_PIPELINE_BYTES as _MAX_PIPELINE_BYTES,
    )
    from skyulf.inference.fitted_pipeline import (
        FittedPipelineArtifact as FittedPipelineArtifact,
    )
    from skyulf.inference.fitted_pipeline import (
        FittedPipelineManifest as FittedPipelineManifest,
    )
    from skyulf.inference.fitted_pipeline import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from skyulf.inference.fitted_pipeline import (
        LocalPipelineManifest as LocalPipelineManifest,
    )
    from skyulf.inference.fitted_pipeline import (
        _check_runtime as _check_runtime,
    )
    from skyulf.inference.fitted_pipeline import (
        _embedding_requirements as _embedding_requirements,
    )
    from skyulf.inference.fitted_pipeline import (
        _load_pipeline_payload as _load_pipeline_payload,
    )
    from skyulf.inference.fitted_pipeline import (
        _manifest as _manifest,
    )
    from skyulf.inference.fitted_pipeline import (
        _prediction_frame as _prediction_frame,
    )
    from skyulf.inference.fitted_pipeline import (
        _project_contract as _project_contract,
    )
    from skyulf.inference.fitted_pipeline import (
        _recorded_schemas as _recorded_schemas,
    )
    from skyulf.inference.fitted_pipeline import (
        _validate_manifest_schema as _validate_manifest_schema,
    )
    from skyulf.inference.fitted_pipeline import (
        _validate_tuned_thresholds as _validate_tuned_thresholds,
    )
    from skyulf.inference.fitted_pipeline import (
        load_local_pipeline as load_local_pipeline,
    )
    from skyulf.inference.fitted_pipeline import (
        load_pipeline as load_pipeline,
    )
    from skyulf.inference.fitted_pipeline import (
        pickle as pickle,
    )
    from skyulf.inference.fitted_pipeline import (
        predict_local_pipeline as predict_local_pipeline,
    )
    from skyulf.inference.fitted_pipeline import (
        predict_pipeline as predict_pipeline,
    )
    from skyulf.inference.fitted_pipeline import (
        read_bounded_artifact as read_bounded_artifact,
    )
    from skyulf.inference.fitted_pipeline import (
        require_local_pipeline_scope as require_local_pipeline_scope,
    )
    from skyulf.inference.fitted_pipeline import (
        require_pipeline_scope as require_pipeline_scope,
    )
    from skyulf.inference.fitted_pipeline import (
        save_local_pipeline as save_local_pipeline,
    )
    from skyulf.inference.fitted_pipeline import (
        save_pipeline as save_pipeline,
    )
    from skyulf.inference.fitted_pipeline import (
        validate_local_input as validate_local_input,
    )
    from skyulf.inference.fitted_pipeline import (
        validate_pipeline_input as validate_pipeline_input,
    )

sys.modules[__name__] = _implementation
