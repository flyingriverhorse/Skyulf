"""Compatibility alias for :mod:`skyulf.integrations.mlflow.models.pipeline_model`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.mlflow.models import pipeline_model as _implementation

if TYPE_CHECKING:
    from skyulf.integrations.mlflow.models.pipeline_model import (
        SkyulfLocalPythonModel as SkyulfLocalPythonModel,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        SkyulfPipelinePythonModel as SkyulfPipelinePythonModel,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        _input_example as _input_example,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        _restore_nullable_dtypes as _restore_nullable_dtypes,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        _signature as _signature,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        local_model_save_options as local_model_save_options,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        log_local_model as log_local_model,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        log_pipeline_model as log_pipeline_model,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        normalized_dtype as normalized_dtype,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        pip_requirements as pip_requirements,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        pipeline_model_save_options as pipeline_model_save_options,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        prepare_pyfunc_input as prepare_pyfunc_input,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        score_local_pipeline as score_local_pipeline,
    )
    from skyulf.integrations.mlflow.models.pipeline_model import (
        validate_local_destination as validate_local_destination,
    )

sys.modules[__name__] = _implementation
