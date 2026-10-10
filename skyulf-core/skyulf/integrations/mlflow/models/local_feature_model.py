"""Compatibility alias for :mod:`skyulf.integrations.mlflow.models.feature_model`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.mlflow.models import feature_model as _implementation

if TYPE_CHECKING:
    from skyulf.integrations.mlflow.models.feature_model import (
        _RUN_LOCK as _RUN_LOCK,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        FEATURE_SPEC_DIGEST_KEY as FEATURE_SPEC_DIGEST_KEY,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        FEATURE_STORE_KEY as FEATURE_STORE_KEY,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        RAW_MODEL_PATH_KEY as RAW_MODEL_PATH_KEY,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _active_logging_run as _active_logging_run,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _contained as _contained,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _end_owned_run as _end_owned_run,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _finish_feature_package as _finish_feature_package,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _log_feature_options as _log_feature_options,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _outer_worker_wheels as _outer_worker_wheels,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _raw_model_path as _raw_model_path,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _restore_environment_run as _restore_environment_run,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _validate_contract as _validate_contract,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _validate_envelope as _validate_envelope,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        _validate_signatures as _validate_signatures,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        copy_feature_package as copy_feature_package,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        feature_package_models as feature_package_models,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        log_feature_model_set as log_feature_model_set,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        log_feature_pipeline_model as log_feature_pipeline_model,
    )
    from skyulf.integrations.mlflow.models.feature_model import (
        log_local_feature_model as log_local_feature_model,
    )

sys.modules[__name__] = _implementation
