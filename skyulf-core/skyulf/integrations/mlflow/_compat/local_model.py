"""Compatibility alias for :mod:`skyulf.integrations.mlflow.models.local_model`."""

import sys

from skyulf.integrations.mlflow.models import local_model as _implementation

sys.modules[__name__] = _implementation
