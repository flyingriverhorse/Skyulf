"""Compatibility alias for :mod:`skyulf.integrations.mlflow.models.model_set`."""

import sys

from skyulf.integrations.mlflow.models import model_set as _implementation

sys.modules[__name__] = _implementation
