"""Compatibility alias for :mod:`skyulf.integrations.mlflow.models.model`."""

import sys

from skyulf.integrations.mlflow.models import model as _implementation

sys.modules[__name__] = _implementation
