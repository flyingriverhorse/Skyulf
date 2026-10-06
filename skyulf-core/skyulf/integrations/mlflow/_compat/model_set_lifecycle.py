"""Compatibility alias for :mod:`skyulf.integrations.mlflow.lifecycle.model_set_lifecycle`."""

import sys

from skyulf.integrations.mlflow.lifecycle import model_set_lifecycle as _implementation

sys.modules[__name__] = _implementation
