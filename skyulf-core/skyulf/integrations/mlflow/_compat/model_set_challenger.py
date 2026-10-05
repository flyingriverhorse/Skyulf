"""Compatibility alias for :mod:`skyulf.integrations.mlflow.lifecycle.model_set_challenger`."""

import sys

from skyulf.integrations.mlflow.lifecycle import model_set_challenger as _implementation

sys.modules[__name__] = _implementation
