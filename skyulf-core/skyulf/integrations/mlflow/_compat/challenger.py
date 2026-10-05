"""Compatibility alias for :mod:`skyulf.integrations.mlflow.lifecycle.challenger`."""

import sys

from skyulf.integrations.mlflow.lifecycle import challenger as _implementation

sys.modules[__name__] = _implementation
