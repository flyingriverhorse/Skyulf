"""Compatibility alias for :mod:`skyulf.integrations.mlflow.runs.tracking`."""

import sys

from skyulf.integrations.mlflow.runs import tracking as _implementation

sys.modules[__name__] = _implementation
