"""Compatibility alias for :mod:`skyulf.integrations.mlflow.lifecycle.promotion`."""

import sys

from skyulf.integrations.mlflow.lifecycle import promotion as _implementation

sys.modules[__name__] = _implementation
