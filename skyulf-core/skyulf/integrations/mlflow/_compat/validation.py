"""Compatibility alias for :mod:`skyulf.integrations.mlflow.lifecycle.validation`."""

import sys

from skyulf.integrations.mlflow.lifecycle import validation as _implementation

sys.modules[__name__] = _implementation
