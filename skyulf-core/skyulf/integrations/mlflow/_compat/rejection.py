"""Compatibility alias for :mod:`skyulf.integrations.mlflow.lifecycle.rejection`."""

import sys

from skyulf.integrations.mlflow.lifecycle import rejection as _implementation

sys.modules[__name__] = _implementation
