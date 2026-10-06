"""Compatibility alias for :mod:`skyulf.integrations.mlflow.registration.registry`."""

import sys

from skyulf.integrations.mlflow.registration import registry as _implementation

sys.modules[__name__] = _implementation
