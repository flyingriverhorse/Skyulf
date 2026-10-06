"""Compatibility alias for :mod:`skyulf.integrations.mlflow.shared._nullable_transport`."""

import sys

from skyulf.integrations.mlflow.shared import _nullable_transport as _implementation

sys.modules[__name__] = _implementation
