"""Compatibility alias for :mod:`skyulf.integrations.mlflow.shared._client`."""

import sys

from skyulf.integrations.mlflow.shared import _client as _implementation

sys.modules[__name__] = _implementation
