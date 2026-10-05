"""Compatibility alias for :mod:`skyulf.integrations.mlflow.shared._model_metadata`."""

import sys

from skyulf.integrations.mlflow.shared import _model_metadata as _implementation

sys.modules[__name__] = _implementation
