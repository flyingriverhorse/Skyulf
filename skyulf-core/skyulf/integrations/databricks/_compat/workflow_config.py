"""Compatibility alias for :mod:`skyulf.integrations.databricks.projects.workflow_config`."""

import sys

from skyulf.integrations.databricks.projects import workflow_config as _implementation

sys.modules[__name__] = _implementation
