"""Compatibility alias for :mod:`skyulf.integrations.databricks.projects.project`."""

import sys

from skyulf.integrations.databricks.projects import project as _implementation

sys.modules[__name__] = _implementation
