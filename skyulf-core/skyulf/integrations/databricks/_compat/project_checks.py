"""Compatibility alias for :mod:`skyulf.integrations.databricks.projects.project_checks`."""

import sys

from skyulf.integrations.databricks.projects import project_checks as _implementation

sys.modules[__name__] = _implementation
