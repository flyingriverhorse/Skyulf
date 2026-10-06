"""Compatibility alias for :mod:`skyulf.integrations.databricks.projects.competition_project`."""

import sys

from skyulf.integrations.databricks.projects import competition_project as _implementation

sys.modules[__name__] = _implementation
