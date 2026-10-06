"""Compatibility alias for :mod:`skyulf.integrations.databricks.projects._project_recipes`."""

import sys

from skyulf.integrations.databricks.projects import _project_recipes as _implementation

sys.modules[__name__] = _implementation
