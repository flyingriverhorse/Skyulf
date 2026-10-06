"""Compatibility alias for :mod:`skyulf.integrations.databricks.projects._project_files`."""

import sys

from skyulf.integrations.databricks.projects import _project_files as _implementation

sys.modules[__name__] = _implementation
