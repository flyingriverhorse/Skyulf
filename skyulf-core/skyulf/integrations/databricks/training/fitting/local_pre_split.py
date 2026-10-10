"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.fitting.pre_split`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.fitting import pre_split as _implementation

if TYPE_CHECKING:
    from .pre_split import (
        FIXED_TYPES as FIXED_TYPES,
    )
    from .pre_split import (
        custom_filter_columns as custom_filter_columns,
    )
    from .pre_split import (
        deduplicate_columns as deduplicate_columns,
    )
    from .pre_split import (
        fixed_columns as fixed_columns,
    )
    from .pre_split import (
        projected_fixed_steps as projected_fixed_steps,
    )
    from .pre_split import (
        target_contract as target_contract,
    )

sys.modules[__name__] = _implementation
