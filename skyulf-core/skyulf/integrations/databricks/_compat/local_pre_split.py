"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.fitting.local_pre_split`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.fitting import pre_split as _implementation

if TYPE_CHECKING:
    from ..training.fitting.pre_split import (
        FIXED_TYPES as FIXED_TYPES,
    )
    from ..training.fitting.pre_split import (
        custom_filter_columns as custom_filter_columns,
    )
    from ..training.fitting.pre_split import (
        deduplicate_columns as deduplicate_columns,
    )
    from ..training.fitting.pre_split import (
        fixed_columns as fixed_columns,
    )
    from ..training.fitting.pre_split import (
        projected_fixed_steps as projected_fixed_steps,
    )
    from ..training.fitting.pre_split import (
        target_contract as target_contract,
    )

sys.modules[__name__] = _implementation
