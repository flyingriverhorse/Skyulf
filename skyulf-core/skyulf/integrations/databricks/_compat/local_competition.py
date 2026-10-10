"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.competition.local_competition`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.competition import competition as _implementation

if TYPE_CHECKING:
    from ..training.competition.competition import (
        LocalCVSpec as LocalCVSpec,
    )
    from ..training.competition.competition import (
        _pipeline_trial_bound as _pipeline_trial_bound,
    )
    from ..training.competition.competition import (
        _trial_bound as _trial_bound,
    )
    from ..training.competition.competition import (
        choose_winner as choose_winner,
    )
    from ..training.competition.competition import (
        prepare_competition as prepare_competition,
    )
    from ..training.competition.competition import (
        selected_request as selected_request,
    )
    from ..training.competition.competition import (
        validate_competition_budget as validate_competition_budget,
    )

sys.modules[__name__] = _implementation
