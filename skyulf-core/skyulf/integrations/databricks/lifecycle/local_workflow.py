"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle.workflow`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.lifecycle import workflow as _implementation

if TYPE_CHECKING:
    from .workflow import (
        AutoTrainingOutcome as AutoTrainingOutcome,
    )
    from .workflow import (
        BundleActionResult as BundleActionResult,
    )
    from .workflow import (
        LocalCandidateResult as LocalCandidateResult,
    )
    from .workflow import (
        LocalCVSpec as LocalCVSpec,
    )
    from .workflow import (
        LocalTrainingSpec as LocalTrainingSpec,
    )
    from .workflow import (
        LocalWorkflowConfig as LocalWorkflowConfig,
    )
    from .workflow import (
        _run_scoring_action as _run_scoring_action,
    )
    from .workflow import (
        _scoring_config as _scoring_config,
    )
    from .workflow import (
        automatic_promotion as automatic_promotion,
    )
    from .workflow import (
        bind_target_name as bind_target_name,
    )
    from .workflow import (
        build_bundle_result as build_bundle_result,
    )
    from .workflow import (
        next_actions as next_actions,
    )
    from .workflow import (
        prepare_training as prepare_training,
    )
    from .workflow import (
        resolve_target_config as resolve_target_config,
    )
    from .workflow import (
        resolve_training_spec as resolve_training_spec,
    )
    from .workflow import (
        run_action as run_action,
    )
    from .workflow import (
        training_settings as training_settings,
    )
    from .workflow import (
        training_spec as training_spec,
    )
    from .workflow import (
        training_window_mode as training_window_mode,
    )
    from .workflow import (
        workflow_policies as workflow_policies,
    )

sys.modules[__name__] = _implementation
