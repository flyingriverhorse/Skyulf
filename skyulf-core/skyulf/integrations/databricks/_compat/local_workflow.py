"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle.local_workflow`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.lifecycle import workflow as _implementation

if TYPE_CHECKING:
    from ..lifecycle.workflow import (
        AutoTrainingOutcome as AutoTrainingOutcome,
    )
    from ..lifecycle.workflow import (
        BundleActionResult as BundleActionResult,
    )
    from ..lifecycle.workflow import (
        LocalCandidateResult as LocalCandidateResult,
    )
    from ..lifecycle.workflow import (
        LocalCVSpec as LocalCVSpec,
    )
    from ..lifecycle.workflow import (
        LocalTrainingSpec as LocalTrainingSpec,
    )
    from ..lifecycle.workflow import (
        LocalWorkflowConfig as LocalWorkflowConfig,
    )
    from ..lifecycle.workflow import (
        _run_scoring_action as _run_scoring_action,
    )
    from ..lifecycle.workflow import (
        _scoring_config as _scoring_config,
    )
    from ..lifecycle.workflow import (
        automatic_promotion as automatic_promotion,
    )
    from ..lifecycle.workflow import (
        bind_target_name as bind_target_name,
    )
    from ..lifecycle.workflow import (
        build_bundle_result as build_bundle_result,
    )
    from ..lifecycle.workflow import (
        next_actions as next_actions,
    )
    from ..lifecycle.workflow import (
        prepare_training as prepare_training,
    )
    from ..lifecycle.workflow import (
        resolve_target_config as resolve_target_config,
    )
    from ..lifecycle.workflow import (
        resolve_training_spec as resolve_training_spec,
    )
    from ..lifecycle.workflow import (
        run_action as run_action,
    )
    from ..lifecycle.workflow import (
        training_settings as training_settings,
    )
    from ..lifecycle.workflow import (
        training_spec as training_spec,
    )
    from ..lifecycle.workflow import (
        training_window_mode as training_window_mode,
    )
    from ..lifecycle.workflow import (
        workflow_policies as workflow_policies,
    )

sys.modules[__name__] = _implementation
