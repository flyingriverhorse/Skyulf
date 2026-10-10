"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.fitting.candidate`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.fitting import candidate as _implementation

if TYPE_CHECKING:
    from .candidate import (
        CandidateResult as CandidateResult,
    )
    from .candidate import (
        LocalCandidateResult as LocalCandidateResult,
    )
    from .candidate import (
        LocalCVSpec as LocalCVSpec,
    )
    from .candidate import (
        LocalTrainingSpec as LocalTrainingSpec,
    )
    from .candidate import (
        NodeRegistry as NodeRegistry,
    )
    from .candidate import (
        TrainingSpec as TrainingSpec,
    )
    from .candidate import (
        _key_digest as _key_digest,
    )
    from .candidate import (
        _log_tuning_evidence as _log_tuning_evidence,
    )
    from .candidate import (
        _materialize_training_rows as _materialize_training_rows,
    )
    from .candidate import (
        apply_pre_split_step as apply_pre_split_step,
    )
    from .candidate import (
        candidate_config as candidate_config,
    )
    from .candidate import (
        compare_candidate as compare_candidate,
    )
    from .candidate import (
        compare_registered_local_models as compare_registered_local_models,
    )
    from .candidate import (
        eligible_training_snapshot as eligible_training_snapshot,
    )
    from .candidate import (
        eligible_training_source as eligible_training_source,
    )
    from .candidate import (
        evaluate_candidate as evaluate_candidate,
    )
    from .candidate import (
        evaluate_local_holdout as evaluate_local_holdout,
    )
    from .candidate import (
        evaluate_training_cv as evaluate_training_cv,
    )
    from .candidate import (
        fit_candidate as fit_candidate,
    )
    from .candidate import (
        fit_local_workflow as fit_local_workflow,
    )
    from .candidate import (
        log_fitted_candidate as log_fitted_candidate,
    )
    from .candidate import (
        log_local_model as log_local_model,
    )
    from .candidate import (
        log_pipeline_model as log_pipeline_model,
    )
    from .candidate import (
        log_training_explanations as log_training_explanations,
    )
    from .candidate import (
        partition_training_rows as partition_training_rows,
    )
    from .candidate import (
        pl as pl,
    )
    from .candidate import (
        read_training_partitions as read_training_partitions,
    )
    from .candidate import (
        read_training_snapshot as read_training_snapshot,
    )
    from .candidate import (
        register_candidate as register_candidate,
    )
    from .candidate import (
        register_model as register_model,
    )
    from .candidate import (
        replace as replace,
    )
    from .candidate import (
        sample_training_source as sample_training_source,
    )
    from .candidate import (
        split_labeled_snapshot as split_labeled_snapshot,
    )
    from .candidate import (
        train_candidate as train_candidate,
    )
    from .candidate import (
        train_local_candidate as train_local_candidate,
    )
    from .candidate import (
        training_spec_payload as training_spec_payload,
    )
    from .candidate import (
        validate_cv_holdout_policy as validate_cv_holdout_policy,
    )
    from .candidate import (
        validate_pre_split_step as validate_pre_split_step,
    )

sys.modules[__name__] = _implementation
