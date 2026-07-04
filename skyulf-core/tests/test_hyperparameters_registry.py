"""Gap-closing test for skyulf.modeling.hyperparameters._registry.build_ensemble_search_space."""

from skyulf.modeling.hyperparameters import build_ensemble_search_space


def test_build_ensemble_search_space_skips_unmapped_base_estimator_key():
    """A base_estimators key with no registry mapping should be silently skipped."""
    space = build_ensemble_search_space(
        "voting_classifier",
        base_estimators=["not_a_real_base_estimator_key"],
        problem_type="classification",
    )
    # No nested keys should be added for the unmapped estimator.
    assert not any(k.startswith("not_a_real_base_estimator_key__") for k in space)
    # The ensemble's own meta-param space should still be present.
    assert "voting" in space
