"""Competition ranking requires comparable complete evidence and bounded search work."""

from copy import deepcopy

import pytest

from skyulf.integrations.databricks.training.competition.competition import (
    _trial_bound,
    choose_winner,
)
from skyulf.integrations.databricks.training.tuning.cv import CVSpec


def _row(name, score=1.0, mode="fixed_cv"):
    """Create two-fold evidence with an explicit common selection contract."""
    return {
        "candidate": name,
        "mean": score,
        "std": 0.0,
        "fold_scores": [score, score],
        "metric": "heldout_rmse",
        "scoring_metric": "neg_root_mean_squared_error",
        "direction": "minimize",
        "evaluation_mode": mode,
        "fold_membership_sha256": "same",
        "split_policy": {"method": "k_fold"},
    }


def test_fixed_and_tuned_candidates_share_ordinary_selection():
    """Ordinary fixed/tuned candidates are comparable despite different bias disclosures."""
    result = choose_winner(
        [_row("fixed", 2), _row("tuned", 1, "post_selection_cv")], {"fixed", "tuned"}
    )
    assert result["winner"] == "tuned"


def test_exact_ties_use_names_independent_of_completion_order():
    """Stable exact ties avoid timing-dependent winner changes."""
    assert choose_winner([_row("z"), _row("a")], {"a", "z"})["winner"] == "a"


@pytest.mark.parametrize(
    "field,value",
    [
        ("fold_membership_sha256", "changed"),
        ("evaluation_mode", "nested_cv"),
        ("mean", float("nan")),
        ("mean", 100.0),
        ("split_policy", {"method": "group_k_fold"}),
    ],
)
def test_incompatible_or_forged_scores_fail(field, value):
    """Incompatible folds or inconsistent score summaries cannot select a winner."""
    changed = deepcopy(_row("b"))
    changed[field] = value
    with pytest.raises(ValueError):
        choose_winner([_row("a"), changed], {"a", "b"})


def test_missing_requested_candidate_fails():
    """Partial success must not silently shrink the candidate set."""
    with pytest.raises(ValueError, match="incomplete"):
        choose_winner([_row("a")], {"a", "b"})


def test_single_call_training_does_not_ignore_competitors():
    """Only the phased training adapter can execute all candidates before registration."""
    from skyulf.integrations.databricks.lifecycle.workflow import run_action

    with pytest.raises(ValueError, match="phased lifecycle"):
        run_action(None, {"training_layout": "model_competition"}, "train")


def test_halving_bound_covers_survivor_rounds_and_nested_searches():
    """Odd candidate counts can require more than twice their count across rounds."""
    model = {
        "type": "hyperparameter_tuner",
        "strategy": "halving_random",
        "n_trials": 9,
        "strategy_params": {"factor": 2},
    }
    cv = CVSpec(enabled=True, method="nested_cv", folds=3)
    assert _trial_bound(model, cv) == (9 + 5 + 3 + 2) * 4
