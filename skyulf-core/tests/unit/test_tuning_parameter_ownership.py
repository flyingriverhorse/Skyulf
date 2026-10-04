"""Tuning model construction must not modify caller-owned estimator defaults."""

from threading import Lock
from typing import Any

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skyulf.modeling._tuning.params import instantiate_model


@pytest.mark.parametrize(
    ("model_class", "tree_class", "container", "prefix"),
    [
        (VotingClassifier, DecisionTreeClassifier, "estimators", "tree"),
        (VotingRegressor, DecisionTreeRegressor, "estimators", "tree"),
        (StackingClassifier, DecisionTreeClassifier, "estimators", "tree"),
        (StackingRegressor, DecisionTreeRegressor, "estimators", "tree"),
        (Pipeline, DecisionTreeClassifier, "steps", "tree"),
        (CalibratedClassifierCV, DecisionTreeClassifier, "estimator", "estimator"),
        (AdaBoostClassifier, DecisionTreeClassifier, "estimator", "estimator"),
    ],
)
def test_nested_candidate_does_not_change_later_default_model(
    model_class, tree_class, container, prefix
):
    """One trial's nested update must not become the next trial's fixed default."""
    original = tree_class(max_depth=4, random_state=17)
    defaults = {container: original if container == "estimator" else [("tree", original)]}
    candidate = instantiate_model(model_class, {**defaults, f"{prefix}__max_depth": 1})
    later = instantiate_model(model_class, defaults)

    assert candidate.get_params()[f"{prefix}__max_depth"] == 1
    assert later.get_params()[f"{prefix}__max_depth"] == 4
    assert original.max_depth == 4
    assert original.random_state == 17
    candidate.get_params()[prefix].fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert not hasattr(original, "tree_")
    assert not hasattr(later.get_params()[prefix], "tree_")


@pytest.mark.parametrize("source", ["constructor", "named", "nested"])
def test_supplied_fitted_children_remain_fitted_and_independent(source):
    """Detaching supplied estimators must retain learned state and replacement precedence."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    fitted = DecisionTreeClassifier(max_depth=4, random_state=17).fit(X, y)
    placeholder = DecisionTreeClassifier(max_depth=7)
    if source == "constructor":
        params = {"steps": [("tree", fitted)], "tree__max_depth": 2}
        prefix = "tree"
    elif source == "named":
        params = {"steps": [("tree", placeholder)], "tree": fitted, "tree__max_depth": 2}
        prefix = "tree"
    else:
        params = {
            "steps": [("calibrated", CalibratedClassifierCV(estimator=placeholder))],
            "calibrated__estimator": fitted,
            "calibrated__estimator__max_depth": 2,
        }
        prefix = "calibrated__estimator"

    model = instantiate_model(Pipeline, params)
    child = model.get_params()[prefix]
    np.testing.assert_array_equal(child.predict(X), y)
    assert child.max_depth == 2
    assert fitted.max_depth == 4
    assert placeholder.max_depth == 7
    child.fit(X, 1 - y)
    np.testing.assert_array_equal(fitted.predict(X), y)


def test_constructor_filtering_still_ignores_unsupported_values():
    """Discarded configuration keys must not introduce new copy or pickle failures."""
    model = instantiate_model(DecisionTreeClassifier, {"max_depth": 2, "ignored": Lock()})
    assert model.max_depth == 2


@pytest.mark.parametrize("source", ["constructor", "named", "nested"])
def test_shared_child_aliases_are_preserved_inside_the_new_model(source):
    """Copying caller state must not split intentionally shared estimator references."""
    original = DecisionTreeClassifier(max_depth=4)
    params: dict[str, Any] = {"estimators": [("left", original)], "left__max_depth": 2}
    if source == "constructor":
        params["final_estimator"] = original
    elif source == "named":
        params["estimators"].append(("right", DecisionTreeClassifier()))
        params["right"] = original
    else:
        params["final_estimator"] = Pipeline([("tree", DecisionTreeClassifier())])
        params["final_estimator__tree"] = original

    model = instantiate_model(StackingClassifier, params)
    right_key = {
        "constructor": "final_estimator",
        "named": "right",
        "nested": "final_estimator__tree",
    }[source]
    assert model.get_params()[right_key] is model.get_params()["left"]
    assert model.get_params()[right_key].max_depth == 2
    assert original.max_depth == 4


def test_failed_nested_update_does_not_partially_mutate_defaults():
    """A rejected candidate must not leave an earlier nested assignment in caller state."""
    original = DecisionTreeClassifier(max_depth=4)
    with pytest.raises(ValueError, match="Invalid parameter"):
        instantiate_model(
            Pipeline,
            {
                "steps": [("tree", original)],
                "tree__max_depth": 2,
                "tree__unsupported": 3,
            },
        )
    assert original.max_depth == 4
