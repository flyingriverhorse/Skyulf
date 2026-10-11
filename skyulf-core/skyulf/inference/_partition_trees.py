"""Exact fitted sklearn tree contracts for independent pandas inference batches."""

from importlib import import_module
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

from ..modeling import classification, regression
from ..registry import NodeRegistry

_tree = import_module("sklearn.tree._tree")

_CONTRACTS = {
    DecisionTreeRegressor: (
        "decision_tree_regressor",
        regression.DecisionTreeRegressorCalculator,
        regression.DecisionTreeRegressorApplier,
    ),
    DecisionTreeClassifier: (
        "decision_tree_classifier",
        classification.DecisionTreeClassifierCalculator,
        classification.DecisionTreeClassifierApplier,
    ),
    RandomForestRegressor: (
        "random_forest_regressor",
        regression.RandomForestRegressorCalculator,
        regression.RandomForestRegressorApplier,
    ),
    RandomForestClassifier: (
        "random_forest_classifier",
        classification.RandomForestClassifierCalculator,
        classification.RandomForestClassifierApplier,
    ),
    ExtraTreesRegressor: (
        "extra_trees_regressor",
        regression.ExtraTreesRegressorCalculator,
        regression.ExtraTreesRegressorApplier,
    ),
    ExtraTreesClassifier: (
        "extra_trees_classifier",
        classification.ExtraTreesClassifierCalculator,
        classification.ExtraTreesClassifierApplier,
    ),
}
_CHILDREN = {
    RandomForestRegressor: DecisionTreeRegressor,
    RandomForestClassifier: DecisionTreeClassifier,
    ExtraTreesRegressor: ExtraTreeRegressor,
    ExtraTreesClassifier: ExtraTreeClassifier,
}
CLASSIFIERS = frozenset((DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier))


def _check_methods(model: Any) -> None:
    """Reject instance callbacks on both the forest and every estimator it executes."""
    if any(callable(value) for value in vars(model).values()):
        raise ValueError("Overridden tree inference methods are unsupported.")


def tree_applier(model: Any) -> type | None:
    """Return a reviewed Core applier only after inspecting exact native fitted trees."""
    contract = _CONTRACTS.get(type(model))
    if contract is None:
        return None
    node, calculator, applier = contract
    if (
        NodeRegistry.get_calculator(node) is not calculator
        or NodeRegistry.get_applier(node) is not applier
    ):
        raise ValueError("Unreviewed tree model registration.")
    _check_methods(model)
    if model.n_outputs_ != 1:
        raise ValueError("Only single-output fitted trees are admitted.")
    if type(model) in CLASSIFIERS:
        _check_class_axis(model)
    child_type = _CHILDREN.get(type(model))
    if child_type is None:
        _check_tree(model, model.n_features_in_)
    else:
        _check_forest(model, child_type)
    return applier


def _check_forest(model: Any, child_type: type) -> None:
    """Disallow substituted children even when their enclosing forest has an exact type."""
    children = model.estimators_
    if type(children) is not list or not children or len(children) != model.n_estimators:
        raise ValueError("A forest requires its complete nonempty fitted estimator list.")
    if type(model.estimator_) is not child_type:
        raise ValueError("Unsupported forest estimator template.")
    _check_methods(model.estimator_)
    for child in children:
        if type(child) is not child_type:
            raise ValueError("Custom forest child estimators are unsupported.")
        _check_methods(child)
        _check_tree(child, model.n_features_in_)
        if type(model) in CLASSIFIERS and child.n_classes_ != model.n_classes_:
            raise ValueError("Forest child probability width disagrees with its parent.")


def _check_tree(model: Any, n_features: int) -> None:
    """Validate the native tree shape and finite outputs without invoking predictions."""
    tree = model.tree_
    if type(tree) is not _tree.Tree:
        raise ValueError("An exact fitted sklearn Tree is required.")
    if model.n_outputs_ != 1 or tree.n_outputs != 1:
        raise ValueError("Only single-output fitted trees are admitted.")
    if tree.n_features != n_features or model.n_features_in_ != n_features:
        raise ValueError("Forest child feature schema disagrees with its parent.")
    if tree.node_count < 1 or not np.isfinite(tree.value).all():
        raise ValueError("Fitted tree values must be nonempty and finite.")
    if type(model) in {DecisionTreeClassifier, ExtraTreeClassifier}:
        _check_class_axis(model)
        if tree.n_classes.tolist() != [model.n_classes_]:
            raise ValueError("Fitted tree probability width disagrees with its class axis.")
    _check_tree_links(tree, n_features)


def _check_class_axis(model: Any) -> None:
    """Prevent class-array subclasses from overriding inference label lookup."""
    labels = model.classes_
    if type(labels) is not np.ndarray or labels.ndim != 1 or not len(labels):
        raise ValueError("A plain nonempty fitted class array is required.")
    if labels.dtype.kind == "O" and any(
        type(value) not in (str, int, float, bool) for value in labels
    ):
        raise ValueError("Custom classifier label objects are unsupported.")
    if len(labels) != model.n_classes_ or len(np.unique(labels)) != len(labels):
        raise ValueError("Classifier labels disagree with its fitted probability width.")


def _check_tree_links(tree: Any, n_features: int) -> None:
    """Reject malformed child pointers and split features before native traversal."""
    left, right = tree.children_left, tree.children_right
    leaves = left == _tree.TREE_LEAF
    if not np.array_equal(leaves, right == _tree.TREE_LEAF):
        raise ValueError("Fitted tree leaf pointers disagree.")
    positions = np.arange(tree.node_count)[~leaves]
    for children in (left[~leaves], right[~leaves]):
        if np.any(children <= positions) or np.any(children >= tree.node_count):
            raise ValueError("Fitted tree child pointers are invalid or cyclic.")
    features = tree.feature[~leaves]
    if np.any(features < 0) or np.any(features >= n_features):
        raise ValueError("Fitted tree split feature is outside its fitted schema.")
    if np.isnan(tree.threshold[~leaves]).any():
        raise ValueError("Fitted tree split thresholds cannot be NaN.")
