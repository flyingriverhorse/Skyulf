"""Tests for the class_weight HyperparameterField added to classifier param lists."""

from skyulf.modeling.hyperparameters import (
    LGBM_CLASSIFIER_PARAMS,
    LGBM_PARAMS,
    MODEL_HYPERPARAMETERS,
    RANDOM_FOREST_CLASSIFIER_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_CLASSIFIER_PARAMS,
    XGBOOST_PARAMS,
)


def _field_names(fields):
    return {f.name for f in fields}


def test_random_forest_classifier_has_class_weight_but_regressor_does_not():
    assert "class_weight" in _field_names(RANDOM_FOREST_CLASSIFIER_PARAMS)
    assert "class_weight" not in _field_names(RANDOM_FOREST_PARAMS)


def test_lgbm_classifier_params_has_class_weight_but_shared_base_does_not():
    assert "class_weight" in _field_names(LGBM_CLASSIFIER_PARAMS)
    assert "class_weight" not in _field_names(LGBM_PARAMS)


def test_xgboost_classifier_params_has_class_weight_but_shared_base_does_not():
    assert "class_weight" in _field_names(XGBOOST_CLASSIFIER_PARAMS)
    assert "class_weight" not in _field_names(XGBOOST_PARAMS)


def test_registry_maps_classifier_keys_to_class_weight_variants():
    assert "class_weight" in _field_names(MODEL_HYPERPARAMETERS["lgbm_classifier"])
    assert "class_weight" not in _field_names(MODEL_HYPERPARAMETERS["lgbm_regressor"])
    assert "class_weight" in _field_names(MODEL_HYPERPARAMETERS["xgboost_classifier"])
    assert "class_weight" not in _field_names(MODEL_HYPERPARAMETERS["xgboost_regressor"])
    assert "class_weight" in _field_names(MODEL_HYPERPARAMETERS["random_forest_classifier"])
    assert "class_weight" not in _field_names(MODEL_HYPERPARAMETERS["random_forest_regressor"])


def test_class_weight_field_default_and_options():
    rf_field = next(f for f in RANDOM_FOREST_CLASSIFIER_PARAMS if f.name == "class_weight")
    assert rf_field.default is None
    assert rf_field.type == "select"
    values = {opt["value"] for opt in rf_field.options}
    assert values == {None, "balanced"}
