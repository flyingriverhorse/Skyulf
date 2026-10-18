"""Native fitted trees must enter semantic seals without hashing process handles."""

import ctypes
import pickle

import numpy as np
import pytest

from skyulf.pipeline.seal import artifact_digest


@pytest.fixture(params=["xgboost", "lightgbm"])
def native_models(request):
    """Train distinct real regressors with identical Python-level hyperparameters."""
    library = pytest.importorskip(request.param)
    features = np.arange(64, dtype=float).reshape(-1, 1)
    labels = (features[:, 0] > 30).astype(int)
    if request.param == "xgboost":
        model_type = library.XGBRegressor
        params = {"max_depth": 2}
    else:
        model_type = library.LGBMRegressor
        params = {"num_leaves": 5, "min_child_samples": 2, "verbosity": -1}
    models = [
        model_type(n_estimators=4, n_jobs=1, **params).fit(features, target)
        for target in (labels, 1 - labels)
    ]
    boosters = [
        model.get_booster() if request.param == "xgboost" else model.booster_ for model in models
    ]
    return library, features, models, boosters


def test_native_tree_swap_invalidates_seal(native_models):
    """A changed native tree cannot retain the seal of an already reviewed regressor."""
    library, features, models, boosters = native_models
    before = artifact_digest(models[0])
    predictions = models[0].predict(features)
    handle = "handle" if library.__name__ == "xgboost" else "_handle"
    first, second = (getattr(booster, handle) for booster in boosters)
    setattr(boosters[0], handle, second)
    setattr(boosters[1], handle, first)
    assert not np.array_equal(predictions, models[0].predict(features))
    assert artifact_digest(models[0]) != before


def test_native_models_survive_pickle_and_distinguish_trees(native_models):
    """Native state must distinguish learned trees and survive ordinary model persistence."""
    _, _, models, boosters = native_models
    for value in (*models, *boosters):
        assert artifact_digest(pickle.loads(pickle.dumps(value))) == artifact_digest(value)
    assert artifact_digest(boosters[0]) != artifact_digest(boosters[1])
    assert artifact_digest(models[0]) != artifact_digest(models[1])


def test_native_serialization_reload_preserves_digest(native_models):
    """A fresh native handle loaded from the same learned representation has the same seal."""
    library, _, _, boosters = native_models
    original = boosters[0]
    if library.__name__ == "xgboost":
        restored = library.Booster(model_file=original.save_raw(raw_format="ubj"))
    else:
        restored = library.Booster(model_str=original.model_to_string(num_iteration=-1))
    assert artifact_digest(restored) == artifact_digest(original)


def test_native_subclasses_fail_closed(native_models):
    """Only the exact reviewed native wrapper class may use the native serializer."""
    _, _, _, boosters = native_models
    booster = boosters[0]
    original_type = type(booster)
    subclass = type("UnreviewedBooster", (original_type,), {})
    booster.__class__ = subclass
    try:
        with pytest.raises(TypeError, match="native booster"):
            artifact_digest(booster)
    finally:
        booster.__class__ = original_type


@pytest.mark.parametrize("attribute", ["predict", "save_raw", "model_to_string"])
def test_injected_native_methods_fail_closed(native_models, attribute):
    """Instance callables must not spoof serialization or alter predictions under a seal."""
    _, _, _, boosters = native_models
    setattr(boosters[0], attribute, lambda *args, **kwargs: b"forged")
    with pytest.raises(TypeError, match="native booster"):
        artifact_digest(boosters[0])


@pytest.mark.parametrize(
    "value",
    [
        ctypes.c_void_p(1),
        ctypes.c_void_p(2),
        ctypes.c_int(7),
        ctypes.pointer(ctypes.c_int(7)),
        (ctypes.c_int * 2)(1, 2),
        type("NativeStruct", (ctypes.Structure,), {"_fields_": [("value", ctypes.c_int)]})(7),
        type("NativeUnion", (ctypes.Union,), {"_fields_": [("value", ctypes.c_int)]})(7),
    ],
)
def test_unrecognized_ctypes_values_fail_closed(value):
    """Unknown native values cannot become identical empty Python-object state."""
    with pytest.raises(TypeError, match="ctypes"):
        artifact_digest(value)


def test_ctypes_function_pointer_fails_closed():
    """Native callbacks cannot disappear from an artifact's semantic identity."""
    callback = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 7)
    with pytest.raises(TypeError, match="ctypes"):
        artifact_digest({"callback": callback})


def test_lightgbm_best_iteration_is_bound():
    """LightGBM's default prediction iteration remains part of its semantic identity."""
    library = pytest.importorskip("lightgbm")
    features = np.arange(64, dtype=float).reshape(-1, 1)
    model = library.LGBMRegressor(n_estimators=4, min_child_samples=2, n_jobs=1, verbosity=-1).fit(
        features, features[:, 0] > 30
    )
    before = artifact_digest(model.booster_)
    model.booster_.best_iteration = 1
    assert artifact_digest(model.booster_) != before
