"""Fixed one-hot vocabularies preserve ordered outputs and strict saved-state admission."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderApplier, OneHotEncoderCalculator


def _frame(values, engine):
    """Retain textual nulls and identical row values on either engine."""
    frame = pd.DataFrame({"city": pd.Series(values, dtype=object)})
    return pl.from_pandas(frame) if engine == "polars" else frame


def _config(**options):
    """Use an ordered vocabulary containing an intentionally unobserved level."""
    return {
        "columns": ["city"],
        "categories": {"city": ["Zürich", "Øl", "Åben"]},
        "max_categories": None,
        **options,
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("style", ["sklearn", "slug"])
@pytest.mark.parametrize("drop_first", [False, True])
def test_fixed_categories_survive_persistence_with_unseen_levels(
    engine, style, drop_first, tmp_path
):
    """Missing training levels and unknown/null requests cannot change the ordered feature space."""
    config = _config(column_names=style, drop_first=drop_first)
    state = OneHotEncoderCalculator().fit(_frame(["Øl", "Zürich", "Øl"], engine), config)
    path = tmp_path / "onehot.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    inspected = OneHotEncoderApplier.validate_fitted_state(restored)
    OneHotEncoderApplier.resolve_fitted_config(config, inspected)
    result = OneHotEncoderApplier().apply(_frame(["Åben", "Øl", "new", None], engine), restored)
    names = (
        ["city_Zürich", "city_Øl", "city_Åben"]
        if style == "sklearn"
        else ["city_zurich", "city_oel", "city_aaben"]
    )
    expected = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    start = int(drop_first)
    assert list(result.columns) == names[start:]
    np.testing.assert_array_equal(result.to_numpy(), expected[:, start:])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("value", ["new", None])
def test_fixed_categories_error_policy_rejects_unknown_and_missing(engine, value):
    """Error policy must reject nulls as well as categories outside the fixed vocabulary."""
    config = _config(handle_unknown="error")
    state = OneHotEncoderCalculator().fit(_frame(["Øl", "Zürich"], engine), config)
    with pytest.raises(ValueError, match="unknown categories"):
        OneHotEncoderApplier().apply(_frame(["Øl", value], engine), state)


@pytest.mark.parametrize(
    "options",
    [
        {"categories": {"city": [1, 2]}},
        {"categories": {"city": []}},
        {"categories": {"city": ["a", "a"]}},
        {"categories": {"other": ["a"]}},
        {"categories": {}},
        {"categories": [["a"]]},
        {"columns": []},
        {"columns": ["city", "city"]},
        {"columns": ["missing"], "categories": {"missing": ["a"]}},
        {"max_categories": 20},
        {"include_missing": True},
        {"column_names": "other"},
        {"target_column": "city"},
    ],
)
def test_fixed_categories_reject_ambiguous_configs(options):
    """A fixed schema cannot silently reorder, omit, cap or coerce configured levels."""
    with pytest.raises(ValueError):
        OneHotEncoderCalculator().fit(_frame(["a", "b"], "pandas"), _config(**options))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fixed_categories_require_explicit_columns(engine):
    """Auto-selection cannot silently change which vocabulary belongs to each feature."""
    config = _config()
    del config["columns"]
    with pytest.raises(ValueError, match="columns"):
        OneHotEncoderCalculator().fit(_frame(["a", "b"], engine), config)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fixed_categories_do_not_stringify_numeric_inputs(engine):
    """Numeric values must not silently match textual vocabulary entries."""
    frame = pd.DataFrame({"city": [1, 2]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    with pytest.raises(ValueError, match="string|categorical"):
        OneHotEncoderCalculator().fit(frame, _config(categories={"city": ["1", "2"]}))


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_fixed_categories_support_nullable_text_at_apply(dtype):
    """Nullable pandas text and all-null batches must obey ignore without sentinel collisions."""
    state = OneHotEncoderCalculator().fit(_frame(["Øl", "Zürich"], "pandas"), _config())
    query = pd.DataFrame({"city": pd.Series([pd.NA, pd.NA], dtype=dtype)})
    result = OneHotEncoderApplier().apply(query, state)
    np.testing.assert_array_equal(result.to_numpy(), np.zeros((2, 3)))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fixed_categories_reject_numeric_scoring_values(engine):
    """A scoring dtype change must not silently reinterpret numbers as fixed string levels."""
    config = _config(categories={"city": ["1", "2"]})
    state = OneHotEncoderCalculator().fit(_frame(["1", "2"], engine), config)
    with pytest.raises(ValueError, match="string"):
        OneHotEncoderApplier().apply(_frame([1, 2], engine), state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fixed_category_columns_use_requested_order_not_mapping_order(engine):
    """Reversed mapping keys must not swap vocabularies between selected columns."""
    frame = pd.DataFrame({"city": ["Øl", "Åben"], "kind": ["B", "A"]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {
        "columns": ["kind", "city"],
        "categories": {"city": ["Åben", "Øl"], "kind": ["B", "A"]},
        "max_categories": None,
        "column_names": "slug",
    }
    state = OneHotEncoderCalculator().fit(frame, config)
    result = OneHotEncoderApplier().apply(frame, state)
    assert list(result.columns) == ["kind_b", "kind_a", "city_aaben", "city_oel"]
    np.testing.assert_array_equal(result.to_numpy(), [[1, 0, 0, 1], [0, 1, 1, 0]])


@pytest.mark.parametrize("levels", [["Øl", "oel"], ["A-B", "a b"], ["!!"]])
def test_slug_names_reject_collisions_and_empty_names(levels):
    """Distinct categories must never overwrite one another after Unicode normalization."""
    with pytest.raises(ValueError, match="collid|duplicate|empty"):
        OneHotEncoderCalculator().fit(
            _frame([levels[0]], "pandas"),
            _config(categories={"city": levels}, column_names="slug"),
        )


def test_slug_names_work_for_automatic_categories_and_fold_unicode():
    """Slug naming is independent of fixed vocabularies and handles Danish and German letters."""
    frame = _frame(["Æble", "Øl", "Åben", "Straße", "Été"], "pandas")
    state = OneHotEncoderCalculator().fit(
        frame, {"columns": ["city"], "max_categories": None, "column_names": "slug"}
    )
    inspected = OneHotEncoderApplier.validate_fitted_state(dict(state))
    result = OneHotEncoderApplier().apply(frame, inspected)
    assert set(result.columns) == {
        "city_aeble",
        "city_oel",
        "city_aaben",
        "city_strasse",
        "city_ete",
    }
    np.testing.assert_array_equal(result.sum(axis=1), np.ones(5))


@pytest.mark.parametrize(
    "field,value",
    [
        ("fixed_categories", False),
        ("fixed_categories", 1),
        ("column_names", "other"),
        ("feature_names", ["a", "b", "c"]),
        ("include_missing", True),
    ],
)
def test_fixed_artifact_rejects_tampered_options(field, value):
    """Saved-state inspection must still reject unsupported policies and invented output names."""
    state = OneHotEncoderCalculator().fit(_frame(["Øl"], "pandas"), _config())
    state[field] = value
    with pytest.raises(ValueError):
        OneHotEncoderApplier.validate_fitted_state(dict(state))


def test_fixed_artifact_binds_both_estimator_categories_and_recipe_order():
    """Changing the estimator's public vocabulary or recipe must invalidate saved admission."""
    config = _config()
    state = OneHotEncoderCalculator().fit(_frame(["Øl"], "pandas"), config)
    changed = deepcopy(state)
    changed["encoder_object"].categories[0] = ["another", "Øl", "Åben"]
    with pytest.raises(ValueError):
        OneHotEncoderApplier.validate_fitted_state(dict(changed))
    changed_config = _config(categories={"city": ["Åben", "Øl", "Zürich"]})
    with pytest.raises(ValueError, match="disagrees"):
        OneHotEncoderApplier.resolve_fitted_config(changed_config, dict(state))


def test_legacy_auto_artifact_replays_without_new_fields():
    """Old automatic-category artifacts keep their original sklearn names and replay values."""
    frame = _frame(["b", "a", "b"], "pandas")
    config = {"columns": ["city"], "max_categories": None}
    state = OneHotEncoderCalculator().fit(frame, config)
    state.pop("fixed_categories", None)
    state.pop("column_names", None)
    inspected = OneHotEncoderApplier.validate_fitted_state(pickle.loads(pickle.dumps(state)))
    OneHotEncoderApplier.resolve_fitted_config(config, inspected)
    result = OneHotEncoderApplier().apply(frame, inspected)
    assert list(result.columns) == ["city_a", "city_b"]
    np.testing.assert_array_equal(result.to_numpy(), [[0, 1], [1, 0], [0, 1]])


def test_fixed_pipeline_save_load_retains_worker_admission(tmp_path, monkeypatch):
    """The complete persisted pipeline must retain strict independent-worker certification."""
    monkeypatch.setenv("SKYULF_ENGINE", "pandas")
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": _config(column_names="slug"),
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(
        pd.DataFrame({"city": ["Øl", "Zürich"] * 10, "target": [1.0, 2.0] * 10}),
        target_column="target",
    )
    save_pipeline(pipeline, tmp_path / "pipeline")
    restored = load_pipeline(tmp_path / "pipeline")
    certificate = require_partition_safe_pipeline(restored)
    query = _frame(["Åben", "Øl", None], "pandas")
    full = predict_pipeline(query, restored)
    pieces = pd.concat([predict_pipeline(query.iloc[[index]], restored) for index in range(3)])
    assert certificate.steps[0].node_type == "OneHotEncoder"
    pd.testing.assert_frame_equal(full, pieces)
