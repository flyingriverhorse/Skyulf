"""Public encoder contracts for unambiguous, stable indicator column names."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import OneHotEncoder

from skyulf.engines.registry import EngineRegistry
from skyulf.preprocessing.drop_and_missing.missing_indicator import (
    MissingIndicatorApplier,
    MissingIndicatorCalculator,
)
from skyulf.preprocessing.encoding.dummy import DummyEncoderApplier, DummyEncoderCalculator
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderApplier, OneHotEncoderCalculator
from skyulf.preprocessing.encoding.target import TargetEncoderApplier, TargetEncoderCalculator


@pytest.fixture(params=["onehot", "dummy"])
def encoder(request):
    """Exercise both public calculator/applier pairs with the same output contract."""
    if request.param == "onehot":
        return OneHotEncoderCalculator(), OneHotEncoderApplier()
    return DummyEncoderCalculator(), DummyEncoderApplier()


@pytest.fixture(params=[("pandas", False), ("pandas", True), ("polars", False), ("polars", True)])
def frame_kind(request):
    """Include the native and wrapped input forms supported by each engine."""
    return request.param


def _frame(data: dict[str, list[Any]], frame_kind: tuple[str, bool]) -> Any:
    """Construct equivalent frames without changing the public caller's input form."""
    engine, wrapped = frame_kind
    native: Any = pd.DataFrame(data)
    if engine == "polars":
        native = pl.from_pandas(native)
    return EngineRegistry.wrap(native) if wrapped else native


def _pandas(frame: Any) -> pd.DataFrame:
    """Expose result values for engine-independent assertions."""
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def test_fit_rejects_indicator_name_matching_a_retained_column(encoder, frame_kind):
    """Fitting must reject an ambiguous schema before a later transform loses input values."""
    calculator, _ = encoder
    frame = _frame({"city": ["a", "b"], "city_a": [9, 8]}, frame_kind)
    original = _pandas(frame).copy(deep=True)

    with pytest.raises(ValueError, match="collid.*city_a"):
        calculator.fit(frame, {"columns": ["city"]})

    pd.testing.assert_frame_equal(_pandas(frame), original)


def test_apply_rejects_extra_column_matching_an_absent_fitted_category(encoder, frame_kind):
    """Held-out columns must not overwrite fitted indicators even when their category is absent."""
    calculator, applier = encoder
    artifact = calculator.fit(_frame({"city": ["a", "b"]}, frame_kind), {"columns": ["city"]})
    other_engine = "polars" if frame_kind[0] == "pandas" else "pandas"
    heldout = _frame({"city": ["b"], "city_a": [91]}, (other_engine, frame_kind[1]))
    original = _pandas(heldout).copy(deep=True)

    with pytest.raises(ValueError, match="collid.*city_a"):
        applier.apply(heldout, artifact)

    pd.testing.assert_frame_equal(_pandas(heldout), original)


def test_fit_rejects_duplicate_indicator_names_from_distinct_sources(encoder, frame_kind):
    """Underscores in source names and categories must not merge independent indicators."""
    calculator, _ = encoder
    frame = _frame({"a": ["b_c", "d"], "a_b": ["c", "e"]}, frame_kind)

    with pytest.raises(ValueError, match="collid.*a_b_c"):
        calculator.fit(frame, {"columns": ["a", "a_b"]})


def test_apply_rejects_duplicate_names_in_legacy_artifacts(encoder, frame_kind):
    """Artifacts fitted before collision validation must fail without mutating their input."""
    calculator, applier = encoder
    frame = _frame({"a": ["b_c", "d"], "a_b": ["c", "e"]}, frame_kind)
    artifact: dict[str, Any] = {"columns": ["a", "a_b"]}
    if isinstance(calculator, OneHotEncoderCalculator):
        fitted = OneHotEncoder(sparse_output=False).fit(_pandas(frame).to_numpy())
        artifact.update(
            encoder_object=fitted, feature_names=fitted.get_feature_names_out(["a", "a_b"]).tolist()
        )
    else:
        artifact["categories"] = {"a": ["b_c", "d"], "a_b": ["c", "e"]}
    original = _pandas(frame).copy(deep=True)

    with pytest.raises(ValueError, match="collid.*a_b_c"):
        applier.apply(frame, artifact)

    pd.testing.assert_frame_equal(_pandas(frame), original)


def test_indicator_can_reuse_a_source_name_that_is_dropped(encoder, frame_kind):
    """Dropping originals must preserve both the replacement indicator and other source values."""
    calculator, applier = encoder
    frame = _frame({"a": ["b", "c"], "a_b": ["x", "y"], "keep": [9, 8]}, frame_kind)
    artifact = calculator.fit(frame, {"columns": ["a", "a_b"]})
    result = _pandas(applier.apply(frame, artifact))

    assert list(result.columns) == ["keep", "a_b", "a_c", "a_b_x", "a_b_y"]
    np.testing.assert_array_equal(result.to_numpy(), [[9, 1, 0, 1, 0], [8, 0, 1, 0, 1]])
    assert _pandas(frame)["a_b"].tolist() == ["x", "y"]


def test_drop_first_excludes_names_that_are_not_emitted(encoder, frame_kind):
    """A retained column matching a dropped reference category must remain a valid input."""
    calculator, applier = encoder
    frame = _frame({"city": ["a", "b"], "city_a": [9, 8]}, frame_kind)
    artifact = calculator.fit(frame, {"columns": ["city"], "drop_first": True})
    result = _pandas(applier.apply(frame, artifact))

    assert list(result.columns) == ["city_a", "city_b"]
    np.testing.assert_array_equal(result.to_numpy(), [[9, 0], [8, 1]])


def test_unseen_category_does_not_create_a_false_heldout_collision(encoder, frame_kind):
    """Only fitted indicators reserve names; an unseen category cannot change the output schema."""
    calculator, applier = encoder
    artifact = calculator.fit(_frame({"city": ["a", "b"]}, frame_kind), {"columns": ["city"]})
    result = _pandas(applier.apply(_frame({"city": ["c"], "city_c": [91]}, frame_kind), artifact))

    assert list(result.columns) == ["city_c", "city_a", "city_b"]
    assert result.iloc[0].tolist() == [91, 0, 0]


def test_numeric_indicators_keep_fitted_names_and_values_across_batches(encoder, frame_kind):
    """Collision checks must preserve numeric dtype coercion and the fitted category contract."""
    calculator, applier = encoder
    artifact = calculator.fit(_frame({"number": [1, 2]}, frame_kind), {"columns": ["number"]})
    alone = _pandas(applier.apply(_frame({"number": [1]}, frame_kind), artifact))
    together = _pandas(applier.apply(_frame({"number": [1.0, 2.5]}, frame_kind), artifact))

    assert list(alone.columns) == list(together.columns) == ["number_1", "number_2"]
    assert alone.iloc[0].tolist() == together.iloc[0].tolist() == [1, 0]
    assert together.iloc[1].tolist() == [0, 0]


def test_onehot_keep_original_rejects_selected_source_name_collision(frame_kind):
    """Keeping originals makes an indicator matching another selected source ambiguous."""
    frame = _frame({"a": ["b", "c"], "a_b": ["x", "y"]}, frame_kind)

    with pytest.raises(ValueError, match="collid.*a_b"):
        OneHotEncoderCalculator().fit(frame, {"columns": ["a", "a_b"], "drop_original": False})


@pytest.mark.parametrize(
    ("values", "config", "name"),
    [
        (["None", None], {}, "city_None"),
        (["infrequent_sklearn"] * 5 + ["a", "b"], {"max_categories": 2}, "city_infrequent_sklearn"),
    ],
)
def test_onehot_rejects_distinct_categories_with_identical_names(values, config, name, frame_kind):
    """Null rendering and the infrequent bucket must not alias a real category indicator."""
    frame = _frame({"city": values}, frame_kind)

    with pytest.raises(ValueError, match=f"collid.*{name}"):
        OneHotEncoderCalculator().fit(frame, {"columns": ["city"], **config})


def test_onehot_sparse_artifact_preserves_unambiguous_output(frame_kind):
    """Legacy sparse sklearn output must retain its fitted feature names and indicator values."""
    frame = _frame({"city": ["a", "b"], "keep": [9, 8]}, frame_kind)
    artifact = OneHotEncoderCalculator().fit(frame, {"columns": ["city"], "drop_original": False})
    artifact["encoder_object"].set_params(sparse_output=True)
    result = _pandas(OneHotEncoderApplier().apply(frame, artifact))

    assert list(result.columns) == ["city", "keep", "city_a", "city_b"]
    np.testing.assert_array_equal(result[["city_a", "city_b"]].to_numpy(), [[1, 0], [0, 1]])
    assert result["keep"].tolist() == [9, 8]


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_indicator_rejects_existing_flag_at_fit(explicit, frame_kind):
    """Automatically detected and selected flags must never overwrite existing data."""
    frame = _frame({"x": [None, 2.0], "x_missing": [91, 92]}, frame_kind)
    config = {"columns": ["x"]} if explicit else {}

    with pytest.raises(ValueError, match="collid.*x_missing"):
        MissingIndicatorCalculator().fit(frame, config)


@pytest.mark.parametrize("suffix", ["_missing", "_flag"])
def test_missing_indicator_rejects_heldout_flag_collision(suffix, frame_kind):
    """A fitted custom flag name must be checked against each incoming frame."""
    artifact = MissingIndicatorCalculator().fit(
        _frame({"x": [None, 2.0]}, frame_kind), {"columns": ["x"], "flag_suffix": suffix}
    )
    frame = _frame({"x": [3.0], f"x{suffix}": [91]}, frame_kind)

    with pytest.raises(ValueError, match=f"collid.*x{suffix}"):
        MissingIndicatorApplier().apply(frame, artifact)

    assert _pandas(frame)[f"x{suffix}"].tolist() == [91]


def test_missing_indicator_skips_absent_source_before_name_validation(frame_kind):
    """A missing source emits no flag, so its would-be name cannot cause a false collision."""
    artifact = MissingIndicatorCalculator().fit(
        _frame({"x": [None, 2.0]}, frame_kind), {"columns": ["x"]}
    )
    frame = _frame({"x_missing": [91]}, frame_kind)
    result = _pandas(MissingIndicatorApplier().apply(frame, artifact))

    assert result.to_dict("list") == {"x_missing": [91]}


def _target_labels(frame_kind: tuple[str, bool], binary: bool = False) -> Any:
    """Keep public tuple labels on the same engine as their feature frame."""
    labels = pd.Series(([0, 1] * 3) if binary else ([0, 1, 2] * 2), name="target")
    return pl.from_pandas(labels) if frame_kind[0] == "polars" else labels


@pytest.mark.parametrize("method", ["fit", "fit_transform_train"])
def test_multiclass_target_rejects_retained_class_name_during_fit(method, frame_kind):
    """Both training entry points must reject class features that would destroy retained data."""
    frame = _frame({"city": ["a", "b"] * 3, "city_cls0": [91] * 6}, frame_kind)
    original = _pandas(frame).copy(deep=True)

    with pytest.raises(ValueError, match="collid.*city_cls0"):
        getattr(TargetEncoderCalculator(), method)(
            (frame, _target_labels(frame_kind)), {"columns": ["city"], "target_type": "multiclass"}
        )

    pd.testing.assert_frame_equal(_pandas(frame), original)


def test_multiclass_target_rejects_extra_heldout_class_column(frame_kind):
    """Multiclass replay must protect held-out columns absent from the fit-time schema."""
    artifact = TargetEncoderCalculator().fit(
        (_frame({"city": ["a", "b"] * 3}, frame_kind), _target_labels(frame_kind)),
        {"columns": ["city"], "target_type": "multiclass"},
    )
    frame = _frame({"city": ["a"], "city_cls2": [91]}, frame_kind)

    with pytest.raises(ValueError, match="collid.*city_cls2"):
        TargetEncoderApplier().apply(frame, artifact)

    assert _pandas(frame)["city_cls2"].tolist() == [91]


@pytest.mark.parametrize("method", ["fit", "fit_transform_train"])
def test_multiclass_target_allows_reusing_dropped_source_names(method, frame_kind):
    """Class indicators may replace selected originals while all original features are encoded."""
    frame = _frame({"city": ["a", "b"] * 3, "city_cls0": ["x", "y"] * 3}, frame_kind)
    fitted = getattr(TargetEncoderCalculator(), method)(
        (frame, _target_labels(frame_kind)),
        {"columns": ["city", "city_cls0"], "target_type": "multiclass", "smooth": 0},
    )
    if method == "fit":
        result = TargetEncoderApplier().apply(frame, fitted)
    else:
        _, (result, _) = fitted
    result = _pandas(result)

    assert list(result.columns) == [
        "city_cls0",
        "city_cls1",
        "city_cls2",
        "city_cls0_cls0",
        "city_cls0_cls1",
        "city_cls0_cls2",
    ]
    np.testing.assert_allclose(result.iloc[:, :3].sum(axis=1), np.ones(6))
    np.testing.assert_allclose(result.iloc[:, 3:].sum(axis=1), np.ones(6))


def test_binary_target_retains_columns_resembling_multiclass_outputs(frame_kind):
    """Binary encoding emits no class suffixes and must keep similarly named input columns."""
    frame = _frame({"city": ["a", "b"] * 3, "city_cls0": [91] * 6}, frame_kind)
    artifact = TargetEncoderCalculator().fit(
        (frame, _target_labels(frame_kind, binary=True)),
        {"columns": ["city"], "target_type": "binary", "smooth": 0},
    )
    result = _pandas(TargetEncoderApplier().apply(frame, artifact))

    assert list(result.columns) == ["city", "city_cls0"]
    assert result["city"].tolist() == [0, 1] * 3
    assert result["city_cls0"].tolist() == [91] * 6
