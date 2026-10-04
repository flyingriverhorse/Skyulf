"""Keep splitting, sampler replay, exact row guards, and weight validation consistent."""

import copy
import os
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.model_selection import train_test_split

import skyulf.modeling._sample_weights as weight_module
import skyulf.preprocessing._weighted_resampling as sampling_module
import skyulf.preprocessing.base as base_module
import skyulf.preprocessing.resampling as resampling_module
import skyulf.preprocessing.split as split_module
from skyulf.data.dataset import SplitDataset
from skyulf.engines import SkyulfPolarsWrapper
from skyulf.modeling._sample_weights import SampleWeightError, validate_sample_weight
from skyulf.modeling.regression import LinearRegressionCalculator
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.preprocessing.split import DataSplitter


@pytest.fixture(scope="module", autouse=True)
def verify_requested_source_root():
    """Isolated verification must import the requested source without restricting normal CI."""
    expected = os.environ.get("SKYULF_EXPECTED_SOURCE_ROOT")
    if expected is not None:
        root = Path(expected).resolve()
        for module in (
            weight_module,
            sampling_module,
            base_module,
            split_module,
            resampling_module,
        ):
            assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
            print(f"Verified source: {module.__file__}")


def _to_engine(frame, engine):
    """Keep equivalent native and wrapped Polars fixtures beside the pandas reference."""
    if engine == "pandas":
        return frame
    native = pl.from_pandas(frame)
    return SkyulfPolarsWrapper(native) if engine == "wrapped_polars" else native


def _native(frame):
    """Use the public engine boundary for value assertions."""
    return frame.to_native() if hasattr(frame, "to_native") else frame


@pytest.mark.parametrize("engine", ["pandas", "polars", "wrapped_polars"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_validation_reconsiders_stratification_after_rare_class_leaves_training(
    engine, paired, weighted, seed
):
    """Validation membership must depend on its current labels, identically on every engine."""
    frame = pd.DataFrame({"id": range(21), "target": [0] * 12 + [1] * 8 + [2]}, index=[3] * 21)
    original = frame.copy(deep=True)
    rows = np.arange(len(frame))
    train_val, test = train_test_split(rows, test_size=0.25, random_state=seed)
    remaining = frame["target"].iloc[train_val]
    validation_labels = remaining if remaining.value_counts().min() >= 2 else None
    train, validation = train_test_split(
        train_val, test_size=0.25 / 0.75, random_state=seed, stratify=validation_labels
    )
    weights = rows.astype(float) + 1 if weighted else None
    splitter = DataSplitter(
        test_size=0.25, validation_size=0.25, random_state=seed, stratify_col="target"
    )
    if paired:
        features = _to_engine(frame[["id"]], engine)
        target = frame["target"] if engine == "pandas" else pl.Series("target", frame["target"])
        result = splitter.split_xy(features, target, sample_weight=weights)
    else:
        result = splitter.split(_to_engine(frame, engine), sample_weight=weights)
    for payload, positions in (
        (result.train, train),
        (result.test, test),
        (result.validation, validation),
    ):
        assert payload is not None
        features, target = payload if paired else (payload, _native(payload)["target"])
        np.testing.assert_array_equal(np.asarray(_native(features)["id"]), positions)
        np.testing.assert_array_equal(np.asarray(target), frame["target"].iloc[positions])
    if weighted:
        assert weights is not None
        np.testing.assert_array_equal(result.train_sample_weight, rows[train] + 1)
        np.testing.assert_array_equal(weights, rows + 1)
    else:
        assert result.train_sample_weight is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.fixture
def sampler_components():
    """Optional samplers cannot skip unrelated split and numeric validation tests."""
    pytest.importorskip("imblearn")
    from skyulf.preprocessing.resampling import (
        OversamplingApplier,
        OversamplingCalculator,
        UndersamplingApplier,
        UndersamplingCalculator,
    )

    return {
        "random_over": (OversamplingCalculator, OversamplingApplier),
        "smote": (OversamplingCalculator, OversamplingApplier),
        "random_under_sampling": (UndersamplingCalculator, UndersamplingApplier),
        "nearmiss": (UndersamplingCalculator, UndersamplingApplier),
        "tomek_links": (UndersamplingCalculator, UndersamplingApplier),
        "edited_nearest_neighbours": (UndersamplingCalculator, UndersamplingApplier),
    }


def _sampler(sampler_components, method):
    """Construct a real training-only sampling node with held-out application disabled."""
    calculator, applier = sampler_components[method]
    return StatefulTransformer(
        calculator(), applier(), "sampling", apply_on_test=False, apply_on_validation=False
    )


def _sampling_config(method):
    """Use deterministic small samplers with an explicit synthetic weight policy."""
    return {
        "method": method,
        "target_column": "target",
        "random_state": 19,
        "k_neighbors": 1,
        "synthetic_weight": "uniform",
    }


@pytest.mark.parametrize("engine", ["pandas", "polars", "wrapped_polars"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("method", ["random_over", "random_under_sampling", "smote"])
def test_split_dataset_replay_never_resamples_training_or_heldout_slots(
    sampler_components, engine, paired, weighted, method
):
    """Adding sample weights cannot change replay into a second training-time sampling run."""
    frame = pd.DataFrame({"x": range(6), "target": [0, 0, 0, 0, 1, 1]})
    native = _to_engine(frame, engine)
    if paired:
        features = _to_engine(frame[["x"]], engine)
        target = frame["target"] if engine == "pandas" else pl.Series("target", frame["target"])
        payload = (features, target)
    else:
        payload = native
    weights = np.arange(1, 7, dtype=float) if weighted else None
    dataset = SplitDataset(
        train=payload, test=payload, validation=payload, train_sample_weight=weights
    )
    transformer = _sampler(sampler_components, method)
    fitted = transformer.fit_transform(dataset, _sampling_config(method))
    assert isinstance(fitted, SplitDataset)
    assert fitted.test is fitted.validation is payload
    # Bare-frame/tuple low-level application remains an explicit sampling operation.
    directly_sampled = transformer.transform(payload)
    sampled_features = directly_sampled[0] if paired else directly_sampled
    assert len(sampled_features) == (4 if method == "random_under_sampling" else 8)
    result = transformer.transform(dataset)
    assert result is dataset
    assert result.train is result.test is result.validation is payload
    assert result.train_sample_weight is weights
    np.testing.assert_array_equal(np.asarray(_native(native)["target"]), [0, 0, 0, 0, 1, 1])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("policy", ["class_mean", "uniform"])
def test_synthetic_sampling_rejects_corrupted_wide_integer_prefix(
    sampler_components, engine, dtype, policy
):
    """The integrity guard must see corruption before mixed columns coerce both sides to float."""
    frame = pd.DataFrame(
        {
            "wide": pd.Series([2**53 + 1 + i for i in range(6)], dtype=dtype),
            "z": np.arange(6, dtype=float),
        }
    )
    source = _to_engine(frame, engine)
    original = frame.copy(deep=True)
    target = pd.Series([0, 0, 0, 0, 1, 1]) if engine == "pandas" else pl.Series([0, 0, 0, 0, 1, 1])
    config = _sampling_config("smote") | {"synthetic_weight": policy}
    with pytest.raises(SampleWeightError, match="original feature rows"):
        _sampler(sampler_components, "smote").fit_transform_weighted(
            (source, target), config, np.arange(1, 7, dtype=float)
        )
    actual = source if engine == "pandas" else source.to_pandas()
    pd.testing.assert_frame_equal(actual, original)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("method", ["random_over", "smote"])
def test_intact_wide_integer_sampling_preserves_exact_original_rows(
    sampler_components, engine, dtype, method
):
    """Wide integers are valid when the sampler really preserves their original values."""
    wide = [2**53 + 1 + i for i in range(6)] if method == "random_over" else [2**54] * 6
    frame = pd.DataFrame({"wide": pd.Series(wide, dtype=dtype), "z": np.arange(6, dtype=float)})
    source = _to_engine(frame, engine)
    target = pd.Series([0, 0, 0, 0, 1, 1]) if engine == "pandas" else pl.Series([0, 0, 0, 0, 1, 1])
    (sampled, _), weights = _sampler(sampler_components, method).fit_transform_weighted(
        (source, target), _sampling_config(method), np.arange(1, 7, dtype=float)
    )
    assert _native(sampled)["wide"].to_list()[:6] == wide
    np.testing.assert_array_equal(weights[:6], np.arange(1, 7, dtype=float))
    assert len(sampled) == len(weights) == 8


@pytest.mark.parametrize("kind", ["ndarray", "series"])
@pytest.mark.parametrize("dtype", ["int64", "uint64", "float32", "float64"])
def test_native_numeric_weight_vectors_do_not_box_and_inspect_each_scalar(monkeypatch, kind, dtype):
    """Repeated fit validation must use native dtype evidence rather than a Python row loop."""
    scalar_checks = []
    original_real = weight_module.Real

    class CountChecks(type):
        """Observe scalar type checks while retaining their genuine numeric result."""

        def __instancecheck__(cls, value):
            """Count the per-value fallback, preserving ordinary numeric validation."""
            scalar_checks.append(type(value))
            return isinstance(value, original_real)

    class CountedReal(metaclass=CountChecks):
        """Stand in for the numeric ABC without changing accepted value types."""

    monkeypatch.setattr(weight_module, "Real", CountedReal)
    original = np.arange(1, 1025, dtype=dtype)[::2]
    values = pd.Series(original) if kind == "series" else original
    result = validate_sample_weight(values, len(values))
    assert result is not None
    np.testing.assert_array_equal(result, original)
    result[0] = 0
    assert values.iloc[0] == 1 if kind == "series" else values[0] == 1
    assert scalar_checks == []


@pytest.mark.parametrize(
    "values",
    [
        [True, 1],
        (1, np.bool_(False)),
        np.array([True, False]),
        pd.Series([True, 1], dtype=object),
        pd.Series([True, False], dtype="boolean"),
        [None, 1],
        [pd.NA, 1],
        pd.Series([1, pd.NA], dtype="Int64"),
        pd.Series([1, pd.NA], dtype="Float64"),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
        np.array([-1, 2]),
        np.zeros(2),
        np.array([1e308, 1e308]),
        np.array([[1.0], [2.0]]),
        np.array([1]),
        np.array(1),
        np.array([1 + 0j, 2 + 0j]),
        np.array(["1", "2"]),
        [Decimal("NaN"), 1],
        [Decimal("Infinity"), 1],
        [2**1024, 1],
    ],
)
def test_native_weight_fast_path_preserves_invalid_value_contract(values):
    """Dtype optimizations cannot admit booleans, missing values, invalid dimensions, or totals."""
    with pytest.raises(SampleWeightError, match="sample_weight"):
        validate_sample_weight(values, 2)


@pytest.mark.parametrize(
    "values",
    [
        [Decimal("0.5"), Fraction(3, 2)],
        np.array([Decimal("0.5"), Fraction(3, 2)], dtype=object),
        pd.Series([0, 2], dtype="Int64"),
        pd.Series([0.5, 1.5], dtype="Float64"),
        pd.Series([0.5, 1.5], dtype="float64[pyarrow]"),
        pl.Series([0.5, 1.5]),
    ],
)
def test_non_native_numeric_weights_keep_fallback_and_copy_semantics(values):
    """Object and extension numeric vectors must retain accepted types and independent storage."""
    original = copy.deepcopy(values)
    result = validate_sample_weight(values, 2)
    assert result is not None
    np.testing.assert_array_equal(result, np.asarray(values, dtype=float))
    result[:] = 99
    np.testing.assert_array_equal(
        np.asarray(values, dtype=object), np.asarray(original, dtype=object)
    )


def test_optimized_weight_validation_keeps_actual_regression_fit_unchanged():
    """The public estimator must receive the copied unnormalized weights after fast validation."""
    from sklearn.linear_model import LinearRegression

    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0.0, 1.0, 2.0, 20.0])
    weights = np.array([1.0, 1.0, 1.0, 0.01])
    fitted = LinearRegressionCalculator().fit(features, target, {}, sample_weight=weights)
    reference = LinearRegression().fit(features.to_numpy(), target, sample_weight=weights)
    np.testing.assert_allclose(
        fitted.predict(features.to_numpy()), reference.predict(features.to_numpy())
    )
    np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0, 0.01])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize(
    "method",
    [
        "random_over",
        "random_under_sampling",
        "nearmiss",
        "tomek_links",
        "edited_nearest_neighbours",
    ],
)
def test_selected_rows_keep_exact_values_indices_names_and_weights(
    sampler_components, engine, weighted, dtype, method
):
    """Selection metadata must gather original values instead of imblearn's rounded mixed frame."""
    from skyulf.preprocessing.resampling import _build_oversampler, _build_undersampler

    frame = pd.DataFrame(
        {
            "wide": pd.Series([2**53 + 1 + i for i in range(8)], dtype=dtype),
            "z": np.arange(8, dtype=float),
        }
    )
    frame.index = [7, 7, 2, 2, 9, 9, 6, 6]
    target = pd.Series(["a"] * 5 + ["b"] * 3, index=[50, 40, 30, 20, 10, 4, 3, 2], name="class")
    config = _sampling_config(method) | {"n_neighbors": 1}
    builder = _build_oversampler if method == "random_over" else _build_undersampler
    sampler = builder(method, config)
    reference_X, reference_y = sampler.fit_resample(frame, target)
    positions = sampler.sample_indices_
    source = _to_engine(frame, engine)
    labels = target if engine == "pandas" else pl.Series("class", target)
    weights = np.arange(1, 9, dtype=float) if weighted else None
    (actual_X, actual_y), actual_weights = _sampler(
        sampler_components, method
    ).fit_transform_weighted((source, labels), config, weights)
    native = _native(actual_X)
    assert native["wide"].to_list() == frame["wide"].iloc[positions].to_list()
    np.testing.assert_array_equal(np.asarray(native["z"]), frame["z"].iloc[positions])
    np.testing.assert_array_equal(np.asarray(actual_y), target.iloc[positions])
    assert actual_y.name == "class"
    if engine == "pandas":
        assert native.index.equals(reference_X.index)
        assert actual_y.index.equals(reference_y.index)
        assert native["wide"].dtype == frame["wide"].dtype
    if weighted:
        np.testing.assert_array_equal(actual_weights, positions + 1)
    else:
        assert actual_weights is None
    assert frame["wide"].iloc[0] == 2**53 + 1


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("positions", [[-1, 1], [0, 6], [0.0, 1.0], [True, False], [[0, 1]], [0]])
def test_invalid_sampler_position_metadata_is_rejected(
    sampler_components, monkeypatch, weighted, positions
):
    """A malformed selection map must fail before fabricating feature, label, or weight alignment."""
    from skyulf.preprocessing import resampling

    class BrokenSampler:
        """Represent a library sampler returning unusable row selection metadata."""

        sample_indices_ = np.asarray(positions)

        def fit_resample(self, X, y):
            """Return ordinary-looking rows with the deliberately malformed map."""
            return X.iloc[:2], y.iloc[:2]

    monkeypatch.setattr(resampling, "_build_oversampler", lambda method, params: BrokenSampler())
    features = pd.DataFrame({"x": range(6)})
    target = pd.Series([0, 0, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="sample_indices"):
        _sampler(sampler_components, "random_over").fit_transform_weighted(
            (features, target), _sampling_config("random_over"), np.ones(6) if weighted else None
        )
