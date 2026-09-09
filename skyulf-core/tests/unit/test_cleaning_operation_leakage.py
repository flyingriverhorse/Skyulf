"""Regression coverage for cleaning, selection, and numeric preprocessing leakage."""

from importlib import import_module

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.preprocessing.bucketing import CustomBinningApplier, CustomBinningCalculator
from skyulf.preprocessing.casting import CastingApplier, CastingCalculator
from skyulf.preprocessing.drop_and_missing.drop_columns import DropMissingColumnsCalculator
from skyulf.preprocessing.drop_and_missing.missing_indicator import MissingIndicatorCalculator
from skyulf.preprocessing.imputation.simple import SimpleImputerCalculator
from skyulf.preprocessing.outliers.elliptic import (
    EllipticEnvelopeApplier,
    EllipticEnvelopeCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_custom_bins_auto_selection_is_declared_learned(engine):
    """Held-out values can activate an auto-selected binning feature before splitting."""
    train = pd.DataFrame({"x": [0.0, 1.0]})
    combined = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    if engine == "polars":
        train, combined = pl.from_pandas(train), pl.from_pandas(combined)
    config = {"bins": [0.0, 1.0, 3.0]}
    calculator = CustomBinningCalculator()
    train_params = calculator.fit(train, config)
    combined_params = calculator.fit(combined, config)
    assert train_params["bin_edges"] == {}
    assert combined_params["bin_edges"] == {"x": [0.0, 1.0, 3.0]}
    metadata = getattr(calculator, "__node_meta__", None)
    assert metadata is not None
    assert metadata.learns_from_data


def test_custom_bins_explicit_columns_ignore_binary_detection():
    """Fixed explicit bins remain usable on columns that happen to be binary in train."""
    train = pd.DataFrame({"x": [0.0, 1.0]})
    params = CustomBinningCalculator().fit(train, {"columns": ["x"], "bins": [0.0, 1.0, 3.0]})
    result = CustomBinningApplier().apply(train, params)
    assert result["x_binned"].tolist() == [0, 0]


@pytest.mark.parametrize("coerce", [True, False])
def test_datetime_cast_does_not_infer_format_from_other_rows(coerce):
    """A held-out date in another format must not turn valid training dates into NaT."""
    train = pd.DataFrame({"date": ["2024-01-02", "2024-03-04"]})
    params = CastingCalculator().fit(
        train, {"columns": ["date"], "target_type": "datetime", "coerce_on_error": coerce}
    )
    combined = pd.concat([pd.DataFrame({"date": ["04/05/2024"]}), train], ignore_index=True)
    result = CastingApplier().apply(combined, params)
    assert result["date"].tolist() == [
        pd.Timestamp("2024-04-05"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-03-04"),
    ]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_category_cast_reuses_training_vocabulary(engine):
    """Held-out categories must not grow the fitted categorical domain."""
    train = pd.DataFrame({"label": ["b", "c"]})
    held_out = pd.DataFrame({"label": ["a", "b"]})
    if engine == "polars":
        train, held_out = pl.from_pandas(train), pl.from_pandas(held_out)
    params = CastingCalculator().fit(train, {"column_types": {"label": "categorical"}})
    result = CastingApplier().apply(held_out, params)
    if engine == "polars":
        assert result["label"].cast(pl.String).to_list() == [None, "b"]
    else:
        assert result["label"].cat.categories.tolist() == ["b", "c"]
        assert result["label"].cat.codes.tolist() == [-1, 0]


def test_category_cast_drops_unobserved_inherited_categories_at_fit():
    """A sliced training frame must not inherit a vocabulary previously inferred on held-out rows."""
    train = pd.DataFrame({"label": pd.Categorical(["b", "c"], categories=["a", "b", "c"])})
    params = CastingCalculator().fit(train, {"columns": ["label"], "target_type": "category"})
    result = CastingApplier().apply(pd.DataFrame({"label": ["a", "b"]}), params)
    assert result["label"].cat.categories.tolist() == ["b", "c"]
    assert result["label"].cat.codes.tolist() == [-1, 0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf, np.nan])
def test_elliptic_nonfinite_companion_does_not_disable_finite_filter(engine, nonfinite):
    """One nonfinite row must not make finite held-out outliers bypass the fitted envelope."""
    train = pd.DataFrame({"x": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]})
    held_out = pd.DataFrame({"x": [0.0, 100.0, nonfinite]}, index=[4, 4, 5])
    y = np.array([10, 20, 30])
    if engine == "polars":
        train, held_out = pl.from_pandas(train), pl.from_pandas(held_out)
    params = EllipticEnvelopeCalculator().fit(train, {"columns": ["x"], "contamination": 0.1})
    result, target = EllipticEnvelopeApplier().apply((held_out, y), params)
    assert len(result) == 2
    assert target.tolist() == [10, 30]


LEARNED_CASES = [
    ("scaling.standard", "StandardScaler", {}),
    ("scaling.standard", "StandardScaler", {"with_mean": False}),
    ("scaling.standard", "StandardScaler", {"with_std": False}),
    ("scaling.minmax", "MinMaxScaler", {"feature_range": [-2, 2]}),
    ("scaling.maxabs", "MaxAbsScaler", {}),
    ("scaling.robust", "RobustScaler", {}),
    ("scaling.robust", "RobustScaler", {"with_centering": False}),
    ("scaling.robust", "RobustScaler", {"with_scaling": False}),
    ("imputation.simple", "SimpleImputer", {"strategy": "mean"}),
    ("imputation.simple", "SimpleImputer", {"strategy": "median"}),
    ("imputation.simple", "SimpleImputer", {"strategy": "most_frequent"}),
    ("imputation.simple", "SimpleImputer", {"strategy": "constant", "fill_value": -1}),
    ("imputation.knn", "KNNImputer", {"n_neighbors": 2}),
    ("imputation.iterative", "IterativeImputer", {}),
    ("bucketing", "GeneralBinning", {"strategy": "equal_width", "n_bins": 3}),
    ("bucketing", "GeneralBinning", {"strategy": "equal_frequency", "n_bins": 3}),
    ("bucketing", "GeneralBinning", {"strategy": "custom", "custom_bins": {"x": [0, 3, 10]}}),
    ("bucketing", "KBinsDiscretizer", {"strategy": "uniform", "n_bins": 3}),
    ("bucketing", "KBinsDiscretizer", {"strategy": "quantile", "n_bins": 3}),
    ("outliers.iqr", "IQR", {}),
    ("outliers.zscore", "ZScore", {"threshold": 1.5}),
    ("outliers.winsorize", "Winsorize", {"lower_percentile": 0, "upper_percentile": 100}),
    ("outliers.elliptic", "EllipticEnvelope", {"contamination": 0.1}),
    ("drop_and_missing.drop_columns", "DropMissingColumns", {"missing_threshold": 30}),
    ("drop_and_missing.missing_indicator", "MissingIndicator", {}),
    ("feature_selection.correlation", "CorrelationThreshold", {"threshold": 0.7}),
    ("feature_selection.variance", "VarianceThreshold", {}),
    ("feature_selection.univariate", "UnivariateSelection", {"k": 1}),
    ("feature_selection.univariate", "UnivariateSelection", {"score_func": "chi2", "k": 1}),
    ("feature_selection.model_based", "ModelBasedSelection", {"estimator": "logistic_regression"}),
    ("feature_selection.model_based", "ModelBasedSelection", {"method": "rfe", "k": 1}),
    ("feature_selection.facade", "FeatureSelection", {"method": "variance"}),
]


@pytest.mark.parametrize("module_name,node_name,config", LEARNED_CASES)
def test_held_out_values_do_not_change_training_representation(module_name, node_name, config):
    """Held-out values, missingness, and numeric eligibility must never refit training state."""
    module = import_module(f"skyulf.preprocessing.{module_name}")
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "z": [5.0, 2.0, 6.0, 3.0, 7.0, 4.0],
            "binary": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "constant": [7.0] * 6,
        }
    )
    target = pd.Series([0, 0, 1, 1, 0, 1], name="target")
    test_a = pd.DataFrame(
        {"x": [3.0, 4.0], "z": [3.0, 4.0], "binary": [0.0, 1.0], "constant": [7.0, 7.0]}
    )
    test_b = pd.DataFrame(
        {
            "x": [np.nan, 10000.0],
            "z": [-9000.0, np.nan],
            "binary": [10.0, 20.0],
            "constant": [-50.0, 50.0],
        }
    )
    train_results = []
    probe_results = []
    for held_out in (test_a, test_b):
        transformer = StatefulTransformer(
            getattr(module, f"{node_name}Calculator")(),
            getattr(module, f"{node_name}Applier")(),
            node_name,
        )
        data = SplitDataset(
            train=(train.copy(), target.copy()),
            test=(held_out, pd.Series([0, 1], name="target")),
            validation=(held_out.iloc[::-1], pd.Series([1, 0], name="target")),
        )
        result = transformer.fit_transform(data, config)
        assert isinstance(result, SplitDataset)
        assert isinstance(result.train, tuple)
        train_frame = result.train[0]
        assert isinstance(train_frame, pd.DataFrame)
        train_results.append(train_frame)
        probe = transformer.transform(test_a)
        assert isinstance(probe, pd.DataFrame)
        probe_results.append(probe)
    assert_frame_equal(train_results[0], train_results[1])
    assert_frame_equal(probe_results[0], probe_results[1])


@pytest.mark.parametrize(
    "module_name,node_name,key,expected",
    [
        ("standard", "StandardScaler", "mean", [3.0]),
        ("minmax", "MinMaxScaler", "data_max", [6.0]),
        ("maxabs", "MaxAbsScaler", "max_abs", [6.0]),
        ("robust", "RobustScaler", "center", [3.0]),
    ],
)
def test_scaler_statistics_and_auto_columns_come_only_from_train(
    module_name, node_name, key, expected
):
    """Binary and constant held-out changes cannot activate a scaler or enter its statistics."""
    module = import_module(f"skyulf.preprocessing.scaling.{module_name}")
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0], "binary": [0.0, 1.0, 0.0, 1.0]})
    held_out = pd.DataFrame({"x": [1000.0], "binary": [999.0]})
    transformer = StatefulTransformer(
        getattr(module, f"{node_name}Calculator")(),
        getattr(module, f"{node_name}Applier")(),
        node_name,
    )
    transformer.fit_transform(SplitDataset(train=train, test=held_out), {})
    assert transformer.params["columns"] == ["x"]
    assert transformer.params[key] == expected


@pytest.mark.parametrize("threshold", [None, 0, -1, "invalid", float("nan")])
def test_drop_columns_nonpositive_threshold_is_explicit_only(threshold):
    """Disabled or invalid thresholds must not learn extra drops from missing held-out values."""
    train = pd.DataFrame({"keep": [1.0, np.nan], "drop": [2.0, 3.0]})
    params = DropMissingColumnsCalculator().fit(
        train, {"columns": ["drop"], "missing_threshold": threshold}
    )
    assert params["columns_to_drop"] == ["drop"]


@pytest.mark.parametrize("selection", [None, []])
def test_missing_indicator_empty_selection_is_data_dependent(selection):
    """An empty picker is auto-detection for this node and cannot receive a static exemption."""
    clean = pd.DataFrame({"x": [1.0, 2.0]})
    missing = pd.DataFrame({"x": [1.0, np.nan]})
    calculator = MissingIndicatorCalculator()
    assert calculator.fit(clean, {"columns": selection})["columns"] == []
    assert calculator.fit(missing, {"columns": selection})["columns"] == ["x"]


@pytest.mark.parametrize("dtype,expected", [("float64", 0), ("object", "missing_value")])
def test_constant_imputer_default_depends_on_dtype_not_observed_values(dtype, expected):
    """Constant default fills may vary by schema, but must not vary with values or missing counts."""
    first = pd.DataFrame({"x": pd.Series([1.0, np.nan], dtype=dtype)})
    second = pd.DataFrame({"x": pd.Series([99.0, 50.0], dtype=dtype)})
    calculator = SimpleImputerCalculator()
    assert calculator.fit(first, {"strategy": "constant"})["fill_values"] == {"x": expected}
    assert calculator.fit(second, {"strategy": "constant"})["fill_values"] == {"x": expected}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_legacy_category_artifact_keeps_historical_conversion(engine):
    """Previously persisted category artifacts remain readable without inventing a vocabulary."""
    frame = pd.DataFrame({"label": ["new", "other"]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    result = CastingApplier().apply(frame, {"type_map": {"label": "category"}})
    values = (
        result["label"].cast(pl.String).to_list()
        if engine == "polars"
        else result["label"].tolist()
    )
    assert values == ["new", "other"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_empty_training_category_domain_does_not_learn_from_held_out_values(engine):
    """An all-missing training category has no observed vocabulary to grow at inference."""
    train = pd.DataFrame({"label": [None, None]})
    held_out = pd.DataFrame({"label": ["unseen"]})
    if engine == "polars":
        train, held_out = pl.from_pandas(train), pl.from_pandas(held_out)
    params = CastingCalculator().fit(train, {"columns": ["label"], "target_type": "category"})
    result = CastingApplier().apply(held_out, params)
    missing = (
        result["label"].null_count() if engine == "polars" else int(result["label"].isna().sum())
    )
    assert missing == 1
