"""Exercise selection and batch boundaries omitted from the registry inventory."""

from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.leakage import step_learns_from_data, validate_leakage_safety
from skyulf.registry import NodeRegistry

_EMPTY_SELECTION_NOOPS = [
    "OneHotEncoder",
    "DummyEncoder",
    "TargetEncoder",
    "WOEEncoder",
    "HashEncoder",
    "PowerTransformer",
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "GeneralBinning",
    "KBinsDiscretizer",
    "CustomBinning",
    "IQR",
    "ZScore",
    "Winsorize",
    "EllipticEnvelope",
]


@pytest.mark.parametrize("node_type", _EMPTY_SELECTION_NOOPS)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_registered_empty_selection_preserves_features_and_labels(node_type, engine):
    """An admission exemption must actually leave rows, features, and labels untouched."""
    frame: Any = pd.DataFrame(
        {"x": [1.0, 3.0, None, 1000.0], "category": ["a", "b", "a", "unseen"]},
        index=[4, 4, 2, 9],
    )
    labels: Any = pd.Series([0, 1, 0, 1], index=frame.index, name="target")
    if engine == "polars":
        frame, labels = pl.from_pandas(frame), pl.from_pandas(labels)
    config = {"columns": [], "target_column": "target"}
    calculator = NodeRegistry.get_calculator(node_type)()
    applier = NodeRegistry.get_applier(node_type)()

    artifact = calculator.fit((frame, labels), config)
    result, actual_labels = applier.apply((frame, labels), artifact)

    if engine == "pandas":
        pd.testing.assert_frame_equal(result, frame)
        pd.testing.assert_series_equal(actual_labels, labels)
    else:
        assert result.equals(frame)
        assert actual_labels.equals(labels)
    assert not step_learns_from_data(node_type, config, target_column="target")


@pytest.mark.parametrize("node_type", ["PolynomialFeatures", "PolynomialFeaturesNode"])
@pytest.mark.parametrize("columns", ["omitted", []])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_polynomial_auto_selection_requires_training_boundary(node_type, columns, engine):
    """Held-out values can activate an auto-selected polynomial feature and require a split."""
    train: Any = pd.DataFrame({"x": [0.0, 1.0]})
    combined: Any = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    if engine == "polars":
        train, combined = pl.from_pandas(train), pl.from_pandas(combined)
    config: dict[str, Any] = {"auto_detect": True, "degree": 2}
    if columns != "omitted":
        config["columns"] = columns
    calculator = NodeRegistry.get_calculator(node_type)()
    applier = NodeRegistry.get_applier(node_type)()

    train_artifact = calculator.fit(train, config)
    combined_artifact = calculator.fit(combined, config)
    assert "poly_x_pow_2" not in applier.apply(train, train_artifact).columns
    assert applier.apply(train, combined_artifact)["poly_x_pow_2"].to_list() == [0.0, 1.0]

    with pytest.raises(ValueError, match=node_type):
        validate_leakage_safety(
            {
                "preprocessing": [
                    {"transformer": node_type, "params": config},
                    {"transformer": "TrainTestSplitter", "params": {}},
                ]
            }
        )


@pytest.mark.parametrize("node_type", ["PolynomialFeatures", "PolynomialFeaturesNode"])
@pytest.mark.parametrize(
    "config",
    [{}, {"columns": []}, {"columns": ["x"]}, {"columns": ["x"], "auto_detect": True}],
)
def test_polynomial_fixed_selection_remains_exempt(node_type, config):
    """Blocking automatic selection must keep explicit bases and disabled nodes usable."""
    assert not step_learns_from_data(node_type, config)


@pytest.mark.parametrize(
    "node_type,config",
    [
        ("DateFeatures", {"columns": ["date"], "features": ["year", "month", "day"]}),
        *[
            (
                node_type,
                {
                    "operations": [
                        {
                            "operation_type": "datetime_extract",
                            "input_columns": ["date"],
                            "datetime_features": ["year", "month", "day"],
                        }
                    ]
                },
            )
            for node_type in ["FeatureGeneration", "FeatureGenerationNode", "FeatureMath"]
        ],
    ],
)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_calendar_features_do_not_infer_format_from_batch_companions(node_type, config, engine):
    """Another date format must not turn valid training dates into missing calendar features."""
    train: Any = pd.DataFrame({"date": ["2024-01-02", "2024-03-04"]})
    combined: Any = pd.DataFrame({"date": ["04/05/2024", "2024-01-02", "2024-03-04"]})
    if engine == "polars":
        train, combined = pl.from_pandas(train), pl.from_pandas(combined)
    artifact = NodeRegistry.get_calculator(node_type)().fit(train, config)
    result = NodeRegistry.get_applier(node_type)().apply(combined, artifact)

    assert not any(pd.isna(value) for value in result["date_year"].to_list())
    assert result["date_year"].to_list() == [2024, 2024, 2024]
    assert result["date_month"].to_list() == [4, 1, 3]
    assert result["date_day"].to_list() == [5, 2, 4]


@pytest.mark.parametrize(
    "node_type",
    [
        "LabelEncoder",
        "OrdinalEncoder",
        "WOEEncoder",
        "DummyEncoder",
        "OneHotEncoder",
        "TargetEncoder",
        "HashEncoder",
    ],
)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_numeric_category_replay_ignores_fractional_batch_sibling(node_type, engine):
    """An unrelated fractional row must not make a known integral category become unknown."""
    train: Any = pd.DataFrame({"category": [1, 1, 2, 2]})
    labels: Any = pd.Series([0, 0, 1, 1], name="target")
    alone: Any = pd.DataFrame({"category": [1]})
    together: Any = pd.DataFrame({"category": [1.0, 2.5]})
    if engine == "polars":
        train, labels = pl.from_pandas(train), pl.from_pandas(labels)
        alone, together = pl.from_pandas(alone), pl.from_pandas(together)
    config = {"columns": ["category"]}
    artifact = NodeRegistry.get_calculator(node_type)().fit((train, labels), config)
    applier = NodeRegistry.get_applier(node_type)()

    single_result = applier.apply(alone, artifact)
    batch_result = applier.apply(together, artifact)

    assert list(single_result.columns) == list(batch_result.columns)
    for column in single_result.columns:
        assert single_result[column].to_list()[0] == batch_result[column].to_list()[0]
