"""Five realistic pandas/Polars pipelines intended for the SM-24a cloud probe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.databricks.local_batch import fit_local_workflow

CASES = [
    (
        "R1",
        "pandas",
        "linear_regression",
        "regression_target",
        ("x", "z"),
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["z"], "strategy": "mean"},
            },
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x", "z"]}},
            {
                "name": "cross",
                "transformer": "FeatureInteraction",
                "params": {"columns": ["x", "z"]},
            },
        ],
    ),
    (
        "R2",
        "polars",
        "random_forest_regressor",
        "regression_target",
        ("x", "z", "city"),
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["z"], "strategy": "median"},
            },
            {"name": "clip", "transformer": "Winsorize", "params": {"columns": ["x"]}},
            {
                "name": "band",
                "transformer": "CustomBinning",
                "params": {
                    "columns": ["x"],
                    "bins": [-1000.0, 0.0, 5.0, 10.0, 1000.0],
                    "output_suffix": "_band",
                },
            },
            {
                "name": "encode",
                "transformer": "OneHotEncoder",
                "params": {
                    "columns": ["city", "x_band"],
                    "handle_unknown": "ignore",
                    "drop_original": True,
                },
            },
        ],
    ),
    (
        "C1",
        "pandas",
        "logistic_regression",
        "classification_target",
        ("x", "z", "city"),
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["z"], "strategy": "mean"},
            },
            {
                "name": "ordinal",
                "transformer": "OrdinalEncoder",
                "params": {
                    "columns": ["city"],
                    "handle_unknown": "use_encoded_value",
                    "unknown_value": -1,
                },
            },
            {"name": "scale", "transformer": "RobustScaler", "params": {"columns": ["x", "z"]}},
        ],
    ),
    (
        "C2",
        "polars",
        "random_forest_classifier",
        "classification_target",
        ("x", "z", "city"),
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["z"], "strategy": "mean"},
            },
            {
                "name": "encode",
                "transformer": "OneHotEncoder",
                "params": {
                    "columns": ["city"],
                    "handle_unknown": "ignore",
                    "drop_original": True,
                },
            },
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x", "z"]}},
            {
                "name": "cross",
                "transformer": "FeatureInteraction",
                "params": {"columns": ["x", "z"]},
            },
        ],
    ),
    (
        "R3",
        "pandas",
        "gradient_boosting_regressor",
        "regression_target",
        ("x", "z"),
        [
            {
                "name": "fill",
                "transformer": "KNNImputer",
                "params": {"columns": ["x", "z"], "n_neighbors": 3},
            },
            {
                "name": "select",
                "transformer": "VarianceThreshold",
                "params": {"columns": ["x", "z"], "threshold": 0.0},
            },
            {"name": "power", "transformer": "PowerTransformer", "params": {"columns": ["x", "z"]}},
        ],
    ),
]


def sample_data(count: int = 120) -> pd.DataFrame:
    """Create deterministic missing values, outliers, labels and categories."""
    positions = np.arange(count)
    x = positions.astype("float64") / 12.0
    x[positions == 17] = 45.0
    z = np.sin(positions / 7.0) * 3.0 + 4.0
    z[positions % 19 == 0] = np.nan
    city = np.array(["Vilnius", "Riga", "Tallinn"])[positions % 3]
    clean_z = np.nan_to_num(z, nan=4.0)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "city": city,
            "regression_target": 2.0 * x + clean_z + (positions % 3),
            "classification_target": (((x % 3.0) + clean_z / 3.0) > 2.4).astype("int64"),
        }
    )


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_five_models_round_trip_on_recorded_local_engine(tmp_path, case) -> None:
    """The planned live model families must survive fit, save, load and prediction."""
    name, engine, model, target, columns, preprocessing = case
    frame = sample_data().loc[:, [*columns, target]]
    native = pl.from_pandas(frame) if engine == "polars" else frame
    train = native[:96]
    test = native[96:]
    artifact = fit_local_workflow(
        {"preprocessing": preprocessing, "modeling": {"type": model}},
        SplitDataset(train=train, test=test),
        target_column=target,
        artifact_path=tmp_path / name,
        max_rows=120,
        max_bytes=128_000,
    )
    query = sample_data(6).loc[:, list(columns)]
    if "city" in query:
        query.loc[0, "city"] = "new-city"
    predictions = predict_local_pipeline(query, artifact)
    assert len(predictions) == len(query)
    assert predictions["prediction"].notna().all()
    native_query = pl.from_pandas(query) if engine == "polars" else query
    transformed = artifact.pipeline.feature_engineer.transform(native_query, preserve_rows=True)
    estimator = artifact.pipeline.model_estimator
    assert estimator is not None
    direct = estimator._unwrap_tuned_model().predict(
        transformed.to_pandas() if isinstance(transformed, pl.DataFrame) else transformed
    )
    if target == "classification_target":
        np.testing.assert_array_equal(predictions["prediction"].to_numpy(), direct)
    else:
        np.testing.assert_allclose(predictions["prediction"].to_numpy(), direct, rtol=0, atol=1e-12)
    if target == "classification_target":
        np.testing.assert_allclose(predictions[["probability_0", "probability_1"]].sum(axis=1), 1.0)
