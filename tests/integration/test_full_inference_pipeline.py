"""Verify full preprocessing and prediction using isolated, persisted job artifacts."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.database.models import Base, TrainingJob
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.service import DeploymentService
from skyulf.modeling.base import extract_xy


@pytest.mark.parametrize("frame_engine", ["pandas", "polars"])
async def test_full_inference_pipeline(tmp_path, monkeypatch, frame_engine):
    """Training, row filtering, encoding and scaling must survive artifact reload and deployment."""
    monkeypatch.setenv("SKYULF_ENGINE", frame_engine)
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", frame_engine)
    base_path = str(tmp_path / "artifacts")
    store = LocalArtifactStore(base_path)
    catalog = FileSystemCatalog(str(tmp_path))
    engine = PipelineEngine(store, catalog=catalog)

    # 2. Create Complex Dummy Data
    # - 'age': Numeric, has outliers (150), has missing (NaN)
    # - 'income': Numeric, needs scaling
    # - 'city': Categorical, needs encoding
    # - 'gender': Categorical, needs encoding
    # - 'target': Binary

    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 150, 40, np.nan, 22, 28, 33, 45],
            "income": [
                50000,
                60000,
                70000,
                80000,
                90000,
                55000,
                45000,
                65000,
                75000,
                85000,
            ],
            "city": ["NY", "LA", "NY", "SF", "LA", "SF", "NY", "LA", "SF", "NY"],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    data_path = str(Path(base_path) / "data.csv")
    df.to_csv(data_path, index=False)

    # 3. Define Pipeline
    nodes = []

    # Node 1: Loader
    nodes.append(NodeConfig(node_id="loader", step_type="data_loader", params={"path": data_path}))

    # Node 2: Value Replacement (Cleaning)
    # Replace 'SF' with 'San Francisco'
    nodes.append(
        NodeConfig(
            node_id="clean_city",
            step_type="ValueReplacement",
            params={"to_replace": {"SF": "San Francisco"}, "columns": ["city"]},
            inputs=["loader"],
        )
    )

    # Node 3: Manual Bounds (Outliers)
    # Drop training and inference rows outside the configured age bounds.
    nodes.append(
        NodeConfig(
            node_id="clip_age",
            step_type="ManualBounds",
            params={"bounds": {"age": {"lower": 0, "upper": 100}}},
            inputs=["clean_city"],
        )
    )

    # Node 4: Splitter
    # Moved up (right after stateless cleaning/clipping) so every
    # data-dependent preprocessing node below fits only on the train
    # portion and is merely applied to test — avoids leaking test-set
    # statistics into imputation/encoding/scaling/feature-selection params.
    nodes.append(
        NodeConfig(
            node_id="splitter",
            step_type="TrainTestSplitter",
            params={"test_size": 0.2, "random_state": 42, "target_column": "target"},
            inputs=["clip_age"],
        )
    )

    # Node 5: Simple Imputer (Imputation)
    # Fill missing age with median (fit on train only, applied to test too)
    nodes.append(
        NodeConfig(
            node_id="impute_age",
            step_type="SimpleImputer",
            params={"strategy": "median", "columns": ["age"]},
            inputs=["splitter"],
        )
    )

    # Node 6: One Hot Encoder (Encoding)
    # Encode city and gender
    nodes.append(
        NodeConfig(
            node_id="encode_cats",
            step_type="OneHotEncoder",
            params={"columns": ["city", "gender"], "handle_unknown": "ignore"},
            inputs=["impute_age"],
        )
    )

    # Node 7: Standard Scaler (Scaling)
    # Scale income
    nodes.append(
        NodeConfig(
            node_id="scale_income",
            step_type="StandardScaler",
            params={"columns": ["income"]},
            inputs=["encode_cats"],
        )
    )

    # Node 8: Variance Threshold (Feature Selection)
    # Remove low variance features (dummy check)
    nodes.append(
        NodeConfig(
            node_id="select_features",
            step_type="VarianceThreshold",
            params={"threshold": 0.0},
            inputs=["scale_income"],
        )
    )

    # Node 9: Model
    nodes.append(
        NodeConfig(
            node_id="model",
            step_type="training",
            params={"algorithm": "logistic_regression", "target_column": "target"},
            inputs=["select_features"],
        )
    )

    config = PipelineConfig(pipeline_id="full_test_pipeline", nodes=nodes)

    result = engine.run(config, job_id="full-inference")
    assert result.status == "success", {
        node_id: node.error for node_id, node in result.node_results.items() if node.error
    }
    assert all(node.status == "success" for node in result.node_results.values())
    # The node artifact is a (model, tuning metadata) tuple; the job artifact
    # carries the fitted preprocessing needed to serve raw records.
    assert isinstance(store.load("model"), tuple)
    reopened = LocalArtifactStore(base_path)
    bundle = reopened.load("full-inference")
    assert isinstance(bundle, dict)
    engineer = bundle["feature_engineer"]

    new_data = pd.DataFrame(
        {
            "age": [50.0, 120.0, np.nan],
            "income": [50000.0, 50000.0, 75000.0],
            "city": ["SF", "NY", "LA"],
            "gender": ["M", "F", "F"],
        }
    )
    transformed = engineer.transform(new_data)
    if hasattr(transformed, "to_pandas"):
        transformed = transformed.to_pandas()
    assert len(transformed) == 2
    assert "city" not in transformed and "gender" not in transformed
    np.testing.assert_array_equal(transformed["city_San Francisco"], [1.0, 0.0])
    np.testing.assert_array_equal(transformed["city_LA"], [0.0, 1.0])
    np.testing.assert_array_equal(transformed["gender_M"], [1.0, 0.0])
    split_features, _ = extract_xy(reopened.load("splitter").train, "target")
    if hasattr(split_features, "to_native"):
        split_features = split_features.to_native()
    if hasattr(split_features, "to_pandas"):
        split_features = split_features.to_pandas()
    np.testing.assert_allclose(transformed["age"], [50.0, split_features["age"].median()])
    expected_income = (
        np.array([50000.0, 75000.0]) - split_features["income"].mean()
    ) / split_features["income"].std(ddof=0)
    np.testing.assert_allclose(transformed["income"], expected_income)
    assert list(transformed.columns) == bundle["feature_columns"]

    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'inference.db'}")
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(database, expire_on_commit=False) as session:
            session.add(
                TrainingJob(
                    id="full-inference",
                    pipeline_id=config.pipeline_id,
                    node_id="model",
                    dataset_source_id="source",
                    status="completed",
                    run_mode="fixed",
                    model_type="logistic_regression",
                    graph=asdict(config),
                    artifact_uri=reopened.get_artifact_uri("full-inference"),
                )
            )
            await session.commit()
            await DeploymentService.deploy_model(session, "full-inference")
            predictions, thresholds = await DeploymentService.predict(
                session, new_data.to_dict("records")
            )
    finally:
        await database.dispose()
    expected = bundle["model"].predict(transformed[bundle["feature_columns"]])
    np.testing.assert_array_equal(predictions, expected)
    assert len(predictions) == 2 and thresholds is None
