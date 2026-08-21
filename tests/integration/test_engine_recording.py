"""F-25: the DataFrame engine a job trained on must be recorded, not guessed.

Covers both recording surfaces:

- The deployment bundle (``_bundle_transformers_with_model``) carries an
  ``engine`` key detected from the actual training frame.
- The training job row records the configured engine in ``job_metadata``.
"""

from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.database.models import Base, TrainingJob
from backend.ml_pipeline._execution.basic_training_manager import BasicTrainingManager
from backend.ml_pipeline._execution.engine._feature_eng import FeatureEngMixin
from backend.ml_pipeline._execution.engine._node_runners import NodeRunnersMixin
from skyulf.data.dataset import SplitDataset
from skyulf.engines.polars_engine import SkyulfPolarsWrapper


class _Harness(NodeRunnersMixin):
    """Minimal stand-in for :class:`PipelineEngine` for pure-logic helpers."""

    def __init__(self):
        self.logs: list[str] = []
        self.artifact_store = MagicMock()
        self.catalog = MagicMock()
        self.executed_transformers: list[dict] = []

    def log(self, msg: str) -> None:
        self.logs.append(msg)


class _BundleHarness(FeatureEngMixin):
    """Minimal harness exposing the bundle-build path with a mock store."""

    def __init__(self):
        self.artifact_store = MagicMock()
        self.executed_transformers: list[dict] = []

    def log(self, msg: str) -> None:
        pass


# ── Training-frame engine detection ───────────────────────────────────────


def test_resolve_train_engine_detects_pandas():
    assert _Harness()._resolve_train_engine(pd.DataFrame({"a": [1]})) == "pandas"


def test_resolve_train_engine_detects_polars():
    assert _Harness()._resolve_train_engine(pl.DataFrame({"a": [1]})) == "polars"


def test_resolve_train_engine_unwraps_skyulf_wrapper():
    wrapped = SkyulfPolarsWrapper(pl.DataFrame({"a": [1]}))
    assert _Harness()._resolve_train_engine(wrapped) == "polars"


def test_resolve_train_engine_unwraps_split_dataset():
    split = SplitDataset(train=pl.DataFrame({"a": [1]}), test=pl.DataFrame(), validation=None)
    assert _Harness()._resolve_train_engine(split) == "polars"


def test_resolve_train_engine_falls_back_to_settings(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "SKYULF_ENGINE", "pandas", raising=False)
    assert _Harness()._resolve_train_engine(None) == "pandas"


def test_resolve_train_engine_non_frame_falls_back_to_settings():
    # An (X, y) tuple of numpy arrays carries no frame to detect from.
    import numpy as np

    assert _Harness()._resolve_train_engine((np.array([[1.0]]), np.array([0]))) in (
        "pandas",
        "polars",
    )


# ── Deployment bundle records the engine ─────────────────────────────────


def test_bundle_records_engine_new_format():
    harness = _BundleHarness()
    harness.artifact_store.load.return_value = MagicMock()  # the model artifact
    feature_engineer = MagicMock()

    harness._bundle_transformers_with_model(
        "model_key",
        job_id="job-1",
        feature_engineer_override=feature_engineer,
        engine="polars",
    )

    _, saved = harness.artifact_store.save.call_args.args
    assert saved["engine"] == "polars"
    assert saved["feature_engineer"] is feature_engineer


def test_bundle_records_engine_legacy_format():
    harness = _BundleHarness()
    harness.artifact_store.load.return_value = MagicMock()

    harness._bundle_transformers_with_model(
        "model_key",
        job_id="job-2",
        engine="pandas",
    )

    _, saved = harness.artifact_store.save.call_args.args
    assert saved["engine"] == "pandas"
    assert "transformers" in saved


# ── Job record carries the engine ─────────────────────────────────────────


@pytest.fixture
async def async_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_training_job_records_engine(async_session, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "SKYULF_ENGINE", "polars", raising=False)

    job_id = await BasicTrainingManager.create_training_job(
        async_session,
        pipeline_id="pipe-1",
        node_id="node-1",
        dataset_id="ds-1",
        model_type="classifier",
    )

    job = await async_session.get(TrainingJob, job_id)
    assert job is not None
    assert job.job_metadata["engine"] == "polars"
    assert job.job_metadata["branch_index"] == 0


# ── JobInfo surfaces the engine ───────────────────────────────────────────


def test_map_training_job_to_info_surfaces_engine():
    job = TrainingJob(
        id="job-3",
        pipeline_id="pipe-1",
        node_id="node-1",
        status="completed",
        run_mode="fixed",
        model_type="classifier",
        job_metadata={"branch_index": 0, "engine": "polars"},
    )
    info = BasicTrainingManager.map_training_job_to_info(job, "dataset")
    assert info.engine == "polars"


def test_map_training_job_to_info_engine_missing_for_legacy_rows():
    job = TrainingJob(
        id="job-4",
        pipeline_id="pipe-1",
        node_id="node-1",
        status="completed",
        run_mode="fixed",
        model_type="classifier",
        job_metadata={"branch_index": 0},
    )
    info = BasicTrainingManager.map_training_job_to_info(job, "dataset")
    assert info.engine is None


def test_map_tuning_job_to_info_surfaces_engine():
    from backend.ml_pipeline._execution.advanced_tuning_manager import AdvancedTuningManager

    job = TrainingJob(
        id="job-5",
        pipeline_id="pipe-1",
        node_id="node-1",
        status="completed",
        run_mode="tuned",
        model_type="classifier",
        job_metadata={"branch_index": 0, "engine": "pandas"},
    )
    info = AdvancedTuningManager.map_tuning_job_to_info(job, "dataset")
    assert info.engine == "pandas"
