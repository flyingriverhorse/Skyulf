import pytest
from sqlalchemy import delete

from backend.database import engine
from backend.database.models import ModelVersionCounter, TrainingJob
from backend.ml_pipeline._execution.jobs import JobManager
from backend.ml_pipeline.constants import StepType


@pytest.mark.asyncio
async def test_unified_versioning():
    # Initialize DB if needed (might be already initialized by other tests or app startup)
    await engine.init_db()
    await engine.create_tables()

    async with engine.async_session_factory() as session:
        pipeline_id = "test_pipeline_unified"
        node_id = "test_node_unified"
        dataset_id = "ds_unified"
        model_type = "rf_unified"

        # Cleanup
        await session.execute(delete(TrainingJob).where(TrainingJob.pipeline_id == pipeline_id))
        # The version counter now lives in its own table (fixes the
        # get_next_version race condition) - it must be reset too, since it's
        # no longer derived purely from the job rows deleted above.
        await session.execute(
            delete(ModelVersionCounter).where(
                ModelVersionCounter.dataset_source_id == dataset_id,
                ModelVersionCounter.model_type == model_type,
            )
        )
        await session.commit()

        # 1. Create Training Job (Should be v1)
        job1_id = await JobManager.create_job(
            session,
            pipeline_id,
            node_id,
            "training",
            dataset_id=dataset_id,
            model_type=model_type,
        )
        job1 = await session.get(TrainingJob, job1_id)
        assert job1.version == 1

        # 2. Create Tuning Job (Should be v2)
        job2_id = await JobManager.create_job(
            session,
            pipeline_id,
            node_id,
            "tuning",
            dataset_id=dataset_id,
            model_type=model_type,
        )
        job2 = await session.get(TrainingJob, job2_id)
        assert job2.version == 2

        # 3. Create Training Job (Should be v3)
        job3_id = await JobManager.create_job(
            session,
            pipeline_id,
            node_id,
            "training",
            dataset_id=dataset_id,
            model_type=model_type,
        )
        job3 = await session.get(TrainingJob, job3_id)
        assert job3.version == 3

        # 4. Create Tuning Job (Should be v4)
        job4_id = await JobManager.create_job(
            session,
            pipeline_id,
            node_id,
            "tuning",
            dataset_id=dataset_id,
            model_type=model_type,
        )
        job4 = await session.get(TrainingJob, job4_id)
        assert job4.version == 4
