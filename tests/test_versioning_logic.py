import asyncio
import sys
import os
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import engine
from core.ml_pipeline.execution.jobs import JobManager
from core.database.models import TrainingJob, HyperparameterTuningJob
from sqlalchemy import select, delete

async def test_versioning():
    print("Initializing Database...")
    await engine.init_db()
    
    async with engine.async_session_factory() as session:
        pipeline_id = "test_pipeline_v1"
        node_id = "test_node_v1"
        
        # Cleanup previous tests
        print("Cleaning up previous test data...")
        await session.execute(delete(TrainingJob).where(TrainingJob.pipeline_id == pipeline_id))
        await session.execute(delete(HyperparameterTuningJob).where(HyperparameterTuningJob.pipeline_id == pipeline_id))
        await session.commit()

        print("\n--- Testing Training Job Versioning ---")
        # Create first job
        job1_id = await JobManager.create_job(
            session, pipeline_id, node_id, "training", 
            dataset_id="ds1", model_type="rf"
        )
        print(f"Created Job 1: {job1_id}")
        
        # Create second job
        job2_id = await JobManager.create_job(
            session, pipeline_id, node_id, "training", 
            dataset_id="ds1", model_type="rf"
        )
        print(f"Created Job 2: {job2_id}")

        # Verify versions
        job1 = await session.get(TrainingJob, job1_id)
        job2 = await session.get(TrainingJob, job2_id)
        
        print(f"Job 1 Version: {job1.version}")
        print(f"Job 2 Version: {job2.version}")
        
        if job1.version == 1 and job2.version == 2:
            print("✅ Training Job Versioning Passed")
        else:
            print("❌ Training Job Versioning Failed")

        print("\n--- Testing Tuning Job Versioning ---")
        # Create first tuning job
        tune1_id = await JobManager.create_job(
            session, pipeline_id, node_id, "tuning", 
            dataset_id="ds1", model_type="rf"
        )
        print(f"Created Tuning Job 1: {tune1_id}")
        
        # Create second tuning job
        tune2_id = await JobManager.create_job(
            session, pipeline_id, node_id, "tuning", 
            dataset_id="ds1", model_type="rf"
        )
        print(f"Created Tuning Job 2: {tune2_id}")

        # Verify run numbers
        tune1 = await session.get(HyperparameterTuningJob, tune1_id)
        tune2 = await session.get(HyperparameterTuningJob, tune2_id)
        
        print(f"Tuning Job 1 Run Number: {tune1.run_number}")
        print(f"Tuning Job 2 Run Number: {tune2.run_number}")
        
        if tune1.run_number == 1 and tune2.run_number == 2:
            print("✅ Tuning Job Versioning Passed")
        else:
            print("❌ Tuning Job Versioning Failed")

if __name__ == "__main__":
    asyncio.run(test_versioning())
