"""Regression tests for the model registry version-allocation race condition.

`ModelRegistryService.get_next_version` previously computed `MAX(version) + 1`
via a plain SELECT with no locking, so concurrent training/tuning job
submissions for the same (dataset_id, model_type) could read the same max and
be handed identical "next" versions. It's now backed by a single
`UPDATE ... RETURNING` against a `ModelVersionCounter` row, which the database
serializes atomically.
"""

import asyncio

import pytest
from sqlalchemy import delete

from backend.database import engine
from backend.database.models import ModelVersionCounter, TrainingJob
from backend.ml_pipeline.model_registry.service import ModelRegistryService


async def _cleanup(dataset_id: str, model_type: str) -> None:
    await engine.init_db()
    async with engine.async_session_factory() as session:
        await session.execute(
            delete(TrainingJob).where(TrainingJob.dataset_source_id == dataset_id)
        )
        await session.execute(
            delete(ModelVersionCounter).where(
                ModelVersionCounter.dataset_source_id == dataset_id,
                ModelVersionCounter.model_type == model_type,
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_concurrent_get_next_version_never_duplicates() -> None:
    """N concurrent callers (each with its own session, simulating N concurrent
    HTTP requests) must all receive distinct, contiguous version numbers -
    never the same version twice."""
    dataset_id = "ds_race_concurrent"
    model_type = "rf_race_concurrent"
    await _cleanup(dataset_id, model_type)

    async def _allocate() -> int:
        async with engine.async_session_factory() as session:
            return await ModelRegistryService.get_next_version(
                session, dataset_id, model_type, "training"
            )

    n = 20
    results = await asyncio.gather(*(_allocate() for _ in range(n)))

    assert len(results) == len(set(results)), (
        f"Duplicate versions allocated under concurrency: {sorted(results)}"
    )
    assert sorted(results) == list(range(1, n + 1))

    await _cleanup(dataset_id, model_type)


@pytest.mark.asyncio
async def test_get_next_version_seeds_from_existing_job_history() -> None:
    """When no counter row exists yet, the first allocation must seed from
    the max of pre-existing TrainingJob.version rows (either run_mode)
    (backward compatibility for jobs created before the counter table existed)."""
    dataset_id = "ds_race_seed"
    model_type = "rf_race_seed"
    await _cleanup(dataset_id, model_type)

    async with engine.async_session_factory() as session:
        session.add(
            TrainingJob(
                id="job-seed-1",
                pipeline_id="p1",
                node_id="n1",
                dataset_source_id=dataset_id,
                status="completed",
                version=5,
                model_type=model_type,
                run_mode="fixed",
                graph={},
            )
        )
        await session.commit()

        next_version = await ModelRegistryService.get_next_version(
            session, dataset_id, model_type, "training"
        )
        assert next_version == 6

    await _cleanup(dataset_id, model_type)


@pytest.mark.asyncio
async def test_get_next_version_is_isolated_per_dataset_and_model_type() -> None:
    """Two different (dataset_id, model_type) pairs must not interfere with
    each other's counters."""
    await _cleanup("ds_race_a", "model_a")
    await _cleanup("ds_race_b", "model_b")

    async with engine.async_session_factory() as session:
        v1 = await ModelRegistryService.get_next_version(
            session, "ds_race_a", "model_a", "training"
        )
        v2 = await ModelRegistryService.get_next_version(
            session, "ds_race_b", "model_b", "training"
        )
        v3 = await ModelRegistryService.get_next_version(
            session, "ds_race_a", "model_a", "training"
        )

    assert (v1, v2, v3) == (1, 1, 2)

    await _cleanup("ds_race_a", "model_a")
    await _cleanup("ds_race_b", "model_b")
