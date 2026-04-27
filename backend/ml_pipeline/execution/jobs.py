"""
Job Management for V2 Pipeline.
Handles persistence of Training and Tuning jobs to the database.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import BasicTrainingJob, AdvancedTuningJob
from backend.ml_pipeline.execution.schemas import JobInfo, JobStatus
from backend.ml_pipeline.execution.basic_training_manager import BasicTrainingManager
from backend.ml_pipeline.execution.advanced_tuning_manager import AdvancedTuningManager


class JobManager:
    """
    Facade for managing training and tuning jobs.
    Delegates to BasicTrainingManager and AdvancedTuningManager.
    """

    @staticmethod
    async def create_job(
        session: AsyncSession,
        pipeline_id: str,
        node_id: str,
        job_type: Literal["basic_training", "advanced_tuning", "preview"],
        dataset_id: str = "unknown",
        user_id: Optional[int] = None,
        model_type: str = "unknown",
        graph: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Creates a new job in the database (Async)."""
        if job_type == "basic_training":
            return await BasicTrainingManager.create_training_job(
                session,
                pipeline_id,
                node_id,
                dataset_id,
                user_id,
                model_type,
                graph,
            )
        elif job_type == "advanced_tuning":
            return await AdvancedTuningManager.create_tuning_job(
                session,
                pipeline_id,
                node_id,
                dataset_id,
                user_id,
                model_type,
                graph,
            )
        elif job_type == "preview":
            return await BasicTrainingManager.create_training_job(
                session,
                pipeline_id,
                node_id,
                dataset_id,
                user_id,
                model_type,
                graph,
                is_preview=True,
            )
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

    @staticmethod
    async def cancel_job(session: AsyncSession, job_id: str) -> bool:
        """Cancels a job if it is running or queued."""
        # Try BasicTrainingJob first
        if await BasicTrainingManager.cancel_training_job(session, job_id):
            return True
        # Then AdvancedTuningJob
        return await AdvancedTuningManager.cancel_tuning_job(session, job_id)

    @staticmethod
    async def attach_celery_task_id(session: AsyncSession, job_id: str, task_id: str) -> None:
        """Stash the Celery task id on the job's metadata so cancel_job can revoke it.

        Stored under `job_metadata.celery_task_id` to avoid a schema migration.
        Tries BasicTrainingJob first, then AdvancedTuningJob; silently no-ops
        if the job row doesn't exist (job creation race shouldn't break submit).
        """
        for model in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model).where(model.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job is None:
                continue
            meta = dict(job.job_metadata) if isinstance(job.job_metadata, dict) else {}
            meta["celery_task_id"] = task_id
            job.job_metadata = meta
            await session.commit()
            return

    @staticmethod
    def update_status_sync(
        session: Session,
        job_id: str,
        status: Optional[JobStatus] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        logs: Optional[List[str]] = None,
    ):
        """Updates job status (Sync - for Background Tasks)."""
        # Try BasicTrainingJob first
        if BasicTrainingManager.update_status_sync(session, job_id, status, error, result, logs):
            return

        # Then AdvancedTuningJob
        AdvancedTuningManager.update_status_sync(session, job_id, status, error, result, logs)

    @staticmethod
    async def get_job(session: AsyncSession, job_id: str) -> Optional[JobInfo]:
        """Retrieves job info (Async)."""
        # Try BasicTrainingJob
        job = await BasicTrainingManager.get_training_job(session, job_id)
        if job:
            return job

        # Then AdvancedTuningJob
        return await AdvancedTuningManager.get_tuning_job(session, job_id)

    @staticmethod
    async def list_jobs(
        session: AsyncSession,
        limit: int = 50,
        skip: int = 0,
        job_type: Optional[str] = None,
    ) -> List[JobInfo]:
        """Lists recent jobs (Async)."""
        jobs = []

        if job_type in ["basic_training", "training"]:
            jobs = await BasicTrainingManager.list_training_jobs(session, limit, skip)
        elif job_type in ["advanced_tuning", "tuning"]:
            jobs = await AdvancedTuningManager.list_tuning_jobs(session, limit, skip)
        else:
            # Combine both
            train_jobs = await BasicTrainingManager.list_training_jobs(session, limit + skip, 0)
            tune_jobs = await AdvancedTuningManager.list_tuning_jobs(session, limit + skip, 0)

            all_jobs = train_jobs + tune_jobs
            # Sort by start_time desc
            all_jobs.sort(key=lambda x: x.start_time or datetime.min, reverse=True)

            # Apply skip and limit
            jobs = all_jobs[skip : skip + limit]

        return jobs

    @staticmethod
    async def get_latest_tuning_job_for_node(
        session: AsyncSession, node_id: str
    ) -> Optional[JobInfo]:
        return await AdvancedTuningManager.get_latest_tuning_job_for_node(session, node_id)

    @staticmethod
    async def get_node_summaries(
        session: AsyncSession, limit: int = 200
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Per-node card summaries from the latest *run group*, keyed by ``node_id``.

        Returns ``{node_id: [entry, ...]}`` where each entry is::

            {
                "summary": "acc 0.87 · f1 0.84",
                "branch_index": 0,                   # Optional[int]
                "pipeline_id": "...__branch_0",
                "parent_pipeline_id": "...",
                "finished_at": "2026-04-27T...",
            }

        For canvases with parallel terminals (one training node fed by N
        branches), a single ``Run All`` produces N completed jobs that
        share a ``parent_pipeline_id`` — we keep all of them so the card
        can render one summary line per branch (Path A / Path B / …).
        Older run groups are dropped: only the most recent
        ``parent_pipeline_id`` per node id is returned, so the card never
        mixes a fresh branch with a stale one.

        Used by the canvas to populate trainer/tuner card lines after a
        ``/pipeline/run`` finishes — the engine stamps the summary into
        ``metadata.summary`` per-node, but only ``job.metrics`` is
        persisted, so the per-node engine metadata never reaches the FE
        store. We expose the same one-liner the inline preview path
        already shows for non-trainer nodes.
        """
        train_jobs = await BasicTrainingManager.list_training_jobs(session, limit, 0)
        tune_jobs = await AdvancedTuningManager.list_tuning_jobs(session, limit, 0)
        # Newest first so the first hit per node_id wins for the
        # "latest run group" key.
        all_jobs = sorted(
            train_jobs + tune_jobs,
            key=lambda j: j.start_time or datetime.min,
            reverse=True,
        )
        # Phase 1: pick the parent_pipeline_id of the most recent
        # completed job for each node. Treat a missing
        # ``parent_pipeline_id`` as a single-branch run; collapse it on
        # the job's own ``pipeline_id`` so single-terminal runs still
        # group together correctly.
        latest_group: Dict[str, str] = {}
        for job in all_jobs:
            if job.status != "completed" or not job.node_id:
                continue
            if job.node_id in latest_group:
                continue
            group_key = job.parent_pipeline_id or job.pipeline_id
            if group_key:
                latest_group[job.node_id] = group_key
        # Phase 2: collect every completed job belonging to that group.
        out: Dict[str, List[Dict[str, Any]]] = {}
        for job in all_jobs:
            if job.status != "completed" or not job.node_id:
                continue
            target_group = latest_group.get(job.node_id)
            if target_group is None:
                continue
            group_key = job.parent_pipeline_id or job.pipeline_id
            if group_key != target_group:
                continue
            summary = (job.metrics or {}).get("summary") if job.metrics else None
            if not isinstance(summary, str) or not summary.strip():
                continue
            entry: Dict[str, Any] = {
                "summary": summary.strip(),
                "branch_index": job.branch_index,
                "pipeline_id": job.pipeline_id,
                "parent_pipeline_id": job.parent_pipeline_id,
                "finished_at": job.end_time.isoformat() if job.end_time else None,
            }
            out.setdefault(job.node_id, []).append(entry)
        # Sort each node's entries by branch_index (None last) so the
        # frontend can render them in deterministic Path-A/B/C order
        # without re-sorting.
        for entries in out.values():
            entries.sort(
                key=lambda e: (
                    e["branch_index"] is None,
                    e["branch_index"] if e["branch_index"] is not None else 0,
                )
            )
        return out

    @staticmethod
    async def get_best_tuning_job_for_model(
        session: AsyncSession, model_type: str
    ) -> Optional[JobInfo]:
        return await AdvancedTuningManager.get_best_tuning_job_for_model(session, model_type)

    @staticmethod
    async def get_tuning_jobs_for_model(
        session: AsyncSession, model_type: str, limit: int = 20
    ) -> List[JobInfo]:
        return await AdvancedTuningManager.get_tuning_jobs_for_model(session, model_type, limit)

    @staticmethod
    async def promote_job(session: AsyncSession, job_id: str) -> bool:
        """Marks a completed job as promoted (winner)."""
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                if job.status != "completed":
                    return False
                job.promoted_at = datetime.now()  # type: ignore[assignment]
                await session.commit()
                return True
        return False

    @staticmethod
    async def unpromote_job(session: AsyncSession, job_id: str) -> bool:
        """Removes promotion from a job."""
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                job.promoted_at = None  # type: ignore[assignment]
                await session.commit()
                return True
        return False
