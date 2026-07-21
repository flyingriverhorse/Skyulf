from datetime import datetime
from typing import Any, cast

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database.models import (
    DataSource,
    Deployment,
    ModelVersionCounter,
    TrainingJob,
)
from backend.ml_pipeline._services.job_service import JobService

from .schemas import ArtifactListResponse, ModelRegistryEntry, ModelVersion, RegistryStats

_MIN_DATETIME = datetime.min
_MAX_VERSION_ALLOCATION_ATTEMPTS = 5


class ModelRegistryService:
    @staticmethod
    async def _compute_seed_version(session: AsyncSession, dataset_id: str, model_type: str) -> int:
        """Best-effort seed for a brand-new counter row from pre-existing job history.

        Only used the first time a (dataset, model_type) pair is versioned via
        the counter table, so historical jobs created before the counter
        existed keep receiving version numbers that continue their sequence.
        """
        stmt = select(func.max(TrainingJob.version)).where(
            TrainingJob.dataset_source_id == dataset_id,
            TrainingJob.model_type == model_type,
        )
        result = await session.execute(stmt)
        max_version = result.scalar() or 0

        return max_version + 1

    @staticmethod
    async def get_next_version(
        session: AsyncSession, dataset_id: str, model_type: str, job_type: str
    ) -> int:
        """Atomically allocates the next version number for a dataset/model_type pair.

        Both "fixed" and "tuned" run_modes of ``TrainingJob`` share the same
        ``version`` column and sequence, backed by a single
        ``ModelVersionCounter`` row per (dataset_id, model_type). The previous
        implementation computed ``max(...) + 1`` via a plain SELECT with no
        locking, so two concurrent job submissions for the same dataset/model
        could read the same max and both be handed the identical "next"
        version. The UPDATE ... RETURNING below is a single atomic statement,
        so concurrent callers are serialized by the database itself instead
        of racing in application code.
        """
        for _attempt in range(_MAX_VERSION_ALLOCATION_ATTEMPTS):
            stmt = (
                update(ModelVersionCounter)
                .where(
                    ModelVersionCounter.dataset_source_id == dataset_id,
                    ModelVersionCounter.model_type == model_type,
                )
                .values(current_version=ModelVersionCounter.current_version + 1)
                .returning(ModelVersionCounter.current_version)
            )
            result = await session.execute(stmt)
            row = result.first()
            if row is not None:
                await session.commit()
                return cast(int, row[0])

            # No counter row yet for this pair - seed one from existing job
            # history. If a concurrent request wins this insert first, our
            # insert raises IntegrityError (PK conflict) and we retry the
            # atomic UPDATE above, which will now succeed against the row
            # the winner just created.
            seed = await ModelRegistryService._compute_seed_version(session, dataset_id, model_type)
            try:
                await session.execute(
                    insert(ModelVersionCounter).values(
                        dataset_source_id=dataset_id,
                        model_type=model_type,
                        current_version=seed,
                    )
                )
                await session.commit()
                return seed
            except IntegrityError:
                await session.rollback()
                continue

        raise RuntimeError(
            f"Failed to allocate a model version for dataset={dataset_id!r} "
            f"model_type={model_type!r} after {_MAX_VERSION_ALLOCATION_ATTEMPTS} attempts"
        )

    @staticmethod
    async def get_registry_stats(session: AsyncSession) -> RegistryStats:
        # Count unique model types (approximate)
        # This is a bit complex with two tables, so we'll just count total jobs for now

        # Count total versions (completed jobs)
        train_count = await session.scalar(
            select(func.count(TrainingJob.id)).where(
                TrainingJob.run_mode == "fixed", TrainingJob.status == "completed"
            )
        )
        tune_count = await session.scalar(
            select(func.count(TrainingJob.id)).where(
                TrainingJob.run_mode == "tuned", TrainingJob.status == "completed"
            )
        )

        # Count active deployments
        deploy_count = await session.scalar(
            select(func.count(Deployment.id)).where(Deployment.is_active)
        )

        return RegistryStats(
            total_models=0,  # Calculated later or ignored for simple stats
            total_versions=(train_count or 0) + (tune_count or 0),
            active_deployments=deploy_count or 0,
        )

    @staticmethod
    async def _get_deployed_job_ids(session: AsyncSession) -> dict[Any, Any]:
        """Fetches active deployments and maps job_id -> deployment_id."""
        deployments_result = await session.execute(select(Deployment))
        deployments = deployments_result.scalars().all()
        return {d.job_id: d.id for d in deployments if d.is_active}

    @staticmethod
    async def _build_dataset_map(session: AsyncSession) -> dict[Any, dict[str, str]]:
        """Builds a lookup of dataset info keyed by integer id, string id, and source_id."""
        data_sources_result = await session.execute(select(DataSource))
        ds_map: dict[Any, dict[str, str]] = {}
        for ds in data_sources_result.scalars().all():
            info = {"name": str(ds.name), "type": str(ds.type)}
            # Map by integer ID (if used)
            ds_map[int(ds.id)] = info
            # Map by string source_id (UUID)
            if ds.source_id:
                ds_map[str(ds.source_id)] = info
                # Also map by string version of integer ID just in case
                ds_map[str(ds.id)] = info
        return ds_map

    @staticmethod
    async def _fetch_completed_train_jobs(session: AsyncSession) -> list[TrainingJob]:
        """Fetches completed fixed-mode training jobs ordered by newest first."""
        train_jobs_result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "fixed", TrainingJob.status == "completed")
            .order_by(TrainingJob.created_at.desc())
        )
        return list(train_jobs_result.scalars().all())

    @staticmethod
    async def _fetch_completed_tune_jobs(session: AsyncSession) -> list[TrainingJob]:
        """Fetches completed tuned-mode tuning jobs ordered by newest first."""
        tune_jobs_result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned", TrainingJob.status == "completed")
            .order_by(TrainingJob.created_at.desc())
        )
        return list(tune_jobs_result.scalars().all())

    @staticmethod
    def _train_job_to_version(
        job: TrainingJob, deployed_job_ids: dict[Any, Any]
    ) -> ModelVersion:
        """Converts a completed training job into a ModelVersion entry."""
        return ModelVersion(
            job_id=job.id,
            pipeline_id=job.pipeline_id,
            node_id=job.node_id,
            model_type=cast(str, job.model_type or "unknown"),
            version=cast(str, job.version),
            source="training",
            status=job.status,
            metrics=cast(dict[str, Any] | None, job.metrics),
            hyperparameters=cast(dict[str, Any] | None, job.hyperparameters),
            created_at=cast(datetime | None, job.created_at),
            artifact_uri=job.artifact_uri,
            is_deployed=job.id in deployed_job_ids,
            deployment_id=deployed_job_ids.get(job.id),
        )

    @staticmethod
    def _tune_job_to_version(
        job: TrainingJob, deployed_job_ids: dict[Any, Any]
    ) -> ModelVersion:
        """Converts a completed tuning job into a ModelVersion entry, using version as the version."""
        # For tuning jobs, best_params is used as hyperparameters
        metrics = dict(cast(dict[str, Any] | None, job.metrics) or {})
        if job.best_score is not None:
            metrics["best_score"] = job.best_score

        return ModelVersion(
            job_id=job.id,
            pipeline_id=job.pipeline_id,
            node_id=job.node_id,
            model_type=cast(str, job.model_type or "unknown"),
            version=job.version,
            source="tuning",
            status=job.status,
            metrics=metrics,
            hyperparameters=cast(dict[str, Any] | None, job.best_params),
            created_at=cast(datetime | None, job.created_at),
            artifact_uri=job.artifact_uri,
            is_deployed=job.id in deployed_job_ids,
            deployment_id=deployed_job_ids.get(job.id),
        )

    @staticmethod
    def _group_versions_by_model_and_dataset(
        train_jobs: list[TrainingJob],
        tune_jobs: list[TrainingJob],
        deployed_job_ids: dict[Any, Any],
    ) -> dict[tuple[str, str], list[ModelVersion]]:
        """Groups training and tuning job versions by (model_type, dataset_source_id)."""
        grouped: dict[tuple[str, str], list[ModelVersion]] = {}

        for job in train_jobs:
            m_type = cast(str, job.model_type or "unknown")
            ds_id = cast(str, job.dataset_source_id or "unknown")
            key = (m_type, ds_id)
            grouped.setdefault(key, []).append(
                ModelRegistryService._train_job_to_version(job, deployed_job_ids)
            )

        for job in tune_jobs:
            m_type = cast(str, job.model_type or "unknown")
            ds_id = cast(str, job.dataset_source_id or "unknown")
            key = (m_type, ds_id)
            grouped.setdefault(key, []).append(
                ModelRegistryService._tune_job_to_version(job, deployed_job_ids)
            )

        return grouped

    @staticmethod
    def _resolve_dataset_info(ds_map: dict[Any, dict[str, str]], ds_id: str) -> tuple[str, str]:
        """Resolves a dataset's display name and type, falling back to a placeholder."""
        # Try exact match, then string conversion
        ds_info = ds_map.get(ds_id)
        if not ds_info and isinstance(ds_id, str) and ds_id.isdigit():
            # Fallback: check if ds_id is numeric string and try int key
            ds_info = ds_map.get(int(ds_id))

        if ds_info:
            return ds_info["name"], ds_info["type"]
        return f"Dataset {ds_id}", "unknown"

    @staticmethod
    def _build_registry_entry(
        m_type: str,
        ds_id: str,
        versions: list[ModelVersion],
        ds_map: dict[Any, dict[str, str]],
    ) -> ModelRegistryEntry:
        """Builds a ModelRegistryEntry from a group of versions, sorted newest-first."""
        # Sort versions by created_at desc
        versions.sort(key=lambda x: x.created_at or _MIN_DATETIME, reverse=True)

        latest = versions[0] if versions else None
        deploy_count = sum(1 for v in versions if v.is_deployed)
        ds_name, ds_type = ModelRegistryService._resolve_dataset_info(ds_map, ds_id)

        return ModelRegistryEntry(
            model_type=m_type,
            dataset_id=ds_id,
            dataset_name=ds_name,
            dataset_type=ds_type,
            latest_version=latest,
            versions=versions,
            deployment_count=deploy_count,
        )

    @staticmethod
    async def list_models(
        session: AsyncSession, skip: int = 0, limit: int | None = None
    ) -> list[ModelRegistryEntry]:
        """
        Lists all model types and their versions.
        Aggregates TrainingJob by (model_type, dataset_source_id), scoped by run_mode.
        """
        deployed_job_ids = await ModelRegistryService._get_deployed_job_ids(session)
        ds_map = await ModelRegistryService._build_dataset_map(session)
        train_jobs = await ModelRegistryService._fetch_completed_train_jobs(session)
        tune_jobs = await ModelRegistryService._fetch_completed_tune_jobs(session)

        grouped = ModelRegistryService._group_versions_by_model_and_dataset(
            train_jobs, tune_jobs, deployed_job_ids
        )

        results = [
            ModelRegistryService._build_registry_entry(m_type, ds_id, versions, ds_map)
            for (m_type, ds_id), versions in grouped.items()
        ]

        # Sort results by latest_version.created_at desc
        results.sort(
            key=lambda x: (
                x.latest_version.created_at
                if x.latest_version and x.latest_version.created_at
                else _MIN_DATETIME
            ),
            reverse=True,
        )

        effective_limit = limit if limit is not None else get_settings().DEFAULT_PAGE_SIZE
        return results[skip : skip + effective_limit]

    @staticmethod
    async def get_model_versions(session: AsyncSession, model_type: str) -> list[ModelVersion]:
        # Similar to list_models but filtered by model_type
        # ... (implementation reuse or copy)
        # For brevity, just filtered the list_models result for now,
        # but in production we should query specifically.

        # Fetch Deployments
        result = await session.execute(select(Deployment))
        deployments = result.scalars().all()
        deployed_job_ids = {d.job_id: d.id for d in deployments if d.is_active}

        versions = []

        # Training Jobs (fixed mode)
        train_jobs = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "fixed")
            .where(TrainingJob.status == "completed")
            .where(TrainingJob.model_type == model_type)
            .order_by(TrainingJob.created_at.desc())
        )
        versions.extend(
            ModelVersion(
                job_id=job.id,
                pipeline_id=job.pipeline_id,
                node_id=job.node_id,
                model_type=model_type,
                version=cast(str, job.version),
                source="training",
                status=job.status,
                metrics=cast(dict[str, Any] | None, job.metrics),
                hyperparameters=cast(dict[str, Any] | None, job.hyperparameters),
                created_at=cast(datetime | None, job.created_at),
                artifact_uri=job.artifact_uri,
                is_deployed=job.id in deployed_job_ids,
                deployment_id=deployed_job_ids.get(job.id),
            )
            for job in train_jobs.scalars().all()
        )

        # Tuning Jobs (tuned mode)
        tune_jobs = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned")
            .where(TrainingJob.status == "completed")
            .where(TrainingJob.model_type == model_type)
            .order_by(TrainingJob.created_at.desc())
        )
        for job in tune_jobs.scalars().all():
            metrics = dict(cast(dict[str, Any] | None, job.metrics) or {})
            if job.best_score is not None:
                metrics["best_score"] = job.best_score

            versions.append(
                ModelVersion(
                    job_id=job.id,
                    pipeline_id=job.pipeline_id,
                    node_id=job.node_id,
                    model_type=model_type,
                    version=job.version,
                    source="tuning",
                    status=job.status,
                    metrics=metrics,
                    hyperparameters=cast(dict[str, Any] | None, job.best_params),
                    created_at=cast(datetime | None, job.created_at),
                    artifact_uri=job.artifact_uri,
                    is_deployed=job.id in deployed_job_ids,
                    deployment_id=deployed_job_ids.get(job.id),
                )
            )

        versions.sort(key=lambda x: x.created_at or _MIN_DATETIME, reverse=True)
        return versions

    @staticmethod
    async def get_job_artifacts(session: AsyncSession, job_id: str) -> ArtifactListResponse:
        """
        List artifacts for a specific job (Training or Tuning).
        """

        from backend.ml_pipeline.artifacts.local import LocalArtifactStore
        from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore

        # 1. Get Job Entity
        job = await JobService.get_job_by_id(session, job_id)

        if not job:
            raise ValueError(f"Job {job_id} not found")

        if not job.artifact_uri:
            return ArtifactListResponse(storage_type="unknown", base_uri="", files=[])
        artifact_uri = str(job.artifact_uri)

        store: S3ArtifactStore | LocalArtifactStore

        # 3. Instantiate the appropriate ArtifactStore
        from backend.ml_pipeline.artifacts.factory import ArtifactFactory

        try:
            store = cast(
                S3ArtifactStore | LocalArtifactStore,
                ArtifactFactory.get_artifact_store(artifact_uri),
            )

            storage_type = "s3" if artifact_uri.startswith("s3://") else "local"

            return ArtifactListResponse(
                storage_type=storage_type, base_uri=artifact_uri, files=store.list_artifacts()
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to list artifacts: {e}")
            raise
