from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import (
    DataSource,
    Deployment,
    HyperparameterTuningJob,
    TrainingJob,
)

from .schemas import ArtifactListResponse, ModelRegistryEntry, ModelVersion, RegistryStats


class ModelRegistryService:

    @staticmethod
    async def get_next_version(
        session: AsyncSession, dataset_id: str, model_type: str, job_type: str
    ) -> int:
        """Calculates the next version number for a given dataset and model type."""
        # Unified versioning: Query both tables and take the max

        # 1. Get max version from TrainingJob
        stmt_train = select(func.max(TrainingJob.version)).where(
            TrainingJob.dataset_source_id == dataset_id,
            TrainingJob.model_type == model_type,
        )
        result_train = await session.execute(stmt_train)
        max_train = result_train.scalar() or 0

        # 2. Get max run_number from HyperparameterTuningJob
        stmt_tune = select(func.max(HyperparameterTuningJob.run_number)).where(
            HyperparameterTuningJob.dataset_source_id == dataset_id,
            HyperparameterTuningJob.model_type == model_type,
        )
        result_tune = await session.execute(stmt_tune)
        max_tune = result_tune.scalar() or 0

        return max(max_train, max_tune) + 1

    @staticmethod
    async def get_registry_stats(session: AsyncSession) -> RegistryStats:
        # Count unique model types (approximate)
        # This is a bit complex with two tables, so we'll just count total jobs for now

        # Count total versions (completed jobs)
        train_count = await session.scalar(
            select(func.count(TrainingJob.id)).where(TrainingJob.status == "completed")
        )
        tune_count = await session.scalar(
            select(func.count(HyperparameterTuningJob.id)).where(
                HyperparameterTuningJob.status == "completed"
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
    async def list_models(
        session: AsyncSession, skip: int = 0, limit: int = 20
    ) -> List[ModelRegistryEntry]:
        """
        Lists all model types and their versions.
        Aggregates TrainingJob and HyperparameterTuningJob by (model_type, dataset_source_id).
        """
        # Fetch all completed jobs

        # Fetch Deployments to mark is_deployed
        deployments_result = await session.execute(select(Deployment))
        deployments = deployments_result.scalars().all()
        deployed_job_ids = {
            cast(str, d.job_id): cast(int, d.id) for d in deployments if d.is_active
        }

        # Fetch DataSources for names
        data_sources_result = await session.execute(select(DataSource))
        ds_map: Dict[Any, Dict[str, str]] = {}
        for ds in data_sources_result.scalars().all():
            info = {"name": str(ds.name), "type": str(ds.type)}
            # Map by integer ID (if used)
            ds_map[int(ds.id)] = info
            # Map by string source_id (UUID)
            if ds.source_id:
                ds_map[str(ds.source_id)] = info
                # Also map by string version of integer ID just in case
                ds_map[str(ds.id)] = info

        # Fetch Training Jobs
        train_jobs_result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.status == "completed")
            .order_by(TrainingJob.created_at.desc())
        )
        train_jobs = train_jobs_result.scalars().all()

        # Fetch Tuning Jobs
        tune_jobs_result = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.status == "completed")
            .order_by(HyperparameterTuningJob.created_at.desc())
        )
        tune_jobs = tune_jobs_result.scalars().all()

        # Group by (model_type, dataset_source_id)
        grouped: Dict[Tuple[str, str], List[ModelVersion]] = {}

        for job in train_jobs:
            m_type = cast(str, job.model_type or "unknown")
            ds_id = cast(str, job.dataset_source_id or "unknown")
            key = (m_type, ds_id)

            if key not in grouped:
                grouped[key] = []

            grouped[key].append(
                ModelVersion(
                    job_id=cast(str, job.id),
                    pipeline_id=cast(str, job.pipeline_id),
                    node_id=cast(str, job.node_id),
                    model_type=m_type,
                    version=cast(str, job.version),
                    source="training",
                    status=cast(str, job.status),
                    metrics=cast(Optional[Dict[str, Any]], job.metrics),
                    hyperparameters=cast(Optional[Dict[str, Any]], job.hyperparameters),
                    created_at=cast(Optional[datetime], job.created_at),
                    artifact_uri=cast(Optional[str], job.artifact_uri),
                    is_deployed=job.id in deployed_job_ids,
                    deployment_id=deployed_job_ids.get(cast(str, job.id)),
                )
            )

        for job in tune_jobs:
            m_type = cast(str, job.model_type or "unknown")
            ds_id = cast(str, job.dataset_source_id or "unknown")
            key = (m_type, ds_id)

            if key not in grouped:
                grouped[key] = []

            # For tuning jobs, we use run_number as version
            # And best_params as hyperparameters
            metrics: Dict[str, Any] = cast(Dict[str, Any], job.metrics) or {}
            if job.best_score:
                metrics["best_score"] = job.best_score

            grouped[key].append(
                ModelVersion(
                    job_id=cast(str, job.id),
                    pipeline_id=cast(str, job.pipeline_id),
                    node_id=cast(str, job.node_id),
                    model_type=m_type,
                    version=cast(int, job.run_number),
                    source="tuning",
                    status=cast(str, job.status),
                    metrics=metrics,
                    hyperparameters=cast(Optional[Dict[str, Any]], job.best_params),
                    created_at=cast(Optional[datetime], job.created_at),
                    artifact_uri=cast(Optional[str], job.artifact_uri),
                    is_deployed=job.id in deployed_job_ids,
                    deployment_id=deployed_job_ids.get(cast(str, job.id)),
                )
            )

        # Build result
        results = []
        for (m_type, ds_id), versions in grouped.items():
            # Sort versions by created_at desc
            versions.sort(key=lambda x: x.created_at or datetime.min, reverse=True)

            latest = versions[0] if versions else None
            deploy_count = sum(1 for v in versions if v.is_deployed)

            # Resolve dataset name and type
            # Try exact match, then string conversion
            ds_info = ds_map.get(ds_id)
            if not ds_info:
                # Fallback: check if ds_id is numeric string and try int key
                if isinstance(ds_id, str) and ds_id.isdigit():
                    ds_info = ds_map.get(int(ds_id))
            
            if ds_info:
                ds_name = ds_info["name"]
                ds_type = ds_info["type"]
            else:
                ds_name = f"Dataset {ds_id}"
                ds_type = "unknown"

            results.append(
                ModelRegistryEntry(
                    model_type=m_type,
                    dataset_id=ds_id,
                    dataset_name=ds_name,
                    dataset_type=ds_type,
                    latest_version=latest,
                    versions=versions,
                    deployment_count=deploy_count,
                )
            )

        # Sort results by latest_version.created_at desc
        results.sort(
            key=lambda x: (
                x.latest_version.created_at
                if x.latest_version and x.latest_version.created_at
                else datetime.min
            ),
            reverse=True,
        )

        return results[skip : skip + limit]

    @staticmethod
    async def get_model_versions(
        session: AsyncSession, model_type: str
    ) -> List[ModelVersion]:
        # Similar to list_models but filtered by model_type
        # ... (implementation reuse or copy)
        # For brevity, just filtered the list_models result for now,
        # but in production we should query specifically.

        # Fetch Deployments
        result = await session.execute(select(Deployment))
        deployments = result.scalars().all()
        deployed_job_ids = {
            cast(str, d.job_id): cast(int, d.id) for d in deployments if d.is_active
        }

        versions = []

        # Training Jobs
        train_jobs = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.status == "completed")
            .where(TrainingJob.model_type == model_type)
            .order_by(TrainingJob.created_at.desc())
        )
        for job in train_jobs.scalars().all():
            versions.append(
                ModelVersion(
                    job_id=cast(str, job.id),
                    pipeline_id=cast(str, job.pipeline_id),
                    node_id=cast(str, job.node_id),
                    model_type=model_type,
                    version=cast(str, job.version),
                    source="training",
                    status=cast(str, job.status),
                    metrics=cast(Optional[Dict[str, Any]], job.metrics),
                    hyperparameters=cast(Optional[Dict[str, Any]], job.hyperparameters),
                    created_at=cast(Optional[datetime], job.created_at),
                    artifact_uri=cast(Optional[str], job.artifact_uri),
                    is_deployed=job.id in deployed_job_ids,
                    deployment_id=deployed_job_ids.get(cast(str, job.id)),
                )
            )

        # Tuning Jobs
        tune_jobs = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.status == "completed")
            .where(HyperparameterTuningJob.model_type == model_type)
            .order_by(HyperparameterTuningJob.created_at.desc())
        )
        for job in tune_jobs.scalars().all():
            metrics: Dict[str, Any] = cast(Dict[str, Any], job.metrics) or {}
            if job.best_score:
                metrics["best_score"] = job.best_score

            versions.append(
                ModelVersion(
                    job_id=cast(str, job.id),
                    pipeline_id=cast(str, job.pipeline_id),
                    node_id=cast(str, job.node_id),
                    model_type=model_type,
                    version=cast(int, job.run_number),
                    source="tuning",
                    status=cast(str, job.status),
                    metrics=metrics,
                    hyperparameters=cast(Optional[Dict[str, Any]], job.best_params),
                    created_at=cast(Optional[datetime], job.created_at),
                    artifact_uri=cast(Optional[str], job.artifact_uri),
                    is_deployed=job.id in deployed_job_ids,
                    deployment_id=deployed_job_ids.get(cast(str, job.id)),
                )
            )

        versions.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        return versions

    @staticmethod
    async def get_job_artifacts(session: AsyncSession, job_id: str) -> ArtifactListResponse:
        """
        List artifacts for a specific job (Training or Tuning).
        """
        from typing import Union
        from backend.ml_pipeline.artifacts.local import LocalArtifactStore
        from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore

        # 1. Try finding a TrainingJob first
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            # 2. If not found, try HyperparameterTuningJob
            stmt = select(HyperparameterTuningJob).where(HyperparameterTuningJob.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        artifact_uri = str(job.artifact_uri)
        if not artifact_uri:
            return ArtifactListResponse(storage_type="unknown", base_uri="", files=[])
        
        store: Union[S3ArtifactStore, LocalArtifactStore]

        # 3. Instantiate the appropriate ArtifactStore
        if artifact_uri.startswith("s3://"):
            # Parse bucket and prefix: s3://bucket/prefix
            parts = artifact_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            # Inject AWS Credentials from Settings
            from backend.config import get_settings
            settings = get_settings()
            
            storage_options = {
                "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                "region_name": settings.AWS_DEFAULT_REGION,
            }
            
            # Filter None values
            storage_options = {k: v for k, v in storage_options.items() if v is not None}
            
            try:
                store = S3ArtifactStore(bucket_name=bucket, prefix=prefix, storage_options=storage_options)
                return ArtifactListResponse(
                    storage_type="s3",
                    base_uri=artifact_uri,
                    files=store.list_artifacts()
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to list S3 artifacts: {e}")
                raise e
        else:
            # Use Local storage
            store = LocalArtifactStore(base_path=artifact_uri)
            return ArtifactListResponse(
                storage_type="local",
                base_uri=artifact_uri,
                files=store.list_artifacts()
            )

