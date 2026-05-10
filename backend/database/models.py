"""
Database Models for FastAPI

SQLAlchemy models that mirror the existing Flask database structure.
These models are compatible with the existing database schema.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, backref, mapped_column, relationship
from sqlalchemy.sql import func

from .engine import Base


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps to models."""

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )


class User(Base, TimestampMixin):
    """
    User model - mirrors the Flask user table structure.
    Compatible with existing Flask-Login users.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # User profile information
    full_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # User status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Authentication tracking
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationship to DataSource model
    data_sources = relationship("DataSource", back_populates="creator")

    def __repr__(self):
        return f"<User {self.username}>"

    def to_dict(self):
        """Convert model to dictionary (excluding password)."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "is_verified": self.is_verified,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "login_count": self.login_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DataSource(Base, TimestampMixin):
    """
    Data Source model for data ingestion connections.
    Compatible with existing data_sources table.
    """

    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # UUID string identifier for file naming
    source_id: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # 'snowflake', 'postgres', 'api', etc.
    type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Connection configuration (stored as JSON)
    config: Mapped[Any] = mapped_column(JSON, nullable=False)

    # Credentials (encrypted in production)
    credentials: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    # Status and metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_tested: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # 'success', 'failed', 'untested'
    test_status: Mapped[str] = mapped_column(String(20), default="untested", nullable=False)

    # User who created this source
    created_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )

    # Relationship to User model
    creator = relationship("User", back_populates="data_sources")

    # Description and documentation
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Additional source metadata (stored as JSON)
    source_metadata: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    def __repr__(self):
        return f"<DataSource {self.name} ({self.type})>"

    def to_dict(self):
        """Convert model to dictionary (excluding sensitive credentials)."""
        # Safely get creator username
        try:
            creator_name = self.creator.username if self.creator else "Unknown"
        except Exception:
            creator_name = "Unknown"

        return {
            "id": self.id,
            "source_id": self.source_id,  # UUID string identifier
            "name": self.name,
            "type": self.type,
            "config": self.config,
            "metadata": self.source_metadata,  # Include source metadata (renamed for API compatibility)
            "is_active": self.is_active,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "test_status": self.test_status,
            "created_by": creator_name,
            "created_by_id": self.created_by,  # Keep the ID as well for internal use
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def has_permission(self, permission: str) -> bool:
        """Check if user has permission. Simplified for this model."""
        # For now, return basic permission logic
        # In a real app, this would check against user roles/permissions
        return True  # Placeholder

    @property
    def is_admin(self) -> bool:
        """Check if creator has admin privileges."""
        # This would be determined by looking up the user
        return False  # Placeholder


class FeatureEngineeringPipeline(Base, TimestampMixin):
    """Stored feature engineering pipelines authored in the canvas."""

    __tablename__ = "feature_engineering_pipelines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_source_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, default="Draft pipeline")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    graph: Mapped[Any] = mapped_column(JSON, nullable=False)
    pipeline_metadata: Mapped[Optional[Any]] = mapped_column("metadata", JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_source_id": self.dataset_source_id,
            "name": self.name,
            "description": self.description,
            "graph": self.graph,
            "metadata": self.pipeline_metadata,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PipelineVersion(Base):
    """Append-only pipeline snapshots (L7 — server-side versioning).

    Replaces the per-browser localStorage Recent ring buffer with a
    durable, cross-device history. Keyed by `dataset_source_id` to
    match how `FeatureEngineeringPipeline` is upserted today (one
    active pipeline per dataset).

    `version_int` is monotonically increasing per dataset and assigned
    by the service on insert (max+1). Pinned rows are exempt from any
    future eviction policy.
    """

    __tablename__ = "pipeline_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_source_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    version_int: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(150), nullable=False, default="Pipeline")
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="manual", index=True)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    graph: Mapped[Any] = mapped_column(JSON, nullable=False)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dataset_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_source_id": self.dataset_source_id,
            "version_int": self.version_int,
            "name": self.name,
            "note": self.note,
            "kind": self.kind,
            "pinned": self.pinned,
            "graph": self.graph,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "dataset_name": self.dataset_name,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MLJob(Base, TimestampMixin):
    """Abstract base class containing common fields for ML jobs."""

    __abstract__ = True

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    pipeline_id: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    node_id: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    dataset_source_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    job_metadata: Mapped[Optional[Any]] = mapped_column("metadata", JSON, nullable=True)
    metrics: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    graph: Mapped[Any] = mapped_column(JSON, nullable=False)
    artifact_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    logs: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict_base(self) -> dict:
        """Convert common model fields to dictionary."""
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "node_id": self.node_id,
            "dataset_source_id": self.dataset_source_id,
            "user_id": self.user_id,
            "status": self.status,
            "model_type": self.model_type,
            "metadata": self.job_metadata,
            "metrics": self.metrics,
            "artifact_uri": self.artifact_uri,
            "error_message": self.error_message,
            "progress": self.progress,
            "current_step": self.current_step,
            "logs": self.logs,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BasicTrainingJob(MLJob):
    """Background model training jobs triggered from the feature canvas."""

    __tablename__ = "basic_training_jobs"

    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    hyperparameters: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    owner = relationship("User", backref="basic_training_jobs")

    def to_dict(self) -> dict:
        data = self.to_dict_base()
        data.update(
            {
                "version": self.version,
                "hyperparameters": self.hyperparameters,
            }
        )
        return data


class AdvancedTuningJob(MLJob):
    """Asynchronous hyperparameter tuning jobs triggered from the feature canvas."""

    __tablename__ = "advanced_tuning_jobs"

    run_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    search_strategy: Mapped[str] = mapped_column(String(20), nullable=False, default="random")
    search_space: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    baseline_hyperparameters: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    n_iterations: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    scoring: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    random_state: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cross_validation: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    results: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    best_params: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    best_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    owner = relationship("User", backref="advanced_tuning_jobs")

    def to_dict(self) -> dict:
        data = self.to_dict_base()
        data.update(
            {
                "run_number": self.run_number,
                "search_strategy": self.search_strategy,
                "search_space": self.search_space,
                "baseline_hyperparameters": self.baseline_hyperparameters,
                "n_iterations": self.n_iterations,
                "scoring": self.scoring,
                "random_state": self.random_state,
                "cross_validation": self.cross_validation,
                "results": self.results,
                "best_params": self.best_params,
                "best_score": self.best_score,
            }
        )
        return data


class Deployment(Base, TimestampMixin):
    """
    Tracks deployed models.
    """

    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # ID of the TrainingJob or HyperparameterTuningJob
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    deployed_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "model_type": self.model_type,
            "artifact_uri": self.artifact_uri,
            "is_active": self.is_active,
            "deployed_by": self.deployed_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class EDAReport(Base, TimestampMixin):
    """
    Stores the results of EDA analysis for a dataset.
    """

    __tablename__ = "eda_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    data_source_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # PENDING, COMPLETED, FAILED
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False)
    config: Mapped[Any] = mapped_column(JSON, nullable=False, default={})
    profile_data: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    test_status: Mapped[str] = mapped_column(String(20), default="untested", nullable=False)

    # Cascade-delete the report rows when the parent DataSource is removed.
    # Without this, deleting a DataSource tries to NULL `data_source_id`
    # (which is NOT NULL) and the DELETE fails with an IntegrityError.
    # `passive_deletes` is intentionally False (default): existing SQLite
    # databases were created without ON DELETE CASCADE on the FK, so we
    # need SQLAlchemy to issue the child DELETEs itself.
    data_source = relationship(
        "DataSource",
        backref=backref(
            "eda_reports",
            cascade="all, delete-orphan",
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "data_source_id": self.data_source_id,
            "status": self.status,
            "config": self.config,
            "profile_data": self.profile_data,
            "error_message": self.error_message,
            "is_active": self.is_active,
            "test_status": self.test_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DriftCheckResult(Base):
    """Stores the result of each drift analysis run for history tracking."""

    __tablename__ = "drift_check_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    reference_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    current_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    drifted_columns_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_columns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    summary: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    column_drifts: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)


class ErrorEvent(Base):
    """Lightweight in-house error tracker — stores unhandled 500s and pipeline crashes."""

    __tablename__ = "error_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # HTTP route that produced the error (empty string for Celery/background tasks)
    route: Mapped[str] = mapped_column(String(500), nullable=False, default="", index=True)
    # Short exception class name, e.g. "PipelineExecutionException"
    error_type: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    # Full Python traceback
    traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Optional pipeline job_id when the error originates from a pipeline run
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # HTTP status code (0 for background task errors)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False, default=500)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False, index=True
    )
    # Set when an operator marks the event as resolved/dismissed
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "route": self.route,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
            "job_id": self.job_id,
            "status_code": self.status_code,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class PipelineRunLog(Base):
    """Per-run record of node failures and warnings captured from a pipeline preview."""

    __tablename__ = "pipeline_run_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pipeline_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    node_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    node_type: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # 'error' for node failures, 'warning' / 'info' for soft advisories
    level: Mapped[str] = mapped_column(String(20), nullable=False, default="error", index=True)
    logger: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    run_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False, index=True
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "level": self.level,
            "logger": self.logger,
            "message": self.message,
            "run_at": self.run_at.isoformat() if self.run_at else None,
        }


# Unused tables removed: DataIngestionJob, SystemLog
# These tables were not used anywhere in the application and caused schema differences


@asynccontextmanager
async def get_database_session(
    engine: Optional[AsyncEngine] = None,
    *,
    expire_on_commit: bool = False,
) -> AsyncIterator[AsyncSession]:
    """Provide an async SQLAlchemy session scoped to the given engine.

    Args:
        engine: Optional preconfigured AsyncEngine. Falls back to the global engine
            initialized via :func:`core.database.engine.init_db` when not provided.
        expire_on_commit: Whether attributes should be expired after commit. Matches
            SQLAlchemy's ``expire_on_commit`` flag for session factories.

    Yields:
        AsyncSession: A managed session with automatic commit/rollback semantics.
    """

    from .engine import get_engine  # Local import to avoid circular dependency

    resolved_engine = engine or get_engine()
    session_factory = async_sessionmaker(
        bind=resolved_engine,
        class_=AsyncSession,
        expire_on_commit=expire_on_commit,
    )

    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
