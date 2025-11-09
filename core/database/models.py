"""
Database Models for FastAPI

SQLAlchemy models that mirror the existing Flask database structure.
These models are compatible with the existing database schema.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from .engine import Base


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps to models."""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class User(Base, TimestampMixin):
    """
    User model - mirrors the Flask user table structure.
    Compatible with existing Flask-Login users.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    # User profile information
    full_name = Column(String(200), nullable=True)

    # User status
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Authentication tracking
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0, nullable=False)

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

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(50), unique=True, nullable=True, index=True)  # UUID string identifier for file naming
    name = Column(String(100), nullable=False, index=True)
    type = Column(String(50), nullable=False)  # 'snowflake', 'postgres', 'api', etc.

    # Connection configuration (stored as JSON)
    config = Column(JSON, nullable=False)

    # Credentials (encrypted in production)
    credentials = Column(JSON, nullable=True)

    # Status and metadata
    is_active = Column(Boolean, default=True, nullable=False)
    last_tested = Column(DateTime, nullable=True)
    test_status = Column(String(20), default="untested", nullable=False)  # 'success', 'failed', 'untested'

    # User who created this source
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)  # Foreign key to users.id

    # Relationship to User model
    creator = relationship("User", back_populates="data_sources")

    # Description and documentation
    description = Column(Text, nullable=True)

    # Additional source metadata (stored as JSON)
    source_metadata = Column(JSON, nullable=True)

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

    id = Column(Integer, primary_key=True, index=True)
    dataset_source_id = Column(String(100), nullable=False, index=True)
    name = Column(String(150), nullable=False, default="Draft pipeline")
    description = Column(Text, nullable=True)
    graph = Column(JSON, nullable=False)
    pipeline_metadata = Column("metadata", JSON, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

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


class TrainingJob(Base, TimestampMixin):
    """Background model training jobs triggered from the feature canvas."""

    __tablename__ = "training_jobs"

    id = Column(String(64), primary_key=True, index=True)
    pipeline_id = Column(String(150), nullable=False, index=True)
    node_id = Column(String(150), nullable=False, index=True)
    dataset_source_id = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    status = Column(String(20), nullable=False, default="queued", index=True)
    version = Column(Integer, nullable=False, default=1)
    model_type = Column(String(100), nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    job_metadata = Column("metadata", JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    graph = Column(JSON, nullable=False)
    artifact_uri = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    owner = relationship("User", backref="training_jobs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "node_id": self.node_id,
            "dataset_source_id": self.dataset_source_id,
            "user_id": self.user_id,
            "status": self.status,
            "version": self.version,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "metadata": self.job_metadata,
            "metrics": self.metrics,
            "artifact_uri": self.artifact_uri,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class HyperparameterTuningJob(Base, TimestampMixin):
    """Asynchronous hyperparameter tuning jobs triggered from the feature canvas."""

    __tablename__ = "hyperparameter_tuning_jobs"

    id = Column(String(64), primary_key=True, index=True)
    pipeline_id = Column(String(150), nullable=False, index=True)
    node_id = Column(String(150), nullable=False, index=True)
    dataset_source_id = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    status = Column(String(20), nullable=False, default="queued", index=True)
    run_number = Column(Integer, nullable=False, default=1)
    model_type = Column(String(100), nullable=False)
    search_strategy = Column(String(20), nullable=False, default="random")
    search_space = Column(JSON, nullable=True)
    baseline_hyperparameters = Column(JSON, nullable=True)
    n_iterations = Column(Integer, nullable=True)
    scoring = Column(String(100), nullable=True)
    random_state = Column(Integer, nullable=True)
    cross_validation = Column(JSON, nullable=True)
    job_metadata = Column("metadata", JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    best_params = Column(JSON, nullable=True)
    best_score = Column(Float, nullable=True)
    graph = Column(JSON, nullable=False)
    artifact_uri = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    owner = relationship("User", backref="hyperparameter_tuning_jobs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "node_id": self.node_id,
            "dataset_source_id": self.dataset_source_id,
            "user_id": self.user_id,
            "status": self.status,
            "run_number": self.run_number,
            "model_type": self.model_type,
            "search_strategy": self.search_strategy,
            "search_space": self.search_space,
            "baseline_hyperparameters": self.baseline_hyperparameters,
            "n_iterations": self.n_iterations,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "cross_validation": self.cross_validation,
            "metadata": self.job_metadata,
            "metrics": self.metrics,
            "results": self.results,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "artifact_uri": self.artifact_uri,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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
