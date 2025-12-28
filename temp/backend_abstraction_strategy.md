# Backend Abstraction Strategy

**Author**: Architecture Review  
**Date**: 2025-01-15  
**Version**: 1.0  
**Status**: Draft for Review

## Executive Summary

This document outlines the backend abstraction strategy for Skyulf, an MLOps platform. The strategy focuses on maintaining clean separation of concerns, enabling extensibility, and supporting production scalability through well-defined abstraction layers.

## Table of Contents

1. [Core Principles](#core-principles)
2. [Abstraction Layers](#abstraction-layers)
3. [Design Patterns](#design-patterns)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Current Architecture](#current-architecture)
6. [Areas for Improvement](#areas-for-improvement)
7. [Migration Strategy](#migration-strategy)

## Core Principles

### 1. Separation of Concerns

The backend is organized into distinct layers:
- **Presentation Layer**: FastAPI routes and endpoints
- **Business Logic Layer**: Service classes and domain logic
- **Data Access Layer**: Database operations and repositories
- **Infrastructure Layer**: External services (S3, Redis, Celery)

### 2. Dependency Inversion

High-level modules should not depend on low-level modules. Both should depend on abstractions.

**Example**:
```python
# Good: Depend on abstraction
class PipelineEngine:
    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

# Bad: Depend on concrete implementation
class PipelineEngine:
    def __init__(self):
        self.artifact_store = LocalArtifactStore()
```

### 3. Open/Closed Principle

Software entities should be open for extension but closed for modification.

**Implementation**: Use interfaces/protocols and factory patterns.

### 4. Single Responsibility

Each module, class, or function should have one reason to change.

## Abstraction Layers

### Layer 1: API/Presentation Layer

**Location**: `backend/*/api.py`, `backend/*/router.py`

**Responsibility**:
- Handle HTTP requests/responses
- Request validation (Pydantic models)
- Response serialization
- Authentication/authorization
- Error handling at HTTP level

**Abstraction**:
```python
from fastapi import APIRouter, Depends
from backend.ml_pipeline.services.job_service import JobService

router = APIRouter()

@router.post("/train")
async def train_model(
    request: TrainRequest,
    job_service: JobService = Depends(get_job_service)
):
    """API endpoint delegates to service layer."""
    job = await job_service.create_training_job(request)
    return job
```

**Key Points**:
- No business logic in routes
- Use dependency injection
- Return domain objects, not database models
- Handle only HTTP concerns

### Layer 2: Service/Business Logic Layer

**Location**: `backend/*/services/`

**Responsibility**:
- Implement business rules
- Orchestrate workflows
- Coordinate between repositories
- Transaction management
- Domain-specific validation

**Abstraction**:
```python
class JobService:
    def __init__(
        self,
        job_repository: JobRepository,
        artifact_store: ArtifactStore,
        task_queue: TaskQueue
    ):
        self.job_repository = job_repository
        self.artifact_store = artifact_store
        self.task_queue = task_queue
    
    async def create_training_job(self, request: TrainRequest) -> Job:
        # Business logic here
        job = Job(...)
        await self.job_repository.save(job)
        await self.task_queue.enqueue_training(job.id)
        return job
```

**Key Points**:
- No HTTP concerns
- No direct database access (use repositories)
- Testable in isolation
- Clear interfaces

### Layer 3: Data Access Layer

**Location**: `backend/database/`, `backend/*/repositories/`

**Responsibility**:
- Database operations (CRUD)
- Query construction
- Data mapping (ORM ↔ domain models)
- Transaction handling

**Abstraction**:
```python
class JobRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, job: Job) -> Job:
        db_job = JobModel.from_domain(job)
        self.session.add(db_job)
        await self.session.commit()
        return job
    
    async def get_by_id(self, job_id: str) -> Optional[Job]:
        result = await self.session.execute(
            select(JobModel).where(JobModel.id == job_id)
        )
        db_job = result.scalar_one_or_none()
        return db_job.to_domain() if db_job else None
```

**Key Points**:
- Encapsulate database logic
- Return domain objects, not ORM models
- Handle database errors
- Support transactions

### Layer 4: Infrastructure Layer

**Location**: `backend/ml_pipeline/artifacts/`, `backend/celery_app.py`

**Responsibility**:
- External service integration (S3, Redis)
- File system operations
- Message queue integration
- Caching

**Abstraction**:
```python
class ArtifactStore(ABC):
    """Abstract interface for artifact storage."""
    
    @abstractmethod
    async def save(self, key: str, data: bytes) -> str:
        """Save artifact and return URL."""
        pass
    
    @abstractmethod
    async def load(self, key: str) -> bytes:
        """Load artifact by key."""
        pass

class LocalArtifactStore(ArtifactStore):
    """Local filesystem implementation."""
    async def save(self, key: str, data: bytes) -> str:
        # Implementation
        pass

class S3ArtifactStore(ArtifactStore):
    """S3 cloud storage implementation."""
    async def save(self, key: str, data: bytes) -> str:
        # Implementation
        pass
```

**Key Points**:
- Define clear interfaces
- Multiple implementations
- Easy to mock for testing
- Configuration-driven selection

## Design Patterns

### 1. Factory Pattern

**Usage**: Create objects without specifying exact class.

**Implementation** (from codebase):
```python
class ArtifactStoreFactory:
    @staticmethod
    def create(storage_type: str, settings: Settings) -> ArtifactStore:
        if storage_type == "local":
            return LocalArtifactStore(settings.MODELS_DIR)
        elif storage_type == "s3":
            return S3ArtifactStore(
                bucket=settings.S3_ARTIFACT_BUCKET,
                region=settings.AWS_DEFAULT_REGION
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
```

**Benefits**:
- Easy to add new storage types
- Configuration-driven
- Testable with mock implementations

### 2. Repository Pattern

**Usage**: Encapsulate data access logic.

**Structure**:
```python
# Domain model (business entity)
class Job:
    id: str
    status: str
    config: dict
    created_at: datetime

# ORM model (database representation)
class JobModel(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    status = Column(String)
    config = Column(JSON)
    created_at = Column(DateTime)
    
    def to_domain(self) -> Job:
        return Job(
            id=self.id,
            status=self.status,
            config=self.config,
            created_at=self.created_at
        )

# Repository (data access)
class JobRepository:
    async def find_by_status(self, status: str) -> List[Job]:
        # Database query logic
        pass
```

**Benefits**:
- Decouples business logic from database
- Easy to switch database implementations
- Testable with in-memory repositories

### 3. Dependency Injection

**Usage**: Inject dependencies rather than creating them.

**Implementation**:
```python
from fastapi import Depends

# Dependency provider
def get_job_service(
    job_repo: JobRepository = Depends(get_job_repository),
    artifact_store: ArtifactStore = Depends(get_artifact_store)
) -> JobService:
    return JobService(job_repo, artifact_store)

# Usage in route
@router.post("/jobs")
async def create_job(
    request: JobRequest,
    service: JobService = Depends(get_job_service)
):
    return await service.create_job(request)
```

**Benefits**:
- Loose coupling
- Easy to test (inject mocks)
- Easy to change implementations

### 4. Strategy Pattern

**Usage**: Select algorithm at runtime.

**Example**: Different storage strategies (local, S3).

### 5. Template Method Pattern

**Usage**: Define skeleton of algorithm, let subclasses implement steps.

**Example**: Base class for calculators in skyulf-core.

## Implementation Guidelines

### Creating New Features

#### Step 1: Define Domain Models

```python
# Domain model (no framework dependencies)
@dataclass
class Pipeline:
    id: str
    name: str
    config: dict
    created_at: datetime
```

#### Step 2: Define Repository Interface

```python
class PipelineRepository(ABC):
    @abstractmethod
    async def save(self, pipeline: Pipeline) -> Pipeline:
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Pipeline]:
        pass
```

#### Step 3: Implement Repository

```python
class SQLPipelineRepository(PipelineRepository):
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, pipeline: Pipeline) -> Pipeline:
        # ORM implementation
        pass
```

#### Step 4: Create Service

```python
class PipelineService:
    def __init__(self, repo: PipelineRepository):
        self.repo = repo
    
    async def create_pipeline(self, data: dict) -> Pipeline:
        pipeline = Pipeline(...)
        return await self.repo.save(pipeline)
```

#### Step 5: Add API Endpoint

```python
@router.post("/pipelines")
async def create_pipeline(
    request: PipelineRequest,
    service: PipelineService = Depends(get_pipeline_service)
):
    pipeline = await service.create_pipeline(request.dict())
    return PipelineResponse.from_domain(pipeline)
```

### Testing Strategy

#### Unit Tests (Services)

```python
async def test_create_pipeline():
    # Arrange
    mock_repo = MockPipelineRepository()
    service = PipelineService(mock_repo)
    
    # Act
    pipeline = await service.create_pipeline({...})
    
    # Assert
    assert pipeline.id is not None
    assert mock_repo.saved_pipelines[0] == pipeline
```

#### Integration Tests (Repositories)

```python
async def test_repository_save(db_session):
    # Arrange
    repo = SQLPipelineRepository(db_session)
    pipeline = Pipeline(...)
    
    # Act
    saved = await repo.save(pipeline)
    
    # Assert
    loaded = await repo.get_by_id(saved.id)
    assert loaded == saved
```

#### API Tests (Routes)

```python
async def test_create_pipeline_endpoint(client, mock_service):
    # Arrange
    app.dependency_overrides[get_pipeline_service] = lambda: mock_service
    
    # Act
    response = await client.post("/pipelines", json={...})
    
    # Assert
    assert response.status_code == 201
```

## Current Architecture

### Strengths

1. **Clear Separation**: Frontend, backend, skyulf-core are well separated
2. **Calculator → Applier Pattern**: Excellent for ML pipelines
3. **Factory Pattern**: Used for artifact stores
4. **Async Support**: Properly implemented throughout
5. **Configuration Management**: Pydantic Settings is well done
6. **Type Hints**: Good coverage

### Current Structure

```
backend/
├── api.py                 # API routes
├── config.py              # Configuration
├── main.py                # Application factory
├── data_ingestion/        # Data handling
├── ml_pipeline/
│   ├── api.py            # Pipeline endpoints
│   ├── services/         # Business logic ✓
│   ├── execution/        # Engine and managers
│   ├── artifacts/        # Storage abstraction ✓
│   └── model_registry/   # Model versioning
└── database/
    └── engine.py         # DB connection
```

## Areas for Improvement

### 1. Repository Pattern Not Fully Implemented

**Issue**: Direct database access mixed with business logic.

**Current**:
```python
# In some services/routes
async with AsyncSession(engine) as session:
    result = await session.execute(select(JobModel))
    jobs = result.scalars().all()
```

**Recommended**:
```python
# Use repository
class JobRepository:
    async def find_all(self) -> List[Job]:
        result = await self.session.execute(select(JobModel))
        db_jobs = result.scalars().all()
        return [j.to_domain() for j in db_jobs]

# In service
jobs = await self.job_repository.find_all()
```

### 2. Domain Models vs ORM Models

**Issue**: Mixing ORM models with business logic.

**Recommended**: Separate concerns:
- **ORM Models**: Database representation (`backend/database/models/`)
- **Domain Models**: Business entities (`backend/domain/`)
- **DTO/Schemas**: API request/response (`backend/schemas/`)

### 3. Service Layer Not Consistent

**Issue**: Some business logic in routes, some in services.

**Recommended**:
```
backend/ml_pipeline/
├── api.py                 # Routes only (HTTP concerns)
├── services/              # All business logic
│   ├── pipeline_service.py
│   ├── job_service.py
│   └── evaluation_service.py
├── repositories/          # Data access
│   ├── pipeline_repository.py
│   └── job_repository.py
└── domain/                # Domain models
    ├── pipeline.py
    └── job.py
```

### 4. Dependency Injection Not Fully Leveraged

**Issue**: Some direct instantiation of dependencies.

**Recommended**: Use `backend/dependencies.py` consistently:

```python
# backend/dependencies.py
def get_artifact_store(
    settings: Settings = Depends(get_settings)
) -> ArtifactStore:
    return ArtifactStoreFactory.create(
        storage_type=determine_storage_type(settings),
        settings=settings
    )

def get_job_repository(
    session: AsyncSession = Depends(get_db)
) -> JobRepository:
    return SQLJobRepository(session)

def get_job_service(
    repo: JobRepository = Depends(get_job_repository),
    store: ArtifactStore = Depends(get_artifact_store)
) -> JobService:
    return JobService(repo, store)
```

### 5. Testing Infrastructure

**Issue**: Limited repository mocking infrastructure.

**Recommended**: Create mock implementations:

```python
# tests/mocks/repositories.py
class InMemoryJobRepository(JobRepository):
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
    
    async def save(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job
    
    async def get_by_id(self, id: str) -> Optional[Job]:
        return self.jobs.get(id)
```

### 6. Error Handling Abstraction

**Issue**: HTTP errors mixed with business errors.

**Recommended**:
```python
# Domain exceptions
class PipelineNotFoundError(Exception):
    pass

# In service
if not pipeline:
    raise PipelineNotFoundError(f"Pipeline {id} not found")

# In API layer
try:
    pipeline = await service.get_pipeline(id)
except PipelineNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
```

## Migration Strategy

### Phase 1: Stabilize Current Architecture

1. Document existing patterns
2. Create coding standards
3. Add repository layer gradually
4. Improve testing infrastructure

### Phase 2: Introduce Repositories

1. Create repository interfaces
2. Implement SQL repositories
3. Migrate services to use repositories
4. Create in-memory repositories for tests

### Phase 3: Separate Domain Models

1. Define domain models
2. Create mapping functions (ORM ↔ domain)
3. Update services to use domain models
4. Keep ORM models internal to repositories

### Phase 4: Enhance Dependency Injection

1. Centralize dependency providers
2. Remove direct instantiation
3. Add factory functions
4. Improve testability

### Implementation Priority

**High Priority** (Do Now):
- [ ] Create repository layer for jobs/pipelines
- [ ] Separate business logic from API routes
- [ ] Improve dependency injection usage
- [ ] Add integration tests

**Medium Priority** (Next Quarter):
- [ ] Separate domain models from ORM models
- [ ] Refactor error handling
- [ ] Create mock implementations for all interfaces
- [ ] Document abstraction patterns

**Low Priority** (Future):
- [ ] Consider event-driven architecture
- [ ] Add caching abstraction layer
- [ ] Implement CQRS pattern (if needed)
- [ ] Add monitoring/observability abstractions

## Best Practices

### 1. Always Code to Interfaces

```python
# Good
def __init__(self, store: ArtifactStore):
    ...

# Bad
def __init__(self, store: S3ArtifactStore):
    ...
```

### 2. Keep Layers Independent

- API layer doesn't know about database
- Service layer doesn't know about HTTP
- Repository doesn't know about business rules

### 3. Use Dependency Injection

- Don't use global state
- Inject all dependencies
- Use FastAPI's Depends()

### 4. Test Each Layer Separately

- Unit test services with mocked repositories
- Integration test repositories with real database
- API test routes with mocked services

### 5. Follow Single Responsibility

Each module should have one reason to change.

## Conclusion

The Skyulf backend has a solid foundation with good separation between frontend/backend/core and uses some excellent patterns (Calculator→Applier, Factory). To improve:

1. **Add Repository Layer**: Encapsulate data access
2. **Separate Domain Models**: Keep business logic independent
3. **Consistent Service Layer**: Move all business logic to services
4. **Improve Dependency Injection**: Use FastAPI's DI consistently
5. **Enhance Testing**: Add mock implementations and integration tests

These improvements will make the codebase more maintainable, testable, and extensible while preserving the existing architecture strengths.

## References

- Project Architecture: `.github/instructions/project_architecture.instructions.md`
- Coding Standards: `.github/instructions/coding_standards.instructions.md`
- Current Implementation: `backend/` directory
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Repository Pattern: Martin Fowler's Patterns of Enterprise Application Architecture

---

**Next Steps**: Review this document with the team, prioritize improvements, and create implementation issues.
