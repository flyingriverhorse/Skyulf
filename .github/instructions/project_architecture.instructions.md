# Project Architecture

## Overview

Skyulf is an MLOps platform split into three main components with strict boundaries:

1. **skyulf-core**: Standalone Python ML library
2. **backend**: FastAPI + Celery orchestration layer
3. **frontend**: React + TypeScript UI

## Architectural Principles

### Separation of Concerns

The project follows a clear separation between:
- **Data layer**: Database, file storage, artifact management
- **Business logic**: ML pipeline execution, model training, feature engineering
- **API layer**: FastAPI routes and endpoints
- **Presentation**: Frontend UI components

### Design Patterns

#### 1. Calculator → Applier Pattern (skyulf-core)

Skyulf-core separates learning from transformation:

- **Calculator**: `fit(data, config) -> params`
  - Learns statistics/encoders/models
  - Returns serializable params dictionary
  
- **Applier**: `apply(data, params) -> transformed_data`
  - Stateless transformer
  - Applies learned parameters

**Benefits**:
- Easier to persist pipelines
- Safer production deployment
- Explicit state management
- Pure and repeatable transformations

#### 2. Registry Pattern

Used for dynamic node discovery and registration:

- **Registration**: Nodes self-register using decorators
  ```python
  @NodeRegistry.register("NodeName", ApplierClass)
  class CalculatorClass:
      ...
  ```
- **Discovery**: Pipeline looks up Calculator/Applier by name at runtime
- **Extensibility**: Add new nodes without modifying core pipeline code

#### 3. Data Catalog Pattern

Decouples data loading from execution engine:

- **Interface**: `DataCatalog` defines contract for loading data
- **Implementation**: `FileSystemCatalog` loads from local filesystem
- **Usage**: PipelineEngine receives catalog via dependency injection
- **Benefits**: Easy to swap storage backends (local, S3, database)

#### 4. Application Factory Pattern

FastAPI application creation follows factory pattern:

```python
def create_app() -> FastAPI:
    app = FastAPI(...)
    _setup_templates_and_static(app)
    _add_middleware(app, settings)
    _include_routers(app)
    _add_exception_handlers(app)
    return app
```

**Benefits**:
- Testable application creation
- Clean configuration
- Easy to create multiple app instances
- Clear initialization steps

#### 5. Dependency Injection

Used throughout FastAPI application:

- Settings injection via `get_settings()`
- Database session injection
- Service layer injection
- Configuration injection

## Directory Structure

### Backend Structure

```
backend/
├── __init__.py
├── celery_app.py          # Celery configuration
├── config.py              # Pydantic settings
├── main.py                # FastAPI application factory
├── dependencies.py        # Dependency injection utilities
├── data/                  # Data models and schemas
├── data_ingestion/        # Data upload and ingestion
├── database/              # Database models and engine
├── exceptions/            # Exception handlers
├── health/                # Health check endpoints
├── middleware/            # Custom middleware
├── ml_pipeline/           # ML pipeline orchestration
│   ├── api.py            # Pipeline API routes
│   ├── tasks.py          # Celery tasks
│   ├── artifacts/        # Artifact storage (local, S3)
│   ├── deployment/       # Model deployment
│   ├── execution/        # Pipeline execution engine
│   ├── model_registry/   # Model versioning
│   └── services/         # Business logic services
└── utils/                 # Utility functions
```

### Key Architectural Files

- **`backend/main.py`**: Application entry point and configuration
- **`backend/config.py`**: Centralized configuration management
- **`backend/ml_pipeline/execution/engine.py`**: Pipeline execution engine
- **`backend/database/engine.py`**: Database connection management
- **`docs/architecture.md`**: Detailed architecture documentation

## Module Organization

### Backend Modules

#### Data Ingestion (`backend/data_ingestion/`)
- Handles file uploads
- Manages data sources
- Provides data catalog

#### ML Pipeline (`backend/ml_pipeline/`)
- **execution/**: Pipeline orchestration and execution
- **artifacts/**: Model artifact storage (Factory pattern)
- **services/**: Business logic (job management, evaluation, prediction)
- **deployment/**: Model deployment and serving
- **model_registry/**: Model versioning and metadata

#### Database (`backend/database/`)
- SQLAlchemy models
- Async database engine
- Connection management
- Migration support via Alembic

#### Middleware (`backend/middleware/`)
- Logging middleware
- Error handling middleware
- Custom request/response processing

#### Exceptions (`backend/exceptions/`)
- Custom exception classes
- Exception handlers for HTTP responses

## Data Flow

### Pipeline Execution Flow

```
1. User creates pipeline config via API/UI
2. Config stored in database
3. Job created with status "pending"
4. Task dispatched to Celery (if enabled) or runs synchronously
5. PipelineEngine executes:
   a. Load data from catalog
   b. Feature engineering (preprocessing steps)
   c. Train/test split
   d. Model training
   e. Evaluation
6. Artifacts saved (local or S3)
7. Results stored in database
8. Job status updated to "completed"
```

### Request Flow

```
1. HTTP Request → FastAPI
2. Middleware (Logging, Error Handling)
3. Route Handler
4. Dependency Injection (settings, db session)
5. Service Layer (business logic)
6. Database/Storage Operations
7. Response serialization (Pydantic)
8. HTTP Response
```

## Configuration Management

### Settings Hierarchy

1. **Base Settings** (`Settings` class)
   - Default values
   - Type validation via Pydantic
   
2. **Environment-Specific**:
   - `DevelopmentSettings`: Debug enabled, relaxed security
   - `ProductionSettings`: Enhanced security, performance optimizations
   - `TestingSettings`: Test database, smaller limits

3. **Environment Variables**:
   - Loaded from `.env` file
   - Override default values
   - See `.env.example` for template

### Configuration Access

```python
from backend.config import get_settings

settings = get_settings()  # Cached via lru_cache
```

## Database Architecture

### Database Support

- **SQLite**: Default, for development
- **PostgreSQL**: Production, with async support

### Connection Management

- Uses SQLAlchemy 2.0+ with async support
- Connection pooling configured
- Async context managers for sessions
- Automatic table creation on startup

### Models Location

Database models are defined using SQLAlchemy's declarative base in:
- `backend/database/models.py` (or split by domain)

## Artifact Storage

### Strategy Pattern for Artifacts

The `ArtifactStore` uses a factory pattern:

```python
class ArtifactStoreFactory:
    @staticmethod
    def create(storage_type: str) -> ArtifactStore:
        if storage_type == "local":
            return LocalArtifactStore()
        elif storage_type == "s3":
            return S3ArtifactStore()
```

**Storage Types**:
- **Local**: Filesystem storage for development
- **S3**: Cloud storage for production

## Asynchronous Architecture

### Async Patterns

1. **Async Routes**: All FastAPI routes are async
2. **Async Database**: Using `asyncpg` and `aiosqlite`
3. **Async File I/O**: Using `aiofiles`
4. **Background Tasks**: Using Celery for long-running tasks

### Celery Integration

- Optional (can be disabled via `USE_CELERY=false`)
- Redis backend for task queue
- Separate worker process
- Task monitoring via Flower

## Security Architecture

### Authentication & Authorization

- JWT-based authentication (planned/optional)
- Token expiration and refresh
- Session management
- Role-based access control (future)

### Security Layers

1. **Middleware**: TrustedHostMiddleware, CORS
2. **Input Validation**: Pydantic models
3. **SQL Injection Prevention**: SQLAlchemy ORM
4. **Secret Management**: Environment variables
5. **HTTPS**: Required in production

## API Design

### RESTful Conventions

- Use HTTP methods correctly (GET, POST, PUT, DELETE)
- Proper status codes (200, 201, 400, 404, 500)
- Resource-based URLs
- Versioning via URL prefix (e.g., `/api/v1/`)

### API Documentation

- Auto-generated via FastAPI/OpenAPI
- Available at `/docs` (Swagger UI)
- Available at `/redoc` (ReDoc)
- Comprehensive schemas and examples

## Extensibility

### Adding New Features

1. **New ML Node**: Create Calculator/Applier, register with decorator
2. **New API Endpoint**: Create router, add to `main.py`
3. **New Storage Backend**: Implement `ArtifactStore` interface
4. **New Data Source**: Extend data ingestion module

### Plugin Architecture

While not fully plugin-based, the registry pattern allows for:
- Dynamic node registration
- Minimal core changes
- Easy testing of new features

## Performance Considerations

### Optimization Strategies

1. **Caching**: Settings cached with `lru_cache`
2. **Connection Pooling**: Database connections reused
3. **Async I/O**: Non-blocking operations
4. **Lazy Loading**: Load data only when needed
5. **Batch Processing**: For large datasets

### Monitoring

- Logging configured per environment
- Health check endpoints
- Performance metrics (future)

## Testing Architecture

### Test Structure

```
tests/
├── test_training_tasks.py
├── test_hyperparameter_tuning_optuna.py
├── test_hyperparameter_tuning_strategies.py
└── ...
```

### Testing Layers

1. **Unit Tests**: Individual functions/classes
2. **Integration Tests**: Multiple components together
3. **API Tests**: HTTP endpoint testing
4. **End-to-End Tests**: Full workflow testing

## Deployment Architecture

### Containerization

- Docker support via `docker-compose.yml`
- Multi-container setup:
  - FastAPI application
  - Redis (for Celery)
  - PostgreSQL (optional)

### Production Deployment

- Use production settings (`FASTAPI_ENV=production`)
- Enable HTTPS
- Configure proper CORS origins
- Use PostgreSQL instead of SQLite
- Enable S3 for artifact storage
- Monitor with health endpoints

## Future Architecture Considerations

- Microservices split (if needed)
- Event-driven architecture
- GraphQL API (optional)
- WebSocket support for real-time updates
- Distributed training support
- Multi-tenancy support
