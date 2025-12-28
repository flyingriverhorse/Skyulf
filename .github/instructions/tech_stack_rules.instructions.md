# Tech Stack Rules

## Overview

This document outlines the approved technology stack and rules for the Skyulf MLOps platform.

## Core Technologies

### Python

- **Version**: Python 3.10+ (specified in `pyproject.toml`)
- **Supported Versions**: 3.10, 3.11, 3.12
- **Rationale**: Modern Python features, type hints, async support

### Package Management

#### Primary: uv (Recommended)

- **Tool**: `uv` (ultra-fast Python package manager)
- **Installation**: `pip install uv`
- **Usage**:
  ```bash
  uv sync --dev          # Install all dependencies
  uv run pytest          # Run tests
  uv run black .         # Format code
  ```
- **Benefits**: Fast, reliable, modern dependency resolution

#### Alternative: pip + venv

For environments without uv:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-fastapi.txt -r requirements-ci.txt
```

### Web Framework

#### FastAPI

- **Version**: 0.104.0+ (< 1.0.0)
- **Purpose**: Modern, fast (high-performance) web framework
- **Features Used**:
  - Async/await support
  - Automatic API documentation (OpenAPI/Swagger)
  - Pydantic integration
  - Dependency injection
  - Type validation
- **Rationale**: Better concurrency than Flask, built-in async support, excellent for ML APIs

### Configuration Management

#### Pydantic Settings

- **Library**: `pydantic-settings`
- **Version**: 2.0.0+ (< 3.0.0)
- **Usage**: Type-safe configuration with environment variable loading
- **Implementation**: See `backend/config.py`

### Database

#### SQLAlchemy 2.0+

- **Version**: 2.0.0+ (< 3.0.0)
- **Mode**: Async mode with `asyncio` support
- **Drivers**:
  - SQLite: `aiosqlite` (development)
  - PostgreSQL: `asyncpg` (production)
- **ORM**: Declarative base pattern

#### Database Choices

1. **SQLite** (Default for Development)
   - File-based database
   - No separate server needed
   - Perfect for development and testing
   
2. **PostgreSQL** (Production)
   - Robust, production-ready
   - Better for concurrent access
   - Advanced features (JSON, full-text search)

#### Migrations

- **Tool**: Alembic
- **Version**: 1.16.5+
- **Usage**: Database schema versioning

### Background Tasks

#### Celery

- **Version**: 5.4.0+
- **Purpose**: Distributed task queue for long-running ML jobs
- **Broker**: Redis
- **Optional**: Can be disabled via `USE_CELERY=false`
- **Monitoring**: Flower (2.0.0+)

#### Redis

- **Version**: 5.0.0+ (< 6.0.0)
- **Purpose**: Message broker for Celery, caching
- **Usage**: Task queue, result backend

## ML and Data Science Stack

### Core ML Libraries

#### Scikit-learn

- **Version**: 1.4.0+
- **Purpose**: Core ML algorithms
- **Integration**: Primary ML backend

#### Pandas

- **Version**: 2.0.0+
- **Features**: Modern Pandas with copy-on-write mode
- **Configuration**: Set in `backend/config.py`

#### NumPy

- **Version**: 1.24.0+ (< 2.0.0)
- **Note**: Stay on 1.x for compatibility

### Advanced ML Libraries

#### XGBoost

- **Version**: 2.1.4+
- **Purpose**: Gradient boosting models

#### Imbalanced-learn

- **Version**: 0.12.0+
- **Purpose**: Handling imbalanced datasets

#### Optuna

- **Version**: 4.0+ (< 5)
- **Purpose**: Hyperparameter optimization
- **Integration**: `optuna-integration` for scikit-learn

### Data Processing

#### Data Formats

- **CSV**: Standard support via Pandas
- **Parquet**: Apache Parquet format
- **Excel**: `.xlsx`, `.xls` support
- **JSON**: Standard JSON format
- **Pickle**: Python serialization
- **Feather**: Arrow-based format
- **HDF5**: `.h5`, `.hdf5` support

#### Polars

- **Version**: 1.36.0+
- **Purpose**: Fast DataFrame library (alternative to Pandas)
- **Usage**: Optional, for performance-critical operations

#### Dask

- **Version**: 2024.8.0+
- **Purpose**: Parallel computing for large datasets
- **Usage**: Optional, for datasets larger than memory

### Visualization

#### Matplotlib

- **Version**: 3.9.4+
- **Purpose**: Static plots

#### Seaborn

- **Version**: 0.13.2+
- **Purpose**: Statistical visualizations

#### Plotly

- **Version**: 6.3.0+
- **Purpose**: Interactive visualizations

### Geospatial (Optional)

#### GeoPandas

- **Version**: 0.14.0+ (< 1.2.0)
- **Dependencies**: `shapely`, `pyproj`, `rtree`
- **Purpose**: Geospatial data processing

## Development Tools

### Code Quality

#### Black

- **Version**: 23.0.0+ (< 24.0.0)
- **Purpose**: Code formatter
- **Configuration**: `.editorconfig`
- **Usage**: `uv run black .`

#### isort

- **Version**: 5.12.0+ (< 6.0.0)
- **Purpose**: Import sorting
- **Usage**: `uv run isort .`

#### Flake8

- **Version**: 6.0.0+ (< 7.0.0)
- **Purpose**: Linting
- **Configuration**: `.flake8`
- **Usage**: `uv run flake8`

#### MyPy

- **Version**: 1.6.0+ (< 2.0.0)
- **Purpose**: Static type checking
- **Configuration**: `mypy.ini`
- **Usage**: `uv run mypy .`

### Testing

#### pytest

- **Version**: 7.4.0+ (< 10.0.0)
- **Extensions**:
  - `pytest-asyncio`: Async test support
  - `pytest-cov`: Coverage reporting

#### Test Helpers

- **factory-boy**: 3.3.0+ (< 4.0.0) - Test data factories
- **faker**: 19.0.0+ (< 20.0.0) - Realistic fake data

### Pre-commit Hooks

- **Tool**: `pre-commit`
- **Version**: 3.4.0+
- **Configuration**: `.pre-commit-config.yaml`
- **Purpose**: Automated code quality checks before commits

## Web Server

### Uvicorn

- **Version**: 0.24.0+ (< 1.0.0)
- **Mode**: With `[standard]` extras for better performance
- **Features**: ASGI server, HTTP/2, WebSocket support
- **Usage**: Production-ready ASGI server

## Authentication & Security

### Password Hashing

- **Library**: `passlib[bcrypt]`
- **Version**: 1.7.4+ (< 2.0.0)
- **Algorithm**: bcrypt with `bcrypt` 4.0.1

### JWT Tokens

- **Library**: `python-jose[cryptography]`
- **Version**: 3.3.0+ (< 4.0.0)
- **Algorithm**: HS256 (configurable)

### HTTPS & TLS

- **Production**: Always use HTTPS
- **Development**: HTTP acceptable
- **Certificates**: Let's Encrypt recommended

## Cloud & Storage

### AWS/S3 Support

- **Purpose**: Artifact storage, data sources
- **Configuration**: Optional, via environment variables
- **Libraries**: Built-in boto3 support

### Local Storage

- **Default**: Local filesystem
- **Directories**:
  - `uploads/data`: User uploads
  - `uploads/models`: Model artifacts
  - `exports/data`: Exported datasets
  - `temp/processing`: Temporary files

## Containerization

### Docker

- **Compose**: Version 3.x
- **File**: `docker-compose.yml`
- **Services**:
  - FastAPI application
  - Redis (optional)
  - PostgreSQL (optional)

## Frontend (Separate Codebase)

### React

- **Purpose**: ML canvas UI
- **Language**: TypeScript
- **Communication**: REST API with backend

## Documentation

### MkDocs

- **File**: `mkdocs.yml`
- **Purpose**: Project documentation
- **Location**: `docs/` directory
- **Deployment**: GitHub Pages via `index.html`

## Version Constraints

### Strict Version Pins

Use strict version constraints for production stability:

```toml
# Exact version (rare)
bcrypt = "==4.0.1"

# Compatible release (common)
fastapi = ">=0.104.0,<1.0.0"

# Minor version constraint
numpy = ">=1.24.0,<2.0.0"
```

### Dependency Groups

From `pyproject.toml`:
- **dependencies**: Core runtime dependencies
- **dev**: Development tools (testing, linting)

## Prohibited Technologies

### ❌ Don't Use

1. **Flask** - Migrated to FastAPI for better async support
2. **Django** - Too heavyweight for this use case
3. **NumPy 2.x** - Compatibility issues with ML libraries
4. **Synchronous database drivers** - Use async drivers
5. **Global state** - Use dependency injection instead

## Technology Selection Guidelines

When adding new dependencies:

1. **Check compatibility** with existing stack
2. **Verify Python 3.10+ support**
3. **Consider async support** for I/O operations
4. **Evaluate maintenance status** (active development)
5. **Check license** (must be compatible with Apache 2.0)
6. **Review security** (check for known vulnerabilities)
7. **Minimize dependencies** (don't add if existing tool suffices)

### Adding New Dependencies

```bash
# Add to pyproject.toml dependencies section
uv add <package-name>

# Or for dev dependencies
uv add --dev <package-name>

# Update lock file
uv lock
```

## Platform Support

### Operating Systems

- **Linux**: Primary target
- **macOS**: Fully supported
- **Windows**: Supported (with notes in `CONTRIBUTING.md`)

### Architecture

- **x86_64**: Primary
- **ARM64**: Supported (Apple Silicon, AWS Graviton)

## Performance Guidelines

### Choose the Right Tool

- **Small datasets (< 1GB)**: Pandas
- **Large datasets (1GB - 10GB)**: Dask or chunked Pandas
- **Very large datasets (> 10GB)**: Polars or Dask
- **Fast I/O**: Use async operations
- **CPU-intensive**: Consider Numba for hot loops

### Async vs Sync

- **I/O operations**: Use async (database, file, network)
- **CPU-bound ML**: Use sync (in Celery workers)
- **API endpoints**: Always async

## Licensing

All dependencies must be compatible with Apache 2.0:

- ✅ MIT, BSD, Apache 2.0, LGPL
- ⚠️ GPL (avoid in main codebase)
- ❌ Proprietary or restrictive licenses

## Updates and Maintenance

### Regular Updates

- Security patches: Apply immediately
- Minor versions: Update quarterly
- Major versions: Evaluate carefully

### Monitoring

- Use Dependabot (`.github/dependabot.yml`)
- Review security advisories
- Test thoroughly after updates

## Support Matrix

| Python | FastAPI | SQLAlchemy | Status |
|--------|---------|------------|--------|
| 3.10   | 0.104+  | 2.0+       | ✅ Supported |
| 3.11   | 0.104+  | 2.0+       | ✅ Supported |
| 3.12   | 0.104+  | 2.0+       | ✅ Supported |
| 3.9    | -       | -          | ❌ Not supported |

## Future Considerations

Technologies under evaluation:
- **GraphQL**: For more flexible APIs
- **WebSockets**: For real-time updates
- **Distributed training**: Ray, Dask-ML
- **Feature stores**: Feast
- **Model serving**: MLflow, BentoML

## Questions?

For technology choices not covered here, discuss in:
- GitHub Issues
- Pull Request discussions
- Team meetings
