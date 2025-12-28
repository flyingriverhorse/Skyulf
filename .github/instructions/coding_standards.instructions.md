# Coding Standards

## General Principles

- Write clean, maintainable, and testable code
- Follow the DRY (Don't Repeat Yourself) principle
- Use meaningful variable and function names
- Keep functions focused and concise (single responsibility)
- Make minimal changes to achieve goals

## Python Code Style

### Formatting and Style
- **Python Version**: Python 3.10+ (specified in `pyproject.toml`)
- **Formatter**: Use Black for code formatting
  - Command: `uv run black .`
- **Import Sorting**: Use isort for consistent import ordering
  - Command: `uv run isort .`
- **Linter**: Use Flake8 for code linting
  - Command: `uv run flake8`
  - Config: `.flake8`
- **Type Checker**: Use MyPy for type checking
  - Command: `uv run mypy .`
  - Config: `mypy.ini`

### Type Hints
- Use type hints for all function signatures
- Use Optional[] for nullable values
- Use List[], Dict[], etc. from typing module
- Example from codebase:
```python
from typing import AsyncGenerator, Optional, List

async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    ...

def get_settings() -> Settings:
    ...
```

### Docstrings
- Use triple-quoted docstrings for modules, classes, and functions
- Follow Google or NumPy docstring format
- Include purpose, parameters, return values, and exceptions
- Example from codebase:
```python
"""
FastAPI MLops Application Entry Point

© 2025 Murat Unsal — Skyulf Project

This module creates and configures the FastAPI application instance.
It provides better concurrency support compared to the Flask implementation.
"""
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `Settings`, `FastAPI`, `LoggingMiddleware`)
- **Functions**: snake_case (e.g., `get_settings`, `create_app`, `init_db`)
- **Variables**: snake_case (e.g., `app`, `settings`, `logger`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `OPENAPI_DESCRIPTION`, `MAX_UPLOAD_SIZE`)
- **Private members**: prefix with underscore (e.g., `_build_swagger_ui_parameters`)

### Imports
- Use absolute imports (e.g., `from backend.config import get_settings`)
- Group imports in this order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Separate groups with blank lines

## FastAPI Specific

### Application Structure
- Use application factory pattern (see `backend/main.py`)
- Use lifespan context manager for startup/shutdown
- Configure middleware in correct order
- Use dependency injection where appropriate

### Routing
- Use APIRouter for modular routing
- Include proper tags for API documentation
- Use proper HTTP status codes
- Example:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/pipeline", tags=["ml-workflow"])
```

### Configuration
- Use Pydantic Settings for configuration (see `backend/config.py`)
- Load from environment variables via `.env` file
- Validate configuration at startup
- Use typed settings with validators

### Error Handling
- Create custom exception handlers
- Return proper error responses (JSON format)
- Log errors appropriately
- Use FastAPI's HTTPException

## Database and ORM

### SQLAlchemy Usage
- Use async SQLAlchemy (sqlalchemy[asyncio])
- Use declarative base for models
- Use Alembic for migrations
- Keep database logic in separate modules

### Database Configuration
- Support multiple database types (SQLite, PostgreSQL)
- Use environment variables for database URLs
- Use connection pooling in production

## Async/Await Patterns

- Use async functions for I/O-bound operations
- Use await for async function calls
- Use AsyncGenerator for async context managers
- Example:
```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()
```

## Logging

- Use Python's logging module
- Configure logging at application startup
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include context in log messages
- Example:
```python
import logging

logger = logging.getLogger(__name__)
logger.info("🚀 Starting FastAPI MLops Application")
```

## Testing

- Write tests using pytest
- Use pytest-asyncio for async tests
- Maintain test coverage
- Keep tests focused and isolated
- See `CONTRIBUTING.md` for test commands:
  - Quick subset: `uv run pytest -q tests/test_training_tasks.py`
  - Full suite: `uv run pytest -q`

## Security

- Never commit secrets to version control
- Use environment variables for sensitive data
- Validate all user inputs
- Use proper authentication and authorization
- Follow FastAPI security best practices
- Use TrustedHostMiddleware and CORS properly

## Performance

- Use async for I/O operations
- Cache expensive computations where appropriate
- Use connection pooling for databases
- Monitor memory usage for large ML datasets
- Use pagination for large result sets

## Copyright and Licensing

- Include copyright notice in module docstrings:
  `© 2025 Murat Unsal — Skyulf Project`
- Follow Apache 2.0 license requirements
- Reference CLA.md for contributions

## Tools and Commands

### Development Setup
```bash
pip install uv
uv sync --dev
```

### Code Quality Commands
```bash
# Format code
uv run black .
uv run isort .

# Lint code
uv run flake8

# Type check
uv run mypy .

# Run tests
uv run pytest -q
```

## Pre-commit Hooks

- Use pre-commit for automated code quality checks
- Config file: `.pre-commit-config.yaml`
- Runs formatters and linters before commits
