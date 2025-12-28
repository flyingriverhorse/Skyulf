# Testing Protocol

## Testing Philosophy

- Write tests that validate behavior, not implementation
- Keep tests focused and isolated
- Tests should be fast and reliable
- Test coverage should be meaningful, not just a number
- Follow the AAA pattern: Arrange, Act, Assert

## Test Framework

### Primary Tools

- **pytest**: Main testing framework
- **pytest-asyncio**: For async test support
- **pytest-cov**: For coverage reporting
- **factory-boy**: For test data generation (optional)
- **faker**: For generating realistic test data (optional)

### Running Tests

From `CONTRIBUTING.md`:

```bash
# Quick subset (fast, backend-only)
uv run pytest -q tests/test_training_tasks.py tests/test_hyperparameter_tuning_optuna.py tests/test_hyperparameter_tuning_strategies.py

# Full suite
uv run pytest -q

# With coverage
uv run pytest --cov=backend --cov-report=html
```

## Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_training_tasks.py         # Training functionality tests
├── test_hyperparameter_tuning_optuna.py
├── test_hyperparameter_tuning_strategies.py
└── ...                            # More test files
```

### Test File Naming

- Prefix test files with `test_`
- Match the module being tested: `test_<module_name>.py`
- Example: `backend/config.py` → `tests/test_config.py`

### Test Function Naming

Use descriptive names that explain what is being tested:

```python
# Good: Clear and descriptive
def test_user_login_with_valid_credentials():
    ...

def test_pipeline_execution_with_missing_data_raises_error():
    ...

# Bad: Too vague
def test_login():
    ...

def test_pipeline():
    ...
```

## Test Types

### 1. Unit Tests

Test individual functions or methods in isolation.

```python
def test_calculate_feature_importance():
    # Arrange
    data = create_sample_dataset()
    model = create_test_model()
    
    # Act
    importance = calculate_feature_importance(model, data)
    
    # Assert
    assert len(importance) == len(data.columns)
    assert all(score >= 0 for score in importance.values())
```

### 2. Integration Tests

Test multiple components working together.

```python
async def test_pipeline_execution_end_to_end():
    # Arrange
    config = create_pipeline_config()
    engine = PipelineEngine()
    
    # Act
    result = await engine.execute(config)
    
    # Assert
    assert result.status == "completed"
    assert result.metrics is not None
```

### 3. API Tests

Test HTTP endpoints.

```python
from fastapi.testclient import TestClient

def test_health_endpoint_returns_ok():
    client = TestClient(app)
    
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### 4. Database Tests

Test database operations.

```python
@pytest.mark.asyncio
async def test_save_pipeline_to_database(db_session):
    # Arrange
    pipeline = Pipeline(name="test", config={})
    
    # Act
    db_session.add(pipeline)
    await db_session.commit()
    
    # Assert
    saved = await db_session.get(Pipeline, pipeline.id)
    assert saved.name == "test"
```

## Fixtures

### Using pytest Fixtures

Define reusable test setup in `conftest.py`:

```python
import pytest
from backend.config import TestingSettings

@pytest.fixture
def settings():
    """Provide test settings."""
    return TestingSettings()

@pytest.fixture
async def db_session():
    """Provide a test database session."""
    # Setup
    engine = create_test_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session = AsyncSession(engine)
    
    yield session
    
    # Teardown
    await session.close()
    await engine.dispose()

@pytest.fixture
def sample_dataset():
    """Provide a sample dataset for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [0, 1, 0, 1, 0]
    })
```

### Fixture Scopes

- `function`: Default, new instance per test
- `class`: Shared across test class
- `module`: Shared across test module
- `session`: Shared across entire test session

```python
@pytest.fixture(scope="session")
def app():
    """Create FastAPI app once for all tests."""
    return create_app()
```

## Async Testing

### Using pytest-asyncio

Configure in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Async Test Examples

```python
@pytest.mark.asyncio
async def test_async_database_operation(db_session):
    result = await db_session.execute(select(User))
    users = result.scalars().all()
    assert len(users) >= 0

@pytest.mark.asyncio
async def test_async_api_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/users")
        assert response.status_code == 200
```

## Test Data

### Creating Test Data

1. **Minimal data**: Use smallest dataset that tests the feature
2. **Realistic data**: Use faker or factory-boy for complex objects
3. **Edge cases**: Test boundary conditions

```python
def create_minimal_dataset():
    """Minimal dataset for basic tests."""
    return pd.DataFrame({'x': [1, 2], 'y': [3, 4]})

def create_realistic_dataset():
    """Realistic dataset with proper distributions."""
    from faker import Faker
    fake = Faker()
    
    return pd.DataFrame({
        'name': [fake.name() for _ in range(100)],
        'age': [fake.random_int(18, 80) for _ in range(100)],
        'email': [fake.email() for _ in range(100)]
    })
```

### Using Factories (Optional)

```python
import factory

class UserFactory(factory.Factory):
    class Meta:
        model = User
    
    username = factory.Faker('user_name')
    email = factory.Faker('email')
    is_active = True

# Usage in tests
def test_user_creation():
    user = UserFactory()
    assert user.username is not None
```

## Mocking and Patching

### Using unittest.mock

```python
from unittest.mock import Mock, patch

def test_external_api_call():
    with patch('backend.services.external_api.call') as mock_call:
        mock_call.return_value = {'status': 'success'}
        
        result = perform_operation_with_external_api()
        
        assert result['status'] == 'success'
        mock_call.assert_called_once()
```

### Mocking Async Functions

```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_service():
    mock_service = AsyncMock()
    mock_service.fetch_data.return_value = {'data': 'test'}
    
    result = await process_with_service(mock_service)
    
    assert result == {'data': 'test'}
```

## Test Configuration

### Testing Settings

Use `TestingSettings` from `backend/config.py`:

```python
class TestingSettings(Settings):
    """Testing environment settings."""
    TESTING: bool = True
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite+aiosqlite:///./test_mlops.db"
    LOG_LEVEL: str = "DEBUG"
    DATA_SAMPLE_SIZE: int = 100
    ML_MODEL_CACHE_SIZE: int = 10
```

### Environment Variables

Set test-specific environment variables:

```python
import os
import pytest

@pytest.fixture(autouse=True)
def test_environment():
    os.environ['FASTAPI_ENV'] = 'testing'
    yield
    os.environ.pop('FASTAPI_ENV', None)
```

## Test Isolation

### Database Isolation

- Use in-memory SQLite for fast tests
- Or use transactions with rollback
- Clean up after each test

```python
@pytest.fixture
async def clean_database(db_session):
    yield
    # Cleanup
    await db_session.execute(delete(User))
    await db_session.commit()
```

### File System Isolation

- Use temporary directories
- Clean up files after tests

```python
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)
```

## Assertions

### Good Assertions

```python
# Be specific
assert user.email == "test@example.com"

# Test multiple aspects separately
assert response.status_code == 200
assert response.json()['id'] is not None
assert response.json()['name'] == "Test"

# Use pytest helpers
assert result in [1, 2, 3]
assert 'key' in dictionary
```

### pytest Assertion Features

```python
# Approximate comparisons
assert abs(result - 3.14) < 0.01
# or
import pytest
assert result == pytest.approx(3.14)

# Exception testing
with pytest.raises(ValueError):
    invalid_operation()

with pytest.raises(ValueError, match="specific message"):
    invalid_operation()
```

## Test Coverage

### Running Coverage

```bash
uv run pytest --cov=backend --cov-report=html
uv run pytest --cov=backend --cov-report=term-missing
```

### Coverage Goals

- Aim for meaningful coverage, not 100%
- Focus on critical paths
- Test edge cases and error conditions
- Don't test trivial code (getters/setters)

### Coverage Exclusions

Mark code that shouldn't be covered:

```python
def debug_only_function():  # pragma: no cover
    print("Debug info")
```

## Test Best Practices

### DO

✅ Write tests before fixing bugs
✅ Test one thing per test function
✅ Use descriptive test names
✅ Keep tests simple and readable
✅ Use fixtures for setup/teardown
✅ Test edge cases and errors
✅ Run tests before committing
✅ Keep tests fast

### DON'T

❌ Test implementation details
❌ Create interdependent tests
❌ Use sleep() for timing (use proper async/await)
❌ Ignore flaky tests
❌ Skip tests without good reason
❌ Test external services directly (mock them)
❌ Commit commented-out tests

## Continuous Integration

Tests run automatically on:
- Pull requests
- Push to main branch
- Python 3.10 and 3.11

CI configuration in `.github/workflows/`

## Debugging Tests

### Running Single Test

```bash
uv run pytest tests/test_file.py::test_function_name -v
```

### Using Print Debugging

```bash
uv run pytest tests/test_file.py -s  # Show print statements
```

### Using pytest-pdb

```bash
uv run pytest tests/test_file.py --pdb  # Drop into debugger on failure
```

### Verbose Output

```bash
uv run pytest -vv  # Very verbose
```

## Test Maintenance

- Remove obsolete tests when removing features
- Update tests when refactoring
- Add tests when fixing bugs
- Review test failures promptly
- Keep test dependencies up to date

## Contributing Tests

From `CONTRIBUTING.md`:

> Add/adjust tests when changing behavior.

All PRs should include:
1. Tests for new features
2. Updated tests for modified features
3. Tests that demonstrate bug fixes
4. All existing tests passing
