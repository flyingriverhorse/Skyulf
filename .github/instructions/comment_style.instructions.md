# Comment Style Guidelines

## General Principles

- Write self-documenting code first; add comments only when necessary
- Comments should explain **why**, not **what**
- Keep comments concise and relevant
- Update comments when code changes
- Remove outdated or redundant comments

## Python Comment Standards

### Module-Level Docstrings

Every Python module should start with a docstring that includes:
1. Brief description of the module's purpose
2. Copyright notice
3. Additional context if needed

**Example from codebase**:
```python
"""
FastAPI MLops Application Entry Point

© 2025 Murat Unsal — Skyulf Project

This module creates and configures the FastAPI application instance.
It provides better concurrency support compared to the Flask implementation.
"""
```

### Class Docstrings

Classes should have docstrings that describe:
- Purpose of the class
- Key attributes (if not obvious)
- Usage examples (if helpful)

**Example**:
```python
class Settings(BaseSettings):
    """
    Comprehensive application settings with automatic environment variable loading.
    Migrated and enhanced from Flask configuration with Pydantic validation.
    """
```

### Function/Method Docstrings

Functions should have docstrings that include:
- Brief description of what the function does
- Parameters (with types if not using type hints)
- Return value description
- Exceptions raised (if any)
- Usage examples (for complex functions)

**Example**:
```python
def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
```

### Inline Comments

Use inline comments sparingly, only when:
- The code is necessarily complex
- There's a non-obvious reason for doing something
- There's a workaround or temporary fix
- There's a reference to external documentation

**Good inline comments**:
```python
# Startup
logger.info("🚀 Starting FastAPI MLops Application")
start_time = time.time()
```

```python
# Add middleware (order matters!)
_add_middleware(app, settings)
```

```python
# Avoid non-ASCII to prevent Windows console encoding issues
print("OK Pandas configured with optimized settings for ML workflows")
```

**Avoid redundant comments**:
```python
# Bad: Comment just repeats the code
user_id = 123  # Set user_id to 123

# Good: Code is self-explanatory
user_id = 123
```

### TODO Comments

Use TODO comments for future improvements:
```python
# TODO: Add support for custom authentication providers
# TODO: Implement caching for model predictions
```

### Configuration Comments

Add comments to configuration sections for clarity:
```python
# === CORE APPLICATION METADATA ===
APP_NAME: str = "Skyulf"
APP_VERSION: str = "0.1.5"

# === AWS CONFIGURATION ===
AWS_ACCESS_KEY_ID: Optional[str] = None
```

## Comment Formatting

### Spacing

- Use a single space after the `#` symbol
- Keep comments aligned with the code they describe
- Use blank lines to separate logical sections

```python
# Good
def process_data():
    # Load data from file
    data = load_data()
    
    # Apply transformations
    transformed = transform(data)
```

### Line Length

- Keep comments within 80-100 characters per line
- Break long comments into multiple lines
- Align continued comment lines

```python
# This is a longer comment that needs to be broken into multiple lines
# to maintain readability and follow line length conventions.
```

## Documentation Comments

### API Documentation

For API endpoints, use FastAPI's built-in documentation features:

```python
@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Train a machine learning model with the provided configuration.
    
    - **request**: Training configuration including dataset and model parameters
    - **db**: Database session for storing results
    
    Returns training job ID and initial status.
    """
```

### Complex Algorithm Comments

For complex algorithms, provide:
1. High-level overview
2. Step-by-step explanation
3. References to papers/documentation (if applicable)

```python
def calculate_feature_importance():
    """
    Calculate feature importance using permutation importance.
    
    Process:
    1. Train baseline model on full dataset
    2. For each feature, shuffle values and measure performance drop
    3. Rank features by importance score
    
    Reference: Breiman, "Random Forests" (2001)
    """
```

## Comments to Avoid

### Don't Comment Obvious Code

```python
# Bad: States the obvious
x = x + 1  # Increment x by 1

# Good: No comment needed
x = x + 1
```

### Don't Use Comments as Version Control

```python
# Bad: Outdated code left in comments
# def old_function():
#     return "old"

# Good: Delete old code, use git history
def new_function():
    return "new"
```

### Don't Blame or Complain in Comments

```python
# Bad: Unprofessional
# This is a terrible hack but it works

# Good: Explain the constraint
# Workaround for upstream library issue #1234
```

## Special Comment Types

### Emoji in Logs (Optional)

The codebase uses emojis in log messages for visual clarity:

```python
logger.info("🚀 Starting FastAPI MLops Application")
logger.info("✅ Database initialized")
logger.info("🛑 Shutting down FastAPI MLops Application")
```

**Guidelines**:
- Use sparingly and consistently
- Helps distinguish important lifecycle events
- Optional: can be disabled in production

### Section Headers

Use section headers for grouping related configuration or code:

```python
# === CORE APPLICATION METADATA ===
APP_NAME: str = "Skyulf"
APP_VERSION: str = "0.1.5"

# === SECURITY ===
SECRET_KEY: str = secrets.token_urlsafe(32)
```

### Implementation Notes

For complex implementation details:

```python
def _configure_openapi(app: FastAPI) -> None:
    """Attach a custom OpenAPI schema builder with enriched metadata."""
    # Note: OpenAPI schema is built lazily on first request
    # to ensure all routes are registered
    ...
```

## Copyright Headers

Always include copyright in module docstrings:

```python
"""
Module Description

© 2025 Murat Unsal — Skyulf Project
"""
```

## Comment Maintenance

- Review comments during code reviews
- Remove comments when refactoring makes them unnecessary
- Update comments when changing related code
- Don't rely on comments to fix bad code—refactor instead

## Comments in Tests

Test functions should have clear names; comments are rarely needed:

```python
# Good: Test name is self-documenting
def test_user_authentication_with_valid_credentials():
    ...

# Comments in tests should explain WHY, not WHAT
def test_edge_case_handling():
    # This edge case occurs when dataset has < 2 samples
    # Model training should fail gracefully
    ...
```

## When in Doubt

Ask yourself:
1. Would another developer understand this code without the comment?
2. Does the comment add value beyond what the code expresses?
3. Is there a way to make the code clearer instead of adding a comment?

If the answer to all three is yes, add the comment. Otherwise, improve the code or skip the comment.
