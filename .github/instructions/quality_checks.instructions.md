# Quality Checks

## Overview

This document defines the quality checks that must pass before code is merged. These checks ensure code quality, consistency, and reliability.

## Automated Quality Checks

### 1. Code Formatting

#### Black

**Purpose**: Automatic code formatting for consistency

**Command**:
```bash
uv run black .
```

**Configuration**: `.editorconfig`

**Rules**:
- Line length: 88 characters (Black default)
- Single quotes or double quotes (consistent)
- Automatic indentation

**CI Check**: Formatting must be applied before commit

#### isort

**Purpose**: Consistent import ordering

**Command**:
```bash
uv run isort .
```

**Configuration**: Compatible with Black

**Rules**:
- Standard library first
- Third-party packages second
- Local imports third
- Blank lines between groups

**CI Check**: Import order must be correct

### 2. Linting

#### Flake8

**Purpose**: Code style and potential error detection

**Command**:
```bash
uv run flake8
```

**Configuration**: `.flake8`

**Key Rules**:
- PEP 8 compliance
- Line length limits
- Unused imports detection
- Complexity checks
- Naming conventions

**Common Issues**:
- E501: Line too long
- F401: Unused import
- E302: Expected 2 blank lines
- E265: Block comment should start with '# '

**Fixing**:
```bash
# Check issues
uv run flake8

# Auto-fix what's possible with Black and isort
uv run black .
uv run isort .
```

### 3. Type Checking

#### MyPy

**Purpose**: Static type checking

**Command**:
```bash
uv run mypy .
```

**Configuration**: `mypy.ini`

**Requirements**:
- Type hints for function signatures
- Proper return type annotations
- No `Any` types without justification

**Example**:
```python
def process_data(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    ...
```

**Ignoring Errors** (when necessary):
```python
# type: ignore  # Reason why ignore is needed
```

### 4. Testing

#### pytest

**Purpose**: Automated testing

**Command**:
```bash
# Quick subset
uv run pytest -q tests/test_training_tasks.py

# Full suite
uv run pytest -q

# With coverage
uv run pytest --cov=backend --cov-report=term-missing
```

**Requirements**:
- All tests must pass
- New features must have tests
- Bug fixes must have regression tests
- Aim for meaningful coverage (not 100% at all costs)

**Coverage Targets**:
- Critical paths: 90%+
- Business logic: 80%+
- Utilities: 70%+
- Overall: 75%+

### 5. Pre-commit Hooks

**Purpose**: Run checks automatically before commit

**Configuration**: `.pre-commit-config.yaml`

**Setup**:
```bash
uv run pre-commit install
```

**What it checks**:
- Trailing whitespace
- File endings
- YAML validity
- Large files
- Code formatting (Black)
- Import sorting (isort)
- Linting (Flake8)

**Running manually**:
```bash
uv run pre-commit run --all-files
```

## Quality Check Workflow

### Before Committing

1. **Format code**:
   ```bash
   uv run black . && uv run isort .
   ```

2. **Run linter**:
   ```bash
   uv run flake8
   ```

3. **Type check**:
   ```bash
   uv run mypy .
   ```

4. **Run tests**:
   ```bash
   uv run pytest -q
   ```

5. **Check pre-commit**:
   ```bash
   uv run pre-commit run --all-files
   ```

### Automated CI Checks

On every pull request, GitHub Actions runs:

1. **Linting**: Flake8, Black check, isort check
2. **Type checking**: MyPy
3. **Testing**: Full test suite on Python 3.10 and 3.11
4. **Coverage**: Report test coverage
5. **Dependencies**: Check for security vulnerabilities (Dependabot)

**Requirements to merge**:
- ✅ All CI checks passing
- ✅ All tests passing
- ✅ Code review approved
- ✅ No merge conflicts

## Code Review Checklist

### Functionality

- [ ] Code solves the stated problem
- [ ] Edge cases are handled
- [ ] Error conditions are handled gracefully
- [ ] No unnecessary complexity
- [ ] Follows single responsibility principle

### Code Quality

- [ ] Follows coding standards
- [ ] Proper naming conventions
- [ ] No commented-out code
- [ ] No debug print statements (use logging)
- [ ] No hardcoded values (use configuration)

### Testing

- [ ] New features have tests
- [ ] Tests are meaningful and focused
- [ ] Tests cover edge cases
- [ ] All tests pass
- [ ] No flaky tests

### Documentation

- [ ] Docstrings for public functions
- [ ] Complex logic is explained
- [ ] README/docs updated if needed
- [ ] API changes documented
- [ ] Breaking changes noted

### Security

- [ ] No secrets in code
- [ ] Input validation present
- [ ] SQL injection prevented (use ORM)
- [ ] XSS prevented (proper escaping)
- [ ] Authentication/authorization correct

### Performance

- [ ] No obvious performance issues
- [ ] Async used for I/O operations
- [ ] No N+1 query problems
- [ ] Efficient algorithms used
- [ ] Memory usage considered

### Dependencies

- [ ] New dependencies justified
- [ ] Version constraints specified
- [ ] License compatible (Apache 2.0)
- [ ] No security vulnerabilities

## Manual Quality Checks

### Code Readability

**Ask yourself**:
- Can another developer understand this code?
- Are variable names meaningful?
- Is the code self-documenting?
- Are functions short and focused?

**Metrics**:
- Function length: < 50 lines (guideline)
- File length: < 500 lines (guideline)
- Cyclomatic complexity: < 10 (Flake8 checks this)

### Architecture Alignment

**Check**:
- Follows project architecture patterns
- Uses dependency injection appropriately
- Separates concerns (API, business logic, data)
- Uses appropriate design patterns

**Reference**: `.github/instructions/project_architecture.instructions.md`

### API Design

For new endpoints:
- [ ] RESTful URL structure
- [ ] Proper HTTP methods (GET, POST, PUT, DELETE)
- [ ] Appropriate status codes
- [ ] Request/response models defined (Pydantic)
- [ ] API documentation complete
- [ ] Versioning considered

### Database Changes

For database modifications:
- [ ] Migration created (Alembic)
- [ ] Backward compatibility considered
- [ ] Indexes added where needed
- [ ] No N+1 queries
- [ ] Proper use of async

## Performance Benchmarks

### Response Time Targets

- Health check: < 50ms
- Simple GET: < 200ms
- Complex query: < 1s
- ML prediction: < 5s
- Model training: Varies (with progress updates)

### Resource Limits

- Memory: Should not exceed 2GB for API server
- CPU: Should not peg CPU on idle
- Database connections: Should release properly

## Security Checks

### Static Security Analysis

Tools to consider:
- **bandit**: Security linter for Python
- **safety**: Check dependencies for vulnerabilities
- **Dependabot**: Automated dependency updates

**Running bandit**:
```bash
pip install bandit
bandit -r backend/
```

### Common Security Issues

**Check for**:
- SQL injection (use ORM)
- XSS (escape output)
- CSRF (use tokens)
- Secrets in code (use environment variables)
- Weak passwords (enforce strong passwords)
- Session management (proper expiry)
- HTTPS usage (production)

## Documentation Quality

### Code Documentation

- [ ] Module docstrings present
- [ ] Function docstrings complete
- [ ] Type hints for all functions
- [ ] Complex logic explained

### External Documentation

- [ ] README.md up to date
- [ ] CONTRIBUTING.md reflects current process
- [ ] API documentation accurate
- [ ] Architecture docs current

## Release Checklist

Before releasing a new version:

1. **Version Bump**
   - [ ] Update version in `pyproject.toml`
   - [ ] Update version in `backend/config.py`
   - [ ] Update `VERSION_UPDATE.md`

2. **Testing**
   - [ ] All tests pass
   - [ ] Manual testing complete
   - [ ] Integration tests pass
   - [ ] Performance acceptable

3. **Documentation**
   - [ ] CHANGELOG updated
   - [ ] Migration guide (if needed)
   - [ ] Breaking changes documented
   - [ ] API docs regenerated

4. **Dependencies**
   - [ ] Lock file updated (`uv.lock`)
   - [ ] No vulnerable dependencies
   - [ ] License compliance checked

5. **Build**
   - [ ] Clean build succeeds
   - [ ] Docker image builds
   - [ ] Smoke tests pass

## Continuous Improvement

### Metrics to Track

- Test coverage percentage
- CI build time
- Code review turnaround time
- Bug count per release
- Performance metrics

### Regular Reviews

- **Weekly**: Review failed CI builds
- **Monthly**: Review and update quality standards
- **Quarterly**: Review and update dependencies
- **Yearly**: Review and update architecture

## Exceptions

### When to Skip Checks

Exceptions should be rare and documented:

1. **Hotfixes**: May skip some checks for critical security fixes
   - Still require: tests, code review
   - Document why fast-tracked

2. **Prototypes**: Experimental branches may have relaxed rules
   - Must meet standards before merging to main
   - Mark as draft PR

3. **Documentation**: Doc-only changes may skip tests
   - Still require: linting, review

### Requesting Exceptions

To request an exception:
1. Comment in PR with justification
2. Get approval from maintainer
3. Document the exception
4. Create follow-up issue if needed

## Tools and Commands Summary

```bash
# Format and lint
uv run black . && uv run isort . && uv run flake8

# Type check
uv run mypy .

# Test
uv run pytest -q

# Coverage
uv run pytest --cov=backend --cov-report=html

# Pre-commit
uv run pre-commit run --all-files

# All checks (one command)
uv run black . && uv run isort . && uv run flake8 && uv run mypy . && uv run pytest -q
```

## Getting Help

If quality checks fail:

1. **Read the error message** - Often explains the issue
2. **Check documentation** - Coding standards and style guides
3. **Ask in PR comments** - Reviewers can help
4. **Search issues** - Others may have faced same problem
5. **Update this document** - If guidance is unclear

## Quality is Everyone's Responsibility

- Write quality code from the start
- Review others' code carefully
- Suggest improvements constructively
- Keep standards up to date
- Automate what can be automated
