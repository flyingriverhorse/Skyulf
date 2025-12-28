# Changelog Discipline

## Overview

Maintaining a clear and consistent changelog helps users, contributors, and maintainers understand what has changed between versions. This document defines how we manage versioning and changelogs for the Skyulf project.

## Versioning Strategy

### Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Examples**:
- `0.1.4` → `0.1.5`: Bug fix
- `0.1.5` → `0.2.0`: New feature
- `0.2.0` → `1.0.0`: Major release, breaking changes

### Pre-release Versions

For pre-releases:
- `1.0.0-alpha.1`: Alpha release
- `1.0.0-beta.1`: Beta release
- `1.0.0-rc.1`: Release candidate

### Version Files

Update version in multiple locations:

1. **`pyproject.toml`**:
   ```toml
   [project]
   version = "0.1.5"
   ```

2. **`backend/config.py`**:
   ```python
   APP_VERSION: str = "0.1.5"
   ```

3. **`VERSION_UPDATE.md`**: Document the change

## Changelog Management

### File: VERSION_UPDATE.md

**Primary changelog** file at project root.

**Structure**:
```markdown
# Version Updates

## [0.1.5] - 2025-01-15

### Added
- New feature X
- Support for Y

### Changed
- Improved Z performance
- Updated dependency versions

### Fixed
- Bug in A
- Issue with B

### Deprecated
- Old feature C will be removed in 0.2.0

### Removed
- Legacy code D

### Security
- Fixed vulnerability E
```

### Changelog Categories

Use these standard categories:

1. **Added**: New features
2. **Changed**: Changes to existing functionality
3. **Deprecated**: Features to be removed in future
4. **Removed**: Removed features
5. **Fixed**: Bug fixes
6. **Security**: Security fixes

### Writing Good Changelog Entries

#### ✅ Good Examples

```markdown
### Added
- FastAPI integration for better async support
- S3 artifact storage for production deployments
- Health check endpoint at `/health`

### Changed
- Migrated from Flask to FastAPI for improved performance
- Updated Pandas to 2.0 with copy-on-write mode enabled
- Improved error messages in pipeline execution

### Fixed
- Fixed race condition in Celery task execution
- Resolved memory leak in large dataset processing
- Corrected type hints in config module

### Security
- Updated dependencies to patch CVE-2024-12345
- Added input validation for file uploads
- Improved JWT token expiration handling
```

#### ❌ Bad Examples

```markdown
### Added
- Stuff  # Too vague

### Changed
- Updates  # Not descriptive

### Fixed
- Bug  # Which bug?
- Fixed it  # Fixed what?
```

### Changelog Entry Guidelines

**DO**:
- ✅ Write from user's perspective
- ✅ Be specific and clear
- ✅ Include issue/PR numbers if relevant
- ✅ Group related changes together
- ✅ Use present tense ("Add feature" not "Added feature")
- ✅ Start with a verb
- ✅ Mention breaking changes explicitly

**DON'T**:
- ❌ Include internal refactoring (unless user-visible)
- ❌ Use jargon without explanation
- ❌ List every single commit
- ❌ Forget to categorize changes
- ❌ Make entries too technical

### Breaking Changes

**Highlight breaking changes** prominently:

```markdown
## [0.2.0] - 2025-02-01

### ⚠️ BREAKING CHANGES
- Database schema migration required
- Configuration file format changed
- API endpoint `/old-path` moved to `/new-path`

### Migration Guide
To upgrade from 0.1.x to 0.2.0:
1. Backup your database
2. Run migration: `alembic upgrade head`
3. Update config file (see `.env.example`)
4. Update API calls to new endpoints
```

## Commit Messages

### Conventional Commits (Recommended)

From `CONTRIBUTING.md`:

```
feat: add data source preview for parquet
fix: handle unicode column names in EDA
docs: update quick start for Windows
chore: update dependencies
test: add tests for pipeline execution
refactor: simplify database connection logic
perf: optimize large dataset loading
style: format code with Black
```

### Format

**Structure**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Example**:
```
feat(ml-pipeline): add XGBoost support

Implement XGBoost integration for gradient boosting models.
Includes hyperparameter tuning via Optuna.

Closes #123
```

### Commit Message Guidelines

**Good commits**:
```
feat: add S3 artifact storage
fix: resolve memory leak in data processing
docs: update architecture documentation
test: add integration tests for API endpoints
```

**Bad commits**:
```
Update stuff
Fix bug
WIP
asdf
```

## Release Process

### Creating a Release

1. **Update version numbers**:
   ```bash
   # pyproject.toml
   version = "0.2.0"
   
   # backend/config.py
   APP_VERSION: str = "0.2.0"
   ```

2. **Update VERSION_UPDATE.md**:
   ```markdown
   ## [0.2.0] - 2025-02-01
   
   ### Added
   - Feature A
   - Feature B
   
   ### Changed
   - Improvement C
   ```

3. **Commit changes**:
   ```bash
   git add pyproject.toml backend/config.py VERSION_UPDATE.md
   git commit -m "chore: bump version to 0.2.0"
   ```

4. **Create git tag**:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

5. **Create GitHub release**:
   - Go to GitHub releases
   - Create new release from tag
   - Copy changelog entries
   - Attach any release artifacts

### Release Cadence

- **Patch releases**: As needed for bug fixes
- **Minor releases**: Monthly or bi-monthly
- **Major releases**: When breaking changes accumulate

### Release Notes

GitHub release notes should include:

1. **Summary**: Brief overview of release
2. **Highlights**: Key features/changes
3. **Changelog**: Full list of changes (from VERSION_UPDATE.md)
4. **Upgrade Guide**: Migration steps if needed
5. **Known Issues**: Any limitations
6. **Contributors**: Thank contributors

**Template**:
```markdown
## Skyulf v0.2.0

### Highlights
- 🚀 Migrated to FastAPI for better performance
- 📦 Added S3 support for artifact storage
- 🔧 Improved configuration management

### What's Changed
[Copy from VERSION_UPDATE.md]

### Upgrade Guide
1. Update dependencies: `uv sync`
2. Update configuration (see .env.example)
3. Run database migrations: `alembic upgrade head`

### Known Issues
- Issue #456: Large datasets may timeout
- Windows users: See QUICKSTART.md for setup

### Contributors
Thanks to @user1, @user2 for their contributions!

**Full Changelog**: https://github.com/flyingriverhorse/Skyulf/compare/v0.1.5...v0.2.0
```

## Pull Request Discipline

### PR Description

Include in PR description:
- **What**: What changes were made
- **Why**: Why these changes are needed
- **How**: How the changes work (if complex)
- **Testing**: How changes were tested
- **Checklist**: Pre-merge checklist

**Template**:
```markdown
## Description
Adds S3 support for artifact storage to enable production deployments.

## Motivation
Users need to store ML artifacts in cloud storage for scalability.

## Changes
- Implemented S3ArtifactStore class
- Added S3 configuration to Settings
- Updated documentation

## Testing
- Added unit tests for S3 operations
- Tested with MinIO (S3-compatible)
- Verified with AWS S3

## Checklist
- [x] Tests added/updated
- [x] Documentation updated
- [x] Follows coding standards
- [x] No breaking changes
```

### PR Labels

Use labels to categorize PRs:
- `enhancement`: New features
- `bug`: Bug fixes
- `documentation`: Doc updates
- `dependencies`: Dependency updates
- `breaking change`: Breaking changes
- `good first issue`: Good for newcomers

## Changelog Automation

### Release Drafter

Configuration: `.github/release-drafter.yml`

**Automatically generates release notes** from PRs:
- Groups by labels
- Extracts PR titles
- Lists contributors

### Dependabot

Configuration: `.github/dependabot.yml`

**Automatically**:
- Updates dependencies
- Creates PRs for security patches
- Generates changelog entries

## Documentation Updates

### When to Update Docs

Update documentation for:
- New features
- Changed APIs
- Configuration changes
- Breaking changes
- Migration guides

### Documentation Files

- **README.md**: Project overview
- **CONTRIBUTING.md**: Contribution guide
- **QUICKSTART.md**: Quick start guide
- **ROADMAP.md**: Future plans
- **docs/**: Detailed documentation

## Version Support

### Supported Versions

- **Latest minor version**: Fully supported
- **Previous minor version**: Security fixes only
- **Older versions**: No longer supported

**Example**:
- Current: 0.2.x (fully supported)
- Previous: 0.1.x (security fixes only)
- Older: < 0.1.x (not supported)

### End-of-Life Policy

Announce end-of-life:
1. **6 months notice** for major versions
2. **3 months notice** for minor versions
3. Update VERSION_UPDATE.md and README.md

## Communication

### Announcing Releases

Communicate releases via:
1. **GitHub Releases**: Primary announcement
2. **README.md**: Link to latest release
3. **Documentation**: Update version badge

### Security Releases

For security fixes:
1. **Immediate release** as patch version
2. **Security advisory** in SECURITY.md
3. **Changelog entry** under Security section
4. **Email notification** (if mailing list exists)

## Best Practices

### Keep It Updated

- Update VERSION_UPDATE.md with each PR (if significant)
- Review changelog before each release
- Archive old versions in changelog

### User-Focused

- Write for users, not developers
- Explain impact of changes
- Provide migration guides
- Link to documentation

### Consistent Format

- Follow the established format
- Use standard categories
- Maintain chronological order
- Include dates

### Review Process

Before release:
- [ ] All changes documented
- [ ] Breaking changes highlighted
- [ ] Migration guide provided (if needed)
- [ ] Version numbers updated everywhere
- [ ] Changelog reviewed and approved

## Examples from Skyulf

### Recent Versions

From `pyproject.toml`:
```toml
version = "0.1.4"
```

From VERSION_UPDATE.md (example entries):
```markdown
## [0.1.5] - 2025-01-XX

### Changed
- Migrated from Flask to FastAPI
- Updated async database support
- Improved error handling

### Fixed
- Fixed issue with large file uploads
- Resolved Celery task timeout

## [0.1.4] - 2024-12-XX

### Added
- Initial FastAPI integration
- Health check endpoints

### Changed
- Configuration management with Pydantic
```

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## Questions?

For questions about versioning or changelogs:
- Review this document
- Check VERSION_UPDATE.md for examples
- Ask in PR comments
- Reference CONTRIBUTING.md
