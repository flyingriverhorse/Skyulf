# Releasing skyulf-core

## Preconditions

- The target change is merged to `master`.
- The skyulf-core test workflow, static checks, security scan, and documentation build are green.
- The version in `skyulf-core/setup.py` follows semantic versioning and is not already tagged as `core-v<version>`.
- `CHANGELOG.md` and the applicable `changelog/<series>.md` entry describe user-visible changes.

## Local distribution verification

```bash
rm -rf skyulf-core/dist
python -m pip install --upgrade build twine
python -m build --outdir skyulf-core/dist skyulf-core
twine check skyulf-core/dist/*
python .github/scripts/verify_skyulf_core_distribution.py skyulf-core/dist/*.whl
```

## Publishing

Merge the version change to `master`. The `Release skyulf-core` workflow uses PyPI Trusted Publishing, publishes the already-validated distribution, and creates the `core-v<version>` tag. Do not add a PyPI API token to the repository.

## Post-release checks

```bash
python -m venv /tmp/skyulf-core-release-check
/tmp/skyulf-core-release-check/bin/python -m pip install --upgrade pip skyulf-core
/tmp/skyulf-core-release-check/bin/python -c "from skyulf import SkyulfPipeline; print(SkyulfPipeline)"
```
