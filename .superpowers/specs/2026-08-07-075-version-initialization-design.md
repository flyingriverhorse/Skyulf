# 0.7.5 Version Initialization Design

## Goal

Initialize branch `075` for application version `0.7.5` and `skyulf-core`
version `0.5.6` without adding release features or substantive changelog
content.

## Version Sources

- Set the root application version in `pyproject.toml` to `0.7.5`.
- Set the core package version in `skyulf-core/setup.py` to `0.5.6`.
- Use `frontend/ml-canvas/scripts/sync-version.mjs` to synchronize
  `frontend/ml-canvas/package.json` and `package-lock.json` to `0.7.5`.
- Refresh `uv.lock` so the workspace package metadata records `0.7.5`, while
  retaining existing third-party dependency versions.
- Add an empty `## v0.7.5` section at the top of `changelog/0.7.x.md`.

## Validation

- Confirm the frontend version-sync check passes.
- Confirm `skyulf-core/setup.py --version` reports `0.5.6`.
- Confirm `uv lock --check` passes and the lock diff contains no unrelated
  dependency upgrades.
- Confirm `changelog/0.7.x.md` starts with the empty `v0.7.5` section.
- Run repository static checks affected by the edited Python metadata.

## Delivery

Commit the version initialization on branch `075` with a DCO sign-off and the
required Copilot co-author trailer.
