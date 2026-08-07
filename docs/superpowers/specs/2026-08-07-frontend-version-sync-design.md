# Frontend Version Sync Design

## Goal

Synchronize the frontend application version with the root Skyulf application
version so the current branch is merge-ready.

## Source of Truth

The root `pyproject.toml` project version is authoritative for the frontend.
The independently versioned `skyulf-core` package does not determine the
frontend version.

## Change

Run the existing `frontend/ml-canvas` version synchronization script. It reads
the root application version and updates:

- `frontend/ml-canvas/package.json`
- the top-level version in `frontend/ml-canvas/package-lock.json`
- the root package version in the lockfile's `packages` map

For this release, all three frontend values will change from `0.7.3` to
`0.7.4`. No dependency versions or application behavior will change.

## Error Handling

Use the existing script's failures for missing or invalid root version data.
Its check mode must report no drift after synchronization.

## Validation

Run the frontend version drift check, lint, TypeScript type check, production
build, and test suite.
