# © 2025 Murat Unsal — Skyulf Project
# Runs skyulf-core's pytest suite in complete isolation from the rest of the
# monorepo (no backend/ dependencies). Run from the repo root or from
# skyulf-core/ — this script cd's into skyulf-core/ itself.

Push-Location "$PSScriptRoot\..\.."
try {
    docker compose -f docker-compose.test.yml run --rm --build unit-tests
} finally {
    Pop-Location
}
