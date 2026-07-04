# © 2025 Murat Unsal — Skyulf Project
# Runs the full pytest suite in an isolated Docker container. No external
# services required (sqlite + no live Redis). See run_integration_tests.ps1
# for the variant that exercises real Celery/Redis wiring.

$ErrorActionPreference = "Stop"

# Resolve docker-compose.test.yml relative to the repo root regardless of the
# caller's current directory (e.g. running this script from tests/docker/).
Push-Location (Join-Path $PSScriptRoot "../..")
try {
    docker compose -f docker-compose.test.yml run --rm --build unit-tests
}
finally {
    Pop-Location
}
