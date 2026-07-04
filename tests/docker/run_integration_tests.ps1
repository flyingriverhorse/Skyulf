# © 2025 Murat Unsal — Skyulf Project
# Runs the pytest suite against a live Redis instance (real Celery
# broker/result-backend), via docker-compose.test.yml.

$ErrorActionPreference = "Stop"

# Resolve docker-compose.test.yml relative to the repo root regardless of the
# caller's current directory (e.g. running this script from tests/docker/).
Push-Location (Join-Path $PSScriptRoot "../..")
try {
    docker compose -f docker-compose.test.yml up --build --abort-on-container-exit integration-tests
}
finally {
    # Always tear down, even if `up` failed (test failures) — prevents
    # leaving containers/networks running and making subsequent runs flaky.
    docker compose -f docker-compose.test.yml down
    Pop-Location
}
