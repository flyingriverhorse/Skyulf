#!/usr/bin/env bash
# © 2025 Murat Unsal — Skyulf Project
# Runs the pytest suite against a live Redis instance (real Celery
# broker/result-backend), via docker-compose.test.yml.
set -euo pipefail
# Resolve docker-compose.test.yml relative to the repo root regardless of the
# caller's cwd (e.g. running this script from tests/docker/ directly).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Ensure `docker compose down` always runs, even if `up` exits non-zero
# (test failures) — otherwise `set -e` would terminate the script early and
# leave containers/networks running, making subsequent runs flaky.
trap 'docker compose -f docker-compose.test.yml down' EXIT
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit integration-tests
