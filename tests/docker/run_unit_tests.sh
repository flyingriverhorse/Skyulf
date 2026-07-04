#!/usr/bin/env bash
# © 2025 Murat Unsal — Skyulf Project
# Runs the full pytest suite in an isolated Docker container. No external
# services required (sqlite + no live Redis). See run_integration_tests.sh
# for the variant that exercises real Celery/Redis wiring.
set -euo pipefail
docker compose -f docker-compose.test.yml run --rm --build unit-tests
