#!/usr/bin/env bash
# © 2025 Murat Unsal — Skyulf Project
# Runs skyulf-core's pytest suite in complete isolation from the rest of the
# monorepo (no backend/ dependencies). Run from the repo root or from
# skyulf-core/ — this script cd's into skyulf-core/ itself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
docker compose -f docker-compose.test.yml run --rm --build unit-tests
