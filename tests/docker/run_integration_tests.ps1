# © 2025 Murat Unsal — Skyulf Project
# Runs the pytest suite against a live Redis instance (real Celery
# broker/result-backend), via docker-compose.test.yml.

docker compose -f docker-compose.test.yml up --build --abort-on-container-exit integration-tests
docker compose -f docker-compose.test.yml down
