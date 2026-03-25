# © 2025 Murat Unsal — Skyulf Project
# Makefile — common shortcuts for developers

.PHONY: start docker stop install lint typecheck test clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

start: ## Start the API server locally (no Docker)
	python run_skyulf.py

docker: ## Start the full stack with Docker Compose
	docker compose up --build

stop: ## Stop Docker Compose services
	docker compose down

install: ## Install all Python dependencies in a virtualenv
	pip install --upgrade pip
	pip install -r requirements-fastapi.txt

lint: ## Run flake8 linting (critical errors only)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

typecheck: ## Run mypy type checking on backend
	python -m mypy -p backend

test: ## Run the test suite
	pytest tests/ -v

clean: ## Remove temp files, caches, and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf temp/processing/* htmlcov/ .coverage
