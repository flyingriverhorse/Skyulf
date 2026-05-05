# © 2025 Murat Unsal — Skyulf Project
# Makefile — common shortcuts for developers

.PHONY: start dev frontend-dev backend-dev docker stop install lint typecheck test clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

start: ## Start the API server locally (no Docker)
	python run_skyulf.py

# `make dev` runs backend (uvicorn with reload) and frontend (vite) in
# parallel using a single command. Ctrl-C stops both.
# Requires GNU Make 4.x. On Windows use `make dev` from Git Bash / WSL or
# install make via choco. Pure-PowerShell users can run `dev-win` instead.
dev: ## One-command local dev stack (backend + frontend with HMR)
	@echo "Starting backend (uvicorn) and frontend (vite)..."
	@$(MAKE) -j 2 backend-dev frontend-dev

backend-dev: ## Backend dev server (auto-reload)
	python run_fastapi.py

frontend-dev: ## Frontend Vite dev server (HMR)
	cd frontend/ml-canvas && npm run dev

# PowerShell-native variant: opens two new terminal windows. Use this on
# Windows when GNU Make's parallel mode (-j) doesn't propagate Ctrl-C
# correctly through MinGW.
dev-win: ## (Windows) Open backend + frontend in two PowerShell windows
	powershell -NoExit -Command "python run_fastapi.py"
	powershell -NoExit -Command "cd frontend/ml-canvas; npm run dev"

docker: ## Start the full stack with Docker Compose
	docker compose up --build

stop: ## Stop Docker Compose services
	docker compose down

install: ## Install all Python dependencies in a virtualenv
	pip install --upgrade pip
	pip install -r requirements-fastapi.txt

lint: ## Run flake8 linting (critical errors only)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

typecheck: ## Run ty (Astral) type checking on backend + skyulf-core
	ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py

test: ## Run the test suite
	pytest tests/ -v

clean: ## Remove temp files, caches, and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf temp/processing/* htmlcov/ .coverage
