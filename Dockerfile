# © 2025 Murat Unsal — Skyulf Project
# Multi-stage Dockerfile with layer caching for fast rebuilds

FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies (cached unless apt list changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached unless requirements change)
COPY requirements-fastapi.txt ./
COPY skyulf-core/ ./skyulf-core/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-fastapi.txt

# Copy application code (changes most often — last layer)
COPY . .

# Create runtime directories
RUN mkdir -p logs uploads/data uploads/models exports/data exports/models exports/pipelines temp/processing

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
