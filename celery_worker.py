"""
Celery worker bootstrap.

© 2025 Murat Unsal — Skyulf Project
"""

from __future__ import annotations

from core.feature_engineering.modeling.model_training_tasks import celery_app
# Ensure tuning tasks register with the shared Celery app
from core.feature_engineering.modeling import hyperparameter_tuning_tasks as _tuning_tasks  # noqa: F401

__all__ = ["celery_app"]

if __name__ == "__main__":  # pragma: no cover - manual worker entrypoint
    celery_app.start()
