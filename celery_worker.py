"""
Celery worker bootstrap.

© 2025 Murat Unsal — Skyulf Project
"""

from __future__ import annotations

from core.feature_engineering.modeling.shared import celery_app
# Ensure tasks register with the shared Celery app
from core.feature_engineering.modeling.training import tasks as _training_tasks  # noqa: F401
from core.feature_engineering.modeling.hyperparameter_tuning import tasks as _tuning_tasks  # noqa: F401

__all__ = ["celery_app"]

if __name__ == "__main__":  # pragma: no cover - manual worker entrypoint
    celery_app.start()
