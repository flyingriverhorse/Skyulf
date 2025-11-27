"""MLops API routes."""

from fastapi import APIRouter

from core.database.engine import get_async_session
from core.feature_engineering.api.analytics import router as analytics_router
from core.feature_engineering.api.catalog import router as catalog_router
from core.feature_engineering.api.datasets import router as datasets_router
from core.feature_engineering.api.pipeline import router as pipeline_router
from core.feature_engineering.api.recommendations import router as recommendations_router
from core.feature_engineering.api.training import router as training_router
from core.feature_engineering.api.tuning import router as tuning_router
from core.feature_engineering.export import export_project_bundle
from core.feature_engineering.split_handler import detect_splits, log_split_processing, remove_split_column

router = APIRouter(prefix="/ml-workflow")

# Mount routers
router.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
router.include_router(catalog_router) # /api/node-catalog
router.include_router(datasets_router) # /api/datasets
router.include_router(pipeline_router, prefix="/api/pipelines", tags=["pipeline"])
router.include_router(recommendations_router, prefix="/api/recommendations", tags=["recommendations"])
router.include_router(training_router, prefix="/api", tags=["training"])
router.include_router(tuning_router, prefix="/api", tags=["tuning"])
