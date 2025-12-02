from fastapi import APIRouter
from .cleaning import router as cleaning_router
from .encoding import router as encoding_router
from .statistics import router as statistics_router
from .discretization import router as discretization_router

router = APIRouter()
router.include_router(cleaning_router)
router.include_router(encoding_router)
router.include_router(statistics_router)
router.include_router(discretization_router)
