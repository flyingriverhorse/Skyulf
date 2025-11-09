"""
FastAPI LLM Core Module

Modern async LLM interface for AI/ML functionality.
Migrated from Flask with improved async patterns and FastAPI integration.
"""

from typing import Optional

from fastapi import APIRouter

from .services import AsyncLLMService, get_llm_service
from .context_service import AsyncDataContextService

# Import router separately to avoid circular dependencies during testing
try:
    from .routes import llm_router as _imported_router

    llm_router: Optional[APIRouter] = _imported_router
    _router_available = True
except ImportError:
    llm_router = None
    _router_available = False

__all__ = [
    "AsyncLLMService",
    "get_llm_service",
    "AsyncDataContextService",
]

# Only export router if it's available
if _router_available:
    __all__.append("llm_router")
