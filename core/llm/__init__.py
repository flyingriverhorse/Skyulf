"""
FastAPI LLM Core Module

Modern async LLM interface for AI/ML functionality.
Migrated from Flask with improved async patterns and FastAPI integration.
"""

from .services import AsyncLLMService, get_llm_service
from .context_service import AsyncDataContextService  
from .file_context_service import AsyncFileBasedContextService

# Import router separately to avoid circular dependencies during testing
try:
    from .routes import llm_router
    _router_available = True
except ImportError:
    _router_available = False
    llm_router = None

__all__ = [
    "AsyncLLMService",
    "get_llm_service",
    "AsyncDataContextService",
    "AsyncFileBasedContextService",
]

# Only export router if it's available
if _router_available:
    __all__.append("llm_router")