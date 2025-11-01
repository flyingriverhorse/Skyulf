"""
Base LLM Provider for FastAPI

Async base class for all LLM providers with FastAPI patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AsyncBaseLLMProvider(ABC):
    """Base class for all async LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def is_configured(self) -> bool:
        """Check if the provider is properly configured"""
        pass

    @abstractmethod
    async def query(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a query to the LLM provider"""
        pass

    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass

    @abstractmethod
    async def get_default_model(self) -> str:
        """Get the default model for this provider"""
        pass

    async def validate_messages(self, messages: List[Dict]) -> bool:
        """Validate message format"""
        if not isinstance(messages, list) or len(messages) == 0:
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["system", "user", "assistant"]:
                return False

        return True

    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": self.provider_name,
            "class_name": self.__class__.__name__,
            "configured": False,  # Will be updated by subclasses
            "available_models": [],
            "default_model": None,
        }