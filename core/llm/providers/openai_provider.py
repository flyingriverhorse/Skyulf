"""
Async OpenAI Provider for FastAPI

OpenAI API provider with async patterns and FastAPI integration.
"""

import os
import logging
from typing import Dict, List, Any, Optional, cast

from .base import AsyncBaseLLMProvider

logger = logging.getLogger(__name__)


class AsyncOpenAIProvider(AsyncBaseLLMProvider):
    """Async OpenAI API provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = "openai"  # Override auto-generated name
        self.api_key = config.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        # Use configured default model or fallback to hardcoded
        self.default_model = config.get("OPENAI_DEFAULT_MODEL") or os.environ.get(
            "OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo"
        )

    async def is_configured(self) -> bool:
        """Check if OpenAI is properly configured"""
        return bool(self.api_key)

    async def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        ]

    async def get_default_model(self) -> str:
        """Get default OpenAI model"""
        return self.default_model

    async def query(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query OpenAI API with async support"""

        if not await self.is_configured():
            return self._format_error_response(
                "OpenAI not configured. Set OPENAI_API_KEY."
            )

        if not await self.validate_messages(messages):
            return self._format_error_response("Invalid message format")

        try:
            from openai import AsyncOpenAI
        except ImportError:
            return self._format_error_response(
                "OpenAI SDK not installed. Run: pip install openai"
            )

        try:
            client = AsyncOpenAI(api_key=self.api_key)

            params: Dict[str, Any] = {
                "model": model or self.default_model,
                "messages": messages,
                "temperature": 0.7,
                "stream": False,
            }

            if max_tokens:
                params["max_tokens"] = max_tokens

            response = await client.chat.completions.create(**params)
            reply = response.choices[0].message.content

            model_name = cast(str, params["model"])

            return self._format_success_response(
                reply,
                model=model_name,
                usage=response.usage._asdict() if hasattr(response, "usage") else None,
            )

        except Exception as e:
            self.logger.error(f"OpenAI error: {e}")
            return self._format_error_response(f"Failed to contact OpenAI: {str(e)}")

    def _format_success_response(
        self,
        reply: str,
        model: str,
        usage: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Format successful response"""
        return {
            "success": True,
            "response": reply,
            "model": model,
            "provider": self.provider_name,
            "tokens_used": usage.get("total_tokens", 0) if usage else 0,
            "usage": usage
        }

    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """Format error response"""
        return {
            "success": False,
            "error": error_message,
            "provider": self.provider_name
        }

    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": self.provider_name,
            "class_name": self.__class__.__name__,
            "configured": False,  # Will be updated async
            "available_models": [],
            "default_model": self.default_model,
            "description": "OpenAI GPT models via OpenAI API"
        }
