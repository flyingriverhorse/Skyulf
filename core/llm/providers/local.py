"""
Async Local Provider for FastAPI

Local LLM provider (Ollama, LM Studio, etc.) with async patterns and FastAPI integration.
"""

import os
import logging
from typing import Dict, List, Any, Optional

import aiohttp

from .base import AsyncBaseLLMProvider

logger = logging.getLogger(__name__)


class AsyncLocalProvider(AsyncBaseLLMProvider):
    """Async Local LLM provider (Ollama, LM Studio, etc.)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = "local"  # Override auto-generated name
        self.api_url = config.get("LOCAL_LLM_URL") or os.environ.get(
            "LOCAL_LLM_URL", "http://localhost:11434"
        )
        self.default_model = config.get("LOCAL_LLM_MODEL") or os.environ.get(
            "LOCAL_LLM_MODEL", "llama2"
        )
        self.api_type = config.get("LOCAL_LLM_TYPE") or os.environ.get(
            "LOCAL_LLM_TYPE", "ollama"
        )  # ollama, lmstudio, textgen

    async def is_configured(self) -> bool:
        """Check if local LLM is available"""
        try:
            endpoint = (
                f"{self.api_url}/api/tags"
                if self.api_type == "ollama"
                else self.api_url
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200

        except Exception:
            return False

    async def get_available_models(self) -> List[str]:
        """Get available local models"""
        try:
            if self.api_type == "ollama":
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.api_url}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return [model["name"] for model in data.get("models", [])]

            # For other local types, return default
            return [self.default_model]

        except Exception:
            return [self.default_model]

    async def get_default_model(self) -> str:
        """Get default local model"""
        return self.default_model

    async def query(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query local LLM with async support"""

        if not await self.is_configured():
            return self._format_error_response(
                f"Local LLM not available at {self.api_url}. Check if the service is running."
            )

        if not await self.validate_messages(messages):
            return self._format_error_response("Invalid message format")

        target_model = model or self.default_model

        try:
            if self.api_type == "ollama":
                return await self._query_ollama(messages, target_model, max_tokens, **kwargs)
            elif self.api_type == "lmstudio":
                return await self._query_lmstudio(messages, target_model, max_tokens, **kwargs)
            else:
                return self._format_error_response(f"Unsupported local LLM type: {self.api_type}")

        except Exception as e:
            self.logger.error(f"Local LLM error: {e}")
            return self._format_error_response(f"Local LLM request failed: {str(e)}")

    async def _query_ollama(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Query Ollama API"""

        # Convert messages to Ollama format
        prompt = await self._messages_to_prompt(messages)

        options: Dict[str, Any] = {
            "temperature": kwargs.get("temperature", 0.7),
        }

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        if max_tokens:
            options["num_predict"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)  # Longer timeout for local models
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    reply = data.get("response", "")

                    return self._format_success_response(
                        reply,
                        model=model,
                        usage=None  # Ollama doesn't provide token usage in this format
                    )
                else:
                    error_data = await response.text()
                    return self._format_error_response(f"Ollama error: {error_data}")

    async def _query_lmstudio(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Query LM Studio API (OpenAI-compatible)"""

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    reply = data["choices"][0]["message"]["content"]

                    return self._format_success_response(
                        reply,
                        model=model,
                        usage=data.get("usage")
                    )
                else:
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                    return self._format_error_response(f"LM Studio error: {error_msg}")

    async def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to a single prompt for models that don't support chat format"""
        prompt_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to local LLM"""
        if not await self.is_configured():
            return {"connected": False, "error": "Service not available"}

        try:
            # Simple test query
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.query(test_messages, max_tokens=10)

            return {
                "connected": response.get("success", False),
                "error": response.get("error") if not response.get("success") else None
            }

        except Exception as e:
            return {"connected": False, "error": str(e)}

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
            "description": f"Local LLM via {self.api_type} at {self.api_url}"
        }
