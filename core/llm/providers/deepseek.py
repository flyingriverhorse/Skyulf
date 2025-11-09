"""
Async DeepSeek Provider for FastAPI

DeepSeek API provider with async patterns and FastAPI integration.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, cast

import aiohttp

from .base import AsyncBaseLLMProvider

logger = logging.getLogger(__name__)


ConfigValue = Union[str, int, float]


class AsyncDeepSeekProvider(AsyncBaseLLMProvider):
    """Async DeepSeek API provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = "deepseek"  # Override auto-generated name
        self.api_url = config.get("DEEPSEEK_API_URL") or os.environ.get(
            "DEEPSEEK_API_URL", "https://api.deepseek.com"
        )
        self.api_key = config.get("DEEPSEEK_API_KEY") or os.environ.get(
            "DEEPSEEK_API_KEY"
        )
        # Use configured default model or fallback to hardcoded
        self.default_model = config.get("DEEPSEEK_DEFAULT_MODEL") or os.environ.get(
            "DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"
        )

        timeout_value = self._coalesce_config_value(
            config.get("DEEPSEEK_TIMEOUT_SECONDS"),
            os.environ.get("DEEPSEEK_TIMEOUT_SECONDS"),
        )
        retries_value = self._coalesce_config_value(
            config.get("DEEPSEEK_MAX_RETRIES"),
            os.environ.get("DEEPSEEK_MAX_RETRIES"),
        )
        backoff_value = self._coalesce_config_value(
            config.get("DEEPSEEK_RETRY_BACKOFF_SECONDS"),
            os.environ.get("DEEPSEEK_RETRY_BACKOFF_SECONDS"),
        )

        timeout_number = self._parse_float(timeout_value)
        self.timeout_seconds = max(10, int(timeout_number)) if timeout_number is not None else 90

        retries_number = self._parse_float(retries_value)
        self.max_retries = max(1, int(retries_number)) if retries_number is not None else 2

        backoff_number = self._parse_float(backoff_value)
        self.retry_backoff_seconds = (
            max(0.5, backoff_number)
            if backoff_number is not None
            else 1.5
        )

    @staticmethod
    def _coalesce_config_value(*values: Any) -> Optional[ConfigValue]:
        for value in values:
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return None

    @staticmethod
    def _parse_float(value: Optional[ConfigValue]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    async def is_configured(self) -> bool:
        """Check if DeepSeek is properly configured"""
        return bool(self.api_url and self.api_key)

    async def get_available_models(self) -> List[str]:
        """Get available DeepSeek models"""
        return [
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-coder-6.7b-base",
            "deepseek-coder-6.7b-instruct",
            "deepseek-coder-1.3b-base",
            "deepseek-coder-1.3b-instruct",
            "deepseek-math",
            "deepseek-reasoner"
        ]

    async def get_default_model(self) -> str:
        """Get default DeepSeek model"""
        return self.default_model

    async def query(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query DeepSeek API with async support and retry logic"""

        if not await self.is_configured():
            return self._format_error_response(
                "DeepSeek not configured. Set DEEPSEEK_API_URL and DEEPSEEK_API_KEY."
            )

        if not await self.validate_messages(messages):
            return self._format_error_response("Invalid message format")

        # Support optional context_data kwarg (if needed for compatibility)
        context_data = kwargs.get("context_data")
        if context_data:
            # Add context to messages if provided
            messages = await self._add_context_to_messages(messages, context_data)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        backoff_cap = 8.0

        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                    ) as response:

                        if response.status == 200:
                            try:
                                data = await response.json()
                                reply = data["choices"][0]["message"]["content"]

                                return self._format_success_response(
                                    reply,
                                    model=cast(str, payload["model"]),
                                    usage=data.get("usage")
                                )
                            except (KeyError, IndexError) as parse_error:
                                self.logger.error(
                                    "DeepSeek response parsing error: %s, Response: %s",
                                    parse_error,
                                    (data if 'data' in locals() else 'N/A'),
                                )
                                return self._format_error_response(
                                    f"DeepSeek returned invalid response format: {str(parse_error)}"
                                )
                        else:
                            try:
                                error_data = await response.json()
                                error_msg = (
                                    error_data.get("error", {}).get("message")
                                    or error_data.get("message")
                                    or f"HTTP {response.status}"
                                )
                                error_type = error_data.get("error", {}).get("type", "unknown_error")
                                self.logger.error(
                                    f"DeepSeek API error [{response.status}]: {error_msg} (type: {error_type})"
                                )
                                return self._format_error_response(
                                    f"DeepSeek API error [{response.status}]: {error_msg}"
                                )
                            except Exception:
                                response_text = await response.text()
                                self.logger.error(
                                    "DeepSeek API error [%s]: Failed to parse error response. Raw: %s",
                                    response.status,
                                    response_text[:500],
                                )
                                return self._format_error_response(
                                    f"DeepSeek API error [{response.status}]: {response_text[:200]}"
                                )

            except asyncio.TimeoutError as timeout_err:
                self.logger.warning(
                    "DeepSeek request timeout after %ss (attempt %s/%s): %s",
                    self.timeout_seconds,
                    attempt,
                    self.max_retries,
                    timeout_err,
                )
                if attempt == self.max_retries:
                    self.logger.error(
                        "DeepSeek request failed after %s attempts due to timeout",
                        self.max_retries,
                    )
                    return self._format_error_response(
                        f"DeepSeek request timed out after {self.timeout_seconds} seconds. "
                        "Try reducing context size or max_tokens, or increase DEEPSEEK_TIMEOUT_SECONDS."
                    )
                await asyncio.sleep(
                    min(self.retry_backoff_seconds * attempt, backoff_cap)
                )
                continue

            except aiohttp.ClientConnectorError as conn_err:
                self.logger.warning(
                    "DeepSeek connection error on attempt %s/%s: %s",
                    attempt,
                    self.max_retries,
                    conn_err,
                )
                if attempt == self.max_retries:
                    self.logger.error(
                        "DeepSeek connection failed after %s attempts: %s",
                        self.max_retries,
                        conn_err,
                    )
                    return self._format_error_response(
                        f"Failed to connect to DeepSeek API after {self.max_retries} attempts: {str(conn_err)}"
                    )
                await asyncio.sleep(
                    min(self.retry_backoff_seconds * attempt, backoff_cap)
                )
                continue

            except aiohttp.ClientError as client_err:
                self.logger.error(f"DeepSeek client error: {client_err}")
                return self._format_error_response(f"DeepSeek client error: {str(client_err)}")

            except Exception as e:
                self.logger.error(
                    f"DeepSeek unexpected error: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                return self._format_error_response(
                    f"DeepSeek request failed: {type(e).__name__}: {str(e)}"
                )

        return self._format_error_response("DeepSeek request failed after retries.")

    async def _add_context_to_messages(self, messages: List[Dict], context_data: str) -> List[Dict]:
        """Add context data to messages (for compatibility)"""
        # Add system message with context if not already present
        if not messages or messages[0].get("role") != "system":
            system_message = {
                "role": "system",
                "content": f"Use the following context to help answer questions:\n\n{context_data}"
            }
            messages = [system_message] + messages

        return messages

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to DeepSeek API"""
        if not await self.is_configured():
            return {"connected": False, "error": "Not configured"}

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
            "description": "DeepSeek AI models via DeepSeek API"
        }
