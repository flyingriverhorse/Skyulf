"""
Async Claude Provider for FastAPI

Anthropic Claude API provider with async patterns and FastAPI integration.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from .base import AsyncBaseLLMProvider

logger = logging.getLogger(__name__)


class AsyncClaudeProvider(AsyncBaseLLMProvider):
    """Async Anthropic Claude API provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = "claude"  # Override auto-generated name
        self.api_key = config.get("ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        # Use configured default model or fallback to hardcoded
        self.default_model = config.get("CLAUDE_DEFAULT_MODEL") or os.environ.get(
            "CLAUDE_DEFAULT_MODEL", "claude-3-haiku-20240307"
        )

    async def is_configured(self) -> bool:
        """Check if Claude is properly configured"""
        return bool(self.api_key)

    async def get_available_models(self) -> List[str]:
        """Get available Claude models"""
        return [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
        ]

    async def get_default_model(self) -> str:
        """Get default Claude model"""
        return self.default_model

    async def query(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query Claude API with async support"""

        if not await self.is_configured():
            return self._format_error_response(
                "Claude not configured. Set ANTHROPIC_API_KEY."
            )

        if not await self.validate_messages(messages):
            return self._format_error_response("Invalid message format")

        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            return self._format_error_response(
                "Anthropic SDK not installed. Run: pip install anthropic"
            )

        try:
            client = AsyncAnthropic(api_key=self.api_key)

            # Claude expects system messages to be separate
            system_content = None
            claude_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    if system_content is None:
                        system_content = msg["content"]
                    else:
                        system_content += f"\n\n{msg['content']}"
                else:
                    claude_messages.append(msg)

            params = {
                "model": model or self.default_model,
                "messages": claude_messages,
                "max_tokens": max_tokens or 1024,
                "temperature": kwargs.get("temperature", 0.7),
            }

            if system_content:
                params["system"] = system_content

            response = await client.messages.create(**params)
            reply = response.content[0].text

            return self._format_success_response(
                reply,
                model=params["model"],
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                } if hasattr(response, "usage") else None
            )

        except Exception as e:
            self.logger.error(f"Claude error: {e}")
            return self._format_error_response(f"Failed to contact Claude: {str(e)}")

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Claude API"""
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
            "description": "Anthropic Claude models via Anthropic API"
        }