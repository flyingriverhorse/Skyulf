"""
Async LLM Service for FastAPI

Central async service for managing LLM providers.
Migrated from Flask with async patterns and FastAPI integration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .providers import (
    AsyncBaseLLMProvider,
    AsyncOpenAIProvider,
    AsyncDeepSeekProvider,
    AsyncClaudeProvider,
    AsyncLocalProvider
)

logger = logging.getLogger(__name__)


class AsyncLLMService:
    """Central async service for managing LLM providers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.providers: Dict[str, AsyncBaseLLMProvider] = {}
        self.default_provider: Optional[str] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def initialize_providers(self):
        """Initialize all available providers"""
        self.logger.info("Initializing LLM providers...")

        # Initialize real provider implementations
        self.providers = {
            "openai": AsyncOpenAIProvider(self.config),
            "deepseek": AsyncDeepSeekProvider(self.config), 
            "claude": AsyncClaudeProvider(self.config),
            "local": AsyncLocalProvider(self.config),
        }

        # Set default provider to first configured one
        for provider_name, provider in self.providers.items():
            if await provider.is_configured():
                if not self.default_provider:
                    self.default_provider = provider_name
                self.logger.info(f"Provider {provider_name} initialized and configured")
            else:
                self.logger.warning(f"Provider {provider_name} not configured")

        if not self.default_provider:
            self.logger.warning("No LLM providers are configured")

    async def query(
        self,
        messages: List[Dict],
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a query to an LLM provider

        Args:
            messages: List of chat messages
            provider_name: Name of provider to use (uses default if None)
            model: Specific model to use
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response dictionary
        """
        try:
            # Use default provider if none specified
            provider_name = provider_name or self.default_provider

            if not provider_name:
                raise ValueError("No LLM provider available")

            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found")

            provider = self.providers[provider_name]

            # Validate messages format
            if not await provider.validate_messages(messages):
                raise ValueError("Invalid message format")

            # Send query to provider
            response = await provider.query(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                **kwargs
            )

            self.logger.debug(f"LLM query completed with provider {provider_name}")
            return response

        except Exception as e:
            self.logger.error(f"Error in LLM query: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": provider_name
            }

    async def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        available = []
        for provider_name, provider in self.providers.items():
            if await provider.is_configured():
                available.append(provider_name)
        return available

    async def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all providers"""
        info = {}
        for provider_name, provider in self.providers.items():
            provider_info = provider.get_info()
            provider_info["configured"] = await provider.is_configured()
            provider_info["available_models"] = await provider.get_available_models()
            provider_info["default_model"] = await provider.get_default_model()
            info[provider_name] = provider_info
        return info

    async def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Get available models for a specific provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")

        provider = self.providers[provider_name]
        return await provider.get_available_models()

    async def set_default_provider(self, provider_name: str) -> bool:
        """Set the default LLM provider"""
        if provider_name not in self.providers:
            return False

        provider = self.providers[provider_name]
        if not await provider.is_configured():
            return False

        self.default_provider = provider_name
        self.logger.info(f"Default provider set to {provider_name}")
        return True

    async def get_optimal_model_for_task(self, task_type: str, provider: Optional[str] = None) -> Dict[str, str]:
        """
        Get the optimal model for a specific task type
        
        Args:
            task_type: Type of task ('chat', 'code', 'math', 'analysis')
            provider: Preferred provider (optional)
            
        Returns:
            Dictionary with provider and model recommendations
        """
        # Default to current provider or first available
        target_provider = provider or self.default_provider
        if not target_provider or target_provider not in self.providers:
            available = await self.get_available_providers()
            target_provider = available[0] if available else None
        
        if not target_provider:
            return {"provider": None, "model": None, "error": "No providers available"}
        
        provider_obj = self.providers[target_provider]
        default_model = await provider_obj.get_default_model()
        
        # Task-specific model selection
        model_map = {
            "chat": {
                "openai": "gpt-3.5-turbo",
                "deepseek": self.config.get("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
                "claude": self.config.get("CLAUDE_DEFAULT_MODEL", "claude-3-haiku-20240307"),
                "local": self.config.get("LOCAL_LLM_MODEL", "llama2")
            },
            "code": {
                "openai": "gpt-4",  # Better for code
                "deepseek": self.config.get("DEEPSEEK_CODE_MODEL", "deepseek-coder"),
                "claude": "claude-3-5-sonnet-20241022",  # Best Claude for code
                "local": self.config.get("LOCAL_LLM_MODEL", "codellama")
            },
            "math": {
                "openai": "gpt-4",
                "deepseek": self.config.get("DEEPSEEK_MATH_MODEL", "deepseek-math"),
                "claude": "claude-3-opus-20240229",  # Most capable Claude
                "local": self.config.get("LOCAL_LLM_MODEL", "llama2")
            },
            "analysis": {
                "openai": "gpt-4-turbo",  # Good for data analysis
                "deepseek": self.config.get("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
                "claude": "claude-3-5-sonnet-20241022",
                "local": self.config.get("LOCAL_LLM_MODEL", "llama2")
            }
        }
        
        recommended_model = model_map.get(task_type, {}).get(target_provider, default_model)
        
        return {
            "provider": target_provider,
            "model": recommended_model,
            "task_type": task_type
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM service"""
        try:
            available_providers = await self.get_available_providers()
            
            return {
                "healthy": len(available_providers) > 0,
                "available_providers": available_providers,
                "default_provider": self.default_provider,
                "total_providers": len(self.providers)
            }

        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }

# Global service instance (would be properly initialized in FastAPI app)
_llm_service: Optional[AsyncLLMService] = None


async def get_llm_service(config: Optional[Dict[str, Any]] = None) -> AsyncLLMService:
    """Get or create the global LLM service instance"""
    global _llm_service
    
    if _llm_service is None:
        _llm_service = AsyncLLMService(config)
        await _llm_service.initialize_providers()
    
    return _llm_service


async def reset_llm_service():
    """Reset the global LLM service instance (useful for testing)"""
    global _llm_service
    _llm_service = None