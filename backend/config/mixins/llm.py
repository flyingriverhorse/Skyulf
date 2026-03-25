"""LLM (Large Language Model) provider settings."""

from typing import Any, Dict, Optional


class LLMMixin:
    """OpenAI, DeepSeek, Anthropic, and local LLM configuration."""

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_ORG_ID: Optional[str] = None
    OPENAI_DEFAULT_MODEL: str = "gpt-4"

    # DeepSeek
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"
    DEEPSEEK_DEFAULT_MODEL: str = "deepseek-chat"
    DEEPSEEK_CODE_MODEL: str = "deepseek-coder"
    DEEPSEEK_MATH_MODEL: str = "deepseek-math"
    DEEPSEEK_TIMEOUT_SECONDS: int = 60
    DEEPSEEK_MAX_RETRIES: int = 3
    DEEPSEEK_RETRY_BACKOFF_SECONDS: int = 2

    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_DEFAULT_MODEL: str = "claude-3-opus-20240229"

    # Local LLM (e.g. Ollama)
    LOCAL_LLM_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "llama3"
    LOCAL_LLM_TYPE: str = "ollama"

    # Defaults
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_LLM_MODEL: str = "gpt-4"
    LLM_MAX_HISTORY_MESSAGES: int = 10
    LLM_MAX_HISTORY_CHAR_LENGTH: int = 4000
    LLM_SYSTEM_PROMPT_CHAR_LIMIT: int = 2000
    LLM_USER_HISTORY_MESSAGES: int = 5
    LLM_USER_HISTORY_CHAR_LENGTH: int = 1000
    LLM_CELL_HISTORY_MESSAGES: int = 5
    LLM_CELL_HISTORY_CHAR_LENGTH: int = 1000

    def get_llm_config(self) -> Dict[str, Any]:
        """Package LLM configuration as a dictionary for the LLM service."""
        return {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,  # type: ignore[attr-defined]
            "OPENAI_ORG_ID": self.OPENAI_ORG_ID,  # type: ignore[attr-defined]
            "OPENAI_DEFAULT_MODEL": self.OPENAI_DEFAULT_MODEL,  # type: ignore[attr-defined]
            "DEEPSEEK_API_KEY": self.DEEPSEEK_API_KEY,  # type: ignore[attr-defined]
            "DEEPSEEK_API_URL": self.DEEPSEEK_API_URL,  # type: ignore[attr-defined]
            "DEEPSEEK_DEFAULT_MODEL": self.DEEPSEEK_DEFAULT_MODEL,  # type: ignore[attr-defined]
            "DEEPSEEK_CODE_MODEL": self.DEEPSEEK_CODE_MODEL,  # type: ignore[attr-defined]
            "DEEPSEEK_MATH_MODEL": self.DEEPSEEK_MATH_MODEL,  # type: ignore[attr-defined]
            "DEEPSEEK_TIMEOUT_SECONDS": self.DEEPSEEK_TIMEOUT_SECONDS,  # type: ignore[attr-defined]
            "DEEPSEEK_MAX_RETRIES": self.DEEPSEEK_MAX_RETRIES,  # type: ignore[attr-defined]
            "DEEPSEEK_RETRY_BACKOFF_SECONDS": self.DEEPSEEK_RETRY_BACKOFF_SECONDS,  # type: ignore[attr-defined]
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,  # type: ignore[attr-defined]
            "CLAUDE_DEFAULT_MODEL": self.CLAUDE_DEFAULT_MODEL,  # type: ignore[attr-defined]
            "LOCAL_LLM_URL": self.LOCAL_LLM_URL,  # type: ignore[attr-defined]
            "LOCAL_LLM_MODEL": self.LOCAL_LLM_MODEL,  # type: ignore[attr-defined]
            "LOCAL_LLM_TYPE": self.LOCAL_LLM_TYPE,  # type: ignore[attr-defined]
            "DEFAULT_LLM_PROVIDER": self.DEFAULT_LLM_PROVIDER,  # type: ignore[attr-defined]
            "DEFAULT_LLM_MODEL": self.DEFAULT_LLM_MODEL,  # type: ignore[attr-defined]
            "LLM_MAX_HISTORY_MESSAGES": self.LLM_MAX_HISTORY_MESSAGES,  # type: ignore[attr-defined]
            "LLM_MAX_HISTORY_CHAR_LENGTH": self.LLM_MAX_HISTORY_CHAR_LENGTH,  # type: ignore[attr-defined]
            "LLM_SYSTEM_PROMPT_CHAR_LIMIT": self.LLM_SYSTEM_PROMPT_CHAR_LIMIT,  # type: ignore[attr-defined]
            "LLM_USER_HISTORY_MESSAGES": self.LLM_USER_HISTORY_MESSAGES,  # type: ignore[attr-defined]
            "LLM_USER_HISTORY_CHAR_LENGTH": self.LLM_USER_HISTORY_CHAR_LENGTH,  # type: ignore[attr-defined]
            "LLM_CELL_HISTORY_MESSAGES": self.LLM_CELL_HISTORY_MESSAGES,  # type: ignore[attr-defined]
            "LLM_CELL_HISTORY_CHAR_LENGTH": self.LLM_CELL_HISTORY_CHAR_LENGTH,  # type: ignore[attr-defined]
        }
