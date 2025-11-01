"""
FastAPI LLM Routes

Modern async LLM endpoints migrated from Flask.
Provides chat functionality, provider management, and context integration.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field

from config import get_settings
from ..auth.dependencies import get_current_user
from ..auth.auth_core import User
from .services import get_llm_service, AsyncLLMService
from .context_service import AsyncDataContextService
from .file_context_service import AsyncFileBasedContextService

logger = logging.getLogger(__name__)

# Create the LLM router
llm_router = APIRouter(prefix="/llm", tags=["llm"])


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class LLMQueryRequest(BaseModel):
    """LLM query request model"""
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    source_id: Optional[str] = Field(None, description="Data source ID for context")
    include_context: bool = Field(True, description="Whether to include data context")
    page_context: Optional[Dict[str, Any]] = Field(None, description="User's current view context")
    provider: Optional[str] = Field(None, description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    cell_id: Optional[str] = Field(None, description="Target cell identifier for scoped notebook chats")
    cell_scope: Optional[str] = Field(None, description="Cell scope (analysis/custom) when scoping notebook chats")


class LLMQueryResponse(BaseModel):
    """LLM query response model"""
    success: bool
    response: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    details: Optional[str] = None


class SetDefaultProviderRequest(BaseModel):
    """Set default provider request model"""
    provider: str = Field(..., description="Provider name to set as default")


# Initialize context services
data_context_service = AsyncDataContextService()
file_context_service = AsyncFileBasedContextService()


def _truncate_text(text: Optional[str], max_chars: int) -> Tuple[Optional[str], bool]:
    """Trim text to the configured character limit for system prompts."""
    if not text:
        return text, False

    if not max_chars or max_chars <= 0:
        return text, False

    if len(text) <= max_chars:
        return text, False

    truncated = text[:max_chars].rstrip()
    truncated += "\n\n[Context truncated to fit model limits.]"
    return truncated, True


def _limit_message_history(
    messages: List[Dict[str, Any]],
    max_messages: Optional[int],
    max_chars: Optional[int],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Limit conversation history by message count and character length."""
    if not messages:
        return [], False

    trimmed = False
    limited = messages[-max_messages:] if max_messages and max_messages > 0 else list(messages)
    if max_messages and len(messages) > len(limited):
        trimmed = True

    if max_chars and max_chars > 0:
        total_chars = sum(len(msg.get("content", "")) for msg in limited)
        if total_chars > max_chars:
            trimmed = True
            result: List[Dict[str, Any]] = []
            running = 0
            for msg in reversed(limited):
                content = msg.get("content", "")
                available = max_chars - running
                if available <= 0:
                    break

                if len(content) <= available:
                    result.append(dict(msg))
                    running += len(content)
                else:
                    truncated_content = content[-available:]
                    result.append({**msg, "content": f"...(trimmed)\n{truncated_content}"})
                    running += len(truncated_content)
                    break

            limited = list(reversed(result))

            if limited:
                first_msg = limited[0]
                limited[0] = {
                    **first_msg,
                    "content": f"(Earlier conversation truncated)\n{first_msg.get('content', '')}"
                }
        else:
            limited = [dict(msg) for msg in limited]
    else:
        limited = [dict(msg) for msg in limited]

    if not limited and messages:
        fallback_msg = dict(messages[-1])
        content = fallback_msg.get("content", "")
        if max_chars and max_chars > 0 and len(content) > max_chars:
            content = content[-max_chars:]
            content = f"...(trimmed)\n{content}"
        fallback_msg["content"] = content or "(Earlier conversation trimmed)"
        limited = [fallback_msg]
        trimmed = True

    return limited, trimmed


def build_general_analysis_system_prompt(context_data: str) -> str:
    """Create a lightweight system message focused on the supplied context."""
    return (
        f"You are a senior data scientist focused on understanding the dataset. Review the dataset notes below and reply in explaning.\n\n"
        f"DATA CONTEXT:\n{context_data}\n\n"
        "Guidelines:\n"
        "- Mention only observations that are directly visible in the context.\n"
        "- Do Recommendations.\n"
        "- Prepare user for next steps in advanced EDA.\n"
        "- Avoid speculative language, extra headings, or sections like 'Deeper Follow-Up Analyses'.\n"
        "Keep bullets short and practical; include code only if a tiny snippet (<=7 lines) is essential."
    )


EDA_ASSISTANT_PROMPT_TEMPLATE = """You are a senior data scientist guiding an exploratory analysis session for each analysis and must stay grounded in the provided evidence.

EDA SESSION CONTEXT:
{eda_context}

Respond with short bullet points that:
- Recap the most important findings already revealed in this cell.
- Recommend immediate follow-up checks or visualizations that can be executed next.
- Highlight concrete feature adjustments.
- Flag any data issues that need rapid confirmation.

Avoid speculative scenarios, avoid headings like 'Deeper Follow-Up Analyses', and keep the answer concise. Provide code only when a short (<=7 line) snippet is essential to perform the next step."""


async def _execute_llm_request(
    request: LLMQueryRequest,
    http_request: Request,
    current_user: User,
    system_prompt: Optional[str] = None,
    query_type: str = "general"
) -> LLMQueryResponse:
    """Shared execution pipeline for all LLM routes."""
    raw_messages = [msg.dict() for msg in request.messages]

    if not raw_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Missing "messages" in request body'
        )

    settings = get_settings()

    base_message_limit = getattr(settings, "LLM_MAX_HISTORY_MESSAGES", 0)
    base_char_limit = getattr(settings, "LLM_MAX_HISTORY_CHAR_LENGTH", 0)

    if request.cell_id:
        max_messages = getattr(settings, "LLM_CELL_HISTORY_MESSAGES", base_message_limit)
        max_chars = getattr(settings, "LLM_CELL_HISTORY_CHAR_LENGTH", base_char_limit)
        history_scope = "cell"
    else:
        max_messages = getattr(settings, "LLM_USER_HISTORY_MESSAGES", base_message_limit)
        max_chars = getattr(settings, "LLM_USER_HISTORY_CHAR_LENGTH", base_char_limit)
        history_scope = "user"

    trimmed_messages, history_trimmed = _limit_message_history(
        raw_messages,
        max_messages,
        max_chars,
    )

    if history_trimmed:
        logger.info(
            "Trimmed %s LLM history from %s to %s messages (char limit %s)",
            history_scope,
            len(raw_messages),
            len(trimmed_messages),
            max_chars or 0,
        )

    enhanced_messages = list(trimmed_messages)

    truncated_prompt, system_truncated = _truncate_text(
        system_prompt,
        getattr(settings, "LLM_SYSTEM_PROMPT_CHAR_LIMIT", 0),
    )

    if system_truncated:
        logger.info(
            "System prompt truncated from %s to limit %s characters",
            len(system_prompt or ""),
            getattr(settings, "LLM_SYSTEM_PROMPT_CHAR_LIMIT", 0),
        )

    if truncated_prompt:
        enhanced_messages.insert(0, {"role": "system", "content": truncated_prompt})

    llm_service = await get_llm_service(settings.get_llm_config())

    response = await llm_service.query(
        messages=enhanced_messages,
        provider_name=request.provider,
        model=request.model,
        max_tokens=request.max_tokens,
    )

    client_ip = http_request.client.host if http_request.client else "unknown"
    logger.info(
        f"LLM {query_type} query by user:{current_user.username} IP:{client_ip} "
        f"provider:{request.provider or 'default'} source:{request.source_id or 'none'}"
    )

    if response.get("success", False):
        return LLMQueryResponse(
            success=True,
            response=response.get("response"),
            model=response.get("model"),
            provider=response.get("provider"),
            tokens_used=response.get("tokens_used")
        )

    return LLMQueryResponse(
        success=False,
        error=response.get("error", "Unknown error"),
        details=response.get("details")
    )


@llm_router.post("/query", response_model=LLMQueryResponse)
async def query_llm(
    request: LLMQueryRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Enhanced LLM query endpoint that includes data context."""
    try:
        system_prompt = None
        if request.include_context and request.source_id:
            try:
                context_data = await file_context_service.format_data_context(
                    request.source_id,
                    request.page_context
                )
                if context_data and context_data.strip():
                    system_prompt = build_general_analysis_system_prompt(context_data)
            except Exception as e:
                logger.warning(f"Could not gather data context: {e}")

        return await _execute_llm_request(
            request,
            http_request,
            current_user,
            system_prompt=system_prompt,
            query_type="analysis"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in LLM query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@llm_router.post("/eda/query", response_model=LLMQueryResponse)
async def query_eda_llm(
    request: LLMQueryRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """LLM endpoint specialized for exploratory data analysis narratives."""
    try:
        eda_prompt = None
        eda_context: Optional[str] = None

        if request.include_context:
            try:
                eda_context = await file_context_service.format_eda_context(
                    request.source_id,
                    request.page_context,
                    request.cell_id,
                    request.cell_scope
                )
            except Exception as e:
                logger.warning(f"Could not gather EDA context: {e}")

            if eda_context and eda_context.strip():
                eda_prompt = EDA_ASSISTANT_PROMPT_TEMPLATE.format(
                    eda_context=eda_context
                )

        return await _execute_llm_request(
            request,
            http_request,
            current_user,
            system_prompt=eda_prompt,
            query_type="eda"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in EDA LLM query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@llm_router.get("/providers")
async def get_providers(
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get available LLM providers and their info"""
    try:
        settings = get_settings()
        llm_service = await get_llm_service(settings.get_llm_config())
        
        available_providers = await llm_service.get_available_providers()
        provider_info = await llm_service.get_provider_info()
        
        info = {
            "available_providers": available_providers,
            "provider_info": provider_info,
            "default_provider": llm_service.default_provider,
        }
        
        logger.debug(f"Provider info requested by user:{current_user.username}")
        return info
        
    except Exception as e:
        logger.error(f"Error getting providers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


class TaskModelRequest(BaseModel):
    """Request for task-optimized model recommendation"""
    task_type: str = Field(..., description="Task type: chat, code, math, analysis")
    provider: Optional[str] = Field(None, description="Preferred provider (optional)")


@llm_router.post("/recommend-model")
async def recommend_model_for_task(
    request: TaskModelRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get optimal model recommendation for a specific task"""
    try:
        settings = get_settings()
        llm_service = await get_llm_service(settings.get_llm_config())
        
        recommendation = await llm_service.get_optimal_model_for_task(
            task_type=request.task_type,
            provider=request.provider
        )
        
        logger.debug(f"Model recommendation for {request.task_type} requested by user:{current_user.username}")
        return recommendation
        
    except Exception as e:
        logger.error(f"Error getting model recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.get("/providers/{provider_name}/models")
async def get_provider_models(
    provider_name: str,
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get available models for a specific provider"""
    try:
        settings = get_settings()
        llm_service = await get_llm_service(settings.get_llm_config())
        models = await llm_service.get_models_for_provider(provider_name)
        
        logger.debug(f"Models for {provider_name} requested by user:{current_user.username}")
        return {"models": models}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.put("/providers/default")
async def set_default_provider(
    request: SetDefaultProviderRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Set the default LLM provider"""
    try:
        settings = get_settings()
        llm_service = await get_llm_service(settings.get_llm_config())
        success = await llm_service.set_default_provider(request.provider)

        if success:
            logger.info(
                f"Default provider set to {request.provider} by user:{current_user.username}"
            )
            return {"success": True, "default_provider": request.provider}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provider not available or not configured"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting default provider: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.get("/health")
async def health_check(
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Health check for LLM service"""
    try:
        settings = get_settings()
        llm_service = await get_llm_service(settings.get_llm_config())
        health_status = await llm_service.health_check()

        http_status = status.HTTP_200_OK if health_status["healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
        
        logger.debug(f"LLM health check by user:{current_user.username} - status:{health_status['healthy']}")
        
        return health_status

    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return {
            "healthy": False,
            "error": str(e)
        }


@llm_router.get("/context/{source_id}")
async def get_context(
    source_id: str,
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get data context for a specific source (for debugging/testing)"""
    try:
        # Use file context service for consistency with Flask version
        context = await file_context_service.get_data_context(source_id)
        
        logger.debug(f"Context for {source_id} requested by user:{current_user.username}")
        return context
        
    except Exception as e:
        logger.error(f"Error getting context: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.get("/ping")
async def llm_ping(
    http_request: Request,
    current_user: User = Depends(get_current_user)
):
    """Simple ping to verify LLM routes are working"""
    try:
        return {
            "success": True,
            "message": "LLM routes registered and working",
            "timestamp": datetime.now().isoformat(),
            "user": current_user.username
        }
    except Exception as e:
        logger.error(f"LLM ping failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM ping failed"
        )