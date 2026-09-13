"""Apply SlowAPI defaults using actual routing order and transparent ASGI delivery."""

from collections.abc import Callable
from typing import Any

from fastapi import routing as fastapi_routing
from slowapi import Limiter
from slowapi.middleware import _get_route_name, async_check_limits
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _first_route_handler(app: Any, scope: Scope) -> Callable | None:
    """Match Starlette's first full route, leaving mounted applications to themselves."""
    # Newer FastAPI includes routers lazily; its public iterator supplies the
    # effective prefixes. Older versions already expose a flat route list.
    route_contexts = getattr(fastapi_routing, "iter_route_contexts", iter)
    for route in route_contexts(app.routes):
        if route.matches(scope)[0] == Match.FULL:
            return getattr(route, "endpoint", None)
    return None


def _skip_default_check(limiter: Limiter, handler: Callable | None) -> bool:
    """Let explicit decorators own static/dynamic limits and exemptions."""
    if not limiter.enabled or handler is None:
        return True
    name = _get_route_name(handler)
    return any(
        name in routes
        for routes in (limiter._exempt_routes, limiter._route_limits, limiter._dynamic_route_limits)
    )


class DefaultRateLimitMiddleware:
    """Enforce configured defaults without consuming streaming bodies or WebSockets.

    SlowAPI's bundled middleware keeps the last matching route, which selects
    the SPA catch-all over an earlier API route. Its ASGI responder also emits
    response-start for each body chunk. Keep routing and delivery here while
    retaining SlowAPI's counter, failure-handler and header behavior.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Store the downstream ASGI application without sharing request state."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check HTTP defaults once before reaching the endpoint."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        app = scope["app"]
        limiter = app.state.limiter
        handler = _first_route_handler(app, scope)
        if _skip_default_check(limiter, handler):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive, send=send)
        response, inject_headers = await async_check_limits(limiter, request, handler, app)
        if response is not None:
            await response(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            """Decorate the single response start while forwarding body frames unchanged."""
            if message["type"] == "http.response.start" and inject_headers:
                view_limit = getattr(request.state, "view_rate_limit", None)
                if view_limit is not None:
                    limiter._inject_asgi_headers(MutableHeaders(scope=message), view_limit)
            await send(message)

        await self.app(scope, receive, send_with_headers)
