"""Apply configured HTTP security headers without buffering response bodies."""

from collections.abc import Mapping

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityHeadersMiddleware:
    """Attach the configured policy to HTTP responses; pass other ASGI scopes through."""

    def __init__(self, app: ASGIApp, headers: Mapping[str, str]) -> None:
        """Copy the startup policy so individual responses cannot change it."""
        self.app = app
        self.headers = dict(headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Add headers when an HTTP response starts, preserving streaming and WebSockets."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            """Apply the server policy before forwarding response headers."""
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).update(self.headers)
            await send(message)

        await self.app(scope, receive, send_with_headers)
