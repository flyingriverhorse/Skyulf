# Exception handlers for FastAPI application
"""Application error types and the FastAPI handlers that render them as JSON.

``core`` holds the ``SkyulfException`` hierarchy; each class carries the
``status_code`` and ``error_code`` it maps to, so raising one is enough to
determine the response. ``handlers`` turns those — and plain HTTP exceptions —
into one consistent JSON envelope that echoes the ``request_id`` the
error-handling middleware stamped on the request, and persists 5xx failures to
the ``error_events`` table on a best-effort basis.

There are no re-exports here; import from the submodules directly.
"""
