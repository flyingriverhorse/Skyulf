"""Middleware package: error handling, request logging and the shared rate limiter.

© 2025 Murat Unsal — Skyulf Project

Nothing is re-exported here — import the submodule you need. Registration order in
``backend.main._add_middleware`` is load-bearing: Starlette's ``add_middleware``
wraps, so the middleware added last is the outermost one.
"""
