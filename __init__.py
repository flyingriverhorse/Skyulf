"""
© 2025 Murat Unsal — Skyulf Project

FastAPI MLops Application Package

This package contains the FastAPI implementation of the MLops platform,
designed to replace the Flask application with better concurrency support.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("skyulf")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__title__ = "FastAPI MLops Platform"
__description__ = "Modern MLops platform with async support and better performance"
