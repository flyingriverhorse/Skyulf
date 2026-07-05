"""Geospatial feature-engineering nodes package.

Importing this package registers the ``GeoDistance`` and ``H3Index`` nodes
and re-exports their public Calculator/Applier classes.

``GeoDistance`` has no optional dependency (haversine/euclidean distance is
pure math). ``H3Index`` requires the optional ``h3`` package (install via
``pip install skyulf-core[geo]`` or ``pip install h3``); the import is lazy
so the rest of skyulf-core works without it installed.
"""

from .distance import GeoDistanceApplier, GeoDistanceCalculator
from .h3_index import H3IndexApplier, H3IndexCalculator

__all__ = [
    "GeoDistanceApplier",
    "GeoDistanceCalculator",
    "H3IndexApplier",
    "H3IndexCalculator",
]
