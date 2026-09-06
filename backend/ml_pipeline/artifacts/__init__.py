"""Artifact storage backends for the outputs of a pipeline run.

Re-exports the :class:`ArtifactStore` ABC and its local-filesystem
implementation. The S3 backend, the factory that picks a backend per job and
the root-level discovery seam live in their own modules.
"""

from .local import LocalArtifactStore
from .store import ArtifactStore

__all__ = ["ArtifactStore", "LocalArtifactStore"]
