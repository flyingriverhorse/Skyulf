"""Pipeline orchestration package.

Groups the top-level :class:`SkyulfPipeline` with the concerns split out of it
in F-19:

- ``_pipeline`` — the pipeline orchestrator (fitting, persistence, model card).
- ``seal`` — the semantic reproducibility digest (:func:`artifact_digest`).
- ``diagram`` — Mermaid topology rendering (:func:`build_mermaid_diagram`).

``SkyulfPipeline`` is re-exported here so ``from skyulf.pipeline import
SkyulfPipeline`` keeps working — that import path is the public contract used
by downstream code and the generated notebook exports.
"""

from ._pipeline import SkyulfPipeline

__all__ = ["SkyulfPipeline"]
