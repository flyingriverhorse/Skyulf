"""Pipeline Execution Engine.

The :class:`PipelineEngine` orchestrates execution of pipelines defined by
:class:`PipelineConfig`. The implementation is split across mixins under the
``_execution`` package to keep this module focused on orchestration:

* :class:`._artifacts.ArtifactsMixin` — feature importances, training-artifact
  finalization, reference-data persistence.
* :class:`._merge.MergeMixin` — frame coercion + multi-input merging.
* :class:`._feature_eng.FeatureEngMixin` — composite FeatureEngineer building,
  ``_run_feature_engineering``, and inference-bundle assembly.
* :class:`._node_runners.NodeRunnersMixin` — per-step runners
  (``_run_data_loader``, ``_run_training`` — unified fixed/tuned training,
  ``_run_transformer``, ``_run_data_preview``).

This module owns the public API (``run``), per-node dispatch (``_execute_node``),
upstream-input resolution helpers, and the ``log`` shim.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from skyulf.data.catalog import DataCatalog

from ...artifacts.store import ArtifactStore
from ...constants import StepType
from .._schema_graph import predict_schemas, schemas_to_dict
from ..schemas import (
    NodeConfig,
    NodeExecutionResult,
    PipelineConfig,
    PipelineExecutionResult,
)
from ..summary import build_summary
from ._artifacts import ArtifactsMixin
from ._feature_eng import FeatureEngMixin
from ._merge import MergeMixin
from ._node_runners import NodeRunnersMixin
from ._warning_capture import WarningCaptureHandler

logger = logging.getLogger(__name__)


class PipelineEngine(ArtifactsMixin, MergeMixin, FeatureEngMixin, NodeRunnersMixin):
    """
    Orchestrates the execution of ML pipelines.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        catalog: DataCatalog,
        log_callback=None,
    ):
        self.artifact_store = artifact_store
        self.catalog = catalog
        self.log_callback = log_callback
        self.executed_transformers: list[
            Any
        ] = []  # Track fitted transformers for inference pipeline
        self._results: dict[str, NodeExecutionResult] = {}
        self._node_configs: dict[str, NodeConfig] = {}
        # Engine-emitted advisories surfaced via PipelineExecutionResult.
        # Initialized here (not just in run()) so direct callers of
        # _merge_inputs / _merge_frames in tests don't hit AttributeError.
        self.merge_warnings: list[dict[str, Any]] = []

    def _pipeline_has_training_node(self) -> bool:
        """Checks if the current pipeline workflow includes a model training step."""
        return any(
            node.step_type == StepType.TRAINING for node in self._node_configs.values()
        )

    def _predict_schemas_safe(self, config: PipelineConfig) -> dict[str, dict[str, Any] | None]:
        """C7 Phase B helper: best-effort pre-run schema prediction.

        Errors are swallowed so a broken predictor never blocks a real run.
        """
        try:
            return schemas_to_dict(predict_schemas(config))
        except Exception:  # noqa: BLE001 - best-effort, never block a run
            logger.exception("Schema prediction failed; continuing without predictions")
            return {}

    def log(self, message: str):
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def _init_run_state(
        self, config: PipelineConfig, dataset_name: str
    ) -> tuple[datetime, PipelineExecutionResult]:
        """Reset per-run engine state and build the optimistic result shell."""
        self.dataset_name = dataset_name
        start_time = datetime.now(UTC)
        self.executed_transformers = []  # Reset for new run
        self._node_configs = {n.node_id: n for n in config.nodes}
        self._topo_order = {n.node_id: i for i, n in enumerate(config.nodes)}
        # Per-run merge advisories surfaced to API/UI so the user understands
        # what fan-in semantics were applied (column union, last-wins, etc.).
        self.merge_warnings = []

        self._current_pipeline_id = config.pipeline_id
        pipeline_result = PipelineExecutionResult(
            pipeline_id=config.pipeline_id,
            status="success",  # Optimistic default
            start_time=start_time,
        )
        return start_time, pipeline_result

    def _run_node_loop(
        self, config: PipelineConfig, job_id: str, pipeline_result: PipelineExecutionResult
    ) -> None:
        """Execute each node in order, updating `pipeline_result` in place.

        Captures per-node warnings via `WarningCaptureHandler` and stops at
        the first failed/erroring node, marking the pipeline as failed.
        """
        warn_handler = WarningCaptureHandler().attach()
        try:
            for node in config.nodes:
                warn_handler.set_current_node(node.node_id, node.step_type)
                try:
                    node_result = self._execute_node(node, job_id=job_id)
                    pipeline_result.node_results[node.node_id] = node_result

                    if node_result.status == "failed":
                        # Emit the real error via warning so the capture handler
                        # surfaces it in the UI notification (not just the server log).
                        error_detail = node_result.error or "Unknown error"
                        logger.warning("Node %s: %s", node.node_id, error_detail)
                        pipeline_result.status = "failed"
                        break

                except Exception as e:
                    # Full traceback goes to the server log; captured message gets
                    # the human-readable str(e) so the UI toast is useful.
                    logger.exception("Unexpected error executing node %s", node.node_id)
                    logger.warning("Node %s: %s", node.node_id, e)
                    pipeline_result.node_results[node.node_id] = NodeExecutionResult(
                        node_id=node.node_id, status="failed", error=str(e)
                    )
                    pipeline_result.status = "failed"
                    break
        finally:
            warn_handler.set_current_node(None, None)
            pipeline_result.node_warnings = warn_handler.drain()
            warn_handler.detach()

    def _dedup_merge_warnings(self) -> list[dict[str, Any]]:
        """Collapse repeated merge advisories from multi-pass branch execution.

        A merge node executed in multiple branches/parts (e.g.
        FeatureTargetSplit hit once per parallel branch) re-appends the same
        advisory each pass. Collapse on (node_id, kind, inputs,
        overlap_columns, dropped_columns, part) so the UI shows one row per
        logically distinct merge instead of N copies.
        """
        seen_keys: set = set()
        deduped: list[dict[str, Any]] = []
        for w in self.merge_warnings:
            key = (
                w.get("node_id"),
                w.get("kind"),
                tuple(sorted(w.get("inputs") or ())),
                tuple(w.get("overlap_columns") or ()),
                tuple(w.get("dropped_columns") or ()),
                w.get("part"),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(w)
        return deduped

    def run(
        self, config: PipelineConfig, job_id: str = "unknown", dataset_name: str = "dataset"
    ) -> PipelineExecutionResult:
        """
        Executes the pipeline defined by the configuration.
        """
        self.log(f"Starting pipeline execution: {config.pipeline_id} (Job: {job_id})")
        _, pipeline_result = self._init_run_state(config, dataset_name)

        # C7 Phase B: walk the topology once and ask each Calculator's
        # ``infer_output_schema`` what its output columns/dtypes will be.
        # Pure addition — does not change runtime behaviour. Loaders are
        # opaque without a catalog seed, so downstream entries will be
        # ``None`` until Phase C wires in dataset-catalog seeding.
        pipeline_result.predicted_schemas = self._predict_schemas_safe(config)

        # Capture per-node `logger.warning(...)` calls (e.g. TargetEncoder
        # coercion notices, OneHotEncoder degenerate-category warnings) so
        # the UI can surface them as toasts / a notification panel instead
        # of silently dropping them in the server log. Tagged with the
        # currently-executing node id via `set_current_node`.
        self._run_node_loop(config, job_id, pipeline_result)

        pipeline_result.end_time = datetime.now(UTC)
        pipeline_result.merge_warnings = self._dedup_merge_warnings()
        return pipeline_result

    def _dispatch_feature_engineering(
        self, node: NodeConfig, job_id: str
    ) -> tuple[str | None, dict[str, Any]]:
        """Run a feature-engineering node, falling back to data-loader if misconfigured."""
        metrics: dict[str, Any] = {}
        # Check if it's actually a misconfigured data loader
        if not node.inputs and "dataset_id" in node.params:
            logger.warning(
                f"Node {node.node_id} has step_type='feature_engineering' but looks like a data loader. "
                "Executing as data loader."
            )
            return self._run_data_loader(node, job_id=job_id), metrics
        return self._run_feature_engineering(node)

    def _dispatch_transformer_fallback(
        self, node: NodeConfig, job_id: str
    ) -> tuple[str | None, dict[str, Any]]:
        """Try to run the node as a single generic transformer step."""
        try:
            logger.debug(f"Running as single transformer: {node.step_type}")
            return self._run_transformer(node, job_id=job_id)
        except Exception as e:
            # If it fails or isn't a valid transformer, re-raise
            if "Unknown transformer type" in str(e):
                raise ValueError(f"Unknown step type: {node.step_type}") from e
            raise e

    def _dispatch_node(self, node: NodeConfig, job_id: str) -> tuple[str | None, dict[str, Any]]:
        """Run a single node's step logic and return ``(output_artifact_id, metrics)``.

        Centralizes the per-step-type dispatch (data loader, feature
        engineering, training, tuning, preview, generic transformer) that
        :meth:`_execute_node` used to inline.
        """
        metrics: dict[str, Any] = {}
        if node.step_type == StepType.DATA_LOADER:
            return self._run_data_loader(node, job_id=job_id), metrics
        if node.step_type == StepType.FEATURE_ENGINEERING:
            return self._dispatch_feature_engineering(node, job_id)
        if node.step_type == StepType.TRAINING:
            return self._run_training(node, job_id=job_id)
        if node.step_type == "data_preview":
            return self._run_data_preview(node)

        # Try to run as a single transformer step
        return self._dispatch_transformer_fallback(node, job_id)

    def _build_node_metadata(self, node: NodeConfig, metrics: dict[str, Any]) -> dict[str, Any]:
        """Best-effort assembly of the one-line node-card summary metadata.

        The artifact store already has the freshly-saved output (every
        ``_run_*`` path writes under node_id), so loading it here is cheap and
        keeps summary logic out of the per-runner methods. Every step here
        tolerates failure - a missing summary just means the card falls back
        to its static description.
        """
        metadata: dict[str, Any] = {}
        # Output / upstream loads are best-effort and isolated from
        # the summary call - for trainers and tuners the summary
        # comes purely from `metrics`, so a failed model load (e.g.
        # an artifact bundle that doesn't unpickle cleanly) must not
        # suppress the card line.
        output: Any = None
        try:
            output = self.artifact_store.load(node.node_id)
        except Exception:
            logger.debug("summary: output load skipped for %s", node.node_id, exc_info=True)
        input_shape: tuple[int, int] | None = None
        try:
            if node.inputs:
                upstream = self.artifact_store.load(node.inputs[0])
                if isinstance(upstream, pd.DataFrame):
                    input_shape = upstream.shape
        except Exception:
            input_shape = None
        try:
            summary = build_summary(
                step_type=node.step_type,
                output=output,
                metrics=metrics,
                input_shape=input_shape,
                params=node.params or {},
            )
            if summary:
                metadata["summary"] = summary
        except Exception:
            logger.debug("summary skipped for node %s", node.node_id, exc_info=True)
        return metadata

    def _execute_node(self, node: NodeConfig, job_id: str = "unknown") -> NodeExecutionResult:
        """Executes a single node based on its type."""
        self.log(f"Executing node: {node.node_id} ({node.step_type})")
        start_ts = time.time()

        try:
            output_artifact_id, metrics = self._dispatch_node(node, job_id)
            duration = time.time() - start_ts
            metadata = self._build_node_metadata(node, metrics)

            return NodeExecutionResult(
                node_id=node.node_id,
                status="success",
                output_artifact_id=output_artifact_id,
                metrics=metrics,
                execution_time=duration,
                step_type=node.step_type,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error in node {node.node_id}")
            duration = time.time() - start_ts
            return NodeExecutionResult(
                node_id=node.node_id,
                status="failed",
                error=str(e),
                execution_time=duration,
                step_type=node.step_type,
            )

    def _resolve_input(self, node: NodeConfig, index: int = 0) -> Any:
        """Helper to get input artifact from the previous node."""
        if not node.inputs or index >= len(node.inputs):
            raise ValueError(f"Node {node.node_id} requires input at index {index}")

        input_node_id = node.inputs[index]
        return self.artifact_store.load(input_node_id)

    def _resolve_all_inputs(self, node: NodeConfig) -> list[Any]:
        """Load artifacts from ALL upstream nodes, ordered by topology.

        Duplicate edges from the same source node (which the frontend can emit
        when a connection is wired multiple times) are collapsed to a single
        load to avoid spurious multi-input merging.
        """
        if not node.inputs:
            raise ValueError(f"Node {node.node_id} has no inputs")
        deduped_ids: list[str] = []
        seen: set[str] = set()
        for nid in node.inputs:
            if nid in seen:
                continue
            seen.add(nid)
            deduped_ids.append(nid)
        sorted_ids = sorted(
            deduped_ids,
            key=lambda nid: self._topo_order.get(nid, 0),
        )
        return [self.artifact_store.load(nid) for nid in sorted_ids]

    def _ancestors_of(self, node_id: str) -> set[str]:
        """Return the set of all ancestor node IDs (transitive parents)."""
        ancestors: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            cfg = self._node_configs.get(current)
            if not cfg or not cfg.inputs:
                continue
            for parent in cfg.inputs:
                if parent in ancestors:
                    continue
                ancestors.add(parent)
                stack.append(parent)
        return ancestors

    def _get_input(self, node: NodeConfig, target_col: str = "") -> Any:
        """Resolve a node's data input, merging when more than one edge exists.

        Duplicate edges to the same source collapse to one input, so a node
        wired twice to the same upstream behaves like a single-input node.
        """
        unique_inputs = list(dict.fromkeys(node.inputs or []))
        if len(unique_inputs) > 1:
            return self._merge_inputs(node, target_col)
        return self._resolve_input(node)
