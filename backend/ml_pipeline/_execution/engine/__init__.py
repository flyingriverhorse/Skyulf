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
  (``_run_data_loader``, ``_run_basic_training``, ``_run_advanced_tuning``,
  ``_run_transformer``, ``_run_data_preview``).

This module owns the public API (``run``), per-node dispatch (``_execute_node``),
upstream-input resolution helpers, and the ``log`` shim.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...artifacts.store import ArtifactStore
from ...constants import StepType
from skyulf.data.catalog import DataCatalog

from ._artifacts import ArtifactsMixin
from ._feature_eng import FeatureEngMixin
from ._merge import MergeMixin
from ._node_runners import NodeRunnersMixin
from ..schemas import (
    NodeConfig,
    NodeExecutionResult,
    PipelineConfig,
    PipelineExecutionResult,
)
from .._schema_graph import predict_schemas, schemas_to_dict
from ..summary import build_summary

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
        self.executed_transformers: List[Any] = (
            []
        )  # Track fitted transformers for inference pipeline
        self._results: Dict[str, NodeExecutionResult] = {}
        self._node_configs: Dict[str, NodeConfig] = {}
        # Engine-emitted advisories surfaced via PipelineExecutionResult.
        # Initialized here (not just in run()) so direct callers of
        # _merge_inputs / _merge_frames in tests don't hit AttributeError.
        self.merge_warnings: List[Dict[str, Any]] = []

    def _pipeline_has_training_node(self) -> bool:
        """Checks if the current pipeline workflow includes a model training step."""
        return any(
            node.step_type in [StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING]
            for node in self._node_configs.values()
        )

    def _predict_schemas_safe(
        self, config: PipelineConfig
    ) -> Dict[str, Optional[Dict[str, Any]]]:
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

    def run(
        self, config: PipelineConfig, job_id: str = "unknown", dataset_name: str = "dataset"
    ) -> PipelineExecutionResult:
        """
        Executes the pipeline defined by the configuration.
        """
        self.log(f"Starting pipeline execution: {config.pipeline_id} (Job: {job_id})")
        self.dataset_name = dataset_name
        start_time = datetime.now()
        self.executed_transformers = []  # Reset for new run
        self._node_configs = {n.node_id: n for n in config.nodes}
        self._topo_order = {n.node_id: i for i, n in enumerate(config.nodes)}
        # Per-run merge advisories surfaced to API/UI so the user understands
        # what fan-in semantics were applied (column union, last-wins, etc.).
        self.merge_warnings: List[Dict[str, Any]] = []

        self._current_pipeline_id = config.pipeline_id
        pipeline_result = PipelineExecutionResult(
            pipeline_id=config.pipeline_id,
            status="success",  # Optimistic default
            start_time=start_time,
        )

        # C7 Phase B: walk the topology once and ask each Calculator's
        # ``infer_output_schema`` what its output columns/dtypes will be.
        # Pure addition — does not change runtime behaviour. Loaders are
        # opaque without a catalog seed, so downstream entries will be
        # ``None`` until Phase C wires in dataset-catalog seeding.
        pipeline_result.predicted_schemas = self._predict_schemas_safe(config)

        for node in config.nodes:
            try:
                node_result = self._execute_node(node, job_id=job_id)
                pipeline_result.node_results[node.node_id] = node_result

                if node_result.status == "failed":
                    logger.error(f"Node {node.node_id} failed. Stopping pipeline.")
                    pipeline_result.status = "failed"
                    break

            except Exception as e:
                logger.exception(f"Unexpected error executing node {node.node_id}")
                pipeline_result.node_results[node.node_id] = NodeExecutionResult(
                    node_id=node.node_id, status="failed", error=str(e)
                )
                pipeline_result.status = "failed"
                break

        pipeline_result.end_time = datetime.now()
        # Dedup advisories: a merge node executed in multiple branches/parts
        # (e.g. FeatureTargetSplit hit once per parallel branch) re-appends
        # the same advisory each pass. Collapse on (node_id, kind, inputs,
        # overlap_columns, dropped_columns, part) so the UI shows one row
        # per logically distinct merge instead of N copies.
        seen_keys: set = set()
        deduped: List[Dict[str, Any]] = []
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
        pipeline_result.merge_warnings = deduped
        return pipeline_result

    def _execute_node(self, node: NodeConfig, job_id: str = "unknown") -> NodeExecutionResult:
        """Executes a single node based on its type."""
        self.log(f"Executing node: {node.node_id} ({node.step_type})")
        start_ts = time.time()

        try:
            output_artifact_id = None
            metrics: Dict[str, Any] = {}

            if node.step_type == StepType.DATA_LOADER:
                output_artifact_id = self._run_data_loader(node, job_id=job_id)
            elif node.step_type == StepType.FEATURE_ENGINEERING:
                # Check if it's actually a misconfigured data loader
                if not node.inputs and "dataset_id" in node.params:
                    logger.warning(
                        f"Node {node.node_id} has step_type='feature_engineering' but looks like a data loader. "
                        "Executing as data loader."
                    )
                    output_artifact_id = self._run_data_loader(node, job_id=job_id)
                else:
                    output_artifact_id, metrics = self._run_feature_engineering(node)
            elif node.step_type == StepType.BASIC_TRAINING:
                output_artifact_id, metrics = self._run_basic_training(node, job_id=job_id)
            elif node.step_type == StepType.ADVANCED_TUNING:
                output_artifact_id, metrics = self._run_advanced_tuning(node, job_id=job_id)
            elif node.step_type == "data_preview":
                output_artifact_id, metrics = self._run_data_preview(node)
            else:
                # Try to run as a single transformer step
                try:
                    logger.debug(f"Running as single transformer: {node.step_type}")
                    output_artifact_id, metrics = self._run_transformer(node, job_id=job_id)
                except Exception as e:
                    # If it fails or isn't a valid transformer, re-raise
                    if "Unknown transformer type" in str(e):
                        raise ValueError(f"Unknown step type: {node.step_type}")
                    raise e

            duration = time.time() - start_ts

            # Build the one-line node-card summary. The artifact store
            # already has the freshly-saved output (every _run_* path
            # writes under node_id), so loading it here is cheap and
            # keeps summary logic out of the per-runner methods. We
            # tolerate any failure - a missing summary just means the
            # card falls back to its static description.
            metadata: Dict[str, Any] = {}
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
            input_shape: Optional[Tuple[int, int]] = None
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

    def _resolve_all_inputs(self, node: NodeConfig) -> List[Any]:
        """Load artifacts from ALL upstream nodes, ordered by topology.

        Duplicate edges from the same source node (which the frontend can emit
        when a connection is wired multiple times) are collapsed to a single
        load to avoid spurious multi-input merging.
        """
        if not node.inputs:
            raise ValueError(f"Node {node.node_id} has no inputs")
        deduped_ids: List[str] = []
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
