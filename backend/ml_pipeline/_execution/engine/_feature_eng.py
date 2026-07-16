"""Feature-engineering composition + bundling for :class:`PipelineEngine`.

Mixin slice — owns: locating per-node FeatureEngineer artifacts, building
a composite FeatureEngineer that spans the upstream pipeline graph, the
``_run_feature_engineering`` step runner, and bundling fitted transformers
with the trained model into a single inference artifact.

Relies on ``self.artifact_store``, ``self._node_configs``, ``self._get_input``,
``self.executed_transformers``, and ``self.log`` from :class:`PipelineEngine`.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.pipeline import FeatureEngineer

from ..schemas import NodeConfig

if TYPE_CHECKING:
    from ...artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)


class FeatureEngMixin:
    """Feature-engineer composition + model bundling helpers."""

    # Type-only stubs so ty can resolve attributes/methods provided by
    # :class:`PipelineEngine` (or its sibling mixins). No runtime impact.
    artifact_store: "ArtifactStore"
    _node_configs: dict[str, NodeConfig]
    executed_transformers: list[dict[str, Any]]
    log: Callable[[str], None]
    _get_input: Any

    def _resolve_feature_engineer_artifact_key(self, node: NodeConfig) -> str | None:
        if not node.inputs:
            return None

        for input_node_id in node.inputs:
            candidate = f"{input_node_id}_pipeline"
            if self.artifact_store.exists(candidate):
                return candidate

            candidate = f"exec_{input_node_id}_pipeline"
            if self.artifact_store.exists(candidate):
                return candidate

        return None

    def _collect_feature_engineer_artifact_keys(self, node_id: str, visited: set[str]) -> list[str]:
        if node_id in visited:
            return []
        visited.add(node_id)

        keys: list[str] = []
        node_cfg = self._node_configs.get(node_id)
        if node_cfg and node_cfg.inputs:
            for upstream_id in node_cfg.inputs:
                keys.extend(self._collect_feature_engineer_artifact_keys(upstream_id, visited))

        # Prefer the execution-time pipeline artifact if present.
        for candidate in (f"exec_{node_id}_pipeline", f"{node_id}_pipeline"):
            if self.artifact_store.exists(candidate):
                keys.append(candidate)
                break

        return keys

    def _merge_fitted_steps(self, artifact_keys: list[str]) -> list[dict[str, Any]]:
        """Load each artifact key's FeatureEngineer and concatenate their fitted steps in order."""
        merged_steps: list[dict[str, Any]] = []
        for key in artifact_keys:
            try:
                fe = self.artifact_store.load(key)
            except Exception as e:
                logger.debug(f"Failed to load pipeline artifact {key}: {e}")
                continue

            fitted_steps = getattr(fe, "fitted_steps", None)
            if isinstance(fitted_steps, list) and fitted_steps:
                merged_steps.extend(fitted_steps)
        return merged_steps

    def _build_composite_feature_engineer(self, node: NodeConfig) -> FeatureEngineer | None:
        """Build a single, ordered FeatureEngineer from all upstream pipeline artifacts.

        Some pipelines are represented as multiple transformer nodes (e.g., encoding -> scaling).
        Each node saves its own FeatureEngineer artifact with only its fitted step(s).
        For inference and label decoding we need a single FeatureEngineer that contains the
        full chain in the correct order.
        """

        if not node.inputs:
            return None

        visited: set[str] = set()
        artifact_keys: list[str] = []
        for input_node_id in node.inputs:
            artifact_keys.extend(
                self._collect_feature_engineer_artifact_keys(input_node_id, visited)
            )

        if not artifact_keys:
            return None

        merged_steps = self._merge_fitted_steps(artifact_keys)

        if not merged_steps:
            return None

        composite = FeatureEngineer([])
        composite.fitted_steps = merged_steps
        return composite

    def _resolve_bundle_feature_engineer(
        self,
        feature_engineer_override: Any | None,
        feature_engineer_artifact_key: str | None,
    ) -> Any | None:
        """Resolve the FeatureEngineer to bundle with the model, if any.

        Prefers an explicit override, then falls back to loading an explicit
        FeatureEngineer artifact key (derived from the pipeline graph) rather
        than scanning the whole artifacts directory, since scanning can pick
        a pipeline from a different run and cause incorrect transforms and
        label decoding.
        """
        feature_engineer = None

        if feature_engineer_override is not None and hasattr(
            feature_engineer_override, "transform"
        ):
            feature_engineer = feature_engineer_override

        if feature_engineer_artifact_key:
            try:
                obj = self.artifact_store.load(feature_engineer_artifact_key)
                if hasattr(obj, "transform"):
                    feature_engineer = obj
            except Exception as e:
                logger.warning(
                    f"Failed to load feature engineer artifact {feature_engineer_artifact_key}: {e}"
                )

        return feature_engineer

    def _build_legacy_transformer_bundle(
        self,
        model_artifact: Any,
        job_id: str,
        target_column: str | None,
        dropped_columns: list[str] | None,
    ) -> dict[str, Any]:
        """Fallback bundle assembled from ``self.executed_transformers`` (manual steps).

        Used when no FeatureEngineer artifact was found for the run.
        """
        transformers = []
        transformer_plan = []

        for t_info in self.executed_transformers:
            try:
                fitted_t = self.artifact_store.load(t_info["artifact_key"])
                if fitted_t:
                    transformers.append(
                        {
                            "node_id": t_info["node_id"],
                            "transformer_name": t_info["transformer_name"],
                            "column_name": t_info["column_name"],
                            "transformer": fitted_t,
                        }
                    )
                    transformer_plan.append(
                        {
                            "node_id": t_info["node_id"],
                            "transformer_name": t_info["transformer_name"],
                            "column_name": t_info["column_name"],
                            "transformer_type": t_info["transformer_type"],
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load transformer artifact {t_info['artifact_key']}: {e}")

        return {
            "model": model_artifact,
            "transformers": transformers,
            "transformer_plan": transformer_plan,
            "job_id": job_id,
            "target_column": target_column,
            "dropped_columns": dropped_columns or [],
        }

    def _bundle_transformers_with_model(
        self,
        model_artifact_key: str,
        job_id: str = "unknown",
        feature_engineer_artifact_key: str | None = None,
        feature_engineer_override: Any | None = None,
        target_column: str | None = None,
        dropped_columns: list[str] | None = None,
    ):
        """Bundles fitted transformers with the model artifact for inference."""
        try:
            model_artifact = self.artifact_store.load(model_artifact_key)

            # Handle tuple artifacts from tuning: (model, metadata/tuning_result)
            if isinstance(model_artifact, tuple) and len(model_artifact) >= 1:
                model_artifact = model_artifact[0]

            # Collect fitted transformer objects
            # In the new SDK, the FeatureEngineer object contains all steps.
            # We should look for the FeatureEngineer artifact.
            feature_engineer = self._resolve_bundle_feature_engineer(
                feature_engineer_override, feature_engineer_artifact_key
            )

            if feature_engineer:
                # Create the new bundle format
                full_artifact = {
                    "model": model_artifact,
                    "feature_engineer": feature_engineer,
                    "job_id": job_id,
                    "target_column": target_column,
                    "dropped_columns": dropped_columns or [],
                }
            else:
                # Fallback to old logic if no FeatureEngineer found (e.g. manual steps)
                full_artifact = self._build_legacy_transformer_bundle(
                    model_artifact, job_id, target_column, dropped_columns
                )

            # Save to job_id key if available - this is the final artifact for the job
            if job_id and job_id != "unknown":
                uri = self.artifact_store.get_artifact_uri(job_id)
                self.log(f"Saving bundled artifact to {uri}")
                self.artifact_store.save(job_id, full_artifact)

        except Exception:
            logger.exception("Failed to bundle transformers with model")
            raise

    def _run_feature_engineering(self, node: NodeConfig) -> tuple[str, dict[str, Any]]:
        # Input: DataFrame or SplitDataset (merged when multiple branches feed in).
        df = self._get_input(node)

        # params: {"steps": [...]}
        engineer = FeatureEngineer(node.params.get("steps", []))

        # SDK FeatureEngineer.fit_transform(data) -> (transformed_data, metrics)
        processed_df, metrics = engineer.fit_transform(df)

        # Manually save state if needed.
        # The SDK keeps state in engineer.fitted_steps.
        # For now, we save the whole engineer object as the artifact for this node?
        # Or we iterate and save individual steps if the artifact store expects that.
        # The original code passed artifact_store to fit_transform, implying internal saving.
        # We will save the engineer object itself to preserve the pipeline state.
        self.artifact_store.save(f"{node.node_id}_pipeline", engineer)

        if hasattr(processed_df, "shape"):
            self.log(f"Feature engineering completed. Output shape: {processed_df.shape}")
        elif isinstance(processed_df, SplitDataset):
            self.log("Feature engineering completed. SplitDataset created.")

        if isinstance(processed_df, tuple):
            # SplitDataset
            train_part = processed_df[0]
            test_part = processed_df[1] if len(processed_df) > 1 else None
            train_shape = getattr(train_part, "shape", None)
            test_shape = getattr(test_part, "shape", None) if test_part is not None else None
            self.log(f"Split details - Train: {train_shape}, Test: {test_shape or 'None'}")

        self.artifact_store.save(node.node_id, processed_df)

        # Track executed transformers
        for step in node.params.get("steps", []):
            self.executed_transformers.append(
                {
                    "node_id": node.node_id,
                    "transformer_name": step["name"],
                    "transformer_type": step["transformer"],
                    # This key might need adjustment if we save the whole engineer
                    "artifact_key": f"{node.node_id}_{step['name']}",
                    "column_name": step.get("params", {}).get("new_column"),
                }
            )

        return node.node_id, metrics
