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
from skyulf.leakage import (
    data_dependent_transformers,
    is_constant_imputation,
    is_explicit_column_drop,
    is_explicit_hash_encoding,
    is_explicit_missing_indicator,
)
from skyulf.modeling.base import extract_xy
from skyulf.preprocessing.fold_adapter import SPLITTER_STEP_TYPES, FeatureEngineerFoldAdapter
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry

from ...constants import StepType
from ..schemas import NodeConfig

if TYPE_CHECKING:
    from skyulf.modeling.fold_preprocessing import FoldPreprocessor

    from ...artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)


def _step_learns_from_data(step: dict[str, Any]) -> bool:
    """Whether a step's ``fit`` reads statistics from the rows it is given.

    Mirrors the leakage gate's per-step verdict (``skyulf.validate_leakage_safety``):
    the param-aware exemptions for stateless modes come first, then the
    registry-derived learner list. Unknown transformers fail closed — they
    cannot be proven stateless, so they are treated as learners.
    """
    transformer = str(step.get("transformer") or "")
    params = step.get("params") or {}
    if is_explicit_column_drop(transformer, params):
        return False
    if is_constant_imputation(transformer, params):
        return False
    if is_explicit_missing_indicator(transformer, params):
        return False
    if is_explicit_hash_encoding(transformer, params):
        return False
    if transformer in data_dependent_transformers():
        return True
    return transformer not in NodeRegistry.get_all_metadata()


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
        feature_columns: list[str] | None = None,
        feature_dtypes: dict[str, str] | None = None,
        engine: str | None = None,
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
            "feature_columns": feature_columns,
            "feature_dtypes": feature_dtypes,
            "engine": engine,
        }

    def _bundle_transformers_with_model(
        self,
        model_artifact_key: str,
        job_id: str = "unknown",
        feature_engineer_artifact_key: str | None = None,
        feature_engineer_override: Any | None = None,
        target_column: str | None = None,
        dropped_columns: list[str] | None = None,
        feature_columns: list[str] | None = None,
        feature_dtypes: dict[str, str] | None = None,
        engine: str | None = None,
    ):
        """Bundles fitted transformers with the model artifact for inference.

        ``feature_columns`` (when known) is the exact, ordered list of column
        names the model was actually fit on — persisted so the deployment
        service and the manual-prediction UI can validate/build the expected
        input shape precisely, instead of guessing from ``feature_names_in_``
        (unreliable for estimators fit on a bare numpy array, e.g. clustering).

        ``engine`` records the DataFrame engine the model was trained on
        ("pandas"/"polars") so serving can detect engine mismatches instead
        of silently assuming pandas (F-25).
        """
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
                    "feature_columns": feature_columns,
                    "feature_dtypes": feature_dtypes,
                    "engine": engine,
                }
            else:
                # Fallback to old logic if no FeatureEngineer found (e.g. manual steps)
                full_artifact = self._build_legacy_transformer_bundle(
                    model_artifact,
                    job_id,
                    target_column,
                    dropped_columns,
                    feature_columns,
                    feature_dtypes,
                    engine,
                )

            # Save to job_id key if available - this is the final artifact for the job
            if job_id and job_id != "unknown":
                uri = self.artifact_store.get_artifact_uri(job_id)
                self.log(f"Saving bundled artifact to {uri}")
                self.artifact_store.save(job_id, full_artifact)

        except Exception:
            logger.exception("Failed to bundle transformers with model")
            raise

    def _upstream_fe_chain(
        self, training_node: NodeConfig
    ) -> tuple[str, list[tuple[str, list[dict[str, Any]]]]] | None:
        """Follow single-input ancestors from ``training_node`` up to the data loader.

        Returns ``(loader_node_id, chain)`` where ``chain`` is
        ``[(node_id, unfitted_step_configs), ...]`` in execution order
        (feature-engineering node steps plus single-transformer nodes wrapped
        as one-step pipelines, matching ``_run_transformer``). Returns
        ``None`` when the upstream graph is not a linear chain of
        feature-engineering/transformer nodes — merged inputs, a missing
        loader, or an unsupported node type (training/preview in between).
        """
        chain: list[tuple[str, list[dict[str, Any]]]] = []
        if len(dict.fromkeys(training_node.inputs or [])) > 1:
            return None
        current_id = training_node.inputs[0] if training_node.inputs else None
        while current_id is not None:
            cfg = self._node_configs.get(current_id)
            if cfg is None:
                return None
            if len(dict.fromkeys(cfg.inputs or [])) > 1:
                return None
            if cfg.step_type == StepType.DATA_LOADER:
                return current_id, list(reversed(chain))
            if cfg.step_type == StepType.FEATURE_ENGINEERING:
                chain.append((cfg.node_id, list((cfg.params or {}).get("steps", []))))
            elif cfg.step_type not in (StepType.TRAINING, "data_preview"):
                # Single-transformer node, wrapped exactly like _run_transformer does.
                chain.append(
                    (
                        cfg.node_id,
                        [
                            {
                                "name": "step",
                                "transformer": cfg.step_type,
                                "params": cfg.params or {},
                            }
                        ],
                    )
                )
            else:
                return None
            current_id = cfg.inputs[0] if cfg.inputs else None
        return None

    @staticmethod
    def _split_train_payload(output: Any, target_col: str) -> tuple[Any, Any]:
        """Extract the pre-transform ``(X, y)`` train payload from a split output."""
        train = output.train if isinstance(output, SplitDataset) else output
        return extract_xy(train, target_col)

    def _resolve_fold_preprocessing(
        self, training_node: NodeConfig, target_col: str
    ) -> tuple["FoldPreprocessor", tuple[Any, Any]] | None:
        """Resolve the per-fold preprocessing adapter + pre-transform train payload (F-15).

        Returns ``(adapter, (X_pre, y_pre))`` so CV/tuning folds slice the rows
        the adapter was built for, or ``None`` to keep pre-transformed scoring.
        Falls back with an explicit job-log warning when the upstream graph is
        not a linear chain or the payload cannot be reconstructed — never
        fails the run.
        """
        warning = None
        try:
            resolved = self._upstream_fe_chain(training_node)
            if resolved is None:
                warning = (
                    "upstream graph is not a linear chain (merged branches or unsupported nodes)"
                )
                return None
            loader_id, chain = resolved

            flat: list[tuple[dict[str, Any], str, int, int]] = [
                (step, node_id, idx, len(steps))
                for node_id, steps in chain
                for idx, step in enumerate(steps)
            ]
            splitter_positions = [
                i
                for i, (step, *_) in enumerate(flat)
                if step.get("transformer") in SPLITTER_STEP_TYPES
            ]
            if splitter_positions:
                last_split = splitter_positions[-1]
                pre_split_learners = [
                    step for step, *_ in flat[:last_split] if _step_learns_from_data(step)
                ]
                if pre_split_learners:
                    # Reconstructing the pre-transform payload would re-fit these
                    # steps on the full frame (held-out rows included), and the
                    # per-fold adapter would then apply them a second time —
                    # leaky statistics plus double-transformed features.
                    names = ", ".join(
                        sorted({str(step.get("transformer")) for step in pre_split_learners})
                    )
                    warning = (
                        f"data-dependent step(s) before the last splitter ({names}) "
                        "cannot be re-fit safely per fold"
                    )
                    return None
                # Stateless pre-split steps were already applied during payload
                # reconstruction (or by the upstream node whose artifact is
                # loaded below); keep them out of the per-fold chain so every
                # step is applied exactly once.
                learning_steps = [
                    step
                    for step, *_ in flat[last_split + 1 :]
                    if step.get("transformer") not in SPLITTER_STEP_TYPES
                ]
            else:
                # No split at all: every non-splitter step refits per fold.
                learning_steps = [
                    step for step, *_ in flat if step.get("transformer") not in SPLITTER_STEP_TYPES
                ]
            if not learning_steps:
                # Only splitters (and stateless pre-split steps) upstream:
                # nothing data-dependent to refit per fold.
                return None

            if not splitter_positions:
                # No split at all: the raw loader frame IS the train payload.
                payload = self._split_train_payload(self.artifact_store.load(loader_id), target_col)
            else:
                _step, node_id, idx, total = flat[splitter_positions[-1]]
                if idx == total - 1:
                    # The last splitter ends at a node boundary, so its stored
                    # output artifact is the pre-transform SplitDataset itself.
                    payload = self._split_train_payload(
                        self.artifact_store.load(node_id), target_col
                    )
                else:
                    # Splitter + learning steps share one FE node; re-run the
                    # splitter-only step prefix on the raw loader frame to
                    # reconstruct the pre-learning train rows.
                    prefix_steps = [s for s, *_ in flat[: splitter_positions[-1] + 1]]
                    split_output, _metrics = FeatureEngineer(prefix_steps).fit_transform(
                        self.artifact_store.load(loader_id)
                    )
                    payload = self._split_train_payload(split_output, target_col)

            adapter = FeatureEngineerFoldAdapter(learning_steps, target_column=target_col)
            self.log(
                f"Per-fold preprocessing refit enabled: {len(learning_steps)} step(s) "
                "re-fit inside every CV/tuning fold (scores are leakage-free)."
            )
            return adapter, payload
        except Exception:
            logger.exception("Failed to resolve per-fold preprocessing")
            warning = "payload reconstruction failed"
            return None
        finally:
            if warning is not None:
                self.log(
                    f"Per-fold preprocessing refit skipped: {warning}; "
                    "CV/tuning scores may be optimistically biased."
                )

    def _run_feature_engineering(self, node: NodeConfig) -> tuple[str, dict[str, Any]]:
        # Input: DataFrame or SplitDataset (merged when multiple branches feed in).
        df = self._get_input(node)

        # params: {"steps": [...]}
        engineer = FeatureEngineer(node.params.get("steps", []))

        # SDK FeatureEngineer.fit_transform(data) -> (transformed_data, metrics)
        processed_df, metrics = engineer.fit_transform(df)

        # Save the fitted FeatureEngineer itself (holds engineer.fitted_steps state)
        # as this node's artifact, so downstream inference can reload the pipeline.
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
