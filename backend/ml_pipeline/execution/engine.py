"""Pipeline Execution Engine."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Phase 1: Data Loading
# from ..data.loader import DataLoader
from skyulf.data.dataset import SplitDataset
from skyulf.data.catalog import DataCatalog
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
)
from skyulf.modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
    RidgeRegressionApplier,
    RidgeRegressionCalculator,
)
from skyulf.modeling.tuning.tuner import TunerApplier, TunerCalculator
from skyulf.preprocessing.pipeline import FeatureEngineer

from ..artifacts.store import ArtifactStore
from .schemas import (
    NodeConfig,
    NodeExecutionResult,
    PipelineConfig,
    PipelineExecutionResult,
)


# Phase 2: Feature Engineering

# Phase 3: Modeling

logger = logging.getLogger(__name__)


class PipelineEngine:
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

    def log(self, message: str):
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def run(
        self, config: PipelineConfig, job_id: str = "unknown"
    ) -> PipelineExecutionResult:
        """
        Executes the pipeline defined by the configuration.
        """
        self.log(f"Starting pipeline execution: {config.pipeline_id} (Job: {job_id})")
        start_time = datetime.now()
        self.executed_transformers = []  # Reset for new run
        self._node_configs = {n.node_id: n for n in config.nodes}

        pipeline_result = PipelineExecutionResult(
            pipeline_id=config.pipeline_id,
            status="success",  # Optimistic default
            start_time=start_time,
        )

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
        return pipeline_result

    def _execute_node(
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> NodeExecutionResult:
        """Executes a single node based on its type."""
        self.log(f"Executing node: {node.node_id} ({node.step_type})")
        start_ts = time.time()

        try:
            output_artifact_id = None
            metrics: Dict[str, Any] = {}

            if node.step_type == "data_loader":
                output_artifact_id = self._run_data_loader(node)
            elif node.step_type == "feature_engineering":
                # Check if it's actually a misconfigured data loader
                if not node.inputs and "dataset_id" in node.params:
                    logger.warning(
                        f"Node {node.node_id} has step_type='feature_engineering' but looks like a data loader. "
                        "Executing as data loader."
                    )
                    output_artifact_id = self._run_data_loader(node)
                else:
                    output_artifact_id, metrics = self._run_feature_engineering(node)
            elif node.step_type == "model_training":
                output_artifact_id, metrics = self._run_model_training(
                    node, job_id=job_id
                )
            elif node.step_type == "model_tuning":
                output_artifact_id, metrics = self._run_model_tuning(
                    node, job_id=job_id
                )
            elif node.step_type == "data_preview":
                output_artifact_id, metrics = self._run_data_preview(node)
            else:
                # Try to run as a single transformer step
                try:
                    logger.debug(f"Running as single transformer: {node.step_type}")
                    output_artifact_id, metrics = self._run_transformer(node)
                except Exception as e:
                    # If it fails or isn't a valid transformer, re-raise
                    if "Unknown transformer type" in str(e):
                        raise ValueError(f"Unknown step type: {node.step_type}")
                    raise e

            duration = time.time() - start_ts
            return NodeExecutionResult(
                node_id=node.node_id,
                status="success",
                output_artifact_id=output_artifact_id,
                metrics=metrics,
                execution_time=duration,
            )

        except Exception as e:
            logger.exception(f"Error in node {node.node_id}")
            duration = time.time() - start_ts
            return NodeExecutionResult(
                node_id=node.node_id,
                status="failed",
                error=str(e),
                execution_time=duration,
            )

    def _resolve_input(self, node: NodeConfig, index: int = 0) -> Any:
        """Helper to get input artifact from the previous node."""
        if not node.inputs or index >= len(node.inputs):
            raise ValueError(f"Node {node.node_id} requires input at index {index}")

        input_node_id = node.inputs[index]
        # In a real graph, we'd look up the artifact ID produced by that node.
        # For now, we assume the artifact ID is the node ID (simple convention)
        # OR we look at self._results if we want to be dynamic.

        # Let's use the artifact store directly.
        # We assume the previous node saved an artifact with its node_id.
        return self.artifact_store.load(input_node_id)

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

    def _collect_feature_engineer_artifact_keys(
        self, node_id: str, visited: set[str]
    ) -> list[str]:
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
            artifact_keys.extend(self._collect_feature_engineer_artifact_keys(input_node_id, visited))

        if not artifact_keys:
            return None

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

        if not merged_steps:
            return None

        composite = FeatureEngineer([])
        composite.fitted_steps = merged_steps
        return composite

    def _bundle_transformers_with_model(
        self,
        model_artifact_key: str,
        job_id: str = "unknown",
        feature_engineer_artifact_key: str | None = None,
        feature_engineer_override: Any | None = None,
        target_column: str | None = None,
        dropped_columns: List[str] | None = None,
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

            feature_engineer = None

            if feature_engineer_override is not None and hasattr(
                feature_engineer_override, "transform"
            ):
                feature_engineer = feature_engineer_override

            # Prefer an explicit FeatureEngineer artifact key (derived from the pipeline graph)
            # rather than scanning the whole artifacts directory. Scanning can pick a pipeline
            # from a different run and cause incorrect transforms and label decoding.
            if feature_engineer_artifact_key:
                try:
                    obj = self.artifact_store.load(feature_engineer_artifact_key)
                    if hasattr(obj, "transform"):
                        feature_engineer = obj
                except Exception as e:
                    logger.warning(
                        f"Failed to load feature engineer artifact {feature_engineer_artifact_key}: {e}"
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
                        logger.warning(
                            f"Failed to load transformer artifact {t_info['artifact_key']}: {e}"
                        )

                full_artifact = {
                    "model": model_artifact,
                    "transformers": transformers,
                    "transformer_plan": transformer_plan,
                    "job_id": job_id,
                    "target_column": target_column,
                    "dropped_columns": dropped_columns or [],
                }

            # Save to job_id key if available - this is the final artifact for the job
            if job_id and job_id != "unknown":
                self.log(f"Saving bundled artifact to {job_id}")
                self.artifact_store.save(job_id, full_artifact)

        except Exception as e:
            logger.error(f"Failed to bundle transformers with model: {e}")

    # --- Step Implementations ---

    def _run_data_loader(self, node: NodeConfig) -> str:
        # params: {"source": "csv", "path": "...", "sample": True/False, "limit": 1000}

        # Some callers use `dataset_id` as a path.
        dataset_id = node.params.get("dataset_id")
        if not dataset_id:
            dataset_id = node.params.get("path")

        if not dataset_id:
            raise KeyError(
                f"Node {node.node_id} missing 'dataset_id' or 'path' in params: {node.params}"
            )

        limit = None
        if node.params.get("sample", False):
            limit = node.params.get("limit", 1000)
            self.log(f"Loading sample data from {dataset_id} (limit={limit})")
        else:
            dataset_name = self.catalog.get_dataset_name(dataset_id)
            log_msg = f"Loading full data from {dataset_name}" if dataset_name else f"Loading full data from {dataset_id}"
            self.log(log_msg)

        # Use the injected catalog
        try:
            df = self.catalog.load(dataset_id, limit=limit)
        except FileNotFoundError:
            # Try to resolve name for better error message
            raise FileNotFoundError(f"Dataset {dataset_id} not found. Please check if the file exists.")

        self.log(
            f"Data loaded successfully. Shape: {df.shape} ({len(df)} rows, {len(df.columns)} columns)"
        )
        self.artifact_store.save(node.node_id, df)
        return node.node_id

    def _run_feature_engineering(self, node: NodeConfig) -> tuple[str, Dict[str, Any]]:
        # Input: DataFrame
        df = self._resolve_input(node)

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
            self.log(
                f"Feature engineering completed. Output shape: {processed_df.shape}"
            )
        elif isinstance(processed_df, SplitDataset):
            self.log("Feature engineering completed. SplitDataset created.")

        if isinstance(processed_df, tuple):
            # SplitDataset
            train_part = processed_df[0]
            test_part = processed_df[1] if len(processed_df) > 1 else None
            train_shape = getattr(train_part, "shape", None)
            test_shape = getattr(test_part, "shape", None) if test_part is not None else None
            self.log(
                f"Split details - Train: {train_shape}, "
                f"Test: {test_shape or 'None'}"
            )

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

    def _run_model_training(  # noqa: C901
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset (from Feature Engineering) or DataFrame
        # Wait, FeatureEngineer returns SplitDataset if split=True, or DataFrame if not.
        # Modeling expects SplitDataset usually.

        data = self._resolve_input(node)

        # Safety check: Ensure data is not a model artifact
        if hasattr(data, "predict") or hasattr(data, "fit"):
            raise ValueError(
                f"Node {node.node_id} received a Model object instead of a Dataset. "
                "Check your pipeline connections. "
                "Did you connect a Tuning/Training node output to a Training node input?"
            )

        target_col = node.params["target_column"]
        algorithm = node.params.get("algorithm") or node.params.get("model_type")
        if not algorithm:
            raise ValueError("Missing 'algorithm' or 'model_type' in node parameters")
        hyperparameters = node.params.get("hyperparameters", {})

        # Factory logic (simplified)
        calculator, applier = self._get_model_components(algorithm)

        # SDK StatefulEstimator(calculator, applier, node_id)
        estimator = StatefulEstimator(calculator, applier, node.node_id)

        # 1. Cross-Validation (Optional)
        cv_metrics = {}
        if node.params.get("cv_enabled", False):
            # Handle DataFrame vs SplitDataset
            cv_data = data
            if isinstance(data, pd.DataFrame):
                from skyulf.data.dataset import SplitDataset

                cv_data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)
            elif isinstance(data, tuple):
                from skyulf.data.dataset import SplitDataset

                # Check if it's (train_df, test_df) or (X, y)
                elem0 = data[0]
                if isinstance(elem0, pd.DataFrame) and target_col in elem0.columns:
                    train_df, test_df = data
                    cv_data = SplitDataset(
                        train=train_df, test=test_df, validation=None
                    )
                else:
                    cv_data = SplitDataset(
                        train=data, test=pd.DataFrame(), validation=None
                    )

            cv_results = estimator.cross_validate(
                cv_data,
                target_col,
                hyperparameters,
                n_folds=node.params.get("cv_folds", 5),
                cv_type=node.params.get("cv_type", "k_fold"),
                shuffle=node.params.get("cv_shuffle", True),
                random_state=node.params.get("cv_random_state", 42),
                log_callback=self.log,
            )

            # Aggregate metrics for the return value
            # cv_results structure: {"accuracy": {"mean": 0.9, "std": 0.01, ...}, ...}
            for metric_name, stats in cv_results.items():
                if isinstance(stats, dict) and "mean" in stats:
                    cv_metrics[f"cv_{metric_name}_mean"] = stats["mean"]
                    cv_metrics[f"cv_{metric_name}_std"] = stats["std"]

        # 2. Train Final Model
        self.log(f"Starting model training with algorithm: {algorithm}")
        # SDK fit_predict(dataset, target_column, config)
        # config expects {"params": ...} usually
        # Ensure hyperparameters are passed correctly.
        # If hyperparameters is already a dict of params, wrap it.
        fit_config = {"params": hyperparameters}

        # Debug log
        self.log(f"Fit config params: {fit_config}")

        estimator.fit_predict(data, target_col, fit_config, log_callback=self.log)

        # Manually save the model artifact
        self.artifact_store.save(node.node_id, estimator.model)
        if job_id and job_id != "unknown":
            self.log(f"Saving model artifact to job key: {job_id}")
            self.artifact_store.save(job_id, estimator.model)

        self.log("Model training finished.")

        # Bundle transformers with the model for inference
        composite_feature_engineer = self._build_composite_feature_engineer(node)
        feature_engineer_key = None
        if composite_feature_engineer is None:
            feature_engineer_key = self._resolve_feature_engineer_artifact_key(node)

        self._bundle_transformers_with_model(
            node.node_id,
            job_id=job_id,
            feature_engineer_artifact_key=feature_engineer_key,
            feature_engineer_override=composite_feature_engineer,
            target_column=target_col,
        )

        # Optional: Evaluate immediately
        metrics = {}
        if node.params.get("evaluate", True):
            # Ensure data is SplitDataset for evaluation
            eval_data = data
            if isinstance(data, pd.DataFrame):
                from skyulf.data.dataset import SplitDataset

                eval_data = SplitDataset(
                    train=data, test=pd.DataFrame(), validation=None
                )
            elif isinstance(data, tuple):
                from skyulf.data.dataset import SplitDataset

                # Check if it's (train_df, test_df) or (X, y)
                elem0 = data[0]
                if isinstance(elem0, pd.DataFrame) and target_col in elem0.columns:
                    train_df, test_df = data
                    eval_data = SplitDataset(
                        train=train_df, test=test_df, validation=None
                    )
                else:
                    eval_data = SplitDataset(
                        train=data, test=pd.DataFrame(), validation=None
                    )

            report = estimator.evaluate(eval_data, target_col, job_id=job_id)

            # Save evaluation data artifact for API
            if "raw_data" in report:
                eval_key = f"{job_id}_evaluation_data"
                self.log(f"Saving evaluation data to {eval_key}")
                self.artifact_store.save(eval_key, report["raw_data"])

            # Flatten metrics for summary with prefixes
            # SDK report is a dict, but splits contain Pydantic models

            splits = report["splits"]
            if "train" in splits and splits["train"]:
                train_metrics = splits["train"].metrics
                for k, v in train_metrics.items():
                    metrics[f"train_{k}"] = v

            if "test" in splits and splits["test"]:
                test_metrics = splits["test"].metrics
                for k, v in test_metrics.items():
                    metrics[f"test_{k}"] = v

            if "validation" in splits and splits["validation"]:
                val_metrics = splits["validation"].metrics
                for k, v in val_metrics.items():
                    metrics[f"val_{k}"] = v

        # Merge CV metrics
        metrics.update(cv_metrics)

        return node.node_id, metrics

    def _run_model_tuning(  # noqa: C901
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset
        data = self._resolve_input(node)
        target_col = node.params["target_column"]
        algorithm = node.params.get("algorithm") or node.params.get("model_type")
        if not algorithm:
            raise ValueError("Missing 'algorithm' or 'model_type' in node parameters")
        tuning_params = node.params["tuning_config"]  # Dict matching TuningConfig

        calculator, applier = self._get_model_components(algorithm)

        # Create Tuner components
        tuner_calc = TunerCalculator(calculator)
        tuner_applier = TunerApplier(applier)

        # Create StatefulEstimator wrapping the Tuner
        # This ensures consistency with how standard models are trained and evaluated
        estimator = StatefulEstimator(tuner_calc, tuner_applier, node.node_id)

        self.log(
            f"Starting hyperparameter tuning (Strategy: {tuning_params.get('strategy', 'random')}, "
            f"Trials: {tuning_params.get('n_trials', 10)})"
        )

        # Ensure data is SplitDataset
        if isinstance(data, pd.DataFrame):
            from skyulf.data.dataset import SplitDataset

            data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)
        elif isinstance(data, tuple):
            from skyulf.data.dataset import SplitDataset

            # Check if it's (train_df, test_df) or (X, y)
            elem0 = data[0]
            if isinstance(elem0, pd.DataFrame) and target_col in elem0.columns:
                train_df, test_df = data
                data = SplitDataset(train=train_df, test=test_df, validation=None)
            else:
                data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)

        def progress_callback(current, total, score=None, params=None):
            msg = f"Tuning progress: Trial {current}/{total}"
            if score is not None:
                msg += f" - Score: {score:.4f}"
            self.log(msg)

        # Run fit_predict
        # This will:
        # 1. Run tuning (TunerCalculator.fit)
        # 2. Refit the best model on the full training set (TunerCalculator.fit)
        # 3. Generate predictions on train/test/val splits (TunerApplier.predict)
        estimator.fit_predict(
            data,
            target_col,
            tuning_params,
            progress_callback=progress_callback,
            log_callback=self.log,
            job_id=job_id,
        )

        # Save model artifact
        # The model artifact is a tuple: (fitted_model, tuning_result)
        self.artifact_store.save(node.node_id, estimator.model)
        if job_id and job_id != "unknown":
            self.artifact_store.save(job_id, estimator.model)

        self.log("Tuning and final model retraining finished.")

        # Bundle transformers with the model for inference
        composite_feature_engineer = self._build_composite_feature_engineer(node)
        feature_engineer_key = None
        if composite_feature_engineer is None:
            feature_engineer_key = self._resolve_feature_engineer_artifact_key(node)

        self._bundle_transformers_with_model(
            node.node_id,
            job_id=job_id,
            feature_engineer_artifact_key=feature_engineer_key,
            feature_engineer_override=composite_feature_engineer,
            target_column=target_col,
        )

        # Extract metrics from tuning result
        # estimator.model is expected to be a tuple (model, tuning_result) for Tuner
        model_artifact = estimator.model
        if isinstance(model_artifact, tuple) and len(model_artifact) == 2:
            _, tuning_result = model_artifact
        else:
            tuning_result = None

        if tuning_result:
            metrics = {
                "best_score": tuning_result.best_score,
                "best_params": tuning_result.best_params,
                "trials": tuning_result.trials,
            }
        else:
            metrics = {}

        # Evaluate the tuned model
        try:
            report = estimator.evaluate(data, target_col, job_id=job_id)

            # Save evaluation data artifact for API
            if "raw_data" in report:
                eval_key = f"{job_id}_evaluation_data"
                self.log(f"Saving evaluation data to {eval_key}")
                self.artifact_store.save(eval_key, report["raw_data"])

            splits = report["splits"]

            if "train" in splits and splits["train"]:
                for k, v in splits["train"].metrics.items():
                    metrics[f"train_{k}"] = v

            if "test" in splits and splits["test"]:
                for k, v in splits["test"].metrics.items():
                    metrics[f"test_{k}"] = v

            if "validation" in splits and splits["validation"]:
                for k, v in splits["validation"].metrics.items():
                    metrics[f"val_{k}"] = v
        except Exception as e:
            logger.warning(f"Failed to evaluate tuned model: {e}")

        return node.node_id, metrics

    def _run_transformer(self, node: NodeConfig) -> tuple[str, Dict[str, Any]]:
        """Runs a single transformer node as a 1-step feature engineering pipeline."""
        # Input: DataFrame or SplitDataset
        data = self._resolve_input(node)

        # Wrap the single node as a 1-step feature engineering pipeline
        step_config = {
            "name": "step",  # Generic name, the artifact will be saved by engine anyway
            "transformer": node.step_type,
            "params": node.params,
        }

        engineer = FeatureEngineer([step_config])

        # SDK FeatureEngineer.fit_transform(data)
        processed_data, run_metrics = engineer.fit_transform(data)

        # Manually save the engineer state if needed
        self.artifact_store.save(f"exec_{node.node_id}_pipeline", engineer)

        self.artifact_store.save(node.node_id, processed_data)

        # Track executed transformer
        self.executed_transformers.append(
            {
                "node_id": node.node_id,
                "transformer_name": "step",
                "transformer_type": node.step_type,
                "artifact_key": f"exec_{node.node_id}_step",
                "column_name": node.params.get("new_column"),
            }
        )

        # Load fitted params to get metrics (e.g. dropped columns)
        metrics = run_metrics.copy()
        # In SDK, metrics are returned directly, so we don't need to load from artifact store.
        # But we might want to inspect engineer.fitted_steps if metrics are missing.

        return node.node_id, metrics

    def _get_model_components(self, algorithm: str):
        """Factory for model components."""
        # Normalize algorithm name
        algo = algorithm.lower().replace(" ", "_").replace("-", "_")

        if algo in ["logistic_regression", "logisticregression"]:
            return LogisticRegressionCalculator(), LogisticRegressionApplier()
        elif algo in [
            "random_forest_classifier",
            "randomforestclassifier",
            "random_forest",
        ]:
            return RandomForestClassifierCalculator(), RandomForestClassifierApplier()
        elif algo in ["ridge_regression", "ridgeregression", "ridge"]:
            return RidgeRegressionCalculator(), RidgeRegressionApplier()
        elif algo in ["random_forest_regressor", "randomforestregressor"]:
            return RandomForestRegressorCalculator(), RandomForestRegressorApplier()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _run_data_preview(self, node: NodeConfig) -> tuple[str, Dict[str, Any]]:
        """
        Generates a detailed preview of the data and pipeline state.
        """
        # Input: DataFrame or SplitDataset
        data = self._resolve_input(node)

        preview_info: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {},
            "applied_transformations": [],
            "operation_mode": "unknown",
        }

        # 1. Analyze Data
        def get_df_info(df: pd.DataFrame, name: str):
            return {
                "name": name,
                "shape": df.shape,
                "columns": list(df.columns),
                # "dtypes": {k: str(v) for k, v in df.dtypes.items()}, # Optional, can be large
                "sample": df.head(20).replace({np.nan: None}).to_dict(orient="records"),
            }

        if isinstance(data, SplitDataset):
            preview_info["operation_mode"] = (
                "Train: fit_transform | Test/Val: transform"
            )

            # Train
            if isinstance(data.train, tuple):
                X, _ = data.train
                preview_info["data_summary"]["train"] = get_df_info(X, "Train (X)")
            else:
                preview_info["data_summary"]["train"] = get_df_info(data.train, "Train")

            # Test
            if data.test is not None:
                if isinstance(data.test, tuple):
                    X_test, _ = data.test
                    preview_info["data_summary"]["test"] = get_df_info(
                        X_test, "Test (X)"
                    )
                elif isinstance(data.test, pd.DataFrame) and not data.test.empty:
                    preview_info["data_summary"]["test"] = get_df_info(
                        data.test, "Test"
                    )

            # Validation
            if data.validation is not None:
                if isinstance(data.validation, tuple):
                    X_val, _ = data.validation
                    preview_info["data_summary"]["validation"] = get_df_info(
                        X_val, "Validation (X)"
                    )
                elif isinstance(data.validation, pd.DataFrame):
                    preview_info["data_summary"]["validation"] = get_df_info(
                        data.validation, "Validation"
                    )

        elif isinstance(data, pd.DataFrame):
            preview_info["operation_mode"] = "fit_transform"
            preview_info["data_summary"]["full"] = get_df_info(data, "Full Dataset")

        # 2. Get History
        # Return the list of transformers executed so far
        preview_info["applied_transformations"] = self.executed_transformers

        # Save the preview artifact
        self.artifact_store.save(node.node_id, preview_info)

        # Return the preview info directly so it's available in the job result
        return node.node_id, preview_info
