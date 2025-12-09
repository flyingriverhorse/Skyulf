"""Pipeline Execution Engine."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from ..artifacts.store import ArtifactStore
from .schemas import PipelineConfig, PipelineExecutionResult, NodeExecutionResult, NodeConfig

# Phase 1: Data Loading
from ..data.loader import DataLoader

# Phase 2: Feature Engineering
from ..preprocessing.pipeline import FeatureEngineer

# Phase 3: Modeling
from ..modeling.base import StatefulEstimator
from ..modeling.classification import (
    LogisticRegressionCalculator, LogisticRegressionApplier,
    RandomForestClassifierCalculator, RandomForestClassifierApplier
)
from ..modeling.regression import (
    RidgeRegressionCalculator, RidgeRegressionApplier,
    RandomForestRegressorCalculator, RandomForestRegressorApplier
)
from ..modeling.tuning.tuner import TunerCalculator
from ..modeling.tuning.schemas import TuningConfig

logger = logging.getLogger(__name__)

class PipelineEngine:
    """
    Orchestrates the execution of ML pipelines.
    """
    def __init__(self, artifact_store: ArtifactStore, log_callback=None):
        self.artifact_store = artifact_store
        self.log_callback = log_callback
        self.executed_transformers = [] # Track fitted transformers for inference pipeline
        self._results: Dict[str, NodeExecutionResult] = {}

    def log(self, message: str):
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def run(self, config: PipelineConfig, job_id: str = "unknown") -> PipelineExecutionResult:
        """
        Executes the pipeline defined by the configuration.
        """
        self.log(f"Starting pipeline execution: {config.pipeline_id} (Job: {job_id})")
        start_time = datetime.now()
        self.executed_transformers = [] # Reset for new run
        
        pipeline_result = PipelineExecutionResult(
            pipeline_id=config.pipeline_id,
            status="success", # Optimistic default
            start_time=start_time
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
                    node_id=node.node_id,
                    status="failed",
                    error=str(e)
                )
                pipeline_result.status = "failed"
                break

        pipeline_result.end_time = datetime.now()
        return pipeline_result

    def _execute_node(self, node: NodeConfig, job_id: str = "unknown") -> NodeExecutionResult:
        """Executes a single node based on its type."""
        self.log(f"Executing node: {node.node_id} ({node.step_type})")
        start_ts = time.time()
        
        try:
            output_artifact_id = None
            metrics = {}

            if node.step_type == "data_loader":
                output_artifact_id = self._run_data_loader(node)
            elif node.step_type == "feature_engineering":
                # Check if it's actually a misconfigured data loader
                if not node.inputs and "dataset_id" in node.params:
                    logger.warning(f"Node {node.node_id} has step_type='feature_engineering' but looks like a data loader. Executing as data loader.")
                    output_artifact_id = self._run_data_loader(node)
                else:
                    output_artifact_id, metrics = self._run_feature_engineering(node)
            elif node.step_type == "model_training":
                output_artifact_id, metrics = self._run_model_training(node, job_id=job_id)
            elif node.step_type == "model_tuning":
                output_artifact_id, metrics = self._run_model_tuning(node, job_id=job_id)
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
                execution_time=duration
            )

        except Exception as e:
            logger.exception(f"Error in node {node.node_id}")
            duration = time.time() - start_ts
            return NodeExecutionResult(
                node_id=node.node_id,
                status="failed",
                error=str(e),
                execution_time=duration
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

    def _bundle_transformers_with_model(self, model_artifact_key: str, job_id: str = "unknown"):
        """Bundles fitted transformers with the model artifact for inference."""
        try:
            model_artifact = self.artifact_store.load(model_artifact_key)
            
            # Collect fitted transformer objects
            transformers = []
            transformer_plan = []
            
            for t_info in self.executed_transformers:
                try:
                    fitted_t = self.artifact_store.load(t_info["artifact_key"])
                    if fitted_t:
                        transformers.append({
                            "node_id": t_info["node_id"],
                            "transformer_name": t_info["transformer_name"],
                            "column_name": t_info["column_name"],
                            "transformer": fitted_t
                        })
                        transformer_plan.append({
                            "node_id": t_info["node_id"],
                            "transformer_name": t_info["transformer_name"],
                            "column_name": t_info["column_name"],
                            "transformer_type": t_info["transformer_type"]
                        })
                except Exception as e:
                    logger.warning(f"Failed to load transformer artifact {t_info['artifact_key']}: {e}")

            # Create the bundle
            full_artifact = {
                "model": model_artifact,
                "transformers": transformers,
                "transformer_plan": transformer_plan,
                "job_id": job_id
            }
            
            # Save back to the same key (overwriting the raw model)
            self.artifact_store.save(model_artifact_key, full_artifact)
            
            # Also save to job_id key if available
            if job_id and job_id != "unknown":
                self.artifact_store.save(job_id, full_artifact)
                
        except Exception as e:
            logger.error(f"Failed to bundle transformers with model: {e}")

    # --- Step Implementations ---

    def _run_data_loader(self, node: NodeConfig) -> str:
        # params: {"source": "csv", "path": "...", "sample": True/False, "limit": 1000}
        loader = DataLoader()
        path = node.params["path"]
        
        if node.params.get("sample", False):
            limit = node.params.get("limit", 1000)
            self.log(f"Loading sample data from {path} (limit={limit})")
            df = loader.load_sample(path, n=limit)
        else:
            self.log(f"Loading full data from {path}")
            df = loader.load_full(path)
            
        self.artifact_store.save(node.node_id, df)
        return node.node_id

    def _run_feature_engineering(self, node: NodeConfig) -> tuple[str, Dict[str, Any]]:
        # Input: DataFrame
        df = self._resolve_input(node)
        
        # params: {"steps": [...]}
        engineer = FeatureEngineer(node.params.get("steps", []))
        # Pass artifact_store and node_id to allow internal steps to save state
        processed_df, metrics = engineer.fit_transform(df, self.artifact_store, node.node_id)
        
        self.artifact_store.save(node.node_id, processed_df)
        
        # Track executed transformers
        for step in node.params.get("steps", []):
            self.executed_transformers.append({
                "node_id": node.node_id,
                "transformer_name": step["name"],
                "transformer_type": step["transformer"],
                "artifact_key": f"{node.node_id}_{step['name']}",
                "column_name": step.get("params", {}).get("new_column")
            })
            
        return node.node_id, metrics

    def _run_model_training(self, node: NodeConfig, job_id: str = "unknown") -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset (from Feature Engineering) or DataFrame
        # Wait, FeatureEngineer returns SplitDataset if split=True, or DataFrame if not.
        # Modeling expects SplitDataset usually.
        
        data = self._resolve_input(node)
        
        # Safety check: Ensure data is not a model artifact
        if hasattr(data, "predict") or hasattr(data, "fit"):
             raise ValueError(f"Node {node.node_id} received a Model object instead of a Dataset. Check your pipeline connections. Did you connect a Tuning/Training node output to a Training node input?")
             
        target_col = node.params["target_column"]
        algorithm = node.params.get("algorithm") or node.params.get("model_type")
        if not algorithm:
             raise ValueError("Missing 'algorithm' or 'model_type' in node parameters")
        hyperparameters = node.params.get("hyperparameters", {})
        
        # Factory logic (simplified)
        calculator, applier = self._get_model_components(algorithm)
        
        estimator = StatefulEstimator(calculator, applier, self.artifact_store, node.node_id)
        
        # 1. Cross-Validation (Optional)
        cv_metrics = {}
        if node.params.get("cv_enabled", False):
            # Handle DataFrame vs SplitDataset
            cv_data = data
            if isinstance(data, pd.DataFrame):
                from ..data.container import SplitDataset
                cv_data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)
            elif isinstance(data, tuple):
                 from ..data.container import SplitDataset
                 cv_data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)

            cv_results = estimator.cross_validate(
                cv_data, 
                target_col, 
                hyperparameters,
                n_folds=node.params.get("cv_folds", 5),
                cv_type=node.params.get("cv_type", "k_fold"),
                shuffle=node.params.get("cv_shuffle", True),
                random_state=node.params.get("cv_random_state", 42)
            )
            
            # Aggregate metrics for the return value
            # cv_results structure: {"accuracy": {"mean": 0.9, "std": 0.01, ...}, ...}
            for metric_name, stats in cv_results.items():
                if isinstance(stats, dict) and "mean" in stats:
                    cv_metrics[f"cv_{metric_name}_mean"] = stats["mean"]
                    cv_metrics[f"cv_{metric_name}_std"] = stats["std"]

        # 2. Train Final Model
        estimator.fit_predict(data, target_col, hyperparameters, job_id=job_id)
        
        # Bundle transformers with the model for inference
        self._bundle_transformers_with_model(node.node_id, job_id=job_id)
        
        # Optional: Evaluate immediately
        metrics = {}
        if node.params.get("evaluate", True):
            report = estimator.evaluate(data, target_col, job_id=job_id)
            # Flatten metrics for summary with prefixes
            if "train" in report.splits and report.splits["train"]:
                for k, v in report.splits["train"].metrics.items():
                    metrics[f"train_{k}"] = v
            
            if "test" in report.splits and report.splits["test"]:
                for k, v in report.splits["test"].metrics.items():
                    metrics[f"test_{k}"] = v
                    
            if "validation" in report.splits and report.splits["validation"]:
                for k, v in report.splits["validation"].metrics.items():
                    metrics[f"val_{k}"] = v
        
        # Merge CV metrics
        metrics.update(cv_metrics)
        
        return node.node_id, metrics

    def _run_model_tuning(self, node: NodeConfig, job_id: str = "unknown") -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset
        data = self._resolve_input(node)
        target_col = node.params["target_column"]
        algorithm = node.params.get("algorithm") or node.params.get("model_type")
        if not algorithm:
             raise ValueError("Missing 'algorithm' or 'model_type' in node parameters")
        tuning_params = node.params["tuning_config"] # Dict matching TuningConfig
        
        calculator, applier = self._get_model_components(algorithm)
        tuner = TunerCalculator(calculator)
        
        # Convert dict to TuningConfig object
        config = TuningConfig(**tuning_params)
        
        # We need X_train, y_train
        # Assuming data is SplitDataset
        if isinstance(data.train, tuple):
            X_train, y_train = data.train
        else:
            X_train = data.train.drop(columns=[target_col])
            y_train = data.train[target_col]
        
        validation_data = None
        if data.validation is not None:
            if isinstance(data.validation, tuple):
                X_val, y_val = data.validation
            else:
                X_val = data.validation.drop(columns=[target_col])
                y_val = data.validation[target_col]
            validation_data = (X_val, y_val)
        
        result = tuner.tune(X_train, y_train, config, validation_data=validation_data)
        
        # Train final model with best params
        estimator = StatefulEstimator(calculator, applier, self.artifact_store, node.node_id)
        estimator.fit_predict(data, target_col, result.best_params, job_id=job_id)
        
        # Bundle transformers with the model for inference
        self._bundle_transformers_with_model(node.node_id, job_id=job_id)
        
        metrics = {"best_score": result.best_score, "best_params": result.best_params}
        
        # Evaluate the tuned model
        try:
            report = estimator.evaluate(data, target_col, job_id=job_id)
            if "train" in report.splits and report.splits["train"]:
                for k, v in report.splits["train"].metrics.items():
                    metrics[f"train_{k}"] = v
            
            if "test" in report.splits and report.splits["test"]:
                for k, v in report.splits["test"].metrics.items():
                    metrics[f"test_{k}"] = v
                    
            if "validation" in report.splits and report.splits["validation"]:
                for k, v in report.splits["validation"].metrics.items():
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
            "name": "step", # Generic name, the artifact will be saved by engine anyway
            "transformer": node.step_type,
            "params": node.params
        }
        
        engineer = FeatureEngineer([step_config])
        
        # We use a prefix that includes the node_id to avoid collisions if FeatureEngineer saves internal state
        # But mostly we care about the final result which we save manually below.
        processed_data, run_metrics = engineer.fit_transform(data, self.artifact_store, node_id_prefix=f"exec_{node.node_id}")
        
        self.artifact_store.save(node.node_id, processed_data)
        
        # Track executed transformer
        self.executed_transformers.append({
            "node_id": node.node_id,
            "transformer_name": "step",
            "transformer_type": node.step_type,
            "artifact_key": f"exec_{node.node_id}_step",
            "column_name": node.params.get("new_column")
        })
        
        # Load fitted params to get metrics (e.g. dropped columns)
        metrics = run_metrics.copy()
        try:
            # FeatureEngineer saves params at {prefix}_{step_name}
            params_id = f"exec_{node.node_id}_step"
            fitted_params = self.artifact_store.load(params_id)
            if isinstance(fitted_params, dict):
                if "columns_to_drop" in fitted_params:
                    metrics["dropped_columns"] = fitted_params["columns_to_drop"]
                if "selected_columns" in fitted_params:
                    metrics["selected_columns"] = fitted_params["selected_columns"]
        except Exception:
            # Artifact might not exist or be loadable, or store might not support load
            pass
            
        return node.node_id, metrics

    def _get_model_components(self, algorithm: str):
        if algorithm == "logistic_regression":
            return LogisticRegressionCalculator(), LogisticRegressionApplier()
        elif algorithm == "random_forest_classifier":
            return RandomForestClassifierCalculator(), RandomForestClassifierApplier()
        elif algorithm == "ridge_regression":
            return RidgeRegressionCalculator(), RidgeRegressionApplier()
        elif algorithm == "random_forest_regressor":
            return RandomForestRegressorCalculator(), RandomForestRegressorApplier()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
