"""Pipeline Execution Engine."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

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
    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store
        self._results: Dict[str, NodeExecutionResult] = {}

    def run(self, config: PipelineConfig) -> PipelineExecutionResult:
        """
        Executes the pipeline defined by the configuration.
        """
        logger.info(f"Starting pipeline execution: {config.pipeline_id}")
        start_time = datetime.now()
        
        pipeline_result = PipelineExecutionResult(
            pipeline_id=config.pipeline_id,
            status="success", # Optimistic default
            start_time=start_time
        )

        for node in config.nodes:
            try:
                node_result = self._execute_node(node)
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

    def _execute_node(self, node: NodeConfig) -> NodeExecutionResult:
        """Executes a single node based on its type."""
        logger.info(f"Executing node: {node.node_id} ({node.step_type})")
        start_ts = time.time()
        
        try:
            output_artifact_id = None
            metrics = {}

            if node.step_type == "data_loader":
                output_artifact_id = self._run_data_loader(node)
            elif node.step_type == "feature_engineering":
                output_artifact_id = self._run_feature_engineering(node)
            elif node.step_type == "model_training":
                output_artifact_id, metrics = self._run_model_training(node)
            elif node.step_type == "model_tuning":
                output_artifact_id, metrics = self._run_model_tuning(node)
            else:
                # Try to run as a single transformer step
                try:
                    output_artifact_id = self._run_transformer(node)
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

    # --- Step Implementations ---

    def _run_data_loader(self, node: NodeConfig) -> str:
        # params: {"source": "csv", "path": "...", "sample": True/False, "limit": 1000}
        loader = DataLoader()
        path = node.params["path"]
        
        if node.params.get("sample", False):
            limit = node.params.get("limit", 1000)
            logger.info(f"Loading sample data from {path} (limit={limit})")
            df = loader.load_sample(path, n=limit)
        else:
            logger.info(f"Loading full data from {path}")
            df = loader.load_full(path)
            
        self.artifact_store.save(node.node_id, df)
        return node.node_id

    def _run_feature_engineering(self, node: NodeConfig) -> str:
        # Input: DataFrame
        df = self._resolve_input(node)
        
        # params: {"steps": [...]}
        engineer = FeatureEngineer(node.params.get("steps", []))
        # Pass artifact_store and node_id to allow internal steps to save state
        processed_df = engineer.fit_transform(df, self.artifact_store, node.node_id)
        
        self.artifact_store.save(node.node_id, processed_df)
        return node.node_id

    def _run_model_training(self, node: NodeConfig) -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset (from Feature Engineering) or DataFrame
        # Wait, FeatureEngineer returns SplitDataset if split=True, or DataFrame if not.
        # Modeling expects SplitDataset usually.
        
        data = self._resolve_input(node)
        target_col = node.params["target_column"]
        algorithm = node.params["algorithm"]
        hyperparameters = node.params.get("hyperparameters", {})
        
        # Factory logic (simplified)
        calculator, applier = self._get_model_components(algorithm)
        
        estimator = StatefulEstimator(calculator, applier, self.artifact_store, node.node_id)
        estimator.fit_predict(data, target_col, hyperparameters)
        
        # Optional: Evaluate immediately
        metrics = {}
        if node.params.get("evaluate", True):
            report = estimator.evaluate(data, target_col)
            # Flatten metrics for summary
            if "test" in report.splits:
                metrics = report.splits["test"].metrics
        
        return node.node_id, metrics

    def _run_model_tuning(self, node: NodeConfig) -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset
        data = self._resolve_input(node)
        target_col = node.params["target_column"]
        algorithm = node.params["algorithm"]
        tuning_params = node.params["tuning_config"] # Dict matching TuningConfig
        
        calculator, applier = self._get_model_components(algorithm)
        tuner = TunerCalculator(calculator)
        
        # Convert dict to TuningConfig object
        config = TuningConfig(**tuning_params)
        
        # We need X_train, y_train
        # Assuming data is SplitDataset
        X_train = data.train.drop(columns=[target_col])
        y_train = data.train[target_col]
        
        validation_data = None
        if data.validation is not None:
            X_val = data.validation.drop(columns=[target_col])
            y_val = data.validation[target_col]
            validation_data = (X_val, y_val)
        
        result = tuner.tune(X_train, y_train, config, validation_data=validation_data)
        
        # Train final model with best params
        estimator = StatefulEstimator(calculator, applier, self.artifact_store, node.node_id)
        estimator.fit_predict(data, target_col, result.best_params)
        
        return node.node_id, {"best_score": result.best_score, "best_params": result.best_params}

    def _run_transformer(self, node: NodeConfig) -> str:
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
        processed_data = engineer.fit_transform(data, self.artifact_store, node_id_prefix=f"exec_{node.node_id}")
        
        self.artifact_store.save(node.node_id, processed_data)
        return node.node_id

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
