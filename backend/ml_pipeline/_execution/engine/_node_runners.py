"""Per-step node-runner methods for :class:`PipelineEngine`.

Mixin slice — owns: ``_run_data_loader``, ``_run_basic_training``,
``_run_advanced_tuning``, ``_run_transformer``, ``_run_data_preview`` and
the algorithm-component factory ``_get_model_components``.

Relies on attributes/methods provided by :class:`PipelineEngine` and its
sibling mixins: ``self.catalog``, ``self.artifact_store``, ``self.log``,
``self._get_input``, ``self._save_reference_data``, ``self.executed_transformers``,
``self._pipeline_has_training_node``, ``self._finalize_training_artifacts``,
``self._build_composite_feature_engineer``,
``self._resolve_feature_engineer_artifact_key``,
``self._bundle_transformers_with_model``, ``self._extract_feature_importances``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from skyulf.data.catalog import DataCatalog
from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling.base import StatefulEstimator
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry

from ..schemas import NodeConfig

if TYPE_CHECKING:
    from ...artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)


class NodeRunnersMixin:
    """Concrete per-step runner implementations."""

    # Type-only stubs so ty can resolve attributes/methods provided by
    # :class:`PipelineEngine` (or its sibling mixins). No runtime impact.
    artifact_store: "ArtifactStore"
    catalog: DataCatalog
    executed_transformers: list[Dict[str, Any]]
    log: Callable[[str], None]
    _get_input: Any
    _save_reference_data: Any
    _finalize_training_artifacts: Any
    _build_composite_feature_engineer: Any
    _resolve_feature_engineer_artifact_key: Any
    _bundle_transformers_with_model: Any
    _extract_feature_importances: Any
    _pipeline_has_training_node: Any

    def _run_data_loader(self, node: NodeConfig, job_id: str = "unknown") -> str:
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
            log_msg = (
                f"Loading full data from {dataset_name}"
                if dataset_name
                else f"Loading full data from {dataset_id}"
            )
            self.log(log_msg)

        # Use the injected catalog
        try:
            df = self.catalog.load(dataset_id, limit=limit)
        except FileNotFoundError:
            # Try to resolve name for better error message
            raise FileNotFoundError(
                f"Dataset {dataset_id} not found. Please check if the file exists."
            )

        self.log(
            f"Data loaded successfully. Shape: {df.shape} ({len(df)} rows, {len(df.columns)} columns)"
        )
        self.artifact_store.save(node.node_id, df)

        # Save as Reference Data for Drift Detection (Raw Initial State)
        if job_id != "unknown" and self._pipeline_has_training_node():
            # We don't have target_col yet, but _save_reference_data splits X/y if tuple.
            # Here df is full dataframe.
            # We'll rely on the fact that target_col is just used to re-assemble if it was a tuple.
            # Since df is a DataFrame, target_col is ignored inside _save_reference_data logic for DF inputs.
            self._save_reference_data(df, job_id, target_col="")

        return node.node_id

    def _run_basic_training(  # noqa: C901
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset (from Feature Engineering) or DataFrame
        # Supports multiple inputs — merges them before training.

        target_col = node.params["target_column"]

        data = self._get_input(node, target_col)

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
                cv_data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)
            elif isinstance(data, tuple):
                # Check if it's (train_df, test_df) or (X, y)
                elem0 = data[0]
                if isinstance(elem0, pd.DataFrame) and target_col in elem0.columns:
                    train_df, test_df = data
                    cv_data = SplitDataset(train=train_df, test=test_df, validation=None)
                else:
                    cv_data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)

            cv_results = estimator.cross_validate(
                cv_data,
                target_col,
                hyperparameters,
                n_folds=node.params.get("cv_folds", 5),
                cv_type=node.params.get("cv_type", "k_fold"),
                shuffle=node.params.get("cv_shuffle", True),
                random_state=node.params.get("cv_random_state", 42),
                time_column=node.params.get("cv_time_column") or None,
                log_callback=self.log,
            )

            # Aggregate metrics for the return value
            # cv_results structure: {"aggregated_metrics": {"accuracy": {"mean": 0.9, ...}}, "folds": [...]}
            agg_metrics = cv_results.get("aggregated_metrics", cv_results)
            for metric_name, stats in agg_metrics.items():
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

        # Finalize and save artifacts
        self._finalize_training_artifacts(data, job_id, target_col, node.node_id, estimator.model)

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
                eval_data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)
            elif isinstance(data, tuple):
                # Check if it's (train_df, test_df) or (X, y)
                elem0 = data[0]
                if isinstance(elem0, pd.DataFrame) and target_col in elem0.columns:
                    train_df, test_df = data
                    eval_data = SplitDataset(train=train_df, test=test_df, validation=None)
                else:
                    eval_data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)

            report = estimator.evaluate(eval_data, target_col, job_id=job_id)

            # Save evaluation data artifact for API
            if "raw_data" in report:
                eval_key = f"{job_id}_evaluation_data"
                uri = self.artifact_store.get_artifact_uri(eval_key)
                self.log(f"Saving evaluation data to {uri}")
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

        # Persist feature importances
        fi = self._extract_feature_importances(estimator.model, data, target_col)
        if fi:
            metrics["feature_importances"] = fi

        # Persist data shape for monitoring
        try:
            if isinstance(data, pd.DataFrame):
                metrics["n_rows"] = len(data)
                metrics["n_features"] = len(data.columns) - 1  # minus target
            elif hasattr(data, "train") and hasattr(data.train, "shape"):
                metrics["n_rows"] = data.train.shape[0]
                metrics["n_features"] = data.train.shape[1] - 1
            elif isinstance(data, tuple) and len(data) >= 1:
                first = data[0]
                if hasattr(first, "shape"):
                    metrics["n_rows"] = first.shape[0]
                    metrics["n_features"] = first.shape[1] - 1
        except Exception:
            pass

        return node.node_id, metrics

    def _run_advanced_tuning(  # noqa: C901
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, Dict[str, Any]]:
        # Input: SplitDataset — supports multiple inputs via merge.
        target_col = node.params["target_column"]

        data = self._get_input(node, target_col)

        algorithm = node.params.get("algorithm") or node.params.get("model_type")
        if not algorithm:
            raise ValueError("Missing 'algorithm' or 'model_type' in node parameters")
        tuning_params = node.params["tuning_config"]  # Dict matching TuningConfig

        calculator, applier = self._get_model_components(algorithm)

        # Create Tuner components
        tuner_calc = TuningCalculator(calculator)
        tuner_applier = TuningApplier(applier)

        # Create StatefulEstimator wrapping the Tuner
        # This ensures consistency with how standard models are trained and evaluated
        estimator = StatefulEstimator(tuner_calc, tuner_applier, node.node_id)

        self.log(
            f"Starting hyperparameter tuning (Strategy: {tuning_params.get('strategy', 'random')}, "
            f"Trials: {tuning_params.get('n_trials', 10)})"
        )

        # Ensure data is SplitDataset
        if isinstance(data, pd.DataFrame):
            data = SplitDataset(train=data, test=pd.DataFrame(), validation=None)
        elif isinstance(data, tuple):
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

        # Finalize and save artifacts
        self._finalize_training_artifacts(data, job_id, target_col, node.node_id, estimator.model)

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
                "scoring_metric": tuning_result.scoring_metric
                or tuning_params.get("tuning_config", {}).get("metric")
                or tuning_params.get("metric"),
            }
        else:
            metrics = {}

        # Cross-Validation on the tuned model (using best params)
        cv_metrics: Dict[str, Any] = {}
        if tuning_params.get("cv_enabled", False):
            best_params: Dict[str, Any] = tuning_result.best_params if tuning_result else {}
            cv_estimator = StatefulEstimator(calculator, applier, node.node_id)

            # For advanced tuning, nested_cv's inner loop already ran during
            # the search. Post-tuning CV only needs the outer evaluation, so
            # downgrade to stratified_k_fold (classification) or k_fold (regression).
            post_cv_type = tuning_params.get("cv_type", "k_fold")
            if post_cv_type == "nested_cv":
                is_classification = getattr(calculator, "problem_type", "") == "classification"
                post_cv_type = "stratified_k_fold" if is_classification else "k_fold"
                self.log(
                    "Nested CV inner loop already ran during tuning. "
                    f"Using {post_cv_type} for post-tuning evaluation."
                )

            self.log("Running cross-validation on tuned model with best parameters...")
            try:
                cv_results = cv_estimator.cross_validate(
                    data,
                    target_col,
                    {"params": best_params},
                    n_folds=tuning_params.get("cv_folds", 5),
                    cv_type=post_cv_type,
                    shuffle=tuning_params.get("cv_shuffle", True),
                    random_state=tuning_params.get("cv_random_state", 42),
                    time_column=tuning_params.get("cv_time_column") or None,
                    log_callback=self.log,
                )

                agg_metrics = cv_results.get("aggregated_metrics", cv_results)
                for metric_name, stats in agg_metrics.items():
                    if isinstance(stats, dict) and "mean" in stats:
                        cv_metrics[f"cv_{metric_name}_mean"] = stats["mean"]
                        cv_metrics[f"cv_{metric_name}_std"] = stats["std"]
            except Exception as e:
                logger.warning(f"Cross-validation failed for tuned model: {e}")

        # Evaluate the tuned model
        try:
            report = estimator.evaluate(data, target_col, job_id=job_id)

            # Save evaluation data artifact for API
            if "raw_data" in report:
                eval_key = f"{job_id}_evaluation_data"
                uri = self.artifact_store.get_artifact_uri(eval_key)
                self.log(f"Saving evaluation data to {uri}")
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

        # Merge CV metrics
        metrics.update(cv_metrics)

        # Persist feature importances
        fi = self._extract_feature_importances(estimator.model, data, target_col)
        if fi:
            metrics["feature_importances"] = fi

        # Persist data shape for monitoring
        try:
            if isinstance(data, pd.DataFrame):
                metrics["n_rows"] = len(data)
                metrics["n_features"] = len(data.columns) - 1
            elif hasattr(data, "train") and hasattr(data.train, "shape"):
                train_frame = cast(Any, data.train)
                metrics["n_rows"] = train_frame.shape[0]
                metrics["n_features"] = train_frame.shape[1] - 1
            elif isinstance(data, tuple) and len(data) >= 1:
                first = cast(Any, data[0])
                if hasattr(first, "shape"):
                    metrics["n_rows"] = first.shape[0]
                    metrics["n_features"] = first.shape[1] - 1
        except Exception:
            pass

        return node.node_id, metrics

    def _run_transformer(
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, Dict[str, Any]]:
        """Runs a single transformer node as a 1-step feature engineering pipeline."""
        # Input: DataFrame or SplitDataset (merged when multiple branches feed in).
        data = self._get_input(node)

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

        # Check if this was a Splitter, if so update Reference Data to be the Train Split
        if (
            "Splitter" in node.step_type
            and job_id != "unknown"
            and self._pipeline_has_training_node()
        ):
            # processed_data is likely a SplitDataset or tuple (train, test)
            # _save_reference_data handles extraction of train part
            self._save_reference_data(processed_data, job_id, target_col="")

        # Load fitted params to get metrics (e.g. dropped columns)
        metrics = run_metrics.copy()
        # In SDK, metrics are returned directly, so we don't need to load from artifact store.
        # But we might want to inspect engineer.fitted_steps if metrics are missing.

        return node.node_id, metrics

    def _get_model_components(self, algorithm: str):
        """Factory for model components."""
        # Normalize algorithm name to match registry IDs
        algo = algorithm.lower().replace(" ", "_").replace("-", "_")

        # Map legacy aliases to registry IDs
        alias_map = {
            "logisticregression": "logistic_regression",
            "randomforestclassifier": "random_forest_classifier",
            "random_forest": "random_forest_classifier",
            "ridgeregression": "ridge_regression",
            "ridge": "ridge_regression",
            "randomforestregressor": "random_forest_regressor",
        }

        registry_id = alias_map.get(algo, algo)

        try:
            calculator_cls = NodeRegistry.get_calculator(registry_id)
            applier_cls = NodeRegistry.get_applier(registry_id)
            return calculator_cls(), applier_cls()
        except ValueError:
            # Fallback: Raise original error if not found in registry
            raise ValueError(f"Unknown algorithm: {algorithm} (Registry ID: {registry_id})")

    def _run_data_preview(self, node: NodeConfig) -> tuple[str, Dict[str, Any]]:
        """
        Generates a detailed preview of the data and pipeline state.
        """
        # Input: DataFrame or SplitDataset (merged when multiple branches feed in).
        data = self._get_input(node)

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
            preview_info["operation_mode"] = "Train: fit_transform | Test/Val: transform"

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
                    preview_info["data_summary"]["test"] = get_df_info(X_test, "Test (X)")
                elif isinstance(data.test, pd.DataFrame) and not data.test.empty:
                    preview_info["data_summary"]["test"] = get_df_info(data.test, "Test")

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
