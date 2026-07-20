"""Per-step node-runner methods for :class:`PipelineEngine`.

Mixin slice — owns: ``_run_data_loader``, ``_run_basic_training``,
``_run_advanced_tuning``, ``_run_transformer``, ``_run_data_preview``,
the algorithm-component factory ``_get_model_components``, and the shared
post-fit orchestration helper ``_finalize_training_run``.

Relies on attributes/methods provided by :class:`PipelineEngine` and its
sibling mixins: ``self.catalog``, ``self.artifact_store``, ``self.log``,
``self._get_input``, ``self._save_reference_data``, ``self.executed_transformers``,
``self._pipeline_has_training_node``, ``self._finalize_training_artifacts``,
``self._build_composite_feature_engineer``,
``self._resolve_feature_engineer_artifact_key``,
``self._bundle_transformers_with_model``, ``self._extract_feature_importances``,
``self._extract_shap_explanation``.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from backend.config import get_settings
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
    executed_transformers: list[dict[str, Any]]
    log: Callable[[str], None]
    _get_input: Any
    _save_reference_data: Any
    _finalize_training_artifacts: Any
    _build_composite_feature_engineer: Any
    _resolve_feature_engineer_artifact_key: Any
    _bundle_transformers_with_model: Any
    _extract_feature_importances: Any
    _extract_shap_explanation: Any
    _pipeline_has_training_node: Any

    def _record_split_dataset_shape_metrics(
        self, metrics: dict[str, Any], data: SplitDataset, target_col: str
    ) -> bool:
        """Populate n_rows/n_features from a SplitDataset's train slot. Returns True if set."""
        train_data = data.train
        if isinstance(train_data, tuple) and len(train_data) >= 1:
            train_x = cast(Any, train_data[0])
            if hasattr(train_x, "shape"):
                metrics["n_rows"] = train_x.shape[0]
                metrics["n_features"] = train_x.shape[1]
                return True
        if hasattr(train_data, "shape"):
            train_frame = cast(Any, train_data)
            metrics["n_rows"] = train_frame.shape[0]
            metrics["n_features"] = train_frame.shape[1] - int(
                target_col in getattr(train_frame, "columns", ())
            )
            return True
        return False

    def _record_tuple_shape_metrics(self, metrics: dict[str, Any], data: tuple) -> None:
        """Populate n_rows/n_features from an ``(X, ...)``-shaped tuple, if possible."""
        if len(data) < 1:
            return
        first = cast(Any, data[0])
        if hasattr(first, "shape"):
            metrics["n_rows"] = first.shape[0]
            metrics["n_features"] = first.shape[1]

    def _record_data_shape_metrics(
        self, metrics: dict[str, Any], data: Any, target_col: str
    ) -> None:
        """Populate ``n_rows``/``n_features`` in ``metrics`` from a resolved data artifact."""
        if isinstance(data, pd.DataFrame):
            metrics["n_rows"] = len(data)
            metrics["n_features"] = len(data.columns) - int(target_col in data.columns)
            return

        if isinstance(data, SplitDataset) and self._record_split_dataset_shape_metrics(
            metrics, data, target_col
        ):
            return

        if isinstance(data, tuple):
            self._record_tuple_shape_metrics(metrics, data)

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
            ) from None

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

    def _get_training_input(self, node: NodeConfig, target_col: str) -> Any:
        """Resolve upstream input for a training node, rejecting Model artifacts."""
        data = self._get_input(node, target_col)
        # Safety check: Ensure data is not a model artifact
        if hasattr(data, "predict") or hasattr(data, "fit"):
            raise ValueError(
                f"Node {node.node_id} received a Model object instead of a Dataset. "
                "Check your pipeline connections. "
                "Did you connect a Tuning/Training node output to a Training node input?"
            )
        return data

    def _to_split_dataset(self, data: Any, target_col: str) -> Any:
        """Coerce a DataFrame or (train, test)/(X, y) tuple into a SplitDataset.

        Returns ``data`` unchanged when it's neither a DataFrame nor tuple
        (e.g. already a SplitDataset).
        """
        if isinstance(data, pd.DataFrame):
            return SplitDataset(train=data, test=pd.DataFrame(), validation=None)
        if isinstance(data, tuple):
            # Check if it's (train_df, test_df) or (X, y)
            elem0 = data[0]
            if isinstance(elem0, pd.DataFrame) and target_col in elem0.columns:
                train_df, test_df = data
                return SplitDataset(train=train_df, test=test_df, validation=None)
            return SplitDataset(train=data, test=pd.DataFrame(), validation=None)
        return data

    def _aggregate_cv_metrics(self, cv_results: dict[str, Any]) -> dict[str, Any]:
        """Flatten a ``cross_validate()`` result's per-metric mean/std into a flat dict.

        cv_results structure: ``{"aggregated_metrics": {"accuracy": {"mean": 0.9, ...}}, "folds": [...]}``
        """
        cv_metrics: dict[str, Any] = {}
        agg_metrics = cv_results.get("aggregated_metrics", cv_results)
        for metric_name, stats in agg_metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                cv_metrics[f"cv_{metric_name}_mean"] = stats["mean"]
                cv_metrics[f"cv_{metric_name}_std"] = stats["std"]
        return cv_metrics

    def _run_basic_training_cv(
        self,
        estimator: Any,
        data: Any,
        target_col: str,
        hyperparameters: dict[str, Any],
        node: NodeConfig,
    ) -> dict[str, Any]:
        """Run optional cross-validation for basic training and return ``cv_`` metrics."""
        if not node.params.get("cv_enabled", False):
            return {}
        cv_data = self._to_split_dataset(data, target_col)
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
        return self._aggregate_cv_metrics(cv_results)

    def _resolve_train_frame(self, data: Any) -> Any:
        """Best-effort extraction of the actual training DataFrame from ``data``.

        ``data`` may be a plain DataFrame, a ``SplitDataset`` (``.train``), or an
        ``(X, y)``/``(train, test)`` tuple — normalize all of these down to the
        single DataFrame that was actually fed into ``estimator.fit_predict()``.
        """
        if isinstance(data, SplitDataset):
            return self._resolve_train_frame(data.train)
        if isinstance(data, tuple) and len(data) >= 1:
            return self._resolve_train_frame(data[0])
        return data

    def _resolve_train_feature_columns(
        self,
        data: Any,
        target_col: str,
        numeric_only: bool = False,
        exclude_columns: list[str] | None = None,
    ) -> list[str] | None:
        """Best-effort list of the exact feature column names used to fit the model.

        Persisted alongside the bundled inference artifact so the deployment
        service and the manual-prediction UI can know precisely which columns
        the model expects, instead of guessing from ``feature_names_in_``
        (which sklearn only sets when ``.fit()`` is called with a DataFrame —
        not the case here since ``SklearnBridge.to_sklearn`` converts to a
        bare numpy array first).

        ``numeric_only`` mirrors clustering models (e.g. K-Means) dropping
        non-numeric columns before fitting — see ``_select_numeric_features``
        in ``skyulf.modeling.clustering`` — so the persisted list matches
        exactly what the model was fit on. ``exclude_columns`` additionally
        drops named columns (e.g. a clustering "reference column") regardless
        of dtype, since those are excluded by name, not by numeric-ness.
        """
        train_frame = self._resolve_train_frame(data)
        if not hasattr(train_frame, "columns"):
            return None
        columns = list(train_frame.columns)
        if target_col and target_col in columns:
            columns.remove(target_col)
        if numeric_only and hasattr(train_frame, "select_dtypes"):
            numeric_cols = set(train_frame.select_dtypes(include=["number", "bool"]).columns)
            columns = [c for c in columns if c in numeric_cols]
        if exclude_columns:
            columns = [c for c in columns if c not in exclude_columns]
        return columns

    def _bundle_model_with_transformers(
        self,
        node: NodeConfig,
        job_id: str,
        target_col: str,
        data: Any = None,
        numeric_only: bool = False,
        exclude_columns: list[str] | None = None,
    ) -> None:
        """Attach fitted transformers to the trained model artifact for inference."""
        composite_feature_engineer = self._build_composite_feature_engineer(node)
        feature_engineer_key = None
        if composite_feature_engineer is None:
            feature_engineer_key = self._resolve_feature_engineer_artifact_key(node)

        feature_columns = (
            self._resolve_train_feature_columns(
                data, target_col, numeric_only=numeric_only, exclude_columns=exclude_columns
            )
            if data is not None
            else None
        )

        self._bundle_transformers_with_model(
            node.node_id,
            job_id=job_id,
            feature_engineer_artifact_key=feature_engineer_key,
            feature_engineer_override=composite_feature_engineer,
            target_column=target_col,
            feature_columns=feature_columns,
        )

    def _flatten_split_metrics(self, splits: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Copy per-split metrics into ``metrics`` with train_/test_/val_ prefixes."""
        for split_name, prefix in (("train", "train"), ("test", "test"), ("validation", "val")):
            split = splits.get(split_name)
            if not split:
                continue
            for k, v in split.metrics.items():
                metrics[f"{prefix}_{k}"] = v

    def _evaluate_and_save_report(
        self,
        estimator: Any,
        data: Any,
        target_col: str,
        job_id: str,
        metrics: dict[str, Any],
        reference_column: str = "",
    ) -> None:
        """Evaluate ``estimator`` on ``data``, save the raw eval artifact, flatten metrics.

        Mutates ``metrics`` in place with the ``train_``/``test_``/``val_`` prefixed
        metrics from the evaluation report. ``reference_column`` is clustering-only
        (see ``StatefulEstimator.evaluate``).
        """
        report = estimator.evaluate(
            data, target_col, job_id=job_id, reference_column=reference_column
        )

        # Save evaluation data artifact for API
        if "raw_data" in report:
            eval_key = f"{job_id}_evaluation_data"
            uri = self.artifact_store.get_artifact_uri(eval_key)
            self.log(f"Saving evaluation data to {uri}")
            self.artifact_store.save(eval_key, report["raw_data"])

        # Flatten metrics for summary with prefixes
        # SDK report is a dict, but splits contain Pydantic models
        self._flatten_split_metrics(report["splits"], metrics)

    def _safe_record_data_shape_metrics(
        self, metrics: dict[str, Any], data: Any, target_col: str, node_id: str
    ) -> None:
        """Best-effort recording of data shape metrics; failures are logged, not raised."""
        try:
            self._record_data_shape_metrics(metrics, data, target_col)
        except Exception:
            logger.debug("Failed to record data shape metrics for node %s", node_id, exc_info=True)

    def _finalize_training_run(
        self,
        node: NodeConfig,
        job_id: str,
        target_col: str,
        data: Any,
        estimator: Any,
        metrics: dict[str, Any],
        cv_metrics: dict[str, Any],
        *,
        completion_log: str,
        is_clustering: bool = False,
        reference_col: str = "",
        numeric_only: bool = False,
        exclude_columns: list[str] | None = None,
        should_evaluate: bool = True,
        swallow_evaluate_errors: bool = False,
    ) -> dict[str, Any]:
        """Shared post-fit steps for both training modes.

        Finalizes artifacts, bundles transformers, optionally evaluates, merges
        CV metrics, extracts feature importances/SHAP, and records data-shape
        metrics. Mutates and returns the completed ``metrics`` dict.

        ``data`` is coerced to ``SplitDataset`` via ``_to_split_dataset`` before
        evaluation — that call is idempotent when ``data`` is already a
        ``SplitDataset`` (the tuning path), so both callers can pass ``data``
        as-is without pre-converting.
        """
        self._finalize_training_artifacts(data, job_id, target_col, node.node_id, estimator.model)

        self.log(completion_log)

        self._bundle_model_with_transformers(
            node,
            job_id,
            target_col,
            data,
            numeric_only=numeric_only,
            exclude_columns=exclude_columns,
        )

        if should_evaluate:
            eval_data = self._to_split_dataset(data, target_col)
            if swallow_evaluate_errors:
                try:
                    self._evaluate_and_save_report(
                        estimator, eval_data, target_col, job_id, metrics,
                        reference_column=reference_col,
                    )
                except Exception:
                    logger.exception("Failed to evaluate tuned model")
            else:
                self._evaluate_and_save_report(
                    estimator, eval_data, target_col, job_id, metrics,
                    reference_column=reference_col,
                )

        metrics.update(cv_metrics)

        if not is_clustering:
            fi = self._extract_feature_importances(estimator.model, data, target_col)
            if fi:
                metrics["feature_importances"] = fi

            shap_explanation = self._extract_shap_explanation(estimator.model, data, target_col)
            if shap_explanation:
                metrics["shap_explanation"] = shap_explanation

        self._safe_record_data_shape_metrics(metrics, data, target_col, node.node_id)

        return metrics

    def _run_basic_training(
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, dict[str, Any]]:
        # Input: SplitDataset (from Feature Engineering) or DataFrame
        # Supports multiple inputs — merges them before training.

        # Clustering (segmentation) nodes have no target column to predict —
        # `""` is the established "no target" sentinel `_get_input`/`_extract_xy`
        # already understand (see `skyulf.modeling.base.StatefulEstimator._extract_xy`).
        target_col = node.params.get("target_column") or ""
        data = self._get_training_input(node, target_col)

        algorithm = node.params.get("algorithm") or node.params.get("model_type")
        if not algorithm:
            raise ValueError("Missing 'algorithm' or 'model_type' in node parameters")
        hyperparameters = node.params.get("hyperparameters", {})

        # Factory logic (simplified)
        calculator, applier = self._get_model_components(algorithm)
        is_clustering = getattr(calculator, "problem_type", "") == "clustering"

        # Clustering-only: an optional column (e.g. a known label like species
        # name) excluded from training features but kept around purely so the
        # user can cross-check "which cluster is which real-world group"
        # after the fact (see `reference_crosstab` in the evaluation report).
        reference_col = (node.params.get("reference_column") or "") if is_clustering else ""

        # SDK StatefulEstimator(calculator, applier, node_id)
        estimator = StatefulEstimator(calculator, applier, node.node_id)

        # 1. Cross-Validation (Optional) — not meaningful for unsupervised
        # clustering (no scorer/target to cross-validate against).
        cv_metrics = (
            {}
            if is_clustering
            else self._run_basic_training_cv(estimator, data, target_col, hyperparameters, node)
        )

        # 2. Train Final Model
        self.log(f"Starting model training with algorithm: {algorithm}")
        # SDK fit_predict(dataset, target_column, config)
        # config expects {"params": ...} usually
        # Ensure hyperparameters are passed correctly.
        # If hyperparameters is already a dict of params, wrap it.
        fit_config: dict[str, Any] = {"params": hyperparameters}
        if reference_col:
            fit_config["reference_column"] = reference_col

        # Debug log
        self.log(f"Fit config params: {fit_config}")

        estimator.fit_predict(data, target_col, fit_config, log_callback=self.log)

        metrics: dict[str, Any] = {}
        return node.node_id, self._finalize_training_run(
            node,
            job_id,
            target_col,
            data,
            estimator,
            metrics,
            cv_metrics,
            completion_log="Model training finished.",
            is_clustering=is_clustering,
            reference_col=reference_col,
            numeric_only=is_clustering,
            exclude_columns=[reference_col] if reference_col else None,
            should_evaluate=node.params.get("evaluate", True),
            swallow_evaluate_errors=False,
        )

    def _prepare_tuning_config(self, node: NodeConfig) -> tuple[Any, Any, dict[str, Any]]:
        """Resolve model components and build the ``tuning_params`` dict for a tuning node.

        Injects server-side parallelism settings and auto-builds a nested
        search space for structural (ensemble) models when the UI sent none.
        """
        algorithm = node.params.get("algorithm") or node.params.get("model_type")
        if not algorithm:
            raise ValueError("Missing 'algorithm' or 'model_type' in node parameters")
        tuning_params = dict(node.params["tuning_config"])  # Dict matching TuningConfig
        # Inject server-side parallelism from settings (not user-configurable via the UI)
        settings = get_settings()
        tuning_params["n_jobs"] = settings.TUNING_N_JOBS
        tuning_params["parallel_backend"] = settings.TUNING_PARALLEL_BACKEND

        calculator, applier = self._get_model_components(algorithm)

        # Advanced Tuning hard-requires a target column and runs supervised
        # `cross_validate()` with a scorer — clustering algorithms (no
        # ground-truth target, no real train/test scorer) would crash deep
        # inside the tuner rather than fail with a clear message. The
        # frontend already hides clustering models from this node's dropdown,
        # but reject defensively here too in case a pipeline JSON is crafted
        # or replayed directly against the API.
        if getattr(calculator, "problem_type", "") == "clustering":
            raise ValueError(
                f"Algorithm '{algorithm}' is a clustering model and is not supported by "
                "Advanced Tuning. Use the Segmentation node for clustering instead."
            )

        # Structural models (ensembles) resolve their base estimators here so the
        # tuner can construct a valid meta-estimator, and may auto-build a nested
        # ``<name>__<param>`` search space for their base learners when the UI
        # sent none. Plain models implement these as no-ops.
        calculator.prepare_tuning_params(tuning_params)
        if not tuning_params.get("search_space"):
            auto_space = calculator.build_tuning_search_space(
                tuning_params, tuning_params.get("strategy", "random")
            )
            if auto_space:
                tuning_params["search_space"] = auto_space

        return calculator, applier, tuning_params

    def _extract_tuning_metrics(
        self, estimator: Any, tuning_params: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Extract the TuningResult (if present) and its summary metrics from the estimator.

        ``estimator.model`` is expected to be a ``(model, tuning_result)`` tuple
        for a Tuner; returns ``(None, {})`` when that shape isn't present.
        """
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
        return tuning_result, metrics

    def _run_tuned_cv(
        self,
        calculator: Any,
        applier: Any,
        data: Any,
        target_col: str,
        tuning_params: dict[str, Any],
        tuning_result: Any,
        node: NodeConfig,
    ) -> dict[str, Any]:
        """Run post-tuning cross-validation with the tuned model's best params.

        For ``nested_cv``, the inner CV loop already ran during the search, so
        post-tuning CV only needs the outer evaluation and downgrades to
        ``stratified_k_fold`` (classification) or ``k_fold`` (regression).
        """
        if not tuning_params.get("cv_enabled", False):
            return {}
        best_params: dict[str, Any] = tuning_result.best_params if tuning_result else {}
        cv_estimator = StatefulEstimator(calculator, applier, node.node_id)

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
            return self._aggregate_cv_metrics(cv_results)
        except Exception:
            logger.exception("Cross-validation failed for tuned model")
            return {}

    def _run_advanced_tuning(
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, dict[str, Any]]:
        # Input: SplitDataset — supports multiple inputs via merge.
        target_col = node.params["target_column"]

        data = self._get_input(node, target_col)

        calculator, applier, tuning_params = self._prepare_tuning_config(node)

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
        data = self._to_split_dataset(data, target_col)

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
        self._bundle_model_with_transformers(node, job_id, target_col, data)

        # Extract metrics from tuning result
        tuning_result, metrics = self._extract_tuning_metrics(estimator, tuning_params)

        # Cross-Validation on the tuned model (using best params)
        cv_metrics = self._run_tuned_cv(
            calculator, applier, data, target_col, tuning_params, tuning_result, node
        )

        return node.node_id, self._finalize_training_run(
            node,
            job_id,
            target_col,
            data,
            estimator,
            metrics,
            cv_metrics,
            completion_log="Tuning and final model retraining finished.",
            is_clustering=False,
            reference_col="",
            numeric_only=False,
            exclude_columns=None,
            should_evaluate=True,
            swallow_evaluate_errors=True,
        )

    def _run_transformer(
        self, node: NodeConfig, job_id: str = "unknown"
    ) -> tuple[str, dict[str, Any]]:
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
            raise ValueError(
                f"Unknown algorithm: {algorithm} (Registry ID: {registry_id})"
            ) from None

    def _data_preview_df_info(self, df: pd.DataFrame, name: str) -> dict[str, Any]:
        """Build the preview payload (shape/columns/sample) for a single DataFrame."""
        return {
            "name": name,
            "shape": df.shape,
            "columns": list(df.columns),
            # "dtypes": {k: str(v) for k, v in df.dtypes.items()}, # Optional, can be large
            "sample": df.head(20).replace({np.nan: None}).to_dict(orient="records"),
        }

    def _preview_slot_info(self, slot: Any, name: str) -> dict[str, Any] | None:
        """Build preview info for a test/validation SplitDataset slot (tuple or DataFrame), or None."""
        if isinstance(slot, tuple):
            X, _ = slot
            return self._data_preview_df_info(X, f"{name} (X)")
        if isinstance(slot, pd.DataFrame) and not slot.empty:
            return self._data_preview_df_info(slot, name)
        return None

    def _build_split_dataset_data_summary(self, data: SplitDataset) -> dict[str, Any]:
        """Build the train/test/validation preview summary for a SplitDataset."""
        summary: dict[str, Any] = {}

        if isinstance(data.train, tuple):
            X, _ = data.train
            summary["train"] = self._data_preview_df_info(cast(pd.DataFrame, X), "Train (X)")
        else:
            summary["train"] = self._data_preview_df_info(cast(pd.DataFrame, data.train), "Train")

        if data.test is not None:
            test_info = self._preview_slot_info(data.test, "Test")
            if test_info is not None:
                summary["test"] = test_info

        if data.validation is not None:
            val_info = self._preview_slot_info(data.validation, "Validation")
            if val_info is not None:
                summary["validation"] = val_info

        return summary

    def _run_data_preview(self, node: NodeConfig) -> tuple[str, dict[str, Any]]:
        """
        Generates a detailed preview of the data and pipeline state.
        """
        # Input: DataFrame or SplitDataset (merged when multiple branches feed in).
        data = self._get_input(node)

        preview_info: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "data_summary": {},
            "applied_transformations": [],
            "operation_mode": "unknown",
        }

        # 1. Analyze Data
        if isinstance(data, SplitDataset):
            preview_info["operation_mode"] = "Train: fit_transform | Test/Val: transform"
            preview_info["data_summary"] = self._build_split_dataset_data_summary(data)
        elif isinstance(data, pd.DataFrame):
            preview_info["operation_mode"] = "fit_transform"
            preview_info["data_summary"]["full"] = self._data_preview_df_info(data, "Full Dataset")

        # 2. Get History
        # Return the list of transformers executed so far
        preview_info["applied_transformations"] = self.executed_transformers

        # Save the preview artifact
        self.artifact_store.save(node.node_id, preview_info)

        # Return the preview info directly so it's available in the job result
        return node.node_id, preview_info
