"""Feature Engineering Pipeline Orchestrator."""

import logging
from typing import Any, Dict, List, Sequence, Union

import pandas as pd

from ..types import PreprocessingStepConfig
from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame
from ..utils import get_data_stats
from ..registry import NodeRegistry
from .base import StatefulTransformer

# Import modules to ensure nodes are registered

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Orchestrates a sequence of feature engineering steps.
    """

    def __init__(
        self,
        steps_config: Sequence[Union[PreprocessingStepConfig, Dict[str, Any]]],
    ):
        # `Sequence` (covariant) accepts list[dict] or list[PreprocessingStepConfig].
        self.steps_config = steps_config
        self.fitted_steps: List[Dict[str, Any]] = []

    def transform(
        self, data: Union[pd.DataFrame, SkyulfDataFrame]
    ) -> Union[pd.DataFrame, SkyulfDataFrame]:
        """
        Apply fitted transformations to new data.
        """
        current_data = data

        for step in self.fitted_steps:
            name = step["name"]
            transformer_type = step["type"]
            applier = step["applier"]
            artifact = step["artifact"]

            # Skip splitters during inference/transform
            if transformer_type in [
                "TrainTestSplitter",
                "feature_target_split",
                "Oversampling",
                "Undersampling",
            ]:
                continue

            logger.debug(f"Applying step: {name} ({transformer_type})")
            current_data = applier.apply(current_data, artifact)

        return current_data

    def fit_transform(
        self, data: Union[pd.DataFrame, SkyulfDataFrame, Any], node_id_prefix=""
    ) -> Any:
        """
        Runs the pipeline on data.
        Returns: (transformed_data, metrics_dict)
        """
        self.fitted_steps = []  # Reset fitted steps
        current_data = data
        metrics: Dict[str, Any] = {}

        for i, step in enumerate(self.steps_config):
            name = step["name"]
            transformer_type = step["transformer"]
            params = step.get("params", {})

            logger.info(f"Running step {i}: {name} ({transformer_type})")
            logger.debug(f"FeatureEngineer running step {i}: {name} ({transformer_type})")
            logger.debug(f"current_data type: {type(current_data)}")

            # Snapshot before for shape-delta + Winsorize value-clipping metrics
            rows_before, cols_before = get_data_stats(current_data)
            data_before = current_data

            calculator, applier = self._get_transformer_components(transformer_type)
            step_node_id = f"{node_id_prefix}_{name}"

            current_data, fitted_params, transformer_inst = self._run_step(
                transformer_type=transformer_type,
                name=name,
                calculator=calculator,
                applier=applier,
                step_node_id=step_node_id,
                current_data=current_data,
                params=params,
            )

            if transformer_inst is not None:
                # Add node-level performance metrics directly into `metrics` dictionary
                metrics["fit_time"] = getattr(transformer_inst, "fit_time", 0.0)
                metrics["peak_memory_bytes"] = getattr(transformer_inst, "peak_memory_bytes", 0)
                metrics["rows_in"] = getattr(transformer_inst, "rows_in", 0)
                metrics["rows_out"] = getattr(transformer_inst, "rows_out", 0)

            logger.debug(f"Step {i} complete. New data type: {type(current_data)}")

            rows_after, cols_after = get_data_stats(current_data)
            self._collect_step_metrics(
                transformer_type=transformer_type,
                fitted_params=fitted_params,
                data_before=data_before,
                current_data=current_data,
                params=params,
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after,
                name=name,
                metrics=metrics,
            )

        return current_data, metrics

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _run_step(
        self,
        *,
        transformer_type: str,
        name: str,
        calculator: Any,
        applier: Any,
        step_node_id: str,
        current_data: Any,
        params: Dict[str, Any],
    ) -> tuple:  # Returns (data, params, transformer)
        """Execute one pipeline step. Returns (new_data, fitted_params).

        Splitters change the data structure (DataFrame -> SplitDataset / (X, y)),
        so they bypass StatefulTransformer; everything else goes through the
        standard fit_transform wrapper and is appended to fitted_steps.
        """
        transformer = StatefulTransformer(calculator, applier, step_node_id)
        fitted_params: Dict[str, Any] = {}

        if transformer_type == "TrainTestSplitter":
            logger.debug("Handling TrainTestSplitter")
            if isinstance(current_data, (pd.DataFrame, SkyulfDataFrame, tuple)):
                params = calculator.fit(current_data, params)
                current_data = applier.apply(current_data, params)
            else:
                logger.debug(f"Skipping TrainTestSplitter. current_data is {type(current_data)}")
                logger.warning(
                    "Attempting to split an already split dataset. Skipping TrainTestSplitter."
                )
            return current_data, fitted_params, None

        if transformer_type == "feature_target_split":
            logger.debug("Handling feature_target_split")
            params = calculator.fit(current_data, params)
            current_data = applier.apply(current_data, params)
            return current_data, fitted_params, None

        logger.debug("Handling standard transformer via StatefulTransformer")
        current_data = transformer.fit_transform(current_data, params)
        fitted_params = transformer.params
        self.fitted_steps.append(
            {
                "name": name,
                "type": transformer_type,
                "applier": applier,
                "artifact": fitted_params,
            }
        )
        return current_data, fitted_params, transformer

    # ------------------------------------------------------------------
    # Metrics collection
    # ------------------------------------------------------------------

    # Transformer-type groups, kept as class constants so dispatch is data-driven.
    _IMPUTATION_TYPES = {"SimpleImputer", "KNNImputer", "IterativeImputer"}
    _FEATURE_SELECTION_TYPES = {
        "feature_selection",
        "UnivariateSelection",
        "ModelBasedSelection",
        "VarianceThreshold",
    }
    _SCALING_TYPES = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"}
    _OUTLIER_TYPES = {"IQR", "Winsorize", "ZScore", "EllipticEnvelope"}
    _BUCKETING_TYPES = {
        "GeneralBinning",
        "EqualWidthBinning",
        "EqualFrequencyBinning",
        "CustomBinning",
        "KBinsDiscretizer",
    }
    _FEATURE_GEN_TYPES = {"FeatureMath", "FeatureGenerationNode"}
    _ROW_DROP_TYPES = {
        "DropMissingRows",
        "Deduplicate",
        "IQR",
        "ZScore",
        "EllipticEnvelope",
        "Winsorize",
    }
    _ENCODER_TYPES = {
        "OneHotEncoder",
        "LabelEncoder",
        "OrdinalEncoder",
        "TargetEncoder",
        "HashEncoder",
        "DummyEncoder",
    }

    def _collect_step_metrics(
        self,
        *,
        transformer_type: str,
        fitted_params: Dict[str, Any],
        data_before: Any,
        current_data: Any,
        params: Dict[str, Any],
        rows_before: int,
        cols_before: Any,
        rows_after: int,
        cols_after: Any,
        name: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Aggregate per-step metrics into the running metrics dict."""
        try:
            if fitted_params:
                self._metrics_from_fitted_params(
                    transformer_type, fitted_params, data_before, current_data, metrics
                )
        except Exception as e:
            logger.warning(f"Failed to retrieve metrics for step {name}: {e}")

        if transformer_type in {"Oversampling", "Undersampling"}:
            self._metrics_resampling(current_data, params, metrics)

        if rows_after > 0 or cols_after:
            self._metrics_shape_change(
                transformer_type,
                data_before,
                current_data,
                params,
                rows_before,
                cols_before,
                rows_after,
                cols_after,
                metrics,
            )

    def _metrics_from_fitted_params(
        self,
        transformer_type: str,
        fitted_params: Dict[str, Any],
        data_before: Any,
        current_data: Any,
        metrics: Dict[str, Any],
    ) -> None:
        if transformer_type in self._IMPUTATION_TYPES:
            for key in ("missing_counts", "total_missing", "fill_values"):
                if key in fitted_params:
                    metrics[key] = fitted_params[key]

        if transformer_type in self._FEATURE_SELECTION_TYPES:
            for key in (
                "feature_scores",
                "p_values",
                "feature_importances",
                "variances",
                "ranking",
                "selected_columns",
            ):
                if key in fitted_params:
                    metrics[key] = fitted_params[key]

        if transformer_type in self._SCALING_TYPES:
            for key in (
                "mean",
                "scale",
                "var",
                "min",
                "data_min",
                "data_max",
                "center",
                "max_abs",
                "columns",
            ):
                if key in fitted_params:
                    metrics[key] = fitted_params[key]

        if transformer_type in self._OUTLIER_TYPES:
            if "warnings" in fitted_params:
                metrics["warnings"] = fitted_params["warnings"]
        if transformer_type in {"IQR", "Winsorize"} and "bounds" in fitted_params:
            metrics["bounds"] = fitted_params["bounds"]
        if transformer_type == "ZScore" and "stats" in fitted_params:
            metrics["stats"] = fitted_params["stats"]
        if transformer_type == "EllipticEnvelope" and "contamination" in fitted_params:
            metrics["contamination"] = fitted_params["contamination"]

        if transformer_type in self._BUCKETING_TYPES:
            for key in ("bin_edges", "n_bins"):
                if key in fitted_params:
                    metrics[key] = fitted_params[key]

        if transformer_type in self._FEATURE_GEN_TYPES:
            if "operations" in fitted_params:
                metrics["operations_count"] = len(fitted_params["operations"])
                metrics["operations"] = fitted_params["operations"]
            new_cols = self._diff_generated_columns(data_before, current_data)
            if new_cols is not None:
                metrics["generated_features"] = new_cols

    @staticmethod
    def _diff_generated_columns(data_before: Any, current_data: Any):
        """Return the set of newly added columns between two pipeline data objects.

        Handles plain DataFrames, SplitDatasets of DataFrames, and (X, y) tuple variants.
        Returns None if the structures don't allow a meaningful diff.
        """
        if isinstance(data_before, (pd.DataFrame, SkyulfDataFrame)) and isinstance(
            current_data, (pd.DataFrame, SkyulfDataFrame)
        ):
            return list(set(current_data.columns) - set(data_before.columns))

        if isinstance(data_before, SplitDataset) and isinstance(current_data, SplitDataset):
            before_train, after_train = data_before.train, current_data.train
            if isinstance(before_train, (pd.DataFrame, SkyulfDataFrame)) and isinstance(
                after_train, (pd.DataFrame, SkyulfDataFrame)
            ):
                return list(set(after_train.columns) - set(before_train.columns))
            if isinstance(before_train, tuple) and isinstance(after_train, tuple):
                x_before, _ = before_train
                x_after, _ = after_train
                if isinstance(x_before, (pd.DataFrame, SkyulfDataFrame)) and isinstance(
                    x_after, (pd.DataFrame, SkyulfDataFrame)
                ):
                    return list(set(x_after.columns) - set(x_before.columns))
        return None

    @staticmethod
    def _extract_y_for_resampling(current_data: Any, params: Dict[str, Any]):
        """Pull the target Series out of whatever shape the resampler produced."""
        if isinstance(current_data, SplitDataset):
            if isinstance(current_data.train, tuple):
                _, y_res = current_data.train
                return y_res
            if isinstance(current_data.train, (pd.DataFrame, SkyulfDataFrame)):
                target_col = params.get("target_column")
                if target_col and target_col in current_data.train.columns:
                    return current_data.train[target_col]
        elif isinstance(current_data, tuple):
            _, y_res = current_data
            return y_res
        elif isinstance(current_data, (pd.DataFrame, SkyulfDataFrame)):
            target_col = params.get("target_column")
            if target_col and target_col in current_data.columns:
                return current_data[target_col]
        return None

    def _metrics_resampling(
        self, current_data: Any, params: Dict[str, Any], metrics: Dict[str, Any]
    ) -> None:
        try:
            y_res: Any = self._extract_y_for_resampling(current_data, params)
            if y_res is None:
                return
            if hasattr(y_res, "to_pandas"):
                y_res = y_res.to_pandas()
            counts = y_res.value_counts().to_dict()
            metrics["class_counts"] = {str(k): int(v) for k, v in counts.items()}
            metrics["total_samples"] = int(len(y_res))
        except Exception as e:
            logger.warning(f"Failed to calculate resampling metrics: {e}")

    @staticmethod
    def _count_winsorize_diffs(d1: Any, d2: Any) -> int:
        """Count cells that differ between two data objects, for Winsorize clipping metric."""
        d1 = d1.to_pandas() if hasattr(d1, "to_pandas") else d1
        d2 = d2.to_pandas() if hasattr(d2, "to_pandas") else d2

        if isinstance(d1, pd.DataFrame) and isinstance(d2, pd.DataFrame):
            if d1.shape == d2.shape:
                return int(d1.ne(d2).sum().sum())
            return 0

        if isinstance(d1, tuple) and isinstance(d2, tuple) and len(d1) == 2 and len(d2) == 2:
            diffs = 0
            x1 = d1[0].to_pandas() if hasattr(d1[0], "to_pandas") else d1[0]
            x2 = d2[0].to_pandas() if hasattr(d2[0], "to_pandas") else d2[0]
            if (
                isinstance(x1, pd.DataFrame)
                and isinstance(x2, pd.DataFrame)
                and x1.shape == x2.shape
            ):
                diffs += int(x1.ne(x2).sum().sum())
            y1 = d1[1].to_pandas() if hasattr(d1[1], "to_pandas") else d1[1]
            y2 = d2[1].to_pandas() if hasattr(d2[1], "to_pandas") else d2[1]
            if (
                isinstance(y1, (pd.DataFrame, pd.Series))
                and isinstance(y2, (pd.DataFrame, pd.Series))
                and y1.shape == y2.shape
            ):
                diffs += int(y1.ne(y2).sum().sum())
            return diffs
        return 0

    def _metrics_winsorize_clipped(
        self, data_before: Any, current_data: Any, metrics: Dict[str, Any]
    ) -> None:
        try:
            clipped_count = 0
            if isinstance(data_before, (pd.DataFrame, SkyulfDataFrame)) and isinstance(
                current_data, (pd.DataFrame, SkyulfDataFrame)
            ):
                clipped_count = self._count_winsorize_diffs(data_before, current_data)
            elif isinstance(data_before, SplitDataset) and isinstance(current_data, SplitDataset):
                clipped_count += self._count_winsorize_diffs(data_before.train, current_data.train)
                clipped_count += self._count_winsorize_diffs(data_before.test, current_data.test)
                clipped_count += self._count_winsorize_diffs(
                    data_before.validation, current_data.validation
                )
            metrics["values_clipped"] = clipped_count
        except Exception as e:
            logger.warning(f"Failed to calculate values_clipped for Winsorize: {e}")

    def _metrics_shape_change(
        self,
        transformer_type: str,
        data_before: Any,
        current_data: Any,
        params: Dict[str, Any],
        rows_before: int,
        cols_before: Any,
        rows_after: int,
        cols_after: Any,
        metrics: Dict[str, Any],
    ) -> None:
        if transformer_type in self._ROW_DROP_TYPES:
            dropped = rows_before - rows_after
            metrics[f"{transformer_type}_rows_removed"] = dropped
            metrics[f"{transformer_type}_rows_remaining"] = rows_after
            metrics[f"{transformer_type}_rows_total"] = rows_before
            metrics["rows_removed"] = dropped
            metrics["rows_total"] = rows_before
            if transformer_type == "Winsorize":
                self._metrics_winsorize_clipped(data_before, current_data, metrics)

        if transformer_type == "MissingIndicator":
            new_cols_set = cols_after - cols_before
            metrics["missing_indicators_created"] = len(new_cols_set)
            metrics["missing_indicators_columns"] = list(new_cols_set)

        if transformer_type in {"DropMissingColumns", "feature_selection"}:
            dropped_cols_set = cols_before - cols_after
            metrics["dropped_columns"] = list(dropped_cols_set)
            metrics["dropped_columns_count"] = len(dropped_cols_set)

        if transformer_type in self._ENCODER_TYPES:
            new_cols_set = cols_after - cols_before
            metrics["new_features_count"] = len(new_cols_set)
            metrics["encoded_columns_count"] = len(params.get("columns", []))
            if "categories_count" in params:
                metrics["categories_count"] = params["categories_count"]
            if "classes_count" in params:
                metrics["classes_count"] = params["classes_count"]

    def _get_transformer_components(self, type_name: str):
        try:
            return (
                NodeRegistry.get_calculator(type_name)(),
                NodeRegistry.get_applier(type_name)(),
            )
        except ValueError:
            raise ValueError(f"Unknown transformer type: {type_name}")
