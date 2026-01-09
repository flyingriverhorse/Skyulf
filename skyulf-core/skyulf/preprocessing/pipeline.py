"""Feature Engineering Pipeline Orchestrator."""

import logging
from typing import Any, Dict, List, Union, cast

import pandas as pd

from ..types import PreprocessingStepConfig
from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame
from ..utils import get_data_stats
from ..registry import NodeRegistry
from .base import StatefulTransformer

# Import modules to ensure nodes are registered
from . import (
    bucketing,
    casting,
    cleaning,
    drop_and_missing,
    encoding,
    feature_generation,
    feature_selection,
    imputation,
    inspection,
    outliers,
    resampling,
    scaling,
    split,
    transformations,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Orchestrates a sequence of feature engineering steps.
    """

    def __init__(self, steps_config: List[PreprocessingStepConfig]):
        self.steps_config = steps_config
        self.fitted_steps: List[Dict[str, Any]] = []

    def transform(self, data: Union[pd.DataFrame, SkyulfDataFrame]) -> Union[pd.DataFrame, SkyulfDataFrame]:
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

    def fit_transform(self, data: Union[pd.DataFrame, SkyulfDataFrame, Any], node_id_prefix="") -> Any:  # noqa: C901
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
            logger.debug(
                f"FeatureEngineer running step {i}: {name} ({transformer_type})"
            )
            logger.debug(f"current_data type: {type(current_data)}")

            # Capture metrics before
            rows_before, cols_before = get_data_stats(current_data)

            # Keep reference for comparison (for Winsorize metrics)
            data_before = current_data

            calculator, applier = self._get_transformer_components(transformer_type)

            # We need a unique ID for this step's artifacts
            step_node_id = f"{node_id_prefix}_{name}"

            transformer = StatefulTransformer(calculator, applier, step_node_id)

            # Handle special transformers that change data structure
            # Splitters return SplitDataset or (X, y) tuples instead of a simple DataFrame,
            # so they bypass the standard StatefulTransformer wrapper.

            # Initialize fitted_params
            fitted_params = {}

            if transformer_type == "TrainTestSplitter":
                logger.debug("Handling TrainTestSplitter")
                # TrainTestSplitter changes DataFrame -> SplitDataset.
                # We bypass StatefulTransformer to allow this structural change.
                # It can also handle (X, y) tuple if FeatureTargetSplit was done first.
                if isinstance(current_data, (pd.DataFrame, SkyulfDataFrame, tuple)):
                    logger.debug("Executing TrainTestSplitter logic")
                    params = calculator.fit(current_data, params)
                    current_data = applier.apply(current_data, params)
                    # In SDK, params are returned but not auto-saved to artifact store here.
                    # The Pipeline object will handle state persistence.
                else:
                    logger.debug(
                        f"Skipping TrainTestSplitter. current_data is {type(current_data)}"
                    )
                    logger.warning(
                        "Attempting to split an already split dataset. Skipping TrainTestSplitter."
                    )

            elif transformer_type == "feature_target_split":
                logger.debug("Handling feature_target_split")
                # FeatureTargetSplitter changes structure to (X, y) or Dict of (X, y).
                # We bypass StatefulTransformer to allow this structural change.
                params = calculator.fit(current_data, params)
                current_data = applier.apply(current_data, params)

            else:
                logger.debug("Handling standard transformer via StatefulTransformer")
                current_data = transformer.fit_transform(current_data, params)
                # In SDK, transformer.params holds the state.
                fitted_params = transformer.params

                self.fitted_steps.append(
                    {
                        "name": name,
                        "type": transformer_type,
                        "applier": applier,
                        "artifact": fitted_params,
                    }
                )

            logger.debug(f"Step {i} complete. New data type: {type(current_data)}")

            # Retrieve fitted params to get metrics from the calculator
            try:
                if fitted_params:
                    # Imputation Metrics
                    if transformer_type in [
                        "SimpleImputer",
                        "KNNImputer",
                        "IterativeImputer",
                    ]:
                        if "missing_counts" in fitted_params:
                            metrics["missing_counts"] = fitted_params["missing_counts"]
                        if "total_missing" in fitted_params:
                            metrics["total_missing"] = fitted_params["total_missing"]
                        if "fill_values" in fitted_params:
                            metrics["fill_values"] = fitted_params["fill_values"]

                    # Feature Selection Metrics
                    if transformer_type in [
                        "feature_selection",
                        "UnivariateSelection",
                        "ModelBasedSelection",
                        "VarianceThreshold",
                    ]:
                        if "feature_scores" in fitted_params:
                            metrics["feature_scores"] = fitted_params["feature_scores"]
                        if "p_values" in fitted_params:
                            metrics["p_values"] = fitted_params["p_values"]
                        if "feature_importances" in fitted_params:
                            metrics["feature_importances"] = fitted_params[
                                "feature_importances"
                            ]
                        if "variances" in fitted_params:
                            metrics["variances"] = fitted_params["variances"]
                        if "ranking" in fitted_params:
                            metrics["ranking"] = fitted_params["ranking"]
                        if "selected_columns" in fitted_params:
                            metrics["selected_columns"] = fitted_params[
                                "selected_columns"
                            ]

                    # Scaling Metrics
                    if transformer_type in [
                        "StandardScaler",
                        "MinMaxScaler",
                        "RobustScaler",
                        "MaxAbsScaler",
                    ]:
                        if "mean" in fitted_params:
                            metrics["mean"] = fitted_params["mean"]
                        if "scale" in fitted_params:
                            metrics["scale"] = fitted_params["scale"]
                        if "var" in fitted_params:
                            metrics["var"] = fitted_params["var"]
                        if "min" in fitted_params:
                            metrics["min"] = fitted_params["min"]
                        if "data_min" in fitted_params:
                            metrics["data_min"] = fitted_params["data_min"]
                        if "data_max" in fitted_params:
                            metrics["data_max"] = fitted_params["data_max"]
                        if "center" in fitted_params:
                            metrics["center"] = fitted_params["center"]
                        if "max_abs" in fitted_params:
                            metrics["max_abs"] = fitted_params["max_abs"]
                        if "columns" in fitted_params:
                            metrics["columns"] = fitted_params["columns"]

                    # Outlier Metrics
                    if transformer_type in [
                        "IQR",
                        "Winsorize",
                        "ZScore",
                        "EllipticEnvelope",
                    ]:
                        if "warnings" in fitted_params:
                            metrics["warnings"] = fitted_params["warnings"]

                    if transformer_type in ["IQR", "Winsorize"]:
                        if "bounds" in fitted_params:
                            metrics["bounds"] = fitted_params["bounds"]

                    if transformer_type == "ZScore":
                        if "stats" in fitted_params:
                            metrics["stats"] = fitted_params["stats"]

                    if transformer_type == "EllipticEnvelope":
                        if "contamination" in fitted_params:
                            metrics["contamination"] = fitted_params["contamination"]

                    # Bucketing Metrics
                    if transformer_type in [
                        "GeneralBinning",
                        "EqualWidthBinning",
                        "EqualFrequencyBinning",
                        "CustomBinning",
                        "KBinsDiscretizer",
                    ]:
                        if "bin_edges" in fitted_params:
                            metrics["bin_edges"] = fitted_params["bin_edges"]
                        if "n_bins" in fitted_params:
                            metrics["n_bins"] = fitted_params["n_bins"]

                    # Feature Generation Metrics
                    if transformer_type in ["FeatureMath", "FeatureGenerationNode"]:
                        if "operations" in fitted_params:
                            metrics["operations_count"] = len(
                                fitted_params["operations"]
                            )
                            metrics["operations"] = fitted_params["operations"]
                        # Calculate generated features by comparing columns
                        if isinstance(data_before, (pd.DataFrame, SkyulfDataFrame)) and isinstance(
                            current_data, (pd.DataFrame, SkyulfDataFrame)
                        ):
                            new_cols = list(
                                set(current_data.columns) - set(data_before.columns)
                            )
                            metrics["generated_features"] = new_cols
                        elif isinstance(data_before, SplitDataset) and isinstance(
                            current_data, SplitDataset
                        ):
                            # Check train set
                            if isinstance(
                                data_before.train, (pd.DataFrame, SkyulfDataFrame)
                            ) and isinstance(current_data.train, (pd.DataFrame, SkyulfDataFrame)):
                                new_cols = list(
                                    set(current_data.train.columns)
                                    - set(data_before.train.columns)
                                )
                                metrics["generated_features"] = new_cols
                            elif isinstance(data_before.train, tuple) and isinstance(
                                current_data.train, tuple
                            ):
                                # (X, y) tuple
                                X_before, _ = data_before.train
                                X_after, _ = current_data.train
                                if isinstance(X_before, (pd.DataFrame, SkyulfDataFrame)) and isinstance(
                                    X_after, (pd.DataFrame, SkyulfDataFrame)
                                ):
                                    new_cols = list(
                                        set(X_after.columns) - set(X_before.columns)
                                    )
                                    metrics["generated_features"] = new_cols

            except Exception as e:
                logger.warning(f"Failed to retrieve metrics for step {name}: {e}")

            # Capture metrics after
            rows_after, cols_after = get_data_stats(current_data)

            # Resampling Metrics (Calculated from data)
            if transformer_type in ["Oversampling", "Undersampling"]:
                try:
                    # Extract y to calculate class counts
                    y_res = None
                    if isinstance(current_data, SplitDataset):
                        if isinstance(current_data.train, tuple):
                            _, y_res = current_data.train
                        elif isinstance(current_data.train, (pd.DataFrame, SkyulfDataFrame)):
                            # Try to find target column from params
                            target_col = params.get("target_column")
                            if target_col and target_col in current_data.train.columns:
                                y_res = current_data.train[target_col]
                    elif isinstance(current_data, tuple):
                        _, y_res = current_data
                    elif isinstance(current_data, (pd.DataFrame, SkyulfDataFrame)):
                        target_col = params.get("target_column")
                        if target_col and target_col in current_data.columns:
                            y_res = current_data[target_col]

                    if y_res is not None:
                        # Convert to pandas for consistent metric calculation if needed
                        if hasattr(y_res, "to_pandas"):
                            y_res = y_res.to_pandas()

                        counts = y_res.value_counts().to_dict()
                        # Convert keys to string to ensure JSON serializability
                        metrics["class_counts"] = {
                            str(k): int(v) for k, v in counts.items()
                        }
                        metrics["total_samples"] = int(len(y_res))
                except Exception as e:
                    logger.warning(f"Failed to calculate resampling metrics: {e}")

            if rows_after > 0 or cols_after:
                if transformer_type in [
                    "DropMissingRows",
                    "Deduplicate",
                    "IQR",
                    "ZScore",
                    "EllipticEnvelope",
                    "Winsorize",
                ]:
                    dropped = rows_before - rows_after
                    metrics[f"{transformer_type}_rows_removed"] = dropped
                    metrics[f"{transformer_type}_rows_remaining"] = rows_after
                    metrics[f"{transformer_type}_rows_total"] = rows_before
                    metrics["rows_removed"] = dropped
                    metrics["rows_total"] = rows_before

                    # Special metric for Winsorize: Values Clipped
                    if transformer_type == "Winsorize":
                        try:
                            clipped_count = 0

                            # Helper to count diffs
                            def count_diffs(df1, df2):
                                # Convert to pandas for comparison
                                d1 = df1.to_pandas() if hasattr(df1, "to_pandas") else df1
                                d2 = df2.to_pandas() if hasattr(df2, "to_pandas") else df2

                                if isinstance(d1, pd.DataFrame) and isinstance(
                                    d2, pd.DataFrame
                                ):
                                    if d1.shape == d2.shape:
                                        return int(d1.ne(d2).sum().sum())
                                elif (
                                    isinstance(d1, tuple)
                                    and isinstance(d2, tuple)
                                    and len(d1) == 2
                                    and len(d2) == 2
                                ):
                                    # Handle (X, y) tuple
                                    diffs = 0
                                    # Compare X (index 0)
                                    x1 = d1[0].to_pandas() if hasattr(d1[0], "to_pandas") else d1[0]
                                    x2 = d2[0].to_pandas() if hasattr(d2[0], "to_pandas") else d2[0]

                                    if isinstance(x1, pd.DataFrame) and isinstance(
                                        x2, pd.DataFrame
                                    ):
                                        if x1.shape == x2.shape:
                                            diffs += int(x1.ne(x2).sum().sum())
                                    # Compare y (index 1) - usually Series
                                    y1 = d1[1].to_pandas() if hasattr(d1[1], "to_pandas") else d1[1]
                                    y2 = d2[1].to_pandas() if hasattr(d2[1], "to_pandas") else d2[1]

                                    if isinstance(
                                        y1, (pd.DataFrame, pd.Series)
                                    ) and isinstance(y2, (pd.DataFrame, pd.Series)):
                                        if y1.shape == y2.shape:
                                            diffs += int(y1.ne(y2).sum().sum())  # type: ignore
                                    return diffs
                                return 0

                            if isinstance(data_before, (pd.DataFrame, SkyulfDataFrame)) and isinstance(
                                current_data, (pd.DataFrame, SkyulfDataFrame)
                            ):
                                clipped_count = count_diffs(data_before, current_data)
                            elif isinstance(data_before, SplitDataset) and isinstance(
                                current_data, SplitDataset
                            ):
                                clipped_count += count_diffs(
                                    data_before.train, current_data.train
                                )
                                clipped_count += count_diffs(
                                    data_before.test, current_data.test
                                )
                                clipped_count += count_diffs(
                                    data_before.validation, current_data.validation
                                )

                            metrics["values_clipped"] = clipped_count
                        except Exception as e:
                            logger.warning(
                                f"Failed to calculate values_clipped for Winsorize: {e}"
                            )
                            pass

                if transformer_type == "MissingIndicator":
                    new_cols_set = cols_after - cols_before
                    metrics["missing_indicators_created"] = len(new_cols_set)
                    cast(Dict[str, Any], metrics)["missing_indicators_columns"] = list(
                        new_cols_set
                    )

                if transformer_type == "DropMissingColumns":
                    dropped_cols_set = cols_before - cols_after
                    cast(Dict[str, Any], metrics)["dropped_columns"] = list(
                        dropped_cols_set
                    )
                    metrics["dropped_columns_count"] = len(dropped_cols_set)

                if transformer_type == "feature_selection":
                    dropped_cols_set = cols_before - cols_after
                    cast(Dict[str, Any], metrics)["dropped_columns"] = list(
                        dropped_cols_set
                    )
                    metrics["dropped_columns_count"] = len(dropped_cols_set)

                if transformer_type in [
                    "OneHotEncoder",
                    "LabelEncoder",
                    "OrdinalEncoder",
                    "TargetEncoder",
                    "HashEncoder",
                    "DummyEncoder",
                ]:
                    new_cols_set = cols_after - cols_before
                    metrics["new_features_count"] = len(new_cols_set)
                    metrics["encoded_columns_count"] = len(params.get("columns", []))

                    if "categories_count" in params:
                        metrics["categories_count"] = params["categories_count"]
                    if "classes_count" in params:
                        metrics["classes_count"] = params["classes_count"]

        return current_data, metrics

    def _get_transformer_components(self, type_name: str):  # noqa: C901
        # Try Registry first
        try:
            return (
                NodeRegistry.get_calculator(type_name)(),
                NodeRegistry.get_applier(type_name)(),
            )
        except ValueError:
            raise ValueError(f"Unknown transformer type: {type_name}")
