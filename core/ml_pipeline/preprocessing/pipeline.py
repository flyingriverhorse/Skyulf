"""Feature Engineering Pipeline Orchestrator."""

from typing import Any, Dict, List, Union
import pandas as pd
import logging
from ..data.container import SplitDataset
from ..utils import get_data_stats

# Import all transformers
from .split import (
    TrainTestSplitterCalculator, TrainTestSplitterApplier,
    FeatureTargetSplitCalculator, FeatureTargetSplitApplier
)
from .cleaning import (
    TextCleaningCalculator, TextCleaningApplier,
    ValueReplacementCalculator, ValueReplacementApplier,
    AliasReplacementCalculator, AliasReplacementApplier,
    InvalidValueReplacementCalculator, InvalidValueReplacementApplier,
    DateStandardizerCalculator, DateStandardizerApplier
)
from .drop_and_missing import (
    DeduplicateCalculator, DeduplicateApplier,
    DropMissingColumnsCalculator, DropMissingColumnsApplier,
    DropMissingRowsCalculator, DropMissingRowsApplier,
    MissingIndicatorCalculator, MissingIndicatorApplier
)
from .imputation import (
    SimpleImputerCalculator, SimpleImputerApplier,
    KNNImputerCalculator, KNNImputerApplier,
    IterativeImputerCalculator, IterativeImputerApplier
)
from .encoding import (
    OneHotEncoderCalculator, OneHotEncoderApplier,
    OrdinalEncoderCalculator, OrdinalEncoderApplier,
    LabelEncoderCalculator, LabelEncoderApplier,
    TargetEncoderCalculator, TargetEncoderApplier,
    HashEncoderCalculator, HashEncoderApplier
)
from .scaling import (
    StandardScalerCalculator, StandardScalerApplier,
    MinMaxScalerCalculator, MinMaxScalerApplier,
    RobustScalerCalculator, RobustScalerApplier,
    MaxAbsScalerCalculator, MaxAbsScalerApplier
)
from .outliers import (
    IQRCalculator, IQRApplier,
    ZScoreCalculator, ZScoreApplier,
    WinsorizeCalculator, WinsorizeApplier,
    EllipticEnvelopeCalculator, EllipticEnvelopeApplier
)
from .transformations import (
    PowerTransformerCalculator, PowerTransformerApplier,
    SimpleTransformationCalculator, SimpleTransformationApplier,
    GeneralTransformationCalculator, GeneralTransformationApplier
)
from .bucketing import (
    GeneralBinningCalculator, BaseBinningApplier as GeneralBinningApplier,
    CustomBinningCalculator, CustomBinningApplier,
    KBinsDiscretizerCalculator, KBinsDiscretizerApplier
)
from .feature_selection import (
    VarianceThresholdCalculator, VarianceThresholdApplier,
    CorrelationThresholdCalculator, CorrelationThresholdApplier,
    UnivariateSelectionCalculator, UnivariateSelectionApplier,
    ModelBasedSelectionCalculator, ModelBasedSelectionApplier,
    FeatureSelectionCalculator, FeatureSelectionApplier
)
from .casting import (
    CastingCalculator, CastingApplier
)
from .feature_generation import (
    PolynomialFeaturesCalculator, PolynomialFeaturesApplier,
    FeatureMathCalculator, FeatureMathApplier
)
from .resampling import (
    OversamplingCalculator, OversamplingApplier,
    UndersamplingCalculator, UndersamplingApplier
)
from .inspection import (
    DatasetProfileCalculator, DatasetProfileApplier,
    DataSnapshotCalculator, DataSnapshotApplier
)
from .base import StatefulTransformer

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Orchestrates a sequence of feature engineering steps.
    """
    def __init__(self, steps_config: List[Dict[str, Any]]):
        self.steps_config = steps_config

    def fit_transform(self, data: Union[pd.DataFrame, Any], artifact_store=None, node_id_prefix="") -> Any:
        """
        Runs the pipeline on data.
        Returns: (transformed_data, metrics_dict)
        """
        current_data = data
        metrics = {}
        
        for i, step in enumerate(self.steps_config):
            name = step["name"]
            transformer_type = step["transformer"]
            params = step.get("params", {})
            
            logger.info(f"Running step {i}: {name} ({transformer_type})")
            logger.debug(f"FeatureEngineer running step {i}: {name} ({transformer_type})")
            logger.debug(f"current_data type: {type(current_data)}")
            
            # Capture metrics before
            rows_before, cols_before = get_data_stats(current_data)
            
            # Keep reference for comparison (for Winsorize metrics)
            data_before = current_data
            
            calculator, applier = self._get_transformer_components(transformer_type)
            
            # We need a unique ID for this step's artifacts
            step_node_id = f"{node_id_prefix}_{name}"
            
            # If artifact_store is None, we might fail if the transformer tries to save.
            if artifact_store is None:
                from ..artifacts.local import LocalArtifactStore
                artifact_store = LocalArtifactStore("./temp_artifacts")

            transformer = StatefulTransformer(calculator, applier, artifact_store, step_node_id)
            
            # Some transformers return a tuple (train, test) like Splitter
            # Or (X, y) like FeatureTargetSplitter
            # We need to handle these special cases or let StatefulTransformer handle them if possible.
            # But StatefulTransformer expects SplitDataset or DataFrame.
            
            # If the transformer is a Splitter, it might return a SplitDataset directly from its apply method
            # But StatefulTransformer wraps it.
            
            # Special handling for Splitters if they don't fit the standard "fit on train, apply on all" pattern
            # Actually, TrainTestSplitter is a bit unique. It takes a DataFrame and returns a SplitDataset.
            # It doesn't really "fit" anything.
            
            if transformer_type == "TrainTestSplitter":
                logger.debug("Handling TrainTestSplitter")
                # TrainTestSplitter changes DataFrame -> SplitDataset.
                # We bypass StatefulTransformer to allow this structural change.
                # It can also handle (X, y) tuple if FeatureTargetSplit was done first.
                if isinstance(current_data, (pd.DataFrame, tuple)):
                    logger.debug("Executing TrainTestSplitter logic")
                    params = calculator.fit(current_data, params)
                    current_data = applier.apply(current_data, params)
                    if artifact_store:
                        artifact_store.save(step_node_id, params)
                else:
                    logger.debug(f"Skipping TrainTestSplitter. current_data is {type(current_data)}")
                    logger.warning("Attempting to split an already split dataset. Skipping TrainTestSplitter.")
            
            elif transformer_type == "feature_target_split":
                 logger.debug("Handling feature_target_split")
                 # FeatureTargetSplitter changes structure to (X, y) or Dict of (X, y).
                 # We bypass StatefulTransformer to allow this structural change.
                 params = calculator.fit(current_data, params)
                 current_data = applier.apply(current_data, params)
                 if artifact_store:
                    artifact_store.save(step_node_id, params)

            else:
                logger.debug("Handling standard transformer via StatefulTransformer")
                current_data = transformer.fit_transform(current_data, params)
            
            logger.debug(f"Step {i} complete. New data type: {type(current_data)}")
            
            # Retrieve fitted params to get metrics from the calculator
            try:
                if artifact_store:
                    fitted_params = artifact_store.load(step_node_id)
                    if fitted_params:
                        # Imputation Metrics
                        if transformer_type in ["SimpleImputer", "KNNImputer", "IterativeImputer"]:
                            if "missing_counts" in fitted_params:
                                metrics["missing_counts"] = fitted_params["missing_counts"]
                            if "total_missing" in fitted_params:
                                metrics["total_missing"] = fitted_params["total_missing"]
                            if "fill_values" in fitted_params:
                                metrics["fill_values"] = fitted_params["fill_values"]
                        
                        # Feature Selection Metrics
                        if transformer_type in ["feature_selection", "UnivariateSelection", "ModelBasedSelection", "VarianceThreshold"]:
                            if "feature_scores" in fitted_params:
                                metrics["feature_scores"] = fitted_params["feature_scores"]
                            if "p_values" in fitted_params:
                                metrics["p_values"] = fitted_params["p_values"]
                            if "feature_importances" in fitted_params:
                                metrics["feature_importances"] = fitted_params["feature_importances"]
                            if "variances" in fitted_params:
                                metrics["variances"] = fitted_params["variances"]
                            if "ranking" in fitted_params:
                                metrics["ranking"] = fitted_params["ranking"]
                            if "selected_columns" in fitted_params:
                                metrics["selected_columns"] = fitted_params["selected_columns"]

                        # Scaling Metrics
                        if transformer_type in ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"]:
                            if "mean" in fitted_params: metrics["mean"] = fitted_params["mean"]
                            if "scale" in fitted_params: metrics["scale"] = fitted_params["scale"]
                            if "var" in fitted_params: metrics["var"] = fitted_params["var"]
                            if "min" in fitted_params: metrics["min"] = fitted_params["min"]
                            if "data_min" in fitted_params: metrics["data_min"] = fitted_params["data_min"]
                            if "data_max" in fitted_params: metrics["data_max"] = fitted_params["data_max"]
                            if "center" in fitted_params: metrics["center"] = fitted_params["center"]
                            if "max_abs" in fitted_params: metrics["max_abs"] = fitted_params["max_abs"]
                            if "columns" in fitted_params: metrics["columns"] = fitted_params["columns"]

                        # Outlier Metrics
                        if transformer_type in ["IQR", "Winsorize", "ZScore", "EllipticEnvelope"]:
                            if "warnings" in fitted_params: metrics["warnings"] = fitted_params["warnings"]

                        if transformer_type in ["IQR", "Winsorize"]:
                            if "bounds" in fitted_params: metrics["bounds"] = fitted_params["bounds"]
                        
                        if transformer_type == "ZScore":
                            if "stats" in fitted_params: metrics["stats"] = fitted_params["stats"]
                            
                        if transformer_type == "EllipticEnvelope":
                            if "contamination" in fitted_params: metrics["contamination"] = fitted_params["contamination"]

            except Exception as e:
                logger.warning(f"Failed to retrieve metrics for step {name}: {e}")

            # Capture metrics after
            rows_after, cols_after = get_data_stats(current_data)

            if rows_after > 0 or cols_after:
                if transformer_type in ["DropMissingRows", "Deduplicate", "IQR", "ZScore", "EllipticEnvelope", "Winsorize"]:
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
                                if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
                                    if df1.shape == df2.shape:
                                        return int(df1.ne(df2).sum().sum())
                                elif isinstance(df1, tuple) and isinstance(df2, tuple) and len(df1) == 2 and len(df2) == 2:
                                    # Handle (X, y) tuple
                                    diffs = 0
                                    # Compare X (index 0)
                                    if isinstance(df1[0], pd.DataFrame) and isinstance(df2[0], pd.DataFrame):
                                         if df1[0].shape == df2[0].shape:
                                             diffs += int(df1[0].ne(df2[0]).sum().sum())
                                    # Compare y (index 1) - usually Series
                                    if isinstance(df1[1], (pd.DataFrame, pd.Series)) and isinstance(df2[1], (pd.DataFrame, pd.Series)):
                                         if df1[1].shape == df2[1].shape:
                                             diffs += int(df1[1].ne(df2[1]).sum().sum())
                                    return diffs
                                return 0

                            if isinstance(data_before, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
                                clipped_count = count_diffs(data_before, current_data)
                            elif isinstance(data_before, SplitDataset) and isinstance(current_data, SplitDataset):
                                clipped_count += count_diffs(data_before.train, current_data.train)
                                clipped_count += count_diffs(data_before.test, current_data.test)
                                clipped_count += count_diffs(data_before.validation, current_data.validation)
                            
                            metrics["values_clipped"] = clipped_count
                        except Exception as e:
                            logger.warning(f"Failed to calculate values_clipped for Winsorize: {e}")
                            pass
                
                if transformer_type == "MissingIndicator":
                    new_cols = cols_after - cols_before
                    metrics["missing_indicators_created"] = len(new_cols)
                    metrics["missing_indicators_columns"] = list(new_cols)
                    
                if transformer_type == "DropMissingColumns":
                    dropped_cols = cols_before - cols_after
                    metrics["dropped_columns"] = list(dropped_cols)
                    metrics["dropped_columns_count"] = len(dropped_cols)

                if transformer_type == "feature_selection":
                    dropped_cols = cols_before - cols_after
                    metrics["dropped_columns"] = list(dropped_cols)
                    metrics["dropped_columns_count"] = len(dropped_cols)

                if transformer_type in ["OneHotEncoder", "LabelEncoder", "OrdinalEncoder", "TargetEncoder", "HashEncoder", "DummyEncoder"]:
                    new_cols = cols_after - cols_before
                    metrics["new_features_count"] = len(new_cols)
                    metrics["encoded_columns_count"] = len(params.get("columns", []))
                    
                    if "categories_count" in params:
                        metrics["categories_count"] = params["categories_count"]
                    if "classes_count" in params:
                        metrics["classes_count"] = params["classes_count"]

        return current_data, metrics

    def _get_transformer_components(self, type_name: str):
        if type_name == "TrainTestSplitter":
            return TrainTestSplitterCalculator(), TrainTestSplitterApplier()
        elif type_name == "feature_target_split":
            return FeatureTargetSplitCalculator(), FeatureTargetSplitApplier()
        elif type_name == "TextCleaning":
            return TextCleaningCalculator(), TextCleaningApplier()
        elif type_name == "ValueReplacement":
            return ValueReplacementCalculator(), ValueReplacementApplier()
        elif type_name == "Deduplicate":
            return DeduplicateCalculator(), DeduplicateApplier()
        elif type_name == "DropMissingColumns":
            return DropMissingColumnsCalculator(), DropMissingColumnsApplier()
        elif type_name == "DropMissingRows":
            return DropMissingRowsCalculator(), DropMissingRowsApplier()
        elif type_name == "MissingIndicator":
            return MissingIndicatorCalculator(), MissingIndicatorApplier()
        elif type_name == "AliasReplacement":
            return AliasReplacementCalculator(), AliasReplacementApplier()
        elif type_name == "InvalidValueReplacement":
            return InvalidValueReplacementCalculator(), InvalidValueReplacementApplier()
        elif type_name == "DateStandardizer":
            return DateStandardizerCalculator(), DateStandardizerApplier()
        elif type_name == "SimpleImputer":
            return SimpleImputerCalculator(), SimpleImputerApplier()
        elif type_name == "KNNImputer":
            return KNNImputerCalculator(), KNNImputerApplier()
        elif type_name == "IterativeImputer":
            return IterativeImputerCalculator(), IterativeImputerApplier()
        elif type_name == "OneHotEncoder":
            return OneHotEncoderCalculator(), OneHotEncoderApplier()
        elif type_name == "DummyEncoder":
            from .encoding import DummyEncoderCalculator, DummyEncoderApplier
            return DummyEncoderCalculator(), DummyEncoderApplier()
        elif type_name == "OrdinalEncoder":
            return OrdinalEncoderCalculator(), OrdinalEncoderApplier()
        elif type_name == "LabelEncoder":
            return LabelEncoderCalculator(), LabelEncoderApplier()
        elif type_name == "TargetEncoder":
            return TargetEncoderCalculator(), TargetEncoderApplier()
        elif type_name == "HashEncoder":
            return HashEncoderCalculator(), HashEncoderApplier()
        elif type_name == "StandardScaler":
            return StandardScalerCalculator(), StandardScalerApplier()
        elif type_name == "MinMaxScaler":
            return MinMaxScalerCalculator(), MinMaxScalerApplier()
        elif type_name == "RobustScaler":
            return RobustScalerCalculator(), RobustScalerApplier()
        elif type_name == "MaxAbsScaler":
            return MaxAbsScalerCalculator(), MaxAbsScalerApplier()
        elif type_name == "IQR":
            return IQRCalculator(), IQRApplier()
        elif type_name == "ZScore":
            return ZScoreCalculator(), ZScoreApplier()
        elif type_name == "Winsorize":
            return WinsorizeCalculator(), WinsorizeApplier()
        elif type_name == "EllipticEnvelope":
            return EllipticEnvelopeCalculator(), EllipticEnvelopeApplier()
        elif type_name == "PowerTransformer":
            return PowerTransformerCalculator(), PowerTransformerApplier()
        elif type_name == "SimpleTransformation":
            return SimpleTransformationCalculator(), SimpleTransformationApplier()
        elif type_name == "GeneralTransformation":
            return GeneralTransformationCalculator(), GeneralTransformationApplier()
        elif type_name == "GeneralBinning":
            return GeneralBinningCalculator(), GeneralBinningApplier()
        elif type_name == "CustomBinning":
            return CustomBinningCalculator(), CustomBinningApplier()
        elif type_name == "KBinsDiscretizer":
            return KBinsDiscretizerCalculator(), KBinsDiscretizerApplier()
        elif type_name == "VarianceThreshold":
            return VarianceThresholdCalculator(), VarianceThresholdApplier()
        elif type_name == "CorrelationThreshold":
            return CorrelationThresholdCalculator(), CorrelationThresholdApplier()
        elif type_name == "UnivariateSelection":
            return UnivariateSelectionCalculator(), UnivariateSelectionApplier()
        elif type_name == "ModelBasedSelection":
            return ModelBasedSelectionCalculator(), ModelBasedSelectionApplier()
        elif type_name == "feature_selection":
            return FeatureSelectionCalculator(), FeatureSelectionApplier()
        elif type_name == "Casting":
            return CastingCalculator(), CastingApplier()
        elif type_name == "PolynomialFeatures":
            return PolynomialFeaturesCalculator(), PolynomialFeaturesApplier()
        elif type_name == "FeatureMath":
            return FeatureMathCalculator(), FeatureMathApplier()
        elif type_name == "Oversampling":
            return OversamplingCalculator(), OversamplingApplier()
        elif type_name == "Undersampling":
            return UndersamplingCalculator(), UndersamplingApplier()
        elif type_name == "DatasetProfile":
            return DatasetProfileCalculator(), DatasetProfileApplier()
        elif type_name == "DataSnapshot":
            return DataSnapshotCalculator(), DataSnapshotApplier()
        else:
            raise ValueError(f"Unknown transformer type: {type_name}")
