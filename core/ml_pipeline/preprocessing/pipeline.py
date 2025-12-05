"""Feature Engineering Pipeline Orchestrator."""

from typing import Any, Dict, List, Union
import pandas as pd
import logging

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
    WinsorizeCalculator, WinsorizeApplier
)
from .transformations import (
    PowerTransformerCalculator, PowerTransformerApplier,
    SimpleTransformationCalculator, SimpleTransformationApplier
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
            
            # Capture metrics before
            rows_before = 0
            cols_before = set()
            if isinstance(current_data, pd.DataFrame):
                rows_before = len(current_data)
                cols_before = set(current_data.columns)
            
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
                # TrainTestSplitter changes DataFrame -> SplitDataset.
                # We bypass StatefulTransformer to allow this structural change.
                # It can also handle (X, y) tuple if FeatureTargetSplit was done first.
                if isinstance(current_data, (pd.DataFrame, tuple)):
                    params = calculator.fit(current_data, params)
                    current_data = applier.apply(current_data, params)
                    if artifact_store:
                        artifact_store.save(step_node_id, params)
                else:
                    logger.warning("Attempting to split an already split dataset. Skipping TrainTestSplitter.")
            
            elif transformer_type == "feature_target_split":
                 # FeatureTargetSplitter changes structure to (X, y) or Dict of (X, y).
                 # We bypass StatefulTransformer to allow this structural change.
                 params = calculator.fit(current_data, params)
                 current_data = applier.apply(current_data, params)
                 if artifact_store:
                    artifact_store.save(step_node_id, params)

            else:
                current_data = transformer.fit_transform(current_data, params)
            
            # Capture metrics after
            if isinstance(current_data, pd.DataFrame):
                rows_after = len(current_data)
                cols_after = set(current_data.columns)
                
                if transformer_type in ["DropMissingRows", "Deduplicate"]:
                    dropped = rows_before - rows_after
                    metrics[f"{transformer_type}_rows_removed"] = dropped
                    metrics[f"{transformer_type}_rows_remaining"] = rows_after
                    metrics[f"{transformer_type}_rows_total"] = rows_before
                
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
        elif type_name == "PowerTransformer":
            return PowerTransformerCalculator(), PowerTransformerApplier()
        elif type_name == "SimpleTransformation":
            return SimpleTransformationCalculator(), SimpleTransformationApplier()
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
