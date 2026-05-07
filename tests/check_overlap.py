import sys
import os

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../skyulf-core")))

from skyulf.registry import NodeRegistry as SkyulfRegistry
from backend.ml_pipeline.constants import StepType  # noqa: F401

dynamic_ids = set(SkyulfRegistry.get_all_metadata().keys())

print(f"Dynamic Nodes Found: {len(dynamic_ids)}")
print(f"Dynamic IDs: {sorted(list(dynamic_ids))}")

# I will recreate the list of static IDs based on the user's file content provided in context
# to verify exactly which ones can be deleted.
static_ids = [
    StepType.DATA_LOADER,
    "TrainTestSplitter",
    "feature_target_split",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "TargetEncoder",
    "HashEncoder",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "IQR",
    "ZScore",
    "Winsorize",
    "PowerTransformer",
    "SimpleTransformation",
    "GeneralBinning",
    "CustomBinning",
    "KBinsDiscretizer",
    "VarianceThreshold",
    "CorrelationThreshold",
    "UnivariateSelection",
    "ModelBasedSelection",
    "feature_selection",
    "Casting",
    "PolynomialFeatures",
    "FeatureGenerationNode",
    "Oversampling",
    "Undersampling",
    "DatasetProfile",
    "DataSnapshot",
    "TextCleaning",
    "ValueReplacement",
    "AliasReplacement",
    "InvalidValueReplacement",
    "Deduplicate",
    "DropMissingColumns",
    "DropMissingRows",
    "MissingIndicator",
    "logistic_regression",
    "random_forest_classifier",
    "ridge_regression",
    "random_forest_regressor",
]

redundant = []
unique_static = []

for sid in static_ids:
    if sid in dynamic_ids:
        redundant.append(sid)
    else:
        unique_static.append(sid)

print(f"\nRedundant Nodes (Covered by Dynamic): {len(redundant)}")
print(redundant)

print(f"\nUnique Static Nodes (Keep these): {len(unique_static)}")
print(unique_static)
