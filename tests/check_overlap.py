import sys
import os
from typing import List

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../skyulf-core')))

# Import backend registry
from backend.ml_pipeline.node_definitions import NodeRegistry, RegistryItem, StepType

# Fetch static nodes by creating a temporary access to the list inside get_all_nodes
# Since get_all_nodes instantiates the list inside the function, we have to copy-paste or inspect it.
# However, we can reconstruct the static list by temporarily disabling SKYULF_AVAILABLE in the module 
# or just manually inspecting.

# Actually, get_all_nodes returns (static - dynamic) + dynamic.
# If we want to see what WAS in static, we need to inspect the source or trust the dynamic override logic.

# Let's verify what dynamic nodes are currently available.
dynamic_nodes = NodeRegistry.get_dynamic_nodes()
dynamic_ids = {n.id for n in dynamic_nodes}

print(f"Dynamic Nodes Found: {len(dynamic_nodes)}")
print(f"Dynamic IDs: {sorted(list(dynamic_ids))}")

# I will recreate the list of static IDs based on the user's file content provided in context
# to verify exactly which ones can be deleted.
static_ids = [
    StepType.DATA_LOADER,
    "TrainTestSplitter", "feature_target_split",
    "SimpleImputer", "KNNImputer", "IterativeImputer",
    "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "TargetEncoder", "HashEncoder",
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "IQR", "ZScore", "Winsorize",
    "PowerTransformer", "SimpleTransformation", "GeneralBinning", "CustomBinning", "KBinsDiscretizer",
    "VarianceThreshold", "CorrelationThreshold", "UnivariateSelection", "ModelBasedSelection", "feature_selection",
    "Casting", "PolynomialFeatures", "FeatureGenerationNode",
    "Oversampling", "Undersampling",
    "DatasetProfile", "DataSnapshot",
    "TextCleaning", "ValueReplacement", "AliasReplacement", "InvalidValueReplacement",
    "Deduplicate", "DropMissingColumns", "DropMissingRows", "MissingIndicator",
    "logistic_regression", "random_forest_classifier", "ridge_regression", "random_forest_regressor"
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
