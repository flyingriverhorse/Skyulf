import importlib
import pkgutil
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../skyulf-core')))

from skyulf.registry import NodeRegistry

def import_submodules(package_name):
    """Import all submodules of a module, recursively, including subpackages"""
    package = importlib.import_module(package_name)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        try:
            results[full_name] = importlib.import_module(full_name)
        except ImportError as e:
            print(f"Failed to import {full_name}: {e}")
            pass
    return results

print("Importing skyulf modules...")
import_submodules("skyulf.preprocessing")
import_submodules("skyulf.modeling")

print("\nChecking Registry...")
metadata = getattr(NodeRegistry, "_metadata", {})
print(f"Found {len(metadata)} nodes with metadata in Registry.")

expected_nodes = [
    "Casting", "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "TargetEncoder", 
    "HashEncoder", "DummyEncoder", "IQR", "ZScore", "Winsorize", "ManualBounds", 
    "EllipticEnvelope", "VarianceThreshold", "CorrelationThreshold", "UnivariateSelection",
    "ModelBasedSelection", "FeatureSelection", "PowerTransformer", "SimpleTransformation",
    "GeneralTransformation", "Split", "feature_target_split", "Oversampling", 
    "Undersampling", "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "DatasetProfile", "DataSnapshot", "SimpleImputer", "KNNImputer", "IterativeImputer",
    "GeneralBinning", "CustomBinning", "KBinsDiscretizer", "TextCleaning", 
    "InvalidValueReplacement", "ValueReplacement", "AliasReplacement", "Deduplicate",
    "DropMissingColumns", "DropMissingRows", "MissingIndicator", "PolynomialFeatures",
    "FeatureGeneration", "logistic_regression", "random_forest_classifier", 
    "ridge_regression", "random_forest_regressor"
]

missing = []
for node in expected_nodes:
    if node not in metadata:
        missing.append(node)

if missing:
    print(f"❌ Missing nodes: {missing}")
    sys.exit(1)
else:
    print("✅ All expected nodes are present.")

print("\nList of registered nodes:")
for k, v in metadata.items():
    print(f" - [{k}]: {v['name']}")
