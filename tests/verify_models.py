import sys
from pathlib import Path

# Add skyulf-core to path
sys.path.append(str(Path(__file__).parent.parent / "skyulf-core"))

from skyulf.registry import NodeRegistry
from skyulf.modeling import classification, regression

def verify_models():
    # Force import to trigger decorators
    print("Importing classification module...")
    print("Importing regression module...")
    
    # Access class methods directly
    all_metadata = NodeRegistry.get_all_metadata()
    
    expected_models = [
        "logistic_regression", "random_forest_classifier",
        "svc", "k_neighbors_classifier", "decision_tree_classifier",
        "gradient_boosting_classifier", "adaboost_classifier", "xgboost_classifier",
        "gaussian_nb",
        
        "ridge_regression", "random_forest_regressor",
        "lasso_regression", "elasticnet_regression",
        "svr", "k_neighbors_regressor", "decision_tree_regressor",
        "gradient_boosting_regressor", "adaboost_regressor", "xgboost_regressor"
    ]
    
    print("\n--- Verifying Models in Registry ---")
    missing = []
    for model_id in expected_models:
        # Check if ID in dict keys
        if model_id in all_metadata:
            print(f"[OK] {model_id}")
        else:
            print(f"[FAIL] {model_id} not found!")
            missing.append(model_id)

    if missing:
        print(f"\nFailed! Missing models: {missing}")
        sys.exit(1)
    else:
        print("\nAll models registered successfully!")

if __name__ == "__main__":
    verify_models()
