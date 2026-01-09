import sys
import os
import logging

# Add workspace root to sys.path to ensure we can import backend and skyulf-core
sys.path.append(os.getcwd())
# Also add skyulf-core specifically if it's not installed in the env
sys.path.append(os.path.join(os.getcwd(), "skyulf-core"))

from backend.ml_pipeline.node_definitions import NodeRegistry
from skyulf.modeling.classification import LogisticRegressionCalculator, RandomForestClassifierCalculator

# Trigger registration by importing (already done via import above, but to be sure)
# The imports above should trigger the decorators @node_meta and @NodeRegistry.register

def test_registry():
    print("Testing Dynamic Node Registry...")
    
    # Check if Skyulf classes have __node_meta__
    if hasattr(LogisticRegressionCalculator, "__node_meta__"):
        print(f"✅ LogisticRegressionCalculator has metadata: {LogisticRegressionCalculator.__node_meta__}")
    else:
        print("❌ LogisticRegressionCalculator MISSING metadata")

    if hasattr(RandomForestClassifierCalculator, "__node_meta__"):
        print(f"✅ RandomForestClassifierCalculator has metadata: {RandomForestClassifierCalculator.__node_meta__}")
    else:
        print("❌ RandomForestClassifierCalculator MISSING metadata")

    # Get all nodes from the backend registry abstraction
    nodes = NodeRegistry.get_all_nodes()
    
    # Check if our dynamic nodes are in the list
    dynamic_ids = ["logistic_regression", "random_forest_classifier"]
    found_map = {id: False for id in dynamic_ids}
    
    for node in nodes:
        if node.id in found_map:
            found_map[node.id] = True
            print(f"✅ Found dynamic node in backend registry: {node.id} ({node.name})")
            # Verify params are passed through
            if node.id == "logistic_regression":
                if "max_iter" in node.params:
                     print(f"   -> Params correctly populated: {node.params}")
                else:
                     print(f"   -> ❌ Params missing: {node.params}")

    for id, found in found_map.items():
        if not found:
            print(f"❌ Failed to find node {id} in NodeRegistry.get_all_nodes()")

if __name__ == "__main__":
    test_registry()
