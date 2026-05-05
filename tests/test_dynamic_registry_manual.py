import sys
import os
import io

# Force UTF-8 output to handle emoji on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add workspace root to sys.path to ensure we can import backend and skyulf-core
sys.path.append(os.getcwd())
# Also add skyulf-core specifically if it's not installed in the env
sys.path.append(os.path.join(os.getcwd(), "skyulf-core"))

from skyulf.registry import NodeRegistry as SkyulfRegistry
from skyulf.modeling.classification import (
    LogisticRegressionCalculator,
    RandomForestClassifierCalculator,
)


def test_registry():
    print("Testing Dynamic Node Registry...")

    # Check if Skyulf classes have __node_meta__
    if hasattr(LogisticRegressionCalculator, "__node_meta__"):
        print(
            f"✅ LogisticRegressionCalculator has metadata: {LogisticRegressionCalculator.__node_meta__}"
        )
    else:
        print("❌ LogisticRegressionCalculator MISSING metadata")

    if hasattr(RandomForestClassifierCalculator, "__node_meta__"):
        print(
            f"✅ RandomForestClassifierCalculator has metadata: {RandomForestClassifierCalculator.__node_meta__}"
        )
    else:
        print("❌ RandomForestClassifierCalculator MISSING metadata")

    # Get all nodes directly from the core registry
    all_metadata = SkyulfRegistry.get_all_metadata()

    # Check if our dynamic nodes are in the list
    dynamic_ids = ["logistic_regression", "random_forest_classifier"]
    found_map = {node_id: False for node_id in dynamic_ids}

    for node_id, meta in all_metadata.items():
        if node_id in found_map:
            found_map[node_id] = True
            print(f"✅ Found dynamic node in registry: {node_id} ({meta.get('name')})")
            if node_id == "logistic_regression":
                if "max_iter" in (meta.get("params") or {}):
                    print(f"   -> Params correctly populated: {meta.get('params')}")
                else:
                    print(f"   -> ❌ Params missing: {meta.get('params')}")

    for id, found in found_map.items():
        if not found:
            print(f"❌ Failed to find node {id} in SkyulfRegistry")


if __name__ == "__main__":
    test_registry()
