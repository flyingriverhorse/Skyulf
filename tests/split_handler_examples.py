"""
Example: Integrating Automatic Split Detection into Routes

This module shows how to integrate the automatic split detection system
into the existing feature engineering routes.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.feature_engineering.split_handler import (
    detect_splits,
    get_node_category,
    log_split_processing,
    remove_split_column,
    SplitType,
    NodeCategory,
)

# Example of how to modify the route handler to use automatic split detection


def execute_pipeline_with_auto_splits(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    working_frame: pd.DataFrame,
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Execute pipeline with automatic split detection and handling.

    This is an example showing how the main pipeline execution can be
    enhanced with automatic split detection without changing individual nodes.

    Args:
        nodes: List of node configurations
        edges: List of edge connections
        working_frame: Input dataframe
        pipeline_id: Optional pipeline identifier

    Returns:
        Tuple of (processed dataframe, applied steps, metadata)
    """
    applied_steps: List[str] = []
    metadata: Dict[str, Any] = {}

    for node in nodes:
        node_id = node.get("id")
        catalog_type = node.get("data", {}).get("catalogType", "")
        node_id_value = str(node_id) if node_id is not None else "unknown"

        # Step 1: Detect splits in incoming data
        split_info = detect_splits(working_frame)

        # Step 2: Log split processing (provides visibility)
        log_split_processing(
            node_id=node_id_value,
            catalog_type=catalog_type,
            split_info=split_info,
            action="processing"
        )

        # Step 3: Get node category (determines processing strategy)
        category = get_node_category(catalog_type)
        categories = metadata.setdefault("node_categories", {})
        categories[node_id_value] = category.value

        # Step 4: Process based on node type
        # The actual node functions remain unchanged!
        # The framework handles split routing automatically

        if catalog_type == "scale_numeric_features":
            from core.feature_engineering.nodes.feature_eng.scaling import (  # type: ignore[import]
                _apply_scale_numeric_features as apply_scale_numeric_features
            )
            working_frame, summary, signal = apply_scale_numeric_features(
                working_frame, node, pipeline_id=pipeline_id
            )

        elif catalog_type == "one_hot_encoding":
            from core.feature_engineering.preprocessing.encoding.one_hot_encoding import (  # type: ignore[import]
                apply_one_hot_encoding
            )
            working_frame, summary, signal = apply_one_hot_encoding(
                working_frame, node, pipeline_id=pipeline_id
            )

        elif catalog_type == "remove_duplicates":
            from core.feature_engineering.nodes.cleaning.duplicates import (  # type: ignore[import]
                apply_remove_duplicates
            )
            working_frame, summary, signal = apply_remove_duplicates(
                working_frame, node
            )

        # ... other node types

        # Step 5: Enhance summary with split information
        if split_info.has_splits:
            split_summary = ", ".join(
                f"{split_type.value}={count}"
                for split_type, count in split_info.split_counts.items()
            )
            summary = f"{summary} [splits: {split_summary}]"

        applied_steps.append(summary)

    # Step 6: Remove internal split column before returning
    final_frame = remove_split_column(working_frame)

    return final_frame, applied_steps, metadata


# Example: Wrapping an existing node with split awareness

def example_wrap_existing_node():
    """
    Example showing how to wrap an existing node function
    to add automatic split handling.
    """
    from core.feature_engineering.split_handler import create_split_aware_wrapper
    from core.feature_engineering.nodes.feature_eng.scaling import (  # type: ignore[import]
        _apply_scale_numeric_features as apply_scale_numeric_features
    )

    # Original function
    original_scaler = apply_scale_numeric_features

    # Wrapped with automatic split handling
    split_aware_scaler = create_split_aware_wrapper(
        node_func=apply_scale_numeric_features,
        catalog_type="scale_numeric_features"
    )

    # Now can be used exactly the same way but handles splits automatically!
    # result, summary, signal = split_aware_scaler(dataframe, node_config)
    return original_scaler, split_aware_scaler


# Example: Checking split information

def example_check_splits(dataframe: pd.DataFrame):
    """
    Example showing how to check split information in a dataframe.
    """
    split_info = detect_splits(dataframe)

    if not split_info.has_splits:
        print("No splits detected - processing as single dataset")
        return

    print(f"Splits detected: {[s.value for s in split_info.split_types]}")
    print(f"Total rows: {split_info.total_rows}")

    if split_info.has_train():
        train_count = split_info.split_counts.get(SplitType.TRAIN, 0)
        train_pct = (train_count / split_info.total_rows) * 100
        print(f"Train: {train_count} rows ({train_pct:.1f}%)")

    if split_info.has_test():
        test_count = split_info.split_counts.get(SplitType.TEST, 0)
        test_pct = (test_count / split_info.total_rows) * 100
        print(f"Test: {test_count} rows ({test_pct:.1f}%)")

    if split_info.has_validation():
        val_count = split_info.split_counts.get(SplitType.VALIDATION, 0)
        val_pct = (val_count / split_info.total_rows) * 100
        print(f"Validation: {val_count} rows ({val_pct:.1f}%)")


# Example: Processing different node categories

def example_process_by_category(
    dataframe: pd.DataFrame,
    node: Dict[str, Any]
) -> pd.DataFrame:
    """
    Example showing how different node categories are processed.
    """
    catalog_type = node.get("data", {}).get("catalogType", "")
    category = get_node_category(catalog_type)

    split_info = detect_splits(dataframe)

    if not split_info.has_splits:
        print(f"Processing {catalog_type} without splits")
        # Normal processing
        return dataframe

    if category == NodeCategory.TRANSFORMER:
        print("Transformer node: will fit on train, transform on all")
        # Framework handles this automatically
        # - Fits on train split
        # - Transforms all splits
        # - Merges results

    elif category == NodeCategory.FILTER:
        print("Filter node: will process each split independently")
        # Framework handles this automatically
        # - Processes train independently
        # - Processes test independently
        # - Processes validation independently
        # - Merges results

    elif category == NodeCategory.TRANSFORMER and catalog_type in [
        "class_oversampling", "class_undersampling"
    ]:
        print("Resampling node: will only process train split")
        # Framework handles this automatically
        # - Resamples train split only
        # - Test and validation unchanged
        # - Merges results

    return dataframe


# Example: Adding a new node type

def example_add_new_node_type():
    """
    Example showing how to add a new node type with automatic split handling.
    """
    from core.feature_engineering.split_handler import NODE_CATEGORY_MAP, NodeCategory

    # Step 1: Implement the core node logic
    def apply_my_new_feature(frame: pd.DataFrame, node: Dict[str, Any]):
        """
        New feature engineering node.
        No need to handle splits manually!
        """
        config = node.get("data", {}).get("config", {})
        multiplier = float(config.get("multiplier", 2.0))

        # Just implement your feature engineering logic
        result = frame.copy()
        result["new_feature"] = result["existing_feature"] * multiplier

        summary = f"Applied new feature engineering (multiplier={multiplier})"
        signal = {
            "columns_created": ["new_feature"],
            "multiplier": multiplier,
        }

        return result, summary, signal

    # Step 2: Register the node category
    NODE_CATEGORY_MAP["my_new_feature"] = NodeCategory.TRANSFORMER

    # Step 3: Done! The node now automatically:
    # - Detects splits
    # - Fits on train if needed
    # - Transforms all splits appropriately
    # - Merges results
    # - Handles all edge cases

    print("New node registered with automatic split handling!")


# Example: Debugging split issues

def example_debug_splits(
    dataframe: pd.DataFrame,
    node: Dict[str, Any]
):
    """
    Example showing how to debug split-related issues.
    """
    from core.feature_engineering.split_handler import (
        SPLIT_TYPE_COLUMN,
        get_split_data,
    )

    node_id = node.get("id", "unknown")
    catalog_type = node.get("data", {}).get("catalogType", "")

    print(f"\n=== Debugging Node {node_id} ({catalog_type}) ===")

    # Check if split column exists
    if SPLIT_TYPE_COLUMN not in dataframe.columns:
        print("❌ No split column found")
        return

    print("✅ Split column found")

    # Check split values
    split_values = dataframe[SPLIT_TYPE_COLUMN].unique()
    print(f"Split values: {split_values}")

    # Check split distribution
    split_counts = dataframe[SPLIT_TYPE_COLUMN].value_counts()
    print(f"Split distribution:\n{split_counts}")

    # Check for null values in split column
    null_count = dataframe[SPLIT_TYPE_COLUMN].isna().sum()
    if null_count > 0:
        print(f"⚠️  Warning: {null_count} rows with null split type")

    # Verify we can extract each split
    for split_value in split_values:
        if split_value in ["train", "test", "validation"]:
            try:
                split_type = SplitType(split_value)
                split_data = get_split_data(dataframe, split_type)
                print(f"✅ Can extract {split_value}: {len(split_data)} rows")
            except Exception as e:
                print(f"❌ Error extracting {split_value}: {e}")

    # Check node category
    category = get_node_category(catalog_type)
    print(f"\nNode category: {category.value}")

    if category == NodeCategory.TRANSFORMER:
        print("→ Will fit on train, transform on all")
    elif category == NodeCategory.FILTER:
        print("→ Will process each split independently")
    elif category == NodeCategory.PASSTHROUGH:
        print("→ Will pass through without special handling")


# Example: Performance monitoring

def example_monitor_split_performance(
    dataframe: pd.DataFrame,
    node: Dict[str, Any]
):
    """
    Example showing how to monitor performance with splits.
    """
    import time

    split_info = detect_splits(dataframe)
    catalog_type = node.get("data", {}).get("catalogType", "")

    if not split_info.has_splits:
        print(f"Processing {catalog_type} without splits")
        return

    print(f"\n=== Processing {catalog_type} with splits ===")

    # Time the processing
    start_time = time.time()

    # Hypothetical processing
    # result = process_node(dataframe, node)

    end_time = time.time()
    duration = end_time - start_time

    # Calculate per-split timing
    total_rows = split_info.total_rows
    rows_per_second = total_rows / duration if duration > 0 else 0

    print(f"Processing time: {duration:.3f}s")
    print(f"Throughput: {rows_per_second:.0f} rows/second")

    for split_type, count in split_info.split_counts.items():
        pct = (count / total_rows) * 100
        est_time = (count / rows_per_second) if rows_per_second > 0 else 0
        print(f"  {split_type.value}: {count} rows ({pct:.1f}%) - ~{est_time:.3f}s")


# Example: Custom split handling for special cases

def example_custom_split_handling(
    dataframe: pd.DataFrame,
    node: Dict[str, Any]
):
    """
    Example showing how to implement custom split handling
    for special cases not covered by the framework.
    """
    from core.feature_engineering.split_handler import (
        get_split_data,
        merge_split_data,
        SPLIT_TYPE_COLUMN,
    )

    split_info = detect_splits(dataframe)

    if not split_info.has_splits:
        # Normal processing
        return process_without_splits(dataframe, node)

    # Custom logic: Apply different parameters to each split
    train_data = get_split_data(dataframe, SplitType.TRAIN)
    test_data = get_split_data(dataframe, SplitType.TEST) if split_info.has_test() else None

    # Process train with aggressive parameters
    train_config = node.copy()
    train_config["data"]["config"]["strength"] = 1.0
    processed_train = custom_process(train_data, train_config)
    processed_train[SPLIT_TYPE_COLUMN] = SplitType.TRAIN.value

    # Process test with conservative parameters
    if test_data is not None:
        test_config = node.copy()
        test_config["data"]["config"]["strength"] = 0.5
        processed_test = custom_process(test_data, test_config)
        processed_test[SPLIT_TYPE_COLUMN] = SplitType.TEST.value
    else:
        processed_test = None

    # Merge back
    result = merge_split_data(
        train=processed_train,
        test=processed_test,
        validation=None,
        original_frame=dataframe
    )

    return result


def custom_process(frame, config):
    """Placeholder for custom processing."""
    return frame.copy()


def process_without_splits(frame, node):
    """Placeholder for processing without splits."""
    return frame.copy()


if __name__ == "__main__":
    # Demo usage
    import pandas as pd

    # Create sample data with splits
    sample_df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "__split_type__": ["train", "train", "train", "test", "validation"]
    })

    print("=== Sample Data ===")
    print(sample_df)

    print("\n=== Split Detection Example ===")
    example_check_splits(sample_df)

    print("\n=== Debug Example ===")
    sample_node = {
        "id": "test-node-1",
        "data": {
            "catalogType": "scale_numeric_features",
            "config": {}
        }
    }
    example_debug_splits(sample_df, sample_node)
