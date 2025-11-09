"""Integration tests for automatic split detection logic."""

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from core.feature_engineering.split_handler import (
        detect_splits,
        SPLIT_TYPE_COLUMN,
        SplitType,
        get_node_category,
        NodeCategory,
    )
except ImportError:  # pragma: no cover - direct test execution fallback
    repo_root_text = str(REPO_ROOT)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    from core.feature_engineering.split_handler import (
        detect_splits,
        SPLIT_TYPE_COLUMN,
        SplitType,
        get_node_category,
        NodeCategory,
    )


def test_split_detection():
    """Test basic split detection functionality."""
    print("=" * 70)
    print("TEST 1: Basic Split Detection")
    print("=" * 70)

    # Create dataframe with splits
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        SPLIT_TYPE_COLUMN: ['train', 'train', 'train', 'test', 'validation']
    })

    # Detect splits
    split_info = detect_splits(df)

    print(f"✓ Has splits: {split_info.has_splits}")
    print(f"✓ Split types: {[s.value for s in split_info.split_types]}")
    print(f"✓ Total rows: {split_info.total_rows}")
    print(f"✓ Train rows: {split_info.split_counts.get(SplitType.TRAIN, 0)}")
    print(f"✓ Test rows: {split_info.split_counts.get(SplitType.TEST, 0)}")
    print(f"✓ Validation rows: {split_info.split_counts.get(SplitType.VALIDATION, 0)}")

    assert split_info.has_splits is True
    assert len(split_info.split_types) == 3
    assert split_info.has_train()
    assert split_info.has_test()
    assert split_info.has_validation()

    print("\n✅ Split detection test PASSED\n")


def test_node_categorization():
    """Test node categorization."""
    print("=" * 70)
    print("TEST 2: Node Categorization")
    print("=" * 70)

    # Test transformer nodes
    transformer_nodes = [
        'scale_numeric_features',
        'one_hot_encoding',
        'label_encoding',
        'simple_imputer'
    ]

    print("\n📊 Transformer Nodes (fit on train, transform on all):")
    for node_type in transformer_nodes:
        category = get_node_category(node_type)
        print(f"   ✓ {node_type}: {category.value}")
        assert category == NodeCategory.TRANSFORMER

    # Test filter nodes
    filter_nodes = [
        'remove_duplicates',
        'drop_missing_rows',
        'outlier_removal',
        'trim_whitespace'
    ]

    print("\n🔍 Filter Nodes (process each split independently):")
    for node_type in filter_nodes:
        category = get_node_category(node_type)
        print(f"   ✓ {node_type}: {category.value}")
        assert category == NodeCategory.FILTER

    # Test resampling nodes
    resampling_nodes = [
        'class_oversampling',
        'class_undersampling'
    ]

    print("\n⚖️  Resampling Nodes (train only):")
    for node_type in resampling_nodes:
        category = get_node_category(node_type)
        print(f"   ✓ {node_type}: {category.value}")
        # They're marked as TRANSFORMER but have special handling
        assert category == NodeCategory.TRANSFORMER

    # Test splitter nodes
    splitter_nodes = [
        'train_test_split',
        'feature_target_split'
    ]

    print("\n✂️  Splitter Nodes (create splits):")
    for node_type in splitter_nodes:
        category = get_node_category(node_type)
        print(f"   ✓ {node_type}: {category.value}")
        assert category == NodeCategory.SPLITTER

    print("\n✅ Node categorization test PASSED\n")


def test_no_splits_detection():
    """Test detection when no splits exist."""
    print("=" * 70)
    print("TEST 3: No Splits Detection")
    print("=" * 70)

    # Create dataframe without splits
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
    })

    # Detect splits
    split_info = detect_splits(df)

    print(f"✓ Has splits: {split_info.has_splits}")
    print(f"✓ Split types: {split_info.split_types}")
    print(f"✓ Total rows: {split_info.total_rows}")

    assert split_info.has_splits is False
    assert len(split_info.split_types) == 0
    assert not split_info.has_train()
    assert not split_info.has_test()
    assert not split_info.has_validation()

    print("\n✅ No splits detection test PASSED\n")


def test_route_integration():
    """Test that routes.py can import the split handler."""
    print("=" * 70)
    print("TEST 4: Route Integration")
    print("=" * 70)

    try:
        # Try to import the route module
        from core.feature_engineering import routes

        # Check if split handler functions are imported
        assert hasattr(routes, 'detect_splits'), "detect_splits not imported"
        assert hasattr(routes, 'log_split_processing'), "log_split_processing not imported"
        assert hasattr(routes, 'remove_split_column'), "remove_split_column not imported"

        from core.feature_engineering.split_handler import get_node_category as _get_node_category
        assert callable(_get_node_category), "get_node_category unavailable"

        print("✓ Routes module imported successfully")
        print("✓ Split handler functions available in routes")
        print("✓ detect_splits function: OK")
        print("✓ log_split_processing function: OK")
        print("✓ remove_split_column function: OK")
        print("✓ get_node_category available via split_handler: OK")

        print("\n✅ Route integration test PASSED\n")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise
    except AssertionError as e:
        print(f"❌ Assertion error: {e}")
        raise


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("AUTOMATIC SPLIT DETECTION - INTEGRATION TESTS")
    print("=" * 70 + "\n")

    try:
        test_split_detection()
        test_node_categorization()
        test_no_splits_detection()
        test_route_integration()

        print("=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Automatic split detection is properly integrated!")
        print("✅ All node categories are correctly mapped!")
        print("✅ Routes can import and use split handler functions!")
        print("\n🚀 Your ML pipeline now has automatic split handling!\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
