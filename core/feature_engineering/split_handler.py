"""Automatic Split Detection and Handling System.

This module provides automatic detection and handling of train/test/validation splits
for all nodes in the ML pipeline. It eliminates the need for manual split handling
in individual node implementations.

Key Features:
- Automatic detection of split column in dataframes
- Smart routing of data to nodes based on split type
- Automatic fit on train, transform on test/validation for transformer nodes
- Split-aware data processing for all node types
- Centralized split management reducing code duplication
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Split type metadata column name
SPLIT_TYPE_COLUMN = "__split_type__"


class SplitType(str, Enum):
    """Enumeration of data split types."""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class NodeCategory(str, Enum):
    """Categorization of nodes by their data processing behavior."""

    # Transformers: Need to fit on train, transform on test/validation
    TRANSFORMER = "transformer"

    # Filters: Apply same logic to all splits independently
    FILTER = "filter"

    # Splitters: Create splits (like train_test_split)
    SPLITTER = "splitter"

    # Models: Fit on train, predict on test/validation
    MODEL = "model"

    # Passthrough: No special handling needed
    PASSTHROUGH = "passthrough"


# Mapping of catalog types to node categories
NODE_CATEGORY_MAP: Dict[str, NodeCategory] = {
    # Transformers (fit/transform pattern)
    "scale_numeric_features": NodeCategory.TRANSFORMER,
    "one_hot_encoding": NodeCategory.TRANSFORMER,
    "label_encoding": NodeCategory.TRANSFORMER,
    "ordinal_encoding": NodeCategory.TRANSFORMER,
    "target_encoding": NodeCategory.TRANSFORMER,
    "hash_encoding": NodeCategory.TRANSFORMER,
    "imputation_methods": NodeCategory.TRANSFORMER,
    "advanced_imputer": NodeCategory.TRANSFORMER,
    "simple_imputer": NodeCategory.TRANSFORMER,
    "binning_discretization": NodeCategory.TRANSFORMER,
    "skewness_transform": NodeCategory.TRANSFORMER,
    "pca_reduction": NodeCategory.TRANSFORMER,
    "polynomial_features": NodeCategory.TRANSFORMER,

    # Filters (apply independently to each split)
    "drop_missing_rows": NodeCategory.FILTER,
    "remove_duplicates": NodeCategory.FILTER,
    "outlier_removal": NodeCategory.FILTER,
    "filter_rows": NodeCategory.FILTER,
    "missing_value_indicator": NodeCategory.FILTER,
    "cast_column_types": NodeCategory.FILTER,
    "trim_whitespace": NodeCategory.FILTER,
    "normalize_text_case": NodeCategory.FILTER,
    "replace_aliases_typos": NodeCategory.FILTER,
    "standardize_date_formats": NodeCategory.FILTER,
    "remove_special_characters": NodeCategory.FILTER,
    "replace_invalid_values": NodeCategory.FILTER,
    "regex_replace_fix": NodeCategory.FILTER,

    # Splitters
    "train_test_split": NodeCategory.SPLITTER,
    "feature_target_split": NodeCategory.SPLITTER,

    # Models
    "train_model_draft": NodeCategory.MODEL,
    "model_training": NodeCategory.MODEL,
    "model_prediction": NodeCategory.MODEL,

    # Resampling (special - only on train)
    "class_undersampling": NodeCategory.TRANSFORMER,
    "class_oversampling": NodeCategory.TRANSFORMER,

    # Passthrough
    "dataset": NodeCategory.PASSTHROUGH,
    "comment": NodeCategory.PASSTHROUGH,
}


@dataclass
class SplitInfo:
    """Information about splits present in a dataframe."""
    has_splits: bool
    split_types: List[SplitType]
    split_counts: Dict[SplitType, int]
    total_rows: int

    def has_train(self) -> bool:
        """Check if train split exists."""
        return SplitType.TRAIN in self.split_types

    def has_test(self) -> bool:
        """Check if test split exists."""
        return SplitType.TEST in self.split_types

    def has_validation(self) -> bool:
        """Check if validation split exists."""
        return SplitType.VALIDATION in self.split_types


def detect_splits(frame: pd.DataFrame) -> SplitInfo:
    """Detect if dataframe contains train/test/validation splits.

    Args:
        frame: Input dataframe

    Returns:
        SplitInfo object containing split information
    """
    if frame.empty or SPLIT_TYPE_COLUMN not in frame.columns:
        return SplitInfo(
            has_splits=False,
            split_types=[],
            split_counts={},
            total_rows=len(frame)
        )

    # Get unique split types
    split_values = frame[SPLIT_TYPE_COLUMN].dropna().unique()
    split_types = []
    split_counts = {}

    for split_value in split_values:
        try:
            split_type = SplitType(split_value)
            split_types.append(split_type)
            count = int((frame[SPLIT_TYPE_COLUMN] == split_value).sum())
            split_counts[split_type] = count
        except ValueError:
            logger.warning(f"Unknown split type detected: {split_value}")

    return SplitInfo(
        has_splits=len(split_types) > 0,
        split_types=split_types,
        split_counts=split_counts,
        total_rows=len(frame)
    )


def get_split_data(frame: pd.DataFrame, split_type: SplitType) -> pd.DataFrame:
    """Extract data for a specific split type.

    Args:
        frame: Input dataframe with split column
        split_type: Split type to extract

    Returns:
        Dataframe containing only the specified split
    """
    if SPLIT_TYPE_COLUMN not in frame.columns:
        return frame.copy()

    mask = frame[SPLIT_TYPE_COLUMN] == split_type.value
    return frame[mask].copy()


def merge_split_data(
    train: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None,
    validation: Optional[pd.DataFrame] = None,
    original_frame: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Merge processed split data back into a single dataframe.

    Args:
        train: Processed training data
        test: Processed test data
        validation: Processed validation data
        original_frame: Original frame to preserve row order (optional)

    Returns:
        Merged dataframe with all splits
    """
    frames_to_concat = []

    if train is not None and not train.empty:
        frames_to_concat.append(train)
    if test is not None and not test.empty:
        frames_to_concat.append(test)
    if validation is not None and not validation.empty:
        frames_to_concat.append(validation)

    if not frames_to_concat:
        return original_frame.copy() if original_frame is not None else pd.DataFrame()

    # Concatenate all splits
    merged = pd.concat(frames_to_concat, ignore_index=False)

    # Try to preserve original row order if possible
    if original_frame is not None and SPLIT_TYPE_COLUMN in original_frame.columns:
        try:
            # Reindex to match original order
            merged = merged.reindex(original_frame.index)
        except Exception as e:
            logger.debug(f"Could not preserve original row order: {e}")

    return merged


def get_node_category(catalog_type: str) -> NodeCategory:
    """Get the processing category for a node type.

    Args:
        catalog_type: Node's catalog type

    Returns:
        NodeCategory enum value
    """
    return NODE_CATEGORY_MAP.get(catalog_type, NodeCategory.PASSTHROUGH)


def should_fit_on_train(catalog_type: str) -> bool:
    """Determine if a node should fit on training data.

    Args:
        catalog_type: Node's catalog type

    Returns:
        True if node should fit on train, transform on test/validation
    """
    category = get_node_category(catalog_type)
    return category in (NodeCategory.TRANSFORMER, NodeCategory.MODEL)


def should_apply_to_each_split(catalog_type: str) -> bool:
    """Determine if a node should be applied independently to each split.

    Args:
        catalog_type: Node's catalog type

    Returns:
        True if node should process each split independently
    """
    category = get_node_category(catalog_type)
    return category == NodeCategory.FILTER


def is_resampling_node(catalog_type: str) -> bool:
    """Check if node is a resampling node (should only apply to train).

    Args:
        catalog_type: Node's catalog type

    Returns:
        True if node is a resampling node
    """
    return catalog_type in ("class_undersampling", "class_oversampling")


def remove_split_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the internal split column from dataframe.

    Args:
        frame: Input dataframe

    Returns:
        Dataframe without split column
    """
    if SPLIT_TYPE_COLUMN in frame.columns:
        return frame.drop(columns=[SPLIT_TYPE_COLUMN])
    return frame


def log_split_processing(
    node_id: str,
    catalog_type: str,
    split_info: SplitInfo,
    action: str = "processing"
) -> None:
    """Log split processing information.

    Args:
        node_id: Node identifier
        catalog_type: Node's catalog type
        split_info: Information about splits
        action: Action being performed
    """
    if not split_info.has_splits:
        logger.debug(
            f"Node {node_id} ({catalog_type}): {action} without splits",
            extra={"node_id": node_id, "catalog_type": catalog_type}
        )
        return

    split_summary = ", ".join(
        f"{split_type.value}={count}"
        for split_type, count in split_info.split_counts.items()
    )

    category = get_node_category(catalog_type)

    logger.info(
        f"Node {node_id} ({catalog_type}): {action} with splits [{split_summary}] - Category: {category.value}",
        extra={
            "node_id": node_id,
            "catalog_type": catalog_type,
            "category": category.value,
            "split_counts": {k.value: v for k, v in split_info.split_counts.items()},
            "action": action
        }
    )


class SplitAwareProcessor:
    """Processor that automatically handles train/test/validation splits.

    This class wraps node processing functions and automatically handles
    split detection and routing without requiring changes to individual nodes.
    """

    def __init__(self, node_func, catalog_type: str, storage=None):
        """Initialize split-aware processor.

        Args:
            node_func: The node processing function to wrap
            catalog_type: Type of node from catalog
            storage: Optional transformer storage for fit/transform tracking
        """
        self.node_func = node_func
        self.catalog_type = catalog_type
        self.storage = storage
        self.category = get_node_category(catalog_type)

    def process(
        self,
        frame: pd.DataFrame,
        node: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.DataFrame, str, Any]:
        """Process dataframe with automatic split handling.

        Args:
            frame: Input dataframe
            node: Node configuration
            **kwargs: Additional arguments for node function

        Returns:
            Tuple of (processed dataframe, summary, signal)
        """
        split_info = detect_splits(frame)
        node_id = node.get("id", "unknown")

        log_split_processing(node_id, self.catalog_type, split_info, "processing")

        # No splits - process normally
        if not split_info.has_splits:
            return self.node_func(frame, node, **kwargs)

        # Handle based on node category
        if self.category == NodeCategory.PASSTHROUGH:
            return self.node_func(frame, node, **kwargs)

        elif self.category == NodeCategory.SPLITTER:
            # Splitter nodes create splits
            return self.node_func(frame, node, **kwargs)

        elif self.category == NodeCategory.FILTER:
            # Apply filter to each split independently
            return self._process_filter(frame, node, split_info, **kwargs)

        elif self.category == NodeCategory.TRANSFORMER:
            # Fit on train, transform on test/validation
            return self._process_transformer(frame, node, split_info, **kwargs)

        elif self.category == NodeCategory.MODEL:
            # Fit on train, predict on test/validation
            return self._process_model(frame, node, split_info, **kwargs)

        else:
            # Fallback to standard processing
            return self.node_func(frame, node, **kwargs)

    def _process_filter(
        self,
        frame: pd.DataFrame,
        node: Dict[str, Any],
        split_info: SplitInfo,
        **kwargs
    ) -> Tuple[pd.DataFrame, str, Any]:
        """Process filter node by applying to each split independently."""

        processed_splits = {}
        summaries = []
        signals = []

        for split_type in split_info.split_types:
            split_data = get_split_data(frame, split_type)

            # Process this split
            processed, summary, signal = self.node_func(split_data, node, **kwargs)

            # Re-add split column if it was removed
            if SPLIT_TYPE_COLUMN not in processed.columns:
                processed[SPLIT_TYPE_COLUMN] = split_type.value

            processed_splits[split_type] = processed
            summaries.append(f"{split_type.value}: {summary}")
            signals.append(signal)

        # Merge all splits back
        merged = merge_split_data(
            train=processed_splits.get(SplitType.TRAIN),
            test=processed_splits.get(SplitType.TEST),
            validation=processed_splits.get(SplitType.VALIDATION),
            original_frame=frame
        )

        # Combine summaries
        combined_summary = f"{self.catalog_type} (split-aware): {'; '.join(summaries)}"

        # Return first signal (they should all be similar)
        return merged, combined_summary, signals[0] if signals else None

    def _process_transformer(
        self,
        frame: pd.DataFrame,
        node: Dict[str, Any],
        split_info: SplitInfo,
        **kwargs
    ) -> Tuple[pd.DataFrame, str, Any]:
        """Process transformer node: fit on train, transform on all."""

        # Special case: resampling nodes only apply to train
        if is_resampling_node(self.catalog_type):
            return self._process_resampling(frame, node, split_info, **kwargs)

        # For transformers, we need the full frame to maintain column structure
        # The node implementation should handle split-aware processing internally
        # This is already implemented in nodes like one_hot_encoding

        # Just pass through - nodes already handle this
        result, summary, signal = self.node_func(frame, node, **kwargs)

        # Add split awareness to summary
        split_summary = ", ".join(
            f"{split_type.value}={count}"
            for split_type, count in split_info.split_counts.items()
        )
        enhanced_summary = f"{summary} [splits: {split_summary}]"

        return result, enhanced_summary, signal

    def _process_resampling(
        self,
        frame: pd.DataFrame,
        node: Dict[str, Any],
        split_info: SplitInfo,
        **kwargs
    ) -> Tuple[pd.DataFrame, str, Any]:
        """Process resampling node: only apply to train split."""
        node_id = node.get("id", "unknown")

        if not split_info.has_train():
            logger.warning(
                f"Resampling node {node_id} requires train split but none found",
                extra={"node_id": node_id, "catalog_type": self.catalog_type}
            )
            return frame, f"{self.catalog_type}: no train split to resample", None

        # Extract train split
        train_data = get_split_data(frame, SplitType.TRAIN)
        test_data = get_split_data(frame, SplitType.TEST) if split_info.has_test() else None
        val_data = get_split_data(frame, SplitType.VALIDATION) if split_info.has_validation() else None

        # Apply resampling only to train
        resampled_train, summary, signal = self.node_func(train_data, node, **kwargs)

        # Re-add split column if removed
        if SPLIT_TYPE_COLUMN not in resampled_train.columns:
            resampled_train[SPLIT_TYPE_COLUMN] = SplitType.TRAIN.value

        # Merge back with unchanged test/validation
        merged = merge_split_data(
            train=resampled_train,
            test=test_data,
            validation=val_data,
            original_frame=frame
        )

        enhanced_summary = f"{summary} (train only: {len(train_data)} → {len(resampled_train)} rows)"

        return merged, enhanced_summary, signal

    def _process_model(
        self,
        frame: pd.DataFrame,
        node: Dict[str, Any],
        split_info: SplitInfo,
        **kwargs
    ) -> Tuple[pd.DataFrame, str, Any]:
        """Process model node: fit on train, predict on test/validation."""
        # Similar to transformer but for model training/prediction
        # Models typically handle this internally
        result, summary, signal = self.node_func(frame, node, **kwargs)

        split_summary = ", ".join(
            f"{split_type.value}={count}"
            for split_type, count in split_info.split_counts.items()
        )
        enhanced_summary = f"{summary} [splits: {split_summary}]"

        return result, enhanced_summary, signal


def create_split_aware_wrapper(
    node_func,
    catalog_type: str,
    storage=None
):
    """Create a split-aware wrapper for a node function.

    This function wraps an existing node function to automatically handle
    train/test/validation splits without modifying the original function.

    Args:
        node_func: The node processing function
        catalog_type: Type of node from catalog
        storage: Optional transformer storage

    Returns:
        Wrapped function that handles splits automatically
    """
    processor = SplitAwareProcessor(node_func, catalog_type, storage)

    def wrapped(frame: pd.DataFrame, node: Dict[str, Any], **kwargs):
        return processor.process(frame, node, **kwargs)

    return wrapped
