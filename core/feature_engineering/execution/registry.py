"""Registry of node transformations and execution specifications."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from core.feature_engineering.execution.graph import resolve_node_label
from core.feature_engineering.shared.utils import _is_node_pending

from core.feature_engineering.preprocessing.bucketing import (
    _apply_binning_discretization,
)
from core.feature_engineering.preprocessing.statistics import (
    _apply_outlier_removal,
    _apply_scale_numeric_features,
    _apply_skewness_transformations,
    apply_imputation_methods as _apply_imputation_methods,
)
from core.feature_engineering.preprocessing.feature_generation import apply_feature_math, apply_polynomial_features
from core.feature_engineering.preprocessing.feature_selection import apply_feature_selection
from core.feature_engineering.preprocessing.inspection import (
    apply_transformer_audit,
)
from core.feature_engineering.preprocessing.encoding.label_encoding import (
    apply_label_encoding,
)
from core.feature_engineering.preprocessing.encoding.hash_encoding import (
    apply_hash_encoding,
)
from core.feature_engineering.preprocessing.resampling import (
    apply_oversampling,
    apply_resampling,
)
from core.feature_engineering.preprocessing.split import apply_feature_target_split
from core.feature_engineering.preprocessing.split import apply_train_test_split
from core.feature_engineering.preprocessing.encoding.ordinal_encoding import (
    apply_ordinal_encoding,
)
from core.feature_engineering.preprocessing.encoding.target_encoding import (
    apply_target_encoding,
)
from core.feature_engineering.preprocessing.encoding.one_hot_encoding import (
    apply_one_hot_encoding,
)
from core.feature_engineering.preprocessing.encoding.dummy_encoding import (
    apply_dummy_encoding,
)
from core.feature_engineering.preprocessing.casting import _apply_cast_column_types
from core.feature_engineering.preprocessing.cleaning import (
    apply_normalize_text_case,
    apply_regex_cleanup,
    apply_remove_special_characters,
    apply_replace_aliases_typos,
    apply_replace_invalid_values,
    apply_standardize_date_formats,
    apply_trim_whitespace,
)
from core.feature_engineering.preprocessing.drop_and_missing import (
    apply_drop_missing_columns,
    apply_drop_missing_rows,
    apply_missing_value_flags,
    apply_remove_duplicates,
)
from core.feature_engineering.modeling.training.train_model_draft import apply_train_model_draft
from core.feature_engineering.modeling.training.evaluation import (
    apply_model_evaluation,
)


NodeTransformSpec = Tuple[Callable[..., Tuple[pd.DataFrame, Any, Any]], bool]

NODE_TRANSFORMS: Dict[str, NodeTransformSpec] = {
    "drop_missing_columns": (apply_drop_missing_columns, False),
    "drop_missing_rows": (apply_drop_missing_rows, False),
    "remove_duplicates": (apply_remove_duplicates, False),
    "missing_value_indicator": (apply_missing_value_flags, False),
    "cast_column_types": (_apply_cast_column_types, False),
    "trim_whitespace": (apply_trim_whitespace, False),
    "normalize_text_case": (apply_normalize_text_case, False),
    "replace_aliases_typos": (apply_replace_aliases_typos, False),
    "standardize_date_formats": (apply_standardize_date_formats, False),
    "remove_special_characters": (apply_remove_special_characters, False),
    "replace_invalid_values": (apply_replace_invalid_values, False),
    "regex_replace_fix": (apply_regex_cleanup, False),
    "feature_math": (apply_feature_math, True),
    "binning_discretization": (_apply_binning_discretization, True),
    "skewness_transform": (_apply_skewness_transformations, True),
    "outlier_removal": (_apply_outlier_removal, True),
    "scale_numeric_features": (_apply_scale_numeric_features, True),
    "feature_target_split": (apply_feature_target_split, False),
    "class_undersampling": (apply_resampling, False),
    "class_oversampling": (apply_oversampling, False),
    "label_encoding": (apply_label_encoding, True),
    "target_encoding": (apply_target_encoding, True),
    "hash_encoding": (apply_hash_encoding, True),
    "train_model_draft": (apply_train_model_draft, False),
    "ordinal_encoding": (apply_ordinal_encoding, True),
    "one_hot_encoding": (apply_one_hot_encoding, True),
    "dummy_encoding": (apply_dummy_encoding, True),
}

for key in ("imputation_methods", "advanced_imputer", "simple_imputer"):
    NODE_TRANSFORMS[key] = (_apply_imputation_methods, True)


@dataclass(frozen=True)
class NodeExecutionContext:
    pipeline_id: Optional[str]
    node_map: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class NodeExecutionResult:
    frame: pd.DataFrame
    summary: str
    signal: Any


@dataclass(frozen=True)
class NodeExecutionSpec:
    handler: Callable[[pd.DataFrame, Dict[str, Any], NodeExecutionContext], NodeExecutionResult]
    signal_attr: Optional[str]
    signal_mode: str = "append"  # append or assign
    update_modeling_metadata: bool = False


def _wrap_node_handler(
    func: Callable[..., Tuple[pd.DataFrame, str, Any]],
    *,
    requires_pipeline: bool = False,
    requires_node_map: bool = False,
) -> Callable[[pd.DataFrame, Dict[str, Any], NodeExecutionContext], NodeExecutionResult]:
    def handler(frame: pd.DataFrame, node: Dict[str, Any], context: NodeExecutionContext) -> NodeExecutionResult:
        kwargs = {"frame": frame, "node": node}
        if requires_pipeline:
            kwargs["pipeline_id"] = context.pipeline_id
        if requires_node_map:
            kwargs["node_map"] = context.node_map

        # Some handlers return 2 values, some 3. We normalize here.
        # Most return (frame, summary, signal)
        # Some might return (frame, summary) -> signal is None
        result = func(**kwargs)
        
        if len(result) == 3:
            new_frame, summary, signal = result
        else:
            new_frame, summary = result
            signal = None
            
        return NodeExecutionResult(new_frame, summary, signal)

    return handler


def _model_registry_overview_handler(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    _: NodeExecutionContext,
) -> NodeExecutionResult:
    label = resolve_node_label(node)
    summary = f"{label}: model registry view – no transformation applied"
    return NodeExecutionResult(frame, summary, None)


NODE_EXECUTION_SPECS: Dict[str, NodeExecutionSpec] = {
    "drop_missing_columns": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_drop_missing_columns),
        signal_attr="drop_missing_columns",
    ),
    "drop_missing_rows": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_drop_missing_rows),
        signal_attr="drop_missing_rows",
    ),
    "remove_duplicates": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_remove_duplicates),
        signal_attr="remove_duplicates",
    ),
    "missing_value_indicator": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_missing_value_flags),
        signal_attr="missing_value_indicator",
    ),
    "cast_column_types": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_cast_column_types),
        signal_attr="cast_column_types",
    ),
    "trim_whitespace": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_trim_whitespace),
        signal_attr="trim_whitespace",
    ),
    "normalize_text_case": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_normalize_text_case),
        signal_attr="normalize_text_case",
    ),
    "replace_aliases_typos": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_replace_aliases_typos),
        signal_attr="replace_aliases",
    ),
    "standardize_date_formats": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_standardize_date_formats),
        signal_attr="standardize_dates",
    ),
    "remove_special_characters": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_remove_special_characters),
        signal_attr="remove_special_characters",
    ),
    "replace_invalid_values": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_replace_invalid_values),
        signal_attr="replace_invalid_values",
    ),
    "feature_math": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_feature_math, requires_pipeline=True),
        signal_attr="feature_math",
    ),
    "regex_replace_fix": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_regex_cleanup),
        signal_attr="regex_cleanup",
    ),
    "imputation_methods": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_imputation_methods, requires_pipeline=True),
        signal_attr="imputation",
    ),
    "advanced_imputer": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_imputation_methods, requires_pipeline=True),
        signal_attr="imputation",
    ),
    "simple_imputer": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_imputation_methods, requires_pipeline=True),
        signal_attr="imputation",
    ),
    "binning_discretization": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_binning_discretization, requires_pipeline=True),
        signal_attr="binning",
    ),
    "skewness_transform": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_skewness_transformations, requires_pipeline=True),
        signal_attr="skewness_transform",
    ),
    "scale_numeric_features": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_scale_numeric_features, requires_pipeline=True),
        signal_attr="scaling",
    ),
    "feature_selection": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_feature_selection, requires_pipeline=True),
        signal_attr="feature_selection",
    ),
    "polynomial_features": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_polynomial_features, requires_pipeline=True),
        signal_attr="polynomial_features",
    ),
    "outlier_removal": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_outlier_removal, requires_pipeline=True),
        signal_attr="outlier_removal",
    ),
    "feature_target_split": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_feature_target_split),
        signal_attr="feature_target_split",
    ),
    "train_test_split": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_train_test_split),
        signal_attr="train_test_split",
    ),
    "class_undersampling": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_resampling),
        signal_attr="class_undersampling",
    ),
    "class_oversampling": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_oversampling),
        signal_attr="class_oversampling",
    ),
    "label_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_label_encoding, requires_pipeline=True),
        signal_attr="label_encoding",
    ),
    "target_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_target_encoding, requires_pipeline=True),
        signal_attr="target_encoding",
    ),
    "hash_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_hash_encoding, requires_pipeline=True),
        signal_attr="hash_encoding",
    ),
    "train_model_draft": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_train_model_draft),
        signal_attr="modeling",
        signal_mode="assign",
        update_modeling_metadata=True,
    ),
    "model_evaluation": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_model_evaluation, requires_pipeline=True),
        signal_attr="model_evaluation",
    ),
    "model_registry_overview": NodeExecutionSpec(
        handler=_model_registry_overview_handler,
        signal_attr=None,
    ),
    "ordinal_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_ordinal_encoding, requires_pipeline=True),
        signal_attr="ordinal_encoding",
    ),
    "one_hot_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_one_hot_encoding, requires_pipeline=True),
        signal_attr="one_hot_encoding",
    ),
    "dummy_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_dummy_encoding, requires_pipeline=True),
        signal_attr="dummy_encoding",
    ),
    "transformer_audit": NodeExecutionSpec(
        handler=_wrap_node_handler(
            apply_transformer_audit,
            requires_pipeline=True,
            requires_node_map=True,
        ),
        signal_attr="transformer_audit",
    ),
}


@dataclass
class PipelineNodeOutcome:
    frame: pd.DataFrame
    summary: str
    modeling_metadata: Optional[Any] = None
