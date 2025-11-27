"""Node catalog API endpoints."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import APIRouter

from core.feature_engineering.preprocessing.bucketing import (
    BINNING_DEFAULT_EQUAL_FREQUENCY_BINS,
    BINNING_DEFAULT_EQUAL_WIDTH_BINS,
    BINNING_DEFAULT_MISSING_LABEL,
    BINNING_DEFAULT_PRECISION,
    BINNING_DEFAULT_SUFFIX,
)
from core.feature_engineering.preprocessing.statistics import (
    DEFAULT_METHOD_PARAMETERS,
    OUTLIER_DEFAULT_METHOD,
    SCALING_DEFAULT_METHOD,
)
from core.feature_engineering.preprocessing.encoding.label_encoding import (
    LABEL_ENCODING_DEFAULT_MAX_UNIQUE,
    LABEL_ENCODING_DEFAULT_SUFFIX,
)
from core.feature_engineering.preprocessing.encoding.hash_encoding import (
    HASH_ENCODING_DEFAULT_BUCKETS,
    HASH_ENCODING_DEFAULT_MAX_CATEGORIES,
    HASH_ENCODING_DEFAULT_SUFFIX,
    HASH_ENCODING_MAX_CARDINALITY_LIMIT,
    HASH_ENCODING_MAX_BUCKETS,
)
from core.feature_engineering.preprocessing.resampling import (
    OVERSAMPLING_DEFAULT_K_NEIGHBORS,
    OVERSAMPLING_DEFAULT_METHOD,
    OVERSAMPLING_DEFAULT_RANDOM_STATE,
    OVERSAMPLING_DEFAULT_REPLACEMENT,
    OVERSAMPLING_METHOD_LABELS,
    RESAMPLING_DEFAULT_METHOD as UNDERSAMPLING_DEFAULT_METHOD,
    RESAMPLING_DEFAULT_RANDOM_STATE as UNDERSAMPLING_DEFAULT_RANDOM_STATE,
    RESAMPLING_DEFAULT_REPLACEMENT as UNDERSAMPLING_DEFAULT_REPLACEMENT,
    RESAMPLING_METHOD_LABELS as UNDERSAMPLING_METHOD_LABELS,
)
from core.feature_engineering.preprocessing.encoding.ordinal_encoding import (
    ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES,
    ORDINAL_ENCODING_DEFAULT_SUFFIX,
    ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT,
    ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE,
)
from core.feature_engineering.modeling.hyperparameter_tuning.registry import (
    get_default_strategy_value,
    get_strategy_choices_for_ui,
)
from core.feature_engineering.preprocessing.encoding.target_encoding import (
    TARGET_ENCODING_DEFAULT_MAX_CATEGORIES,
    TARGET_ENCODING_DEFAULT_SUFFIX,
    TARGET_ENCODING_DEFAULT_SMOOTHING,
    TARGET_ENCODING_MAX_CARDINALITY_LIMIT,
    TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN,
)
from core.feature_engineering.preprocessing.encoding.one_hot_encoding import (
    ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES,
    ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR,
    ONE_HOT_ENCODING_MAX_CARDINALITY_LIMIT,
)
from core.feature_engineering.preprocessing.encoding.dummy_encoding import (
    DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES,
    DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR,
    DUMMY_ENCODING_MAX_CARDINALITY_LIMIT,
)
from core.feature_engineering.modeling.training.registry import list_registered_models

router = APIRouter()


class FeatureNodeParameterOption(TypedDict, total=False):
    """Enumerated option for select-style parameters."""

    value: str
    label: str
    description: Optional[str]
    metadata: Dict[str, Any]


class FeatureNodeParameterSource(TypedDict, total=False):
    """External source definition for dynamic parameter options."""

    type: str
    endpoint: str
    value_key: str


class FeatureNodeParameter(TypedDict, total=False):
    """Schema describing configurable parameters for a node."""

    name: str
    label: str
    description: Optional[str]
    type: str
    default: Optional[Any]
    min: Optional[float]
    max: Optional[float]
    step: Optional[float]
    unit: Optional[str]
    placeholder: Optional[str]
    options: List[FeatureNodeParameterOption]
    source: FeatureNodeParameterSource


class FeatureNodeCatalogEntryBase(TypedDict):
    """Schema for node catalog entries."""

    type: str
    label: str
    description: str
    inputs: List[str]
    outputs: List[str]


class FeatureNodeCatalogEntry(FeatureNodeCatalogEntryBase, total=False):
    category: str
    tags: List[str]
    parameters: List[FeatureNodeParameter]
    default_config: Dict[str, Any]


def _make_option(
    value: str,
    label: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureNodeParameterOption:
    option: FeatureNodeParameterOption = {"value": value, "label": label}
    if description:
        option["description"] = description
    if metadata:
        option["metadata"] = metadata
    return option


@router.get("/api/node-catalog", response_model=List[FeatureNodeCatalogEntry])
async def get_node_catalog() -> List[FeatureNodeCatalogEntry]:
    """Return the prototype node catalog for the feature engineering canvas."""

    def _build_preprocessing_nodes() -> List[FeatureNodeCatalogEntry]:
        drop_missing_node: FeatureNodeCatalogEntry = {
            "type": "drop_missing_columns",
            "label": "Drop high-missing columns",
            "description": "Drop columns",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["missing_data", "cleanup"],
            "parameters": [
                {
                    "name": "missing_threshold",
                    "label": "Missingness threshold (%)",
                    "description": "Columns at or above this missing percentage will be removed.",
                    "type": "number",
                    "default": 40.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "unit": "%",
                },
                {
                    "name": "columns",
                    "label": "Columns to drop (recommended)",
                    "description": (
                        "Pre-populated with EDA suggestions covering high-missing, empty, or constant columns."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "drop_column_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/drop-columns",
                        "value_key": "candidates",
                    },
                },
            ],
            "default_config": {
                "missing_threshold": 40.0,
                "columns": [],
            },
        }

        drop_missing_rows_node: FeatureNodeCatalogEntry = {
            "type": "drop_missing_rows",
            "label": "Drop high-missing rows",
            "description": "Remove rows with excessive missing values.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["missing_data", "cleanup"],
            "parameters": [
                {
                    "name": "missing_threshold",
                    "label": "Missingness threshold (%)",
                    "description": "Rows at or above this missing percentage will be removed.",
                    "type": "number",
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "unit": "%",
                },
                {
                    "name": "drop_if_any_missing",
                    "label": "Drop rows with any missing value",
                    "description": "Override the threshold and drop rows that contain any missing value.",
                    "type": "boolean",
                    "default": False,
                },
            ],
            "default_config": {
                "missing_threshold": 50.0,
                "drop_if_any_missing": False,
            },

        }

        outlier_removal_node: FeatureNodeCatalogEntry = {
            "type": "outlier_removal",
            "label": "Remove outliers",
            "description": "Identify and handle numeric outliers using z-score, IQR, winsorization, or manual bounds.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["numeric", "cleanup", "quality"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to inspect",
                    "description": "Select numeric columns manually or rely on auto-detection.",
                    "type": "multi_select",
                },
                {
                    "name": "default_method",
                    "label": "Default outlier strategy",
                    "description": "Method applied when a column override is not configured.",
                    "type": "text",
                    "default": OUTLIER_DEFAULT_METHOD,
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include numeric columns that pass quality checks.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "default_method": OUTLIER_DEFAULT_METHOD,
                "column_methods": {},
                "auto_detect": True,
                "skipped_columns": [],
                "method_parameters": {
                    key: dict(value)
                    for key, value in DEFAULT_METHOD_PARAMETERS.items()
                },
                "column_parameters": {},
            },
        }

        missing_indicator_node: FeatureNodeCatalogEntry = {
            "type": "missing_value_indicator",
            "label": "Missing value indicator",
            "description": "Append binary *_was_missing columns that flag rows with missing data.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["missing_data", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to flag",
                    "description": (
                        "Choose columns to generate missing-value indicator fields "
                        "(default: columns with missing data)."
                    ),
                    "type": "multi_select",
                },
                {
                    "name": "flag_suffix",
                    "label": "Indicator suffix",
                    "description": (
                        "Suffix appended to the original column name when "
                        "creating the flag column."
                    ),
                    "type": "text",
                    "default": "_was_missing",
                },
            ],
            "default_config": {
                "columns": [],
                "flag_suffix": "_was_missing",
            },
        }

        remove_duplicates_node: FeatureNodeCatalogEntry = {
            "type": "remove_duplicates",
            "label": "Remove duplicate rows",
            "description": "Drop duplicate rows based on all columns or a selected subset.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["duplicates", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to compare",
                    "description": "Leave blank to compare all columns when detecting duplicates.",
                    "type": "multi_select",
                },
                {
                    "name": "keep",
                    "label": "Keep strategy",
                    "description": "Accepts 'first', 'last', or 'none' (drop all duplicates).",
                    "type": "text",
                    "default": "first",
                },
            ],
            "default_config": {
                "columns": [],
                "keep": "first",
            },
        }

        cast_column_types_node: FeatureNodeCatalogEntry = {
            "type": "cast_column_types",
            "label": "Cast column types",
            "description": "Convert selected columns to a target pandas dtype (e.g., float64, Int64, datetime).",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["typing", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to cast",
                    "description": "Select columns to convert to the target dtype.",
                    "type": "multi_select",
                },
                {
                    "name": "target_dtype",
                    "label": "Target dtype",
                    "description": "Enter a pandas dtype such as float64, Int64, string, boolean, or datetime64[ns].",
                    "type": "text",
                },
                {
                    "name": "coerce_on_error",
                    "label": "Coerce invalid values",
                    "description": "Convert unparseable values to missing instead of raising errors.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "target_dtype": "float64",
                "coerce_on_error": True,
            },
        }

        imputation_methods_node: FeatureNodeCatalogEntry = {
            "type": "imputation_methods",
            "label": "Imputation methods",
            "description": "Configure statistical or model-driven fills (mean/median/mode, KNN, regression, MICE).",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["missing_data", "imputation", "advanced"],
            "parameters": [
                {
                    "name": "strategies",
                    "label": "Imputation recipes",
                    "description": "Configure multivariate strategies like KNN, regression, or MICE per column group.",
                    "type": "text",
                }
            ],
            "default_config": {
                "strategies": [],
            },
        }

        trim_whitespace_node: FeatureNodeCatalogEntry = {
            "type": "trim_whitespace",
            "label": "Trim whitespace",
            "description": "Remove leading/trailing whitespace from text columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "cleanup", "standardization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Trim mode",
                    "description": "Choose whether to trim leading, trailing, or both sides of whitespace.",
                    "type": "select",
                    "default": "both",
                    "options": [
                        {"value": "both", "label": "Leading and trailing"},
                        {"value": "leading", "label": "Leading only"},
                        {"value": "trailing", "label": "Trailing only"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "both",
            },
        }

        normalize_text_case_node: FeatureNodeCatalogEntry = {
            "type": "normalize_text_case",
            "label": "Normalize text case",
            "description": "Convert text columns to a consistent case (lower, upper, title, sentence).",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "standardization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Case style",
                    "description": "Supported modes: lower, upper, title, sentence.",
                    "type": "select",
                    "default": "lower",
                    "options": [
                        {"value": "lower", "label": "Lowercase"},
                        {"value": "upper", "label": "Uppercase"},
                        {"value": "title", "label": "Title case"},
                        {"value": "sentence", "label": "Sentence case"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "lower",
            },
        }

        replace_aliases_node: FeatureNodeCatalogEntry = {
            "type": "replace_aliases_typos",
            "label": "Standardize aliases & typos",
            "description": "Normalize common aliases (countries, booleans) or apply custom replacements.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "standardization", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Replacement strategy",
                    "description": "Use presets for common aliases or switch to custom mappings.",
                    "type": "select",
                    "default": "canonicalize_country_codes",
                    "options": [
                        {"value": "canonicalize_country_codes", "label": "Country aliases"},
                        {"value": "normalize_boolean", "label": "Boolean tokens"},
                        {"value": "punctuation", "label": "Strip punctuation"},
                        {"value": "custom", "label": "Custom mappings"},
                    ],
                },
                {
                    "name": "custom_pairs",
                    "label": "Custom pairs",
                    "description": (
                        "Provide alias => replacement pairs (one per line). Applies "
                        "when mode is set to Custom."
                    ),
                    "type": "textarea",
                    "placeholder": "st -> Street\nrd => Road",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "canonicalize_country_codes",
                "custom_pairs": "",
            },
        }

        standardize_dates_node: FeatureNodeCatalogEntry = {
            "type": "standardize_date_formats",
            "label": "Standardize date formats",
            "description": "Parse and rewrite date/time strings into a consistent format.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["datetime", "cleanup", "standardization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect datetime-like columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Output format",
                    "description": "Select the desired date format for parsed values.",
                    "type": "select",
                    "default": "iso_date",
                    "options": [
                        {"value": "iso_date", "label": "ISO date (YYYY-MM-DD)"},
                        {"value": "iso_datetime", "label": "ISO datetime (YYYY-MM-DD HH:MM:SS)"},
                        {"value": "month_day_year", "label": "MM/DD/YYYY"},
                        {"value": "day_month_year", "label": "DD/MM/YYYY"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "iso_date",
            },
        }

        remove_special_chars_node: FeatureNodeCatalogEntry = {
            "type": "remove_special_characters",
            "label": "Remove special characters",
            "description": "Strip or replace non-alphanumeric characters in text columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Retention rule",
                    "description": "Choose which character classes should be preserved.",
                    "type": "select",
                    "default": "keep_alphanumeric",
                    "options": [
                        {"value": "keep_alphanumeric", "label": "Keep letters & digits"},
                        {"value": "keep_alphanumeric_space", "label": "Keep letters, digits & spaces"},
                        {"value": "letters_only", "label": "Letters only"},
                        {"value": "digits_only", "label": "Digits only"},
                    ],
                },
                {
                    "name": "replacement",
                    "label": "Replacement",
                    "description": "Text to insert when removing characters (default: remove).",
                    "type": "text",
                    "default": "",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "keep_alphanumeric",
                "replacement": "",
            },
        }

        replace_invalid_values_node: FeatureNodeCatalogEntry = {
            "type": "replace_invalid_values",
            "label": "Replace invalid numeric values",
            "description": "Convert out-of-range or placeholder numeric values to missing.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["numeric", "cleanup", "quality"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect numeric columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Rule",
                    "description": "Select how invalid values are detected and replaced.",
                    "type": "select",
                    "default": "negative_to_nan",
                    "options": [
                        {"value": "negative_to_nan", "label": "Negative to missing"},
                        {"value": "zero_to_nan", "label": "Zero to missing"},
                        {"value": "percentage_bounds", "label": "Percentage bounds (0-100%)"},
                        {"value": "age_bounds", "label": "Age bounds (0-120)"},
                        {"value": "custom_range", "label": "Custom numeric range"},
                    ],
                },
                {
                    "name": "min_value",
                    "label": "Minimum value",
                    "description": "Optional lower bound (applies to percentage, age, or custom modes).",
                    "type": "number",
                },
                {
                    "name": "max_value",
                    "label": "Maximum value",
                    "description": "Optional upper bound (applies to percentage, age, or custom modes).",
                    "type": "number",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "negative_to_nan",
                "min_value": None,
                "max_value": None,
            },
        }

        regex_replace_node: FeatureNodeCatalogEntry = {
            "type": "regex_replace_fix",
            "label": "Regex cleanup",
            "description": "Apply preset or custom regular expression replacements to text columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "regex", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Regex preset",
                    "description": "Use a preset cleanup or switch to custom pattern/replacement.",
                    "type": "select",
                    "default": "normalize_slash_dates",
                    "options": [
                        {"value": "normalize_slash_dates", "label": "Normalize slash dates"},
                        {"value": "collapse_whitespace", "label": "Collapse whitespace"},
                        {"value": "extract_digits", "label": "Extract digits"},
                        {"value": "custom", "label": "Custom pattern"},
                    ],
                },
                {
                    "name": "pattern",
                    "label": "Custom pattern",
                    "description": "Regular expression applied when mode is Custom.",
                    "type": "text",
                    "placeholder": r"(?i)acct#:?",
                },
                {
                    "name": "replacement",
                    "label": "Replacement text",
                    "description": "Replacement string for the regex substitution (supports backreferences).",
                    "type": "text",
                    "default": "",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "normalize_slash_dates",
                "pattern": "",
                "replacement": "",
            },
        }

        binning_node: FeatureNodeCatalogEntry = {
            "type": "binning_discretization",
            "label": "Bin numeric columns",
            "description": "Discretize numeric columns with equal-width, equal-frequency, or custom thresholds.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "discretization", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to bin",
                    "description": "Select numeric columns to convert into discrete bins.",
                    "type": "multi_select",
                }
            ],
            "default_config": {
                "strategy": "equal_width",
                "columns": [],
                "equal_width_bins": BINNING_DEFAULT_EQUAL_WIDTH_BINS,
                "equal_frequency_bins": BINNING_DEFAULT_EQUAL_FREQUENCY_BINS,
                "include_lowest": True,
                "precision": BINNING_DEFAULT_PRECISION,
                "duplicates": "raise",
                "output_suffix": BINNING_DEFAULT_SUFFIX,
                "drop_original": False,
                "label_format": "range",
                "missing_strategy": "keep",
                "missing_label": BINNING_DEFAULT_MISSING_LABEL,
                "custom_bins": {},
                "custom_labels": {},
            },
        }

        polynomial_features_node: FeatureNodeCatalogEntry = {
            "type": "polynomial_features",
            "label": "Polynomial features",
            "description": "Generate polynomial and interaction terms for numeric columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "feature_engineering", "transformation"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to expand",
                    "description": "Select numeric columns manually or leave blank to auto-detect.",
                    "type": "multi_select",
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include numeric columns detected at runtime.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "degree",
                    "label": "Maximum degree",
                    "description": "Upper polynomial degree to generate (higher degrees create more features).",
                    "type": "number",
                    "default": 2,
                    "min": 2.0,
                    "max": 5.0,
                    "step": 1.0,
                },
                {
                    "name": "include_bias",
                    "label": "Add bias column",
                    "description": "Include a constant bias column when enabled.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "interaction_only",
                    "label": "Interaction only",
                    "description": "Generate only interaction features without power terms.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "include_input_features",
                    "label": "Include original features",
                    "description": "Retain degree-1 terms alongside the generated features.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "output_prefix",
                    "label": "Feature prefix",
                    "description": "Prefix applied to generated feature columns.",
                    "type": "text",
                    "default": "poly",
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "degree": 2,
                "include_bias": False,
                "interaction_only": False,
                "include_input_features": False,
                "output_prefix": "poly",
            },
        }

        feature_selection_node: FeatureNodeCatalogEntry = {
            "type": "feature_selection",
            "label": "Feature selection",
            "description": (
                "Score features and retain the most informative columns using "
                "univariate tests or model-based selectors."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["feature_selection", "numeric", "modeling"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Candidate columns",
                    "description": "Select columns to score or leave blank to rely on auto-detection.",
                    "type": "multi_select",
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include numeric columns detected at runtime.",
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Supervised selectors require a target column for scoring "
                        "(e.g., regression/classification target)."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "method",
                    "label": "Selection method",
                    "description": "Choose the feature selection strategy to apply.",
                    "type": "select",
                    "default": "select_k_best",
                    "options": [
                        {"value": "select_k_best", "label": "Select K Best"},
                        {"value": "select_percentile", "label": "Select Percentile"},
                        {"value": "generic_univariate_select", "label": "Generic univariate (mode driven)"},
                        {"value": "select_fpr", "label": "Select FPR"},
                        {"value": "select_fdr", "label": "Select FDR"},
                        {"value": "select_fwe", "label": "Select FWE"},
                        {"value": "select_from_model", "label": "Select From Model"},
                        {"value": "variance_threshold", "label": "Variance Threshold"},
                        {"value": "rfe", "label": "Recursive Feature Elimination"},
                    ],
                },
                {
                    "name": "score_func",
                    "label": "Scoring function",
                    "description": "Statistical test used for univariate selectors.",
                    "type": "select",
                    "default": "f_classif",
                    "options": [
                        {"value": "f_classif", "label": "ANOVA F-value (classification)"},
                        {"value": "f_regression", "label": "F-value (regression)"},
                        {"value": "mutual_info_classif", "label": "Mutual information (classification)"},
                        {"value": "mutual_info_regression", "label": "Mutual information (regression)"},
                        {"value": "chi2", "label": "Chi-squared (non-negative features)"},
                        {"value": "r_regression", "label": "Pearson r"},
                    ],
                },
                {
                    "name": "problem_type",
                    "label": "Problem type",
                    "description": "Guides default scoring function and estimator selection.",
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto detect"},
                        {"value": "classification", "label": "Classification"},
                        {"value": "regression", "label": "Regression"},
                    ],
                },
                {
                    "name": "k",
                    "label": "Top K features",
                    "description": "Number of features to keep when using K-based strategies.",
                    "type": "number",
                    "min": 1.0,
                    "step": 1.0,
                    "default": 10.0,
                },
                {
                    "name": "percentile",
                    "label": "Percentile",
                    "description": "Percentile of features to retain when using percentile strategies.",
                    "type": "number",
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "default": 10.0,
                },
                {
                    "name": "alpha",
                    "label": "Alpha",
                    "description": "Significance threshold for FPR/FDR/FWE modes.",
                    "type": "number",
                    "min": 0.0,
                    "step": 0.001,
                    "default": 0.05,
                },
                {
                    "name": "threshold",
                    "label": "Threshold",
                    "description": "Threshold for variance or model-based selectors (leave blank for defaults).",
                    "type": "number",
                },
                {
                    "name": "mode",
                    "label": "Generic mode",
                    "description": "Mode parameter for GenericUnivariateSelect (k_best, percentile, fpr, fdr, fwe).",
                    "type": "select",
                    "default": "k_best",
                    "options": [
                        {"value": "k_best", "label": "K Best"},
                        {"value": "percentile", "label": "Percentile"},
                        {"value": "fpr", "label": "FPR"},
                        {"value": "fdr", "label": "FDR"},
                        {"value": "fwe", "label": "FWE"},
                    ],
                },
                {
                    "name": "estimator",
                    "label": "Estimator",
                    "description": "Base estimator used for model-based selectors (SelectFromModel / RFE).",
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto"},
                        {"value": "logistic_regression", "label": "Logistic regression"},
                        {"value": "random_forest", "label": "Random forest"},
                        {"value": "linear_regression", "label": "Linear regression"},
                    ],
                },
                {
                    "name": "step",
                    "label": "RFE step",
                    "description": "Number (or fraction) of features to remove at each RFE iteration.",
                    "type": "number",
                    "step": 0.1,
                    "default": 1.0,
                },
                {
                    "name": "min_features",
                    "label": "Minimum features",
                    "description": "Optional lower bound on features to keep (used by some estimators).",
                    "type": "number",
                },
                {
                    "name": "max_features",
                    "label": "Maximum features",
                    "description": "Optional upper bound on features to keep (used by some estimators).",
                    "type": "number",
                },
                {
                    "name": "drop_unselected",
                    "label": "Drop unselected columns",
                    "description": "Remove columns that fail the selection criteria from the dataset.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": True,
                "target_column": "",
                "method": "select_k_best",
                "score_func": "f_classif",
                "problem_type": "auto",
                "k": 10,
                "percentile": 10.0,
                "alpha": 0.05,
                "threshold": None,
                "mode": "k_best",
                "estimator": "auto",
                "step": 1.0,
                "min_features": None,
                "max_features": None,
                "drop_unselected": True,
            },
        }

        feature_math_node: FeatureNodeCatalogEntry = {
            "type": "feature_math",
            "label": "Feature math lab",
            "description": (
                "Combine columns using arithmetic, ratios, statistics, similarity "
                "scores, and datetime extraction steps."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "datetime", "text", "feature_engineering"],
            "parameters": [
                {
                    "name": "error_handling",
                    "label": "On operation error",
                    "description": (
                        "Choose whether the pipeline should continue or fail when "
                        "an operation encounters invalid inputs."
                    ),
                    "type": "select",
                    "default": "skip",
                    "options": [
                        {"value": "skip", "label": "Skip and continue"},
                        {"value": "fail", "label": "Fail immediately"},
                    ],
                },
                {
                    "name": "allow_overwrite",
                    "label": "Allow overwriting columns",
                    "description": (
                        "Permit operations to overwrite existing columns when the "
                        "chosen output already exists."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "default_timezone",
                    "label": "Default timezone",
                    "description": "Timezone applied when extracting datetime features (IANA name).",
                    "type": "text",
                    "default": "UTC",
                },
                {
                    "name": "epsilon",
                    "label": "Division epsilon",
                    "description": (
                        "Small constant added to denominators to avoid "
                        "divide-by-zero when computing ratios."
                    ),
                    "type": "number",
                    "default": 1e-9,
                    "min": 0.0,
                    "step": 1e-9,
                },
            ],
            "default_config": {
                "operations": [],
                "error_handling": "skip",
                "allow_overwrite": False,
                "default_timezone": "UTC",
                "epsilon": 1e-9,
            },
        }

        undersampling_method_options: List[FeatureNodeParameterOption] = [
            _make_option(method, label)
            for method, label in UNDERSAMPLING_METHOD_LABELS.items()
        ]

        oversampling_method_options: List[FeatureNodeParameterOption] = [
            _make_option(method, label)
            for method, label in OVERSAMPLING_METHOD_LABELS.items()
        ]

        undersampling_node: FeatureNodeCatalogEntry = {
            "type": "class_undersampling",
            "label": "Class undersampling",
            "description": (
                "Reduce majority-class rows with random under-sampling to "
                "improve balance."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "RESAMPLING DATASET",
            "tags": ["undersampling", "class_balance", "imbalanced"],
            "parameters": [
                {
                    "name": "method",
                    "label": "Resampling method",
                    "description": (
                        "Choose the sampling approach (under-sampling available now)."
                    ),
                    "type": "select",
                    "default": UNDERSAMPLING_DEFAULT_METHOD,
                    "options": undersampling_method_options,
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": "Categorical target column used to guide sampling.",
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "sampling_strategy",
                    "label": "Sampling ratio",
                    "description": "Minority-to-majority ratio (0 < ratio ≤ 1). Leave blank for auto.",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "name": "random_state",
                    "label": "Random state",
                    "description": "Optional random seed for reproducible sampling.",
                    "type": "number",
                    "default": UNDERSAMPLING_DEFAULT_RANDOM_STATE,
                    "step": 1.0,
                },
                {
                    "name": "replacement",
                    "label": "Sample with replacement",
                    "description": "Allow sampling with replacement when reducing the majority class.",
                    "type": "boolean",
                    "default": UNDERSAMPLING_DEFAULT_REPLACEMENT,
                },
            ],
            "default_config": {
                "method": UNDERSAMPLING_DEFAULT_METHOD,
                "target_column": "",
                "sampling_strategy": None,
                "random_state": UNDERSAMPLING_DEFAULT_RANDOM_STATE,
                "replacement": UNDERSAMPLING_DEFAULT_REPLACEMENT,
            },
        }

        oversampling_node: FeatureNodeCatalogEntry = {
            "type": "class_oversampling",
            "label": "Class oversampling",
            "description": (
                "Boost minority-class representation with synthetic "
                "over-sampling techniques."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "RESAMPLING DATASET",
            "tags": ["oversampling", "class_balance", "imbalanced"],
            "parameters": [
                {
                    "name": "method",
                    "label": "Resampling method",
                    "description": (
                        "Choose the synthetic sampling strategy for balancing classes."
                    ),
                    "type": "select",
                    "default": OVERSAMPLING_DEFAULT_METHOD,
                    "options": oversampling_method_options,
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": "Categorical target column used to guide sampling.",
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "sampling_strategy",
                    "label": "Sampling ratio",
                    "description": "Minority-to-majority ratio (> 0). Leave blank for auto.",
                    "type": "number",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                },
                {
                    "name": "k_neighbors",
                    "label": "K-neighbors",
                    "description": (
                        "Nearest neighbors considered when synthesising new "
                        "minority samples."
                    ),
                    "type": "number",
                    "default": OVERSAMPLING_DEFAULT_K_NEIGHBORS,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                },
                {
                    "name": "random_state",
                    "label": "Random state",
                    "description": "Optional random seed for reproducible sampling.",
                    "type": "number",
                    "default": OVERSAMPLING_DEFAULT_RANDOM_STATE,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "method": OVERSAMPLING_DEFAULT_METHOD,
                "target_column": "",
                "sampling_strategy": None,
                "k_neighbors": OVERSAMPLING_DEFAULT_K_NEIGHBORS,
                "random_state": OVERSAMPLING_DEFAULT_RANDOM_STATE,
                "replacement": OVERSAMPLING_DEFAULT_REPLACEMENT,
            },
        }

        label_encoding_node: FeatureNodeCatalogEntry = {
            "type": "label_encoding",
            "label": "Label encode categories",
            "description": "Convert categorical columns into integer codes for model-ready features.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": "Choose categorical columns or leave blank to rely on auto-detection.",
                    "type": "multi_select",
                    "source": {
                        "type": "label_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/label-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": "Automatically include text columns that stay within the cardinality threshold.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_unique_values",
                    "label": "Max categories for auto-detect",
                    "description": "Upper bound on unique values when auto-detect is enabled (0 disables the cap).",
                    "type": "number",
                    "default": LABEL_ENCODING_DEFAULT_MAX_UNIQUE,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": "Suffix added to new encoded columns when originals are retained.",
                    "type": "text",
                    "default": LABEL_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": "Overwrite the source column instead of creating a suffixed copy.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "missing_strategy",
                    "label": "Missing value strategy",
                    "description": "Choose whether to keep missing values as <NA> or assign a dedicated code.",
                    "type": "select",
                    "default": "keep_na",
                    "options": [
                        {"value": "keep_na", "label": "Keep as <NA>"},
                        {"value": "encode", "label": "Assign code"},
                    ],
                },
                {
                    "name": "missing_code",
                    "label": "Missing value code",
                    "description": "Integer code applied to missing values when the assign-code strategy is selected.",
                    "type": "number",
                    "default": -1,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_unique_values": LABEL_ENCODING_DEFAULT_MAX_UNIQUE,
                "output_suffix": LABEL_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "missing_strategy": "keep_na",
                "missing_code": -1,
                "skipped_columns": [],
            },
        }

        target_encoding_node: FeatureNodeCatalogEntry = {
            "type": "target_encoding",
            "label": "Target encode categories",
            "description": (
                "Replace categorical values with smoothed averages of a "
                "numeric target column."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "when safe."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "target_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/target-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Numeric target column used to compute per-category "
                        "averages (e.g., regression value or 0/1 outcome)."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": (
                        "Automatically include text columns that stay within "
                        "the category limit."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": TARGET_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(TARGET_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": (
                        "Suffix added to new encoded columns when originals "
                        "are retained."
                    ),
                    "type": "text",
                    "default": TARGET_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Overwrite the source column instead of creating a "
                        "suffixed copy."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "smoothing",
                    "label": "Smoothing strength",
                    "description": (
                        "Higher values pull category means toward the global "
                        "average to reduce overfitting."
                    ),
                    "type": "number",
                    "default": TARGET_ENCODING_DEFAULT_SMOOTHING,
                    "min": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "encode_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Assign the global mean to rows where the categorical "
                        "value is missing."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "handle_unknown",
                    "label": "Handle unseen categories",
                    "description": (
                        "Choose whether to assign the global mean or raise an "
                        "error for categories not observed during fitting."
                    ),
                    "type": "select",
                    "default": TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN,
                    "options": [
                        {"value": "global_mean", "label": "Use global mean"},
                        {"value": "error", "label": "Raise error"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "target_column": "",
                "auto_detect": False,
                "max_categories": TARGET_ENCODING_DEFAULT_MAX_CATEGORIES,
                "output_suffix": TARGET_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "smoothing": TARGET_ENCODING_DEFAULT_SMOOTHING,
                "encode_missing": False,
                "handle_unknown": TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN,
                "skipped_columns": [],
            },
        }

        hash_encoding_node: FeatureNodeCatalogEntry = {
            "type": "hash_encoding",
            "label": "Hash encode categories",
            "description": (
                "Project categorical values into deterministic hash buckets "
                "to bound feature width."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "to handle high-cardinality features."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "hash_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/hash-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": (
                        "Automatically include text columns that stay within "
                        "the category limit."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": HASH_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(HASH_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "n_buckets",
                    "label": "Hash buckets",
                    "description": (
                        "Number of hash buckets used to encode each column "
                        "(higher values reduce collisions)."
                    ),
                    "type": "number",
                    "default": HASH_ENCODING_DEFAULT_BUCKETS,
                    "min": float(2),
                    "max": float(HASH_ENCODING_MAX_BUCKETS),
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": (
                        "Suffix added to new hashed columns when originals "
                        "are retained."
                    ),
                    "type": "text",
                    "default": HASH_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Overwrite the source column instead of creating a "
                        "suffixed copy."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "encode_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Assign a dedicated bucket to missing values instead "
                        "of leaving them null."
                    ),
                    "type": "boolean",
                    "default": False,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": HASH_ENCODING_DEFAULT_MAX_CATEGORIES,
                "n_buckets": HASH_ENCODING_DEFAULT_BUCKETS,
                "output_suffix": HASH_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "encode_missing": False,
                "skipped_columns": [],
            },
        }

        model_registry = list_registered_models()
        model_type_options: List[FeatureNodeParameterOption] = []
        for spec_key, spec in model_registry.items():
            label = spec_key.replace("_", " ").title()
            description = f"{spec.problem_type.title()} baseline"
            model_type_options.append(
                _make_option(
                    spec_key,
                    label,
                    description=description,
                    metadata={
                        "problem_type": spec.problem_type,
                        "default_params": dict(spec.default_params),
                    },
                )
            )

        model_type_options.sort(key=lambda option: option["label"])
        if not model_type_options:
            model_type_options.append(_make_option("logistic_regression", "Logistic Regression"))

        preferred_default_model_type = "logistic_regression"
        default_model_type = preferred_default_model_type
        preferred_exists = any(
            option["value"] == preferred_default_model_type for option in model_type_options
        )
        if not preferred_exists:
            default_model_type = model_type_options[0]["value"]

        default_model_spec = model_registry.get(default_model_type)
        default_hyperparameters_text = ""
        if default_model_spec and default_model_spec.default_params:
            try:
                default_hyperparameters_text = json.dumps(
                    default_model_spec.default_params,
                    indent=2,
                    sort_keys=True,
                )
            except TypeError:
                default_hyperparameters_text = json.dumps(default_model_spec.default_params, indent=2)

        feature_target_split_node: FeatureNodeCatalogEntry = {
            "type": "feature_target_split",
            "label": "Separate features & target",
            "description": (
                "Designate the supervised target column. All remaining "
                "columns are treated as features (X)."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "MODELING",
            "tags": ["modeling", "features", "target"],
            "parameters": [
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Column treated as the target (y) for downstream "
                        "modeling."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
            ],
            "default_config": {
                "target_column": "",
            },
        }

        train_test_split_node: FeatureNodeCatalogEntry = {
            "type": "train_test_split",
            "label": "Train/Test Split",
            "description": (
                "Split dataset into training, testing, and optionally "
                "validation sets. Supports stratification."
            ),
            "inputs": ["dataset"],
            "outputs": ["train", "test", "validation"],
            "category": "MODELING",
            "tags": ["modeling", "split", "train_test"],
            "parameters": [
                {
                    "name": "test_size",
                    "label": "Test size",
                    "description": (
                        "Proportion of the dataset to include in the test "
                        "split."
                    ),
                    "type": "number",
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "name": "validation_size",
                    "label": "Validation size",
                    "description": (
                        "Proportion of the dataset to include in the "
                        "validation split (optional)."
                    ),
                    "type": "number",
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "name": "random_state",
                    "label": "Random state",
                    "description": "Seed for reproducible splits. Leave empty for random.",
                    "type": "number",
                    "default": 42,
                    "min": 0,
                    "step": 1,
                },
                {
                    "name": "shuffle",
                    "label": "Shuffle",
                    "description": "Whether to shuffle the data before splitting.",
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "stratify",
                    "label": "Stratify",
                    "description": "Preserve class distribution in splits using the target column.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "target_column",
                    "label": "Target column (for stratification)",
                    "description": "Column to use for stratified splitting (required if stratify is enabled).",
                    "type": "text",
                    "placeholder": "target",
                },
            ],
            "default_config": {
                "test_size": 0.2,
                "validation_size": 0.0,
                "random_state": 42,
                "shuffle": True,
                "stratify": False,
                "target_column": "",
            },
        }

        problem_type_options: List[FeatureNodeParameterOption] = [
            _make_option("classification", "Classification"),
            _make_option("regression", "Regression"),
        ]

        train_model_draft_node: FeatureNodeCatalogEntry = {
            "type": "train_model_draft",
            "label": "Train model",
            "description": (
                "Validate the pipeline output, launch background training "
                "jobs, and expose trained models downstream."
            ),
            "inputs": ["dataset"],
            "outputs": ["models"],
            "category": "MODELING",
            "tags": ["modeling", "validation", "preview"],
            "parameters": [
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Name of the response column to model (leave blank "
                        "to map via downstream configuration)."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "problem_type",
                    "label": "Problem type",
                    "description": (
                        "Choose the expected modeling task for downstream "
                        "training."
                    ),
                    "type": "select",
                    "default": "classification",
                    "options": problem_type_options,
                },
                {
                    "name": "model_type",
                    "label": "Model template",
                    "description": (
                        "Select the registered estimator used when launching "
                        "background jobs."
                    ),
                    "type": "select",
                    "default": default_model_type,
                    "options": model_type_options,
                },
                {
                    "name": "hyperparameters",
                    "label": "Hyperparameters (JSON)",
                    "description": (
                        "Optional JSON object merged with the default "
                        "parameters for the selected model."
                    ),
                    "type": "textarea",
                    "placeholder": "{\n  \"n_estimators\": 200\n}",
                },
                {
                    "name": "cv_enabled",
                    "label": "Enable cross-validation",
                    "description": (
                        "Run k-fold cross-validation on the training split "
                        "before finalising the model."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "cv_strategy",
                    "label": "Cross-validation strategy",
                    "description": (
                        "Choose how folds are generated when cross-validation "
                        "is enabled."
                    ),
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto"},
                        {"value": "kfold", "label": "K-Fold"},
                        {"value": "stratified_kfold", "label": "Stratified K-Fold"},
                    ],
                },
                {
                    "name": "cv_folds",
                    "label": "Number of folds",
                    "description": (
                        "How many folds to use when cross-validation is "
                        "enabled (minimum 2)."
                    ),
                    "type": "number",
                    "default": 5,
                    "min": 2.0,
                    "max": 20.0,
                    "step": 1.0,
                },
                {
                    "name": "cv_shuffle",
                    "label": "Shuffle before splitting",
                    "description": (
                        "Shuffle the training rows before generating "
                        "cross-validation folds."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "cv_random_state",
                    "label": "Shuffle random state",
                    "description": (
                        "Optional random seed applied when shuffling folds "
                        "(leave blank for random)."
                    ),
                    "type": "number",
                    "default": 42,
                    "min": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "cv_refit_strategy",
                    "label": "Refit using",
                    "description": (
                        "Choose whether the final model should be refit on "
                        "the training split only or on training+validation "
                        "after cross-validation."
                    ),
                    "type": "select",
                    "default": "train_plus_validation",
                    "options": [
                        {"value": "train_only", "label": "Train split only"},
                        {"value": "train_plus_validation", "label": "Train + validation"},
                    ],
                },
            ],
            "default_config": {
                "target_column": "",
                "problem_type": "classification",
                "model_type": default_model_type,
                "hyperparameters": default_hyperparameters_text,
                "cv_enabled": False,
                "cv_strategy": "auto",
                "cv_folds": 5,
                "cv_shuffle": True,
                "cv_random_state": 42,
                "cv_refit_strategy": "train_plus_validation",
            },
        }

        model_evaluation_node: FeatureNodeCatalogEntry = {
            "type": "model_evaluation",
            "label": "Model evaluation",
            "description": (
                "Review confusion matrices, ROC/PR curves, and residual "
                "diagnostics for trained models without leaving the canvas."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "MODELING",
            "tags": ["modeling", "evaluation", "diagnostics"],
            "parameters": [],
            "default_config": {
                "training_job_id": "",
                "splits": ["test"],
                "include_curves": True,
                "include_confusion": True,
                "include_residuals": True,
                "last_evaluated_at": None,
            },
        }

        model_registry_node: FeatureNodeCatalogEntry = {
            "type": "model_registry_overview",
            "label": "Model registry",
            "description": (
                "Review trained model versions, metrics, and artifacts from "
                "the connected training node."
            ),
            "inputs": ["models"],
            "outputs": ["models"],
            "category": "MODELING",
            "tags": ["modeling", "registry", "metrics"],
            "parameters": [
                {
                    "name": "default_problem_type",
                    "label": "Default problem tab",
                    "description": (
                        "Problem type tab that opens by default when viewing "
                        "the registry."
                    ),
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto (latest model)"},
                        {"value": "classification", "label": "Classification"},
                        {"value": "regression", "label": "Regression"},
                    ],
                },
                {
                    "name": "default_method",
                    "label": "Default model method",
                    "description": (
                        "Optional model template to spotlight initially (e.g. "
                        "logistic_regression)."
                    ),
                    "type": "text",
                    "placeholder": "logistic_regression",
                },
                {
                    "name": "show_non_success",
                    "label": "Show non-successful runs",
                    "description": (
                        "Include queued, running, failed, and cancelled "
                        "versions in comparisons."
                    ),
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "default_problem_type": "auto",
                "default_method": "",
                "show_non_success": True,
            },
        }

        tuning_strategy_choices = [
            _make_option(
                choice.get("value", ""),
                choice.get("label", ""),
                description=choice.get("description"),
            )
            for choice in get_strategy_choices_for_ui()
        ]
        tuning_strategy_default = get_default_strategy_value()

        tuning_default_search_space = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"],
        }
        default_search_space_text = json.dumps(tuning_default_search_space, indent=2)

        hyperparameter_tuning_node: FeatureNodeCatalogEntry = {
            "type": "hyperparameter_tuning",
            "label": "Hyperparameter tuning",
            "description": (
                "Search over candidate hyperparameter combinations using "
                "cross-validation and background workers."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "MODELING",
            "tags": ["modeling", "tuning", "optimization"],
            "parameters": [
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Name of the response column to optimize against."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "problem_type",
                    "label": "Problem type",
                    "description": (
                        "Select the expected modeling task so metrics align."
                    ),
                    "type": "select",
                    "default": "classification",
                    "options": problem_type_options,
                },
                {
                    "name": "model_type",
                    "label": "Model template",
                    "description": (
                        "Choose which registered estimator to tune."
                    ),
                    "type": "select",
                    "default": default_model_type,
                    "options": model_type_options,
                },
                {
                    "name": "baseline_hyperparameters",
                    "label": "Baseline hyperparameters (JSON)",
                    "description": (
                        "Optional JSON object merged into the estimator "
                        "before tuning."
                    ),
                    "type": "textarea",
                    "placeholder": "{\n  \"max_iter\": 1000\n}",
                },
                {
                    "name": "search_strategy",
                    "label": "Search strategy",
                    "description": (
                        "Choose how to explore the search space. Available "
                        "strategies are configured in application settings."
                    ),
                    "type": "select",
                    "default": tuning_strategy_default,
                    "options": tuning_strategy_choices,
                },
                {
                    "name": "search_space",
                    "label": "Search space (JSON)",
                    "description": (
                        "JSON object mapping hyperparameter names to lists "
                        "of candidate values."
                    ),
                    "type": "textarea",
                    "placeholder": "{\n  \"C\": [0.1, 1.0, 10.0],\n  \"solver\": [\"lbfgs\", \"saga\"]\n}",
                },
                {
                    "name": "search_iterations",
                    "label": "Max iterations",
                    "description": (
                        "Maximum sampled combinations when random or Optuna "
                        "search is enabled (ignored for grid and halving)."
                    ),
                    "type": "number",
                    "default": 20,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 1.0,
                },
                {
                    "name": "search_random_state",
                    "label": "Random state",
                    "description": (
                        "Optional seed controlling candidate sampling order "
                        "for random/Optuna search."
                    ),
                    "type": "number",
                    "default": 42,
                    "min": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "scoring_metric",
                    "label": "Scoring metric",
                    "description": (
                        "Optional sklearn-compatible scoring string (leave "
                        "blank for model default)."
                    ),
                    "type": "text",
                    "placeholder": "accuracy",
                },
                {
                    "name": "cv_enabled",
                    "label": "Enable cross-validation",
                    "description": (
                        "Toggle K-fold cross-validation for evaluating each "
                        "candidate."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "cv_strategy",
                    "label": "Cross-validation strategy",
                    "description": (
                        "Choose how folds are generated when tuning."
                    ),
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto"},
                        {"value": "kfold", "label": "K-Fold"},
                        {"value": "stratified_kfold", "label": "Stratified K-Fold"},
                    ],
                },
                {
                    "name": "cv_folds",
                    "label": "Number of folds",
                    "description": (
                        "How many folds to use when tuning (minimum 2)."
                    ),
                    "type": "number",
                    "default": 5,
                    "min": 2.0,
                    "max": 20.0,
                    "step": 1.0,
                },
                {
                    "name": "cv_shuffle",
                    "label": "Shuffle before splitting",
                    "description": (
                        "Shuffle rows before generating folds."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "cv_random_state",
                    "label": "Shuffle random state",
                    "description": (
                        "Optional random seed applied when shuffling folds."
                    ),
                    "type": "number",
                    "default": 42,
                    "min": 0.0,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "target_column": "",
                "problem_type": "classification",
                "model_type": default_model_type,
                "baseline_hyperparameters": default_hyperparameters_text,
                "search_strategy": tuning_strategy_default,
                "search_space": default_search_space_text,
                "search_iterations": 20,
                "search_random_state": 42,
                "scoring_metric": "",
                "cv_enabled": True,
                "cv_strategy": "auto",
                "cv_folds": 5,
                "cv_shuffle": True,
                "cv_random_state": 42,
            },
        }

        ordinal_encoding_node: FeatureNodeCatalogEntry = {
            "type": "ordinal_encoding",
            "label": "Ordinal encode categories",
            "description": (
                "Map categorical columns to ordered codes with optional "
                "unknown fallbacks."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "when safe."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "ordinal_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/ordinal-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": "Automatically include text columns that stay within the category limit.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": (
                        "Suffix added to new encoded columns when originals "
                        "are retained."
                    ),
                    "type": "text",
                    "default": ORDINAL_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Overwrite the source column instead of creating a "
                        "suffixed copy."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "encode_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Assign the fallback code to missing values instead "
                        "of leaving them null."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "handle_unknown",
                    "label": "Handle unseen categories",
                    "description": (
                        "Choose whether to assign a fallback code or raise an "
                        "error for unknown categories."
                    ),
                    "type": "select",
                    "default": "use_encoded_value",
                    "options": [
                        {"value": "use_encoded_value", "label": "Assign fallback code"},
                        {"value": "error", "label": "Raise error"},
                    ],
                },
                {
                    "name": "unknown_value",
                    "label": "Fallback code",
                    "description": (
                        "Integer code applied to unseen categories or missing "
                        "values when enabled."
                    ),
                    "type": "number",
                    "default": ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES,
                "output_suffix": ORDINAL_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "encode_missing": False,
                "handle_unknown": "use_encoded_value",
                "unknown_value": ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE,
                "skipped_columns": [],
            },
        }

        dummy_encoding_node: FeatureNodeCatalogEntry = {
            "type": "dummy_encoding",
            "label": "Dummy encode categories",
            "description": (
                "Expand categorical columns while dropping a reference level "
                "per feature."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "to capture safe candidates."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "dummy_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/dummy-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": (
                        "Automatically include text columns that keep dummy "
                        "width manageable."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(DUMMY_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "drop_first",
                    "label": "Drop reference level",
                    "description": (
                        "Always drop the first dummy column per feature to "
                        "mitigate multicollinearity."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "include_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Include a dedicated dummy column for missing values."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Remove the source column after dummy expansion."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "prefix_separator",
                    "label": "Dummy prefix separator",
                    "description": (
                        "Separator between the column name and category when "
                        "naming dummy columns."
                    ),
                    "type": "text",
                    "default": DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES,
                "drop_first": True,
                "include_missing": False,
                "drop_original": False,
                "prefix_separator": DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                "skipped_columns": [],
            },
        }

        one_hot_encoding_node: FeatureNodeCatalogEntry = {
            "type": "one_hot_encoding",
            "label": "One-hot encode categories",
            "description": "Expand categorical columns into binary indicator features.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": "Choose categorical columns or rely on auto-detection to keep configs light.",
                    "type": "multi_select",
                    "source": {
                        "type": "one_hot_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/one-hot-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": "Automatically include text columns when their dummy expansion stays manageable.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": "Upper bound on unique values when auto-detect is enabled (0 disables the cap).",
                    "type": "number",
                    "default": ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(ONE_HOT_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "drop_first",
                    "label": "Drop first dummy",
                    "description": "Avoid multicollinearity by dropping the first dummy column per feature.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "include_missing",
                    "label": "Encode missing values",
                    "description": "Include a dedicated dummy column for missing values.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": "Remove the source column after dummy expansion.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "prefix_separator",
                    "label": "Dummy prefix separator",
                    "description": "Separator between the column name and category when naming dummy columns.",
                    "type": "text",
                    "default": ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES,
                "drop_first": False,
                "include_missing": False,
                "drop_original": False,
                "prefix_separator": ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                "skipped_columns": [],
            },
        }

        scaling_node: FeatureNodeCatalogEntry = {
            "type": "scale_numeric_features",
            "label": "Scale numeric features",
            "description": "Standardize or normalize numeric columns with recommended scalers.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "scaling", "normalization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to scale",
                    "description": "Select numeric columns manually or leave blank to rely on auto-detection.",
                    "type": "multi_select",
                },
                {
                    "name": "default_method",
                    "label": "Default scaling method",
                    "description": "Fallback scaler applied whenever a column override is not configured.",
                    "type": "text",
                    "default": SCALING_DEFAULT_METHOD,
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include continuous numeric columns that pass quality checks.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "default_method": SCALING_DEFAULT_METHOD,
                "column_methods": {},
                "auto_detect": True,
                "skipped_columns": [],
            },
        }

        skewness_transform_node: FeatureNodeCatalogEntry = {
            "type": "skewness_transform",
            "label": "Fix skewed columns",
            "description": "Surface skewness diagnostics and apply variance-stabilizing transforms.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["skewness", "transformation", "numeric"],
            "parameters": [
                {
                    "name": "transformations",
                    "label": "Column transformations",
                    "description": "Pick recommended transforms per column to reduce skewness.",
                    "type": "text",
                    "source": {
                        "type": "skewness_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/skewness",
                        "value_key": "columns",
                    },
                }
            ],
            "default_config": {
                "transformations": [],
            },
        }

        binned_distribution_node: FeatureNodeCatalogEntry = {
            "type": "binned_distribution",
            "label": "Binned column distributions",
            "description": "Visualize category counts for columns generated by binning.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["binning", "visualization"],
            "parameters": [],
            "default_config": {},
        }

        skewness_distribution_node: FeatureNodeCatalogEntry = {
            "type": "skewness_distribution",
            "label": "Skewness distributions",
            "description": "Visualize distributions for skewed numeric columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["skewness", "visualization"],
            "parameters": [],
            "default_config": {},
        }

        data_preview_node: FeatureNodeCatalogEntry = {
            "type": "data_preview",
            "label": "Data snapshot",
            "description": "Inspect a sample of the dataset after upstream transforms.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["preview", "validation"],
            "parameters": [],
            "default_config": {},
        }

        transformer_audit_node: FeatureNodeCatalogEntry = {
            "type": "transformer_audit",
            "label": "Transformer audit",
            "description": "Review transformer fit/transform activity across train, test, and validation splits.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["monitoring", "transformers", "splits"],
            "parameters": [],
            "default_config": {},
        }

        dataset_profile_node: FeatureNodeCatalogEntry = {
            "type": "dataset_profile",
            "label": "Dataset profile",
            "description": "Generate a lightweight dataset profile with summary statistics.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["profiling", "eda", "quality"],
            "parameters": [],
            "default_config": {},
        }

        return [
            drop_missing_node,
            drop_missing_rows_node,
            missing_indicator_node,
            remove_duplicates_node,
            cast_column_types_node,
            imputation_methods_node,
            trim_whitespace_node,
            normalize_text_case_node,
            replace_aliases_node,
            standardize_dates_node,
            remove_special_chars_node,
            replace_invalid_values_node,
            regex_replace_node,
            binning_node,
            polynomial_features_node,
            feature_selection_node,
            feature_math_node,
            undersampling_node,
            oversampling_node,
            label_encoding_node,
            target_encoding_node,
            hash_encoding_node,
            feature_target_split_node,
            train_test_split_node,
            train_model_draft_node,
            model_evaluation_node,
            model_registry_node,
            hyperparameter_tuning_node,
            ordinal_encoding_node,
            dummy_encoding_node,
            one_hot_encoding_node,
            outlier_removal_node,
            scaling_node,
            skewness_transform_node,
            binned_distribution_node,
            skewness_distribution_node,
            data_preview_node,
            transformer_audit_node,
            dataset_profile_node,
        ]

    return _build_preprocessing_nodes()
