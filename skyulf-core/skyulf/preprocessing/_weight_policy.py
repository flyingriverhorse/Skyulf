"""Transport training weights through audited positional row transformations.

Custom appliers may implement ``apply_with_row_mapping(data, artifact)`` returning
``(output, positions)``. Positions explicitly identify source observations, including
repeats. The orchestrator selects the original target and weights with that mapping;
this hook does not authorize target re-encoding or synthetic observations. Custom
training representations require their own audited implementation and are rejected.
"""

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np

from ..data.dataset import SplitDataset
from ..modeling._sample_weights import SampleWeightError, validate_sample_weight
from ..registry import NodeRegistry
from ..utils import pack_pipeline_output, unpack_pipeline_input
from ._helpers import select_rows_by_position

# Resolve the actual exported classes, not metadata or user-declared safety flags.
# Row selectors use the exact same single application that produces their output.
_ROW_MAPPING_MODULES = {
    "DropMissingRows": "drop_and_missing.drop_rows",
    "Deduplicate": "drop_and_missing.deduplicate",
    "IQR": "outliers.iqr",
    "ZScore": "outliers.zscore",
    "Winsorize": "outliers.winsorize",
    "EllipticEnvelope": "outliers.elliptic",
    "ManualBounds": "outliers.manual_bounds",
    "LagFeatures": "time_series.lag",
    "RollingAggregate": "time_series.rolling",
    "RowFilterFunction": "function_steps",
}

_SAFE_MODULES = {
    **_ROW_MAPPING_MODULES,
    "CustomBinning": "bucketing",
    "GeneralBinning": "bucketing",
    "KBinsDiscretizer": "bucketing",
    "DataSnapshot": "inspection",
    "DatasetProfile": "inspection",
    "FeatureGeneration": "feature_generation.generation",
    "GroupImputer": "imputation.group",
    "GeoDistance": "geo.distance",
    "H3Index": "geo.h3_index",
    "count_vectorizer": "vectorization.count_vectorizer",
    "tfidf_vectorizer": "vectorization.tfidf_vectorizer",
    "hashing_vectorizer": "vectorization.hashing_vectorizer",
    "sentence_embedder": "vectorization.sentence_embedder",
    "tokenizer": "vectorization.tokenizer",
    "feature_selection": "feature_selection.facade",
    "Oversampling": "resampling",
    "Undersampling": "resampling",
    "ColumnFunction": "function_steps",
    "FittedFunction": "function_steps",
    "StandardScaler": "scaling.standard",
    "MinMaxScaler": "scaling.minmax",
    "MaxAbsScaler": "scaling.maxabs",
    "RobustScaler": "scaling.robust",
    "SimpleImputer": "imputation.simple",
    "KNNImputer": "imputation.knn",
    "IterativeImputer": "imputation.iterative",
    "OneHotEncoder": "encoding.one_hot",
    "OrdinalEncoder": "encoding.ordinal",
    "LabelEncoder": "encoding.label",
    "DummyEncoder": "encoding.dummy",
    "HashEncoder": "encoding.hash",
    "TargetEncoder": "encoding.target",
    "WOEEncoder": "encoding.woe",
    "DropMissingColumns": "drop_and_missing.drop_columns",
    "MissingIndicator": "drop_and_missing.missing_indicator",
    "VarianceThreshold": "feature_selection.variance",
    "UnivariateSelection": "feature_selection.univariate",
    "ModelBasedSelection": "feature_selection.model_based",
    "CorrelationThreshold": "feature_selection.correlation",
    "ClipValues": "outliers.clip_values",
    "DateFeatures": "time_series.date_features",
    "Casting": "casting",
    "SimpleTransformation": "transformations.simple",
    "GeneralTransformation": "transformations.general",
    "PowerTransformer": "transformations.power",
    "PolynomialFeatures": "feature_generation.polynomial",
    "FeatureInteraction": "feature_generation.interaction",
    "TextCleaning": "cleaning.text",
    "AliasReplacement": "cleaning.alias",
    "ValueReplacement": "cleaning.value_replacement",
    "InvalidValueReplacement": "cleaning.invalid_value",
    "feature_target_split": "split",
}


_CLASS_STEMS = {
    "feature_target_split": "FeatureTargetSplit",
    "count_vectorizer": "CountVectorizer",
    "tfidf_vectorizer": "TfidfVectorizer",
    "hashing_vectorizer": "HashingVectorizer",
    "sentence_embedder": "SentenceEmbedder",
    "tokenizer": "Tokenizer",
    "feature_selection": "FeatureSelection",
}


def _trusted_components(name: str, *, allow_split: bool) -> tuple[type, type] | None:
    """Return exact built-in classes for an explicitly supported node name."""
    if allow_split and name in {"TrainTestSplitter", "Split"}:
        module_name, stem = "split", "Split"
    else:
        module_name = _SAFE_MODULES.get(name)
        stem = _CLASS_STEMS.get(name, name)
    if module_name is None:
        return _trusted_row_preserving_alias(name)
    module = import_module(f"skyulf.preprocessing.{module_name}")
    return getattr(module, f"{stem}Calculator"), getattr(module, f"{stem}Applier")


def validate_weighted_steps(
    steps: Sequence[Any], *, allow_split: bool = False, allow_custom_mapping: bool = True
) -> None:
    """Reject custom replacements and any step without a proven positional contract."""
    for step in steps:
        name = step["transformer"]
        expected = _trusted_components(name, allow_split=allow_split)
        if allow_custom_mapping and callable(
            getattr(NodeRegistry.get_applier(name), "apply_with_row_mapping", None)
        ):
            continue
        if expected is None or expected != (
            NodeRegistry.get_calculator(name),
            NodeRegistry.get_applier(name),
        ):
            raise SampleWeightError(
                f"Step '{name}' is unsupported for weighted preprocessing; "
                "sample_weight requires an explicit row-mapping contract."
            )


def validate_weighted_preprocessing(preprocessor: Any) -> None:
    """Validate exact adapter types recursively, never trusting subclass declarations."""
    if preprocessor is None:
        return
    # These modules import this policy during initialization; resolve after import.
    adapters = import_module("skyulf.preprocessing.fold_adapter")
    pipeline = import_module("skyulf.pipeline._pipeline")
    if type(preprocessor) is adapters.AuditedFoldPreprocessor:
        validate_weighted_preprocessing(preprocessor.inner)
    elif type(preprocessor) in (
        adapters.FeatureEngineerFoldAdapter,
        pipeline._PipelineTuningPreprocessor,
    ):
        validate_weighted_steps(preprocessor._steps_config)
    elif type(preprocessor) is adapters.MergedBranchFoldAdapter:
        for steps in preprocessor._branch_step_lists:
            validate_weighted_steps(steps, allow_custom_mapping=False)
    else:
        raise SampleWeightError("Custom preprocessing is unsupported for sample_weight.")


def prepare_pipeline_weights(data: Any, sample_weight: Any, steps: Sequence[Any]) -> Any:
    """Validate raw or stored weights before any preprocessing learns from data."""
    if isinstance(data, SplitDataset):
        if sample_weight is not None:
            raise SampleWeightError("SplitDataset uses train_sample_weight; omit sample_weight.")
        sample_weight = data.train_sample_weight
        data = data.train
    if sample_weight is None:
        return None
    features = data[0] if isinstance(data, tuple) else data
    weights = validate_sample_weight(sample_weight, len(features))
    validate_weighted_steps(steps, allow_split=True)
    return weights


def validate_weighted_transformer(dataset: Any, calculator: Any, applier: Any) -> None:
    """Apply the same exact-class policy to standalone stateful transformers."""
    if not isinstance(dataset, SplitDataset) or dataset.train_sample_weight is None:
        return
    prepare_pipeline_weights(dataset, None, [])
    validate_weighted_components(calculator, applier)


def validate_weighted_components(calculator: Any, applier: Any) -> None:
    """Require an audited builtin pair or an explicit custom positional hook."""
    actual = type(calculator), type(applier)
    if callable(getattr(applier, "apply_with_row_mapping", None)):
        return
    if not any(actual == _trusted_components(name, allow_split=False) for name in _SAFE_MODULES):
        raise SampleWeightError("Custom transformer is unsupported for weighted preprocessing.")


def apply_weighted(applier: Any, data: Any, params: Any, weights: Any) -> tuple[Any, Any]:
    """Apply once with explicit positions, never interpreting frame index labels."""
    X, y, paired = unpack_pipeline_input(data)
    hook = getattr(applier, "apply_with_row_mapping", None)
    if callable(hook):
        result, positions = hook(data, params)
        output_X = unpack_pipeline_input(result)[0]
        positions = validate_row_mapping(positions, len(X), len(output_X))
        result = pack_pipeline_output(output_X, select_rows_by_position(y, positions), paired)
    elif has_builtin_row_mapping(applier):
        output_X, positions = applier.apply((X, np.arange(len(X))), params)
        positions = validate_row_mapping(positions, len(X), len(output_X))
        result = pack_pipeline_output(output_X, select_rows_by_position(y, positions), paired)
    else:
        if not _matches_applier(applier, _SAFE_MODULES) or is_sampling_applier(applier):
            raise SampleWeightError("Applier requires an explicit weighted row-mapping contract.")
        result = applier.apply(data, params)
        output_X = unpack_pipeline_input(result)[0]
        positions = np.arange(len(X))
    positions = validate_row_mapping(positions, len(X), len(output_X))
    return result, validate_sample_weight(weights[positions], len(output_X))


def validate_row_mapping(positions: Any, input_rows: int, output_rows: int) -> Any:
    """Reject malformed provenance instead of guessing from lengths or labels."""
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.dtype.kind not in "iu" or len(positions) != output_rows:
        raise SampleWeightError("Row mapping must contain one integer position per output row.")
    if np.any(positions < 0) or np.any(positions >= input_rows):
        raise SampleWeightError("Row mapping contains an out-of-range input position.")
    return positions


def has_builtin_row_mapping(applier: Any) -> bool:
    """Identify audited appliers whose target is only sliced with their features."""
    return _matches_applier(applier, _ROW_MAPPING_MODULES)


def validate_training_representation(calculator: Any, applier: Any) -> None:
    """Reject custom train hooks that bypass the applier's explicit row mapping."""
    actual = type(calculator), type(applier)
    if not any(actual == _trusted_components(name, allow_split=False) for name in _SAFE_MODULES):
        raise SampleWeightError("Custom fit_transform_train is unsupported for sample_weight.")


def is_sampling_applier(applier: Any) -> bool:
    """Recognize exact registered samplers for their training-only transform policy."""
    return _matches_applier(applier, ("Oversampling", "Undersampling"))


def _matches_applier(applier: Any, names: Any) -> bool:
    """Compare only resolved builtin classes, keeping optional lookup explicit."""
    for name in names:
        components = _trusted_components(name, allow_split=False)
        if components is not None and type(applier) is components[1]:
            return True
    return False


def _trusted_row_preserving_alias(name: str) -> tuple[type, type] | None:
    """Admit aliases only by exact identity with audited row-preserving pairs.

    Row-changing and structural nodes have additional name-dependent execution
    rules, so their aliases need explicit pipeline registration instead.
    """
    actual = NodeRegistry.get_calculator(name), NodeRegistry.get_applier(name)
    structural = {*_ROW_MAPPING_MODULES, "Oversampling", "Undersampling", "feature_target_split"}
    for canonical in _SAFE_MODULES.keys() - structural:
        if actual == _trusted_components(canonical, allow_split=False):
            return actual
    return None
