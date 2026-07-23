"""Static diagnostics for preprocessing configurations that risk data leakage."""

from typing import Any

from .types import PipelineConfig

_DATA_DEPENDENT_TRANSFORMERS = frozenset(
    {
        "SimpleImputer",
        "KNNImputer",
        "IterativeImputer",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "MaxAbsScaler",
        "OneHotEncoder",
        "LabelEncoder",
        "OrdinalEncoder",
        "DummyEncoder",
        "TargetEncoder",
        "WOEEncoder",
        "IQR",
        "ZScore",
        "Winsorize",
        "EllipticEnvelope",
        "VarianceThreshold",
        "CorrelationThreshold",
        "UnivariateSelection",
        "ModelBasedSelection",
        "feature_selection",
        "GeneralBinning",
        "EqualWidthBinning",
        "EqualFrequencyBinning",
        "KBinsDiscretizer",
        "PowerTransformer",
        "count_vectorizer",
        "tfidf_vectorizer",
    }
)
_TRAIN_TEST_SPLITTERS = frozenset({"TrainTestSplitter", "Split"})


def validate_leakage_safety(pipeline_config: PipelineConfig | dict[str, Any]) -> list[str]:
    """Return warnings for learned preprocessing configured before a train/test split."""
    preprocessing = pipeline_config.get("preprocessing", [])
    splitter = next(
        (
            (index, step.get("transformer"))
            for index, step in enumerate(preprocessing)
            if step.get("transformer") in _TRAIN_TEST_SPLITTERS
        ),
        None,
    )
    if splitter is None:
        return []

    splitter_index, splitter_name = splitter
    return [
        (
            f"Step {index} ('{transformer}') is configured before the train/test split "
            f"(step {splitter_index}, '{splitter_name}') and will fit its statistics on "
            "the full dataset including the test set — move it after the splitter."
        )
        for index, step in enumerate(preprocessing[:splitter_index])
        if (transformer := step.get("transformer")) in _DATA_DEPENDENT_TRANSFORMERS
    ]
