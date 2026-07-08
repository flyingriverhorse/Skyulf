from typing import Any, TypedDict


class PreprocessingStepConfig(TypedDict, total=False):
    """Configuration for a single preprocessing step."""

    name: str
    transformer: str
    params: dict[str, Any]


class ModelConfig(TypedDict, total=False):
    """Configuration for the modeling step."""

    type: str  # e.g. "random_forest_classifier"
    node_id: str
    params: dict[str, Any]
    base_model: dict[str, Any]  # nested ModelConfig (recursive type support varies)


class PipelineConfig(TypedDict, total=False):
    """Configuration for the full pipeline."""

    preprocessing: list[PreprocessingStepConfig]
    modeling: ModelConfig


class NodeMetadataDict(TypedDict, total=False):
    """Dictionary representation of node metadata for the registry."""

    id: str
    name: str
    category: str
    description: str
    params: dict[str, Any]
    tags: list[str]
