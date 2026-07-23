"""Permissive structural validation for public pipeline configuration dictionaries."""

from collections.abc import Mapping, Sequence
from difflib import get_close_matches
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _PreprocessingStepModel(BaseModel):
    """Validate only the universal structural fields of a preprocessing step."""

    model_config = ConfigDict(extra="allow", strict=True)

    name: str
    transformer: str
    params: Any = Field(default_factory=dict)


class _ModelConfigModel(BaseModel):
    """Validate only the universal structural fields of a modeling config."""

    model_config = ConfigDict(extra="allow", strict=True)

    type: str | None = None
    params: Any = Field(default_factory=dict)


class _PipelineConfigModel(BaseModel):
    """Validate the outer shape while preserving unrecognized config keys."""

    model_config = ConfigDict(extra="allow", strict=True)

    preprocessing: Sequence[_PreprocessingStepModel] = Field(default_factory=list)
    modeling: _ModelConfigModel | None = None


class _PreprocessingConfigModel(BaseModel):
    """Validate a standalone FeatureEngineer step sequence."""

    model_config = ConfigDict(extra="allow", strict=True)

    preprocessing: Sequence[_PreprocessingStepModel]


def _format_location(location: tuple[str | int, ...]) -> str:
    """Render a Pydantic error location using pipeline-config notation."""
    if not location:
        return "config"

    result = str(location[0])
    for item in location[1:]:
        result += f"[{item}]" if isinstance(item, int) else f".{item}"
    return result


def _missing_key_suggestion(error: Mapping[str, Any]) -> str:
    """Return a typo hint when an input mapping has a close key match."""
    missing_key = str(error["loc"][-1])
    input_value = error.get("input")
    if not isinstance(input_value, Mapping):
        return ""

    matches = get_close_matches(missing_key, input_value.keys(), n=1)
    return f" (did you mean '{matches[0]}'?)" if matches else ""


def _format_pydantic_error(error: Mapping[str, Any]) -> str:
    """Turn a Pydantic field error into a concise public config diagnostic."""
    location = _format_location(tuple(error["loc"]))
    error_type = str(error["type"])

    if error_type == "missing":
        key = str(error["loc"][-1])
        return f"{location.rsplit('.', 1)[0]}: missing required key '{key}'{_missing_key_suggestion(error)}"
    if error_type == "string_type":
        return f"{location}: must be a string"
    if error_type in {"list_type", "sequence_str"}:
        return f"{location}: must be a list or sequence of preprocessing steps"
    if error_type in {"model_type", "dict_type"}:
        return f"{location}: must be a dictionary"
    return f"{location}: {error['msg']}"


def _raise_validation_error(errors: list[str]) -> None:
    """Raise one aggregated ValueError for the supplied config diagnostics."""
    if errors:
        count = len(errors)
        raise ValueError(
            f"Invalid pipeline config ({count} problem{'s' if count != 1 else ''} found):\n"
            + "\n".join(f"  - {error}" for error in errors)
        )


def _validate_model(model: type[BaseModel], value: Any) -> BaseModel | None:
    """Validate a Pydantic model and raise its aggregated structural errors."""
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        _raise_validation_error([_format_pydantic_error(error) for error in exc.errors()])
    return None


def validate_preprocessing_steps(steps_config: Sequence[Mapping[str, Any]]) -> None:
    """Validate standalone FeatureEngineer steps before they can raise KeyError."""
    _validate_model(_PreprocessingConfigModel, {"preprocessing": steps_config})


def validate_pipeline_config(config: Mapping[str, Any]) -> None:
    """Validate a pipeline's outer shape without inspecting node-specific params."""
    _validate_model(_PipelineConfigModel, config)
