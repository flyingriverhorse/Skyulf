"""Preserve original class identities at each row-preserving target encoding step."""

from typing import Any

import numpy as np

from ..data.dataset import SplitDataset


def _target_values(data: Any, target: str) -> Any:
    """Read only training labels, whether embedded or explicitly separated."""
    payload = data.train if isinstance(data, SplitDataset) else data
    return payload[1] if isinstance(payload, tuple) else payload[target]


def _comparable_labels(values: np.ndarray) -> list[Any]:
    """Give repeated floating NaNs one identity without changing other class labels."""
    return [
        np.nan if isinstance(value, float | np.floating) and np.isnan(value) else value
        for value in values.tolist()
    ]


def record_target_labels(
    before: Any, after: Any, target: str | None, kind: str, artifact: dict
) -> None:
    """Record a bijection where an encoder preserves row identity, before later sampling."""
    if target is None or kind not in {"LabelEncoder", "OrdinalEncoder"}:
        return
    original = np.asarray(_target_values(before, target))
    encoded = np.asarray(_target_values(after, target))
    if original.shape != encoded.shape:
        raise ValueError("Target encoding must preserve row identity.")
    original_values = _comparable_labels(original)
    encoded_values = _comparable_labels(encoded)
    if original.dtype.kind == encoded.dtype.kind and original_values == encoded_values:
        return
    pairs = set(zip(encoded_values, original_values, strict=True))
    mapping = dict(pairs)
    if len(mapping) != len(pairs) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Target encoding must preserve a one-to-one class mapping.")
    artifact["target_label_map"] = mapping


def original_labels(pipeline: Any, labels: Any) -> np.ndarray:
    """Decode with the recorded encoder chain, never inferring codes from label order."""
    return original_target_labels(pipeline.feature_engineer.fitted_steps, labels)


def original_target_labels(fitted_steps: list[dict[str, Any]], labels: Any) -> np.ndarray:
    """Decode fitted target encoders without relying on training row count or order."""
    values = np.asarray(labels)
    for step in reversed(fitted_steps):
        mapping = step["artifact"].get("target_label_map")
        if mapping:
            try:
                values = np.asarray([mapping[value] for value in values])
            except KeyError as exc:
                raise ValueError(
                    "Prediction class is absent from the fitted target mapping."
                ) from exc
    return values


def encoded_label(pipeline: Any, label: Any, classes: Any) -> Any:
    """Resolve a user-facing original label on the actual fitted probability axis."""
    raw = original_labels(pipeline, classes).tolist()
    for position, candidate in enumerate(raw):
        if type(candidate) is type(label) and candidate == label:
            return np.asarray(classes).tolist()[position]
    raise ValueError(f"Decision threshold label {label!r} must match an original target class.")


def encoded_labels(pipeline: Any, labels: Any, classes: Any) -> np.ndarray:
    """Translate a batch using one fitted mapping, without repeated pipeline walks."""
    mapping = dict(
        zip(original_labels(pipeline, classes).tolist(), np.asarray(classes).tolist(), strict=True)
    )
    try:
        return np.asarray([mapping[label] for label in labels])
    except KeyError as exc:
        raise ValueError(
            "Evaluation label is absent from the fitted original target classes."
        ) from exc
