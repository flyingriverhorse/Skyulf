"""Preserve original class identities at each row-preserving target encoding step."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.dataset import SplitDataset


def _target_values(data: Any, target: str) -> Any:
    """Read only training labels, whether embedded or explicitly separated."""
    payload = data.train if isinstance(data, SplitDataset) else data
    return payload[1] if isinstance(payload, tuple) else payload[target]


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
    if original.dtype.kind == encoded.dtype.kind and np.array_equal(original, encoded):
        return
    pairs = set(zip(encoded.tolist(), original.tolist(), strict=True))
    mapping = dict(pairs)
    if len(mapping) != len(pairs) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Target encoding must preserve a one-to-one class mapping.")
    artifact["target_label_map"] = mapping


def original_labels(pipeline: Any, labels: Any) -> np.ndarray:
    """Decode with the recorded encoder chain, never inferring codes from label order."""
    values = np.asarray(labels)
    for step in reversed(pipeline.feature_engineer.fitted_steps):
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
    if label not in raw:
        raise ValueError(f"Decision threshold label {label!r} must match an original target class.")
    return np.asarray(classes).tolist()[raw.index(label)]


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
