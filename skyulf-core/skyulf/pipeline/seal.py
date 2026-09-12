"""Reproducibility seal: semantic digests of fitted pipeline artifacts.

Extracted from ``pipeline.py`` (F-19). The digest walks an artifact's
meaningful content — hyperparameters, fitted arrays, tree structures —
instead of its pickle bytes, so :meth:`SkyulfPipeline.fingerprint` stays
stable across library and pickle-protocol versions while still changing
whenever the learned model changes.
"""

import dataclasses
import hashlib
import inspect
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import numpy as np


def artifact_digest(obj: Any) -> bytes:
    """Stable semantic digest of a fitted artifact.

    Walks the object's meaningful content — hyperparameters, fitted weights,
    tree node arrays — instead of its pickle bytes, so the digest is stable
    across library/pickle-protocol versions and still changes whenever the
    learned model changes. Raises ``TypeError`` for anything it cannot
    canonicalize: an artifact that cannot be digested must fail the seal,
    not silently pass it via ``repr``. Cyclic references are unsupported;
    shared references in an acyclic graph are digested by value. Structured
    NumPy dtypes containing objects are unsupported because their raw bytes
    contain process-dependent pointers. Nesting beyond Python's recursion
    capacity also raises ``TypeError``, with an excessive-depth explanation.

    The length-framed encoding replaces the original ambiguous encoding;
    previously stored artifact digests and fitted fingerprints must be
    recomputed when upgrading to this format.
    """
    try:
        return _canonical_digest(obj, set())
    except RecursionError as exc:
        raise TypeError(
            "Cannot digest artifact: nesting exceeds supported recursion depth"
        ) from exc


def _canonical_digest(obj: Any, active: set[int]) -> bytes:
    """Digest a value while retaining the caller's active recursion path."""
    hasher = hashlib.sha256()
    _feed_canonical(hasher, obj, active)
    return hasher.digest()


def _feed_bytes(h: Any, tag: bytes, value: bytes) -> None:
    """Frame a scalar payload so embedded tags cannot change its boundaries."""
    h.update(tag + b":" + str(len(value)).encode() + b":")
    h.update(value)


def _feed_canonical(h: Any, obj: Any, active: set[int]) -> None:
    """Feed framed values, rejecting only references on the active recursion path."""
    identity = id(obj)
    if identity in active:
        raise TypeError(f"Cannot digest cyclic reference to object of type {type(obj)!r}")
    active.add(identity)
    try:
        if isinstance(obj, dict):
            h.update(f"dict:{len(obj)}:".encode())
            # Fixed-size child digests provide boundaries and stable ordering.
            entries = sorted(
                (_canonical_digest(key, active), _canonical_digest(value, active))
                for key, value in obj.items()
            )
            for key_digest, value_digest in entries:
                h.update(key_digest)
                h.update(value_digest)
        elif isinstance(obj, (set, frozenset)):
            h.update(f"set:{len(obj)}:".encode())
            for digest in sorted(_canonical_digest(item, active) for item in obj):
                h.update(digest)
        else:
            # Encoders yield before recursion and resume framing afterward.
            # Their suspended frames do not reduce supported container depth.
            for child in _canonical_children(h, obj):
                _feed_canonical(h, child, active)
    finally:
        active.remove(identity)


def _feed_scalar(h: Any, obj: Any) -> bool:
    """Encode supported scalar values, reporting whether the value was handled."""
    if obj is None:
        h.update(b"none:")
    elif isinstance(obj, (bool, np.bool_)):
        _feed_bytes(h, b"bool", b"1" if obj else b"0")
    elif isinstance(obj, (int, np.integer)):
        _feed_bytes(h, b"int", str(int(obj)).encode())
    elif isinstance(obj, (float, np.floating)):
        _feed_bytes(h, b"float", repr(float(obj)).encode())
    elif isinstance(obj, complex):
        _feed_bytes(h, b"complex", repr(obj).encode())
    elif isinstance(obj, str):
        _feed_bytes(h, b"str", obj.encode())
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        _feed_bytes(h, b"bytes", bytes(obj))
    else:
        return False
    return True


def _array_children(h: Any, obj: np.ndarray) -> Iterator[Any]:
    """Frame arrays around child values without hashing pointers or record padding."""
    arr = np.ascontiguousarray(obj)
    if arr.dtype == object:
        # Raw object-array bytes contain allocator-dependent PyObject* pointers.
        h.update(f"ndarray-object:{arr.size}:".encode())
        yield obj.shape
        yield from arr.flat
        return
    if arr.dtype.hasobject:
        raise TypeError(
            f"Cannot digest array with object-containing dtype {arr.dtype}: "
            "no canonical representation"
        )
    h.update(b"ndarray:")
    yield str(arr.dtype)
    yield obj.shape
    if arr.dtype.names is not None:
        # Padding between record fields is not a learned value.
        yield {name: arr[name] for name in arr.dtype.names}
    else:
        _feed_bytes(h, b"bytes", arr.tobytes())


def _canonical_children(h: Any, obj: Any) -> Iterator[Any]:
    """Write framing and yield ordered children for recursion in the caller's frame."""
    if _feed_scalar(h, obj):
        return
    if isinstance(obj, np.ndarray):
        yield from _array_children(h, obj)
    elif isinstance(obj, np.random.RandomState):
        h.update(b"randomstate:")
        yield obj.get_state()
    elif isinstance(obj, (tuple, list)):
        tag = "tuple" if isinstance(obj, tuple) else "list"
        h.update(f"{tag}:{len(obj)}:".encode())
        yield from obj
    else:
        yield from _object_children(h, obj)


def _tree_children(h: Any, obj: Any) -> Iterator[Any]:
    """Yield the node arrays that determine a sklearn tree's predictions."""
    h.update(b"tree:")
    yield obj.node_count
    for attr in (
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "impurity",
        "n_node_samples",
        "weighted_n_node_samples",
        "value",
    ):
        yield np.asarray(getattr(obj, attr))


def _object_children(h: Any, obj: Any) -> Iterator[Any]:
    """Yield supported object state and reject values with no canonical representation."""
    if isinstance(obj, type):
        h.update(b"type:")
        yield obj.__module__
        yield obj.__qualname__
        return
    if dataclasses.is_dataclass(obj):
        h.update(b"dataclass:")
        yield type(obj)
        yield {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
        return
    # sklearn trees are C-extension objects without a __dict__.
    if (
        type(obj).__name__ == "Tree"
        and hasattr(obj, "node_count")
        and hasattr(obj, "children_left")
    ):
        yield from _tree_children(h, obj)
        return
    if hasattr(obj, "__dict__"):
        h.update(b"obj:")
        yield type(obj)
        yield _object_state(obj)
        return
    raise TypeError(f"Cannot digest object of type {type(obj)!r}: no canonical representation")


def _object_state(obj: Any) -> dict[str, Any]:
    """Exclude routines and imported modules from an object's meaningful fitted state."""
    return {
        name: value
        for name, value in vars(obj).items()
        if not inspect.isroutine(value) and not isinstance(value, ModuleType)
    }
