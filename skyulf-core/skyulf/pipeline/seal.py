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
        if obj is None:
            h.update(b"none:")
            return
        if isinstance(obj, (bool, np.bool_)):
            _feed_bytes(h, b"bool", b"1" if obj else b"0")
            return
        if isinstance(obj, (int, np.integer)):
            _feed_bytes(h, b"int", str(int(obj)).encode())
            return
        if isinstance(obj, (float, np.floating)):
            _feed_bytes(h, b"float", repr(float(obj)).encode())
            return
        if isinstance(obj, complex):
            _feed_bytes(h, b"complex", repr(obj).encode())
            return
        if isinstance(obj, str):
            _feed_bytes(h, b"str", obj.encode())
            return
        if isinstance(obj, (bytes, bytearray, memoryview)):
            _feed_bytes(h, b"bytes", bytes(obj))
            return
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            if arr.dtype == object:
                # tobytes() on object arrays serialises raw PyObject* pointers,
                # which are allocator/ASLR dependent and differ across processes.
                # Digest the elements instead so the digest reflects values.
                h.update(f"ndarray-object:{arr.size}:".encode())
                _feed_canonical(h, obj.shape, active)
                for x in arr.flat:
                    _feed_canonical(h, x, active)
                return
            if arr.dtype.hasobject:
                raise TypeError(
                    f"Cannot digest array with object-containing dtype {arr.dtype}: "
                    "no canonical representation"
                )
            h.update(b"ndarray:")
            _feed_canonical(h, str(arr.dtype), active)
            _feed_canonical(h, obj.shape, active)
            if arr.dtype.names is not None:
                # Padding between record fields is not a learned value and
                # can contain different uninitialized bytes after a copy.
                fields = {name: arr[name] for name in arr.dtype.names}
                _feed_canonical(h, fields, active)
            else:
                _feed_bytes(h, b"bytes", arr.tobytes())
            return
        if isinstance(obj, np.random.RandomState):
            h.update(b"randomstate:")
            _feed_canonical(h, obj.get_state(), active)
            return
        if isinstance(obj, dict):
            h.update(f"dict:{len(obj)}:".encode())
            # Fixed-size child digests provide boundaries and stable ordering even
            # for keys whose repr contains addresses or unordered containers.
            entries = sorted(
                (_canonical_digest(key, active), _canonical_digest(value, active))
                for key, value in obj.items()
            )
            for key_digest, value_digest in entries:
                h.update(key_digest)
                h.update(value_digest)
            return
        if isinstance(obj, tuple):
            h.update(f"tuple:{len(obj)}:".encode())
            for item in obj:
                _feed_canonical(h, item, active)
            return
        if isinstance(obj, list):
            h.update(f"list:{len(obj)}:".encode())
            for item in obj:
                _feed_canonical(h, item, active)
            return
        if isinstance(obj, (set, frozenset)):
            h.update(f"set:{len(obj)}:".encode())
            for digest in sorted(_canonical_digest(item, active) for item in obj):
                h.update(digest)
            return
        if isinstance(obj, type):
            h.update(b"type:")
            _feed_canonical(h, obj.__module__, active)
            _feed_canonical(h, obj.__qualname__, active)
            return
        if dataclasses.is_dataclass(obj):
            h.update(b"dataclass:")
            _feed_canonical(h, type(obj), active)
            state = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
            _feed_canonical(h, state, active)
            return
        # sklearn decision trees are C-extension objects without a __dict__; walk
        # the node arrays that fully determine the tree's structure and predictions.
        if (
            type(obj).__name__ == "Tree"
            and hasattr(obj, "node_count")
            and hasattr(obj, "children_left")
        ):
            h.update(b"tree:")
            _feed_canonical(h, obj.node_count, active)
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
                _feed_canonical(h, np.asarray(getattr(obj, attr)), active)
            return
        if hasattr(obj, "__dict__"):
            h.update(b"obj:")
            _feed_canonical(h, type(obj), active)
            state = {
                name: value
                for name, value in vars(obj).items()
                if not inspect.isroutine(value) and not isinstance(value, ModuleType)
            }
            _feed_canonical(h, state, active)
            return
        raise TypeError(f"Cannot digest object of type {type(obj)!r}: no canonical representation")
    finally:
        active.remove(identity)
