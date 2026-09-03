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
    not silently pass it via ``repr``.
    """
    hasher = hashlib.sha256()
    _feed_canonical(hasher, obj)
    return hasher.digest()


def _feed_canonical(h: Any, obj: Any) -> None:
    """Recursively feed a type-tagged canonical form of ``obj`` into ``h``."""
    if obj is None:
        h.update(b"none")
        return
    if isinstance(obj, (bool, np.bool_)):
        h.update(b"bool:" + (b"1" if obj else b"0"))
        return
    if isinstance(obj, (int, np.integer)):
        h.update(b"int:" + str(int(obj)).encode())
        return
    if isinstance(obj, (float, np.floating)):
        h.update(b"float:" + repr(float(obj)).encode())
        return
    if isinstance(obj, complex):
        h.update(b"complex:" + repr(obj).encode())
        return
    if isinstance(obj, str):
        h.update(b"str:" + obj.encode())
        return
    if isinstance(obj, (bytes, bytearray, memoryview)):
        h.update(b"bytes:" + bytes(obj))
        return
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        if arr.dtype == object:
            # tobytes() on object arrays serialises raw PyObject* pointers,
            # which are allocator/ASLR dependent and differ across processes.
            # Digest the elements instead so the digest reflects values.
            h.update(f"ndarray-object:{arr.shape}|".encode())
            for x in arr.flat:
                _feed_canonical(h, x)
            return
        h.update(f"ndarray:{arr.dtype}|{arr.shape}|".encode())
        h.update(arr.tobytes())
        return
    if isinstance(obj, np.random.RandomState):
        h.update(b"randomstate:")
        _feed_canonical(h, obj.get_state())
        return
    if isinstance(obj, dict):
        h.update(f"dict:{len(obj)}[".encode())
        for key in sorted(obj, key=repr):
            _feed_canonical(h, key)
            h.update(b"=")
            _feed_canonical(h, obj[key])
            h.update(b";")
        h.update(b"]")
        return
    if isinstance(obj, tuple):
        h.update(f"tuple:{len(obj)}(".encode())
        for item in obj:
            _feed_canonical(h, item)
            h.update(b",")
        h.update(b")")
        return
    if isinstance(obj, list):
        h.update(f"list:{len(obj)}[".encode())
        for item in obj:
            _feed_canonical(h, item)
            h.update(b",")
        h.update(b"]")
        return
    if isinstance(obj, (set, frozenset)):
        h.update(f"set:{len(obj)}{{".encode())
        for item in sorted(obj, key=repr):
            _feed_canonical(h, item)
            h.update(b",")
        h.update(b"}")
        return
    if isinstance(obj, type):
        h.update(b"type:" + obj.__module__.encode() + b"." + obj.__qualname__.encode())
        return
    if dataclasses.is_dataclass(obj):
        h.update(b"dataclass:" + type(obj).__qualname__.encode() + b"{")
        for field in dataclasses.fields(obj):
            h.update(field.name.encode() + b"=")
            _feed_canonical(h, getattr(obj, field.name))
            h.update(b";")
        h.update(b"}")
        return
    # sklearn decision trees are C-extension objects without a __dict__; walk
    # the node arrays that fully determine the tree's structure and predictions.
    if (
        type(obj).__name__ == "Tree"
        and hasattr(obj, "node_count")
        and hasattr(obj, "children_left")
    ):
        h.update(f"tree:{obj.node_count}|".encode())
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
            _feed_canonical(h, np.asarray(getattr(obj, attr)))
        return
    if hasattr(obj, "__dict__"):
        cls = type(obj)
        h.update(b"obj:" + cls.__module__.encode() + b"." + cls.__qualname__.encode() + b"{")
        state = vars(obj)
        for name in sorted(state):
            value = state[name]
            if inspect.isroutine(value) or isinstance(value, ModuleType):
                continue
            h.update(name.encode() + b"=")
            _feed_canonical(h, value)
            h.update(b";")
        h.update(b"}")
        return
    raise TypeError(f"Cannot digest object of type {type(obj)!r}: no canonical representation")
