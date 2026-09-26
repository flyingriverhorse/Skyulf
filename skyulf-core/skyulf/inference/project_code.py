"""Load trusted, self-contained project code under a source-specific module name.

Like the pipeline pickle, this source is executable trusted model content, not
untrusted input or a sandbox. Imported third-party packages must be installed.
"""

import hashlib
import sys
from threading import RLock
from types import ModuleType
from typing import Any

from ..registry import NodeRegistry

MAX_PROJECT_SOURCE_BYTES = 64 * 1024
_PREFIX = "_skyulf_project_"
_LOCK = RLock()


def project_source_digest(source: str) -> str:
    """Validate source size and identify the exact UTF-8 code used for fitting."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Project preprocessing source must be nonempty Python text.")
    payload = source.encode("utf-8")
    if len(payload) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError("Project preprocessing source exceeds 64 KiB.")
    return hashlib.sha256(payload).hexdigest()


def load_project_module(source: str) -> ModuleType:
    """Load trusted source once per digest so different model versions stay isolated."""
    name = _PREFIX + project_source_digest(source)
    with _LOCK:
        existing = sys.modules.get(name)
        if existing is not None:
            if getattr(existing, "__skyulf_source__", None) != source:
                raise ValueError("Project module identity conflicts with its source.")
            return existing
        module = ModuleType(name)
        module.__file__ = f"<{name}>"
        module.__dict__["__skyulf_source__"] = source
        sys.modules[name] = module
        try:
            exec(compile(source, module.__file__, "exec"), module.__dict__)  # nosec B102 - trusted project/model code
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module


def custom_step(
    name: str,
    calculator: type,
    applier: type,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a project's top-level fit/apply classes with a source-specific ID.

    Call inside ``build_preprocessing``. Classes must live in the same source
    file; they must preserve input row order and keep learned state in the
    calculator's returned artifact, not in mutable module globals.
    """
    if (
        not isinstance(calculator, type)
        or not isinstance(applier, type)
        or not calculator.__module__.startswith(_PREFIX)
        or calculator.__module__ != applier.__module__
        or "<locals>" in calculator.__qualname__
        or "<locals>" in applier.__qualname__
    ):
        raise ValueError("Custom classes must be top-level definitions in preprocessing.py.")
    if not callable(getattr(calculator, "fit", None)) or not callable(
        getattr(applier, "apply", None)
    ):
        raise ValueError("Custom preprocessing needs calculator.fit and applier.apply.")
    identity = f"{calculator.__module__}.{calculator.__qualname__}.{applier.__qualname__}"
    if identity not in NodeRegistry.list_transformers():
        NodeRegistry.register(identity, applier)(calculator)
    return {"name": name, "transformer": identity, "params": {} if params is None else params}
