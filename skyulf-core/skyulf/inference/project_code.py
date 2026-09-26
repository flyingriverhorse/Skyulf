"""Load trusted, self-contained project code under a source-specific module name.

Like the pipeline pickle, this source is executable trusted model content, not
untrusted input or a sandbox. Imported third-party packages must be installed.
"""

import hashlib
import re
import sys
from copy import deepcopy
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
    *,
    pre_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a project's top-level fit/apply classes with a source-specific ID.

    Call inside ``build_preprocessing`` or ``build_pre_split_steps``. Classes
    must live in the same source file. Pre-split use requires an explicit
    filter-only declaration; it is an assertion by trusted project code.
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
    if pre_split is not None:
        columns = pre_split.get("required_columns") if type(pre_split) is dict else None
        if (
            type(pre_split) is not dict
            or set(pre_split) != {"effect", "required_columns", "learns_from_data"}
            or pre_split["effect"] != "filter"
            or pre_split["learns_from_data"] is not False
            or type(columns) is not list
            or not columns
            or any(
                type(column) is not str
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column)
                or column.casefold().startswith("__skyulf_")
                for column in columns
            )
            or len({column.casefold() for column in columns}) != len(columns)
        ):
            raise ValueError(
                "Custom pre_split declaration requires effect='filter', distinct simple "
                "required_columns, and learns_from_data=False; use ordinary custom "
                "preprocessing for value changes."
            )
    identity = f"{calculator.__module__}.{calculator.__qualname__}.{applier.__qualname__}"
    try:
        registered = NodeRegistry.get_calculator(identity)
    except ValueError:
        NodeRegistry.register(identity, applier)(calculator)
    else:
        if registered is not calculator or NodeRegistry.get_applier(identity) is not applier:
            raise ValueError("Custom step identity conflicts with an existing registration.")
    step = {"name": name, "transformer": identity, "params": {} if params is None else params}
    if pre_split is not None:
        step["pre_split"] = deepcopy(pre_split)
    return step


def is_registered_project_step(identity: str) -> bool:
    """Recognize only an exact class pair registered from isolated project source."""
    if not isinstance(identity, str) or not identity.startswith(_PREFIX):
        return False
    try:
        calculator = NodeRegistry.get_calculator(identity)
        applier = NodeRegistry.get_applier(identity)
    except ValueError:
        return False
    module = calculator.__module__
    return (
        module == applier.__module__
        and module.startswith(_PREFIX)
        and sys.modules.get(module) is not None
        and identity == f"{module}.{calculator.__qualname__}.{applier.__qualname__}"
    )
