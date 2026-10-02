"""Load trusted project source under a source-specific module or package name.

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

from ..preprocessing.function_steps import FILTER_STEP
from ..registry import NodeRegistry
from .project_package import discard_project_package

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
            discard_project_package(name)
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
    must live together in one saved module, including package submodules.
    Pre-split use requires an explicit
    filter-only declaration; it is an assertion by trusted project code.
    """
    _validate_custom_classes(calculator, applier)
    if pre_split is not None:
        _validate_pre_split(pre_split)
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


def _validate_custom_classes(calculator: type, applier: type) -> None:
    """Require project-owned top-level implementations of the fit/apply pair."""
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


def _validate_pre_split(pre_split: dict[str, Any]) -> None:
    """Require an explicit filter-only declaration with valid required columns."""
    columns = pre_split.get("required_columns") if type(pre_split) is dict else None
    if (
        type(pre_split) is not dict
        or set(pre_split) != {"effect", "required_columns", "learns_from_data"}
        or pre_split["effect"] != "filter"
        or pre_split["learns_from_data"] is not False
        or not _valid_required_columns(columns)
    ):
        raise ValueError(
            "Custom pre_split declaration requires effect='filter', distinct simple "
            "required_columns, and learns_from_data=False; use ordinary custom "
            "preprocessing for value changes."
        )


def _valid_required_columns(columns: Any) -> bool:
    """Recognize distinct simple column identifiers outside the reserved namespace."""
    if type(columns) is not list or not columns:
        return False
    if any(
        type(column) is not str
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column)
        or column.casefold().startswith("__skyulf_")
        for column in columns
    ):
        return False
    return len({column.casefold() for column in columns}) == len(columns)


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


def is_project_filter_step(step: Any) -> bool:
    """Recognize a project filter class pair or a function filter from loaded project source."""
    if type(step) is not dict:
        return False
    if step.get("transformer") != FILTER_STEP:
        return is_registered_project_step(step.get("transformer"))
    module = project_step_source(step)
    return module.startswith(_PREFIX) and sys.modules.get(module) is not None


def project_step_source(step: dict[str, Any]) -> str:
    """Return the project identity owning a custom filter: class identity or function module."""
    if step.get("transformer") == FILTER_STEP:
        ref = step.get("params", {}).get("function")
        return ref.partition(":")[0] if isinstance(ref, str) else ""
    return step.get("transformer", "")
