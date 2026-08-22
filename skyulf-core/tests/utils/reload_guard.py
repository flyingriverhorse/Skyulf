"""Guard for ``importlib.reload``-based tests that would pollute the NodeRegistry.

``importlib.reload`` re-executes a module, which re-runs every
``@NodeRegistry.register`` decorator with brand-new class objects. Other test
modules already hold the original classes from their own imports, so their
identity checks (``NodeRegistry.get_calculator(...) is SomeCalculator``) fail
afterwards. This helper captures the currently-registered classes for the
reloaded module and re-registers them once the module is restored.
"""

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from skyulf.registry import NodeRegistry


@contextmanager
def reload_module_preserving_registry(
    module: Any, monkeypatch: Any, blocked_pkg: str
) -> Iterator[Any]:
    """Reload ``module`` with ``blocked_pkg`` unimportable, then restore the
    module *and* the registry's original class objects for every node the
    module registers.

    Capture happens from the registry (not from module attributes), so the
    helper stays correct when several reload tests run back-to-back: the
    registry always holds the canonical classes other test modules see.
    """
    original_pairs = [
        (name, NodeRegistry.get_calculator(name), NodeRegistry.get_applier(name))
        for name, cls in list(NodeRegistry._calculators.items())
        if getattr(cls, "__module__", "") == module.__name__
    ]
    monkeypatch.setitem(sys.modules, blocked_pkg, None)
    try:
        importlib.reload(module)
        yield module
    finally:
        monkeypatch.delitem(sys.modules, blocked_pkg, raising=False)
        importlib.reload(module)
        for name, calc_cls, applier_cls in original_pairs:
            NodeRegistry.register(name, applier_cls)(calc_cls)
            # Pickle resolves a class via getattr(sys.modules[mod], qualname),
            # so leave the module attributes pointing at the reloaded classes
            # and pickling an instance of an original raises "it's not the
            # same object as". Restore both attributes too.
            setattr(module, calc_cls.__name__, calc_cls)
            setattr(module, applier_cls.__name__, applier_cls)
