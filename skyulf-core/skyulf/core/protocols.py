"""Structural (duck-typed) protocols for pipeline steps.

``preprocessing.base.BaseCalculator``/``BaseApplier`` are the ABCs every
built-in node subclasses. ``pipeline.py``/``FeatureEngineer`` and
``StatefulTransformer`` only ever *call* ``calculator.fit(df, config)`` and
``applier.apply(df, params)`` though — they never rely on ABC machinery
(``isinstance`` checks against the ABC, shared ABC state, etc.). These
``Protocol`` classes formalise that structural contract so callers can pass
any object with the right shape (a plain class, a dataclass, a
``functools.partial``-wrapped function object) into the pipeline without
subclassing ``BaseCalculator``/``BaseApplier``.

Both ABC subclasses and plain duck-typed objects satisfy these protocols
automatically — ``@runtime_checkable`` uses structural ``isinstance`` checks
(method presence only, not signatures), so no migration of existing nodes is
required.
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ..engines import SkyulfDataFrame

__all__ = ["CalculatorProtocol", "ApplierProtocol", "PipelineStep"]


@runtime_checkable
class CalculatorProtocol(Protocol):
    """Structural contract for the "fit" half of a pipeline step.

    Mirrors ``preprocessing.base.BaseCalculator.fit``: given the training
    data and a config dict, return a ``Mapping`` of fitted parameters
    (typically a ``TypedDict`` artifact).
    """

    def fit(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple,
        config: dict[str, Any],
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class ApplierProtocol(Protocol):
    """Structural contract for the "transform" half of a pipeline step.

    Mirrors ``preprocessing.base.BaseApplier.apply``: given data and the
    fitted parameters produced by a matching :class:`CalculatorProtocol`,
    return the transformed data (shape depends on the input — DataFrame in,
    DataFrame out; ``(X, y)`` tuple in, tuple out; etc.).
    """

    def apply(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple,
        params: dict[str, Any],
    ) -> Any: ...


@runtime_checkable
class PipelineStep(Protocol):
    """Structural contract for a complete pipeline step (calculator + applier).

    Satisfied by any object exposing both ``.calculator`` and ``.applier``
    attributes that each satisfy :class:`CalculatorProtocol` /
    :class:`ApplierProtocol` — e.g. ``preprocessing.base.StatefulTransformer``,
    or a lightweight custom wrapper that doesn't subclass it.
    """

    calculator: CalculatorProtocol
    applier: ApplierProtocol
