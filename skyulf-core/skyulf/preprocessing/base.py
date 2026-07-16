import functools
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, cast

import pandas as pd

from ..core.protocols import ApplierProtocol, CalculatorProtocol
from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame
from ..utils import get_data_stats, pack_pipeline_output, unpack_pipeline_input
from ._schema import SkyulfSchema

# TypeVar lets the specific NodeArtifact TypedDict flow through fit_method
# so callers see the concrete return type. Bound to Mapping (not Dict) so
# TypedDicts — which are not LSP-substitutable for Dict — are valid returns.


def apply_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that handles unpack/pack boilerplate around an Applier's `apply`.

    The decorated method is written with signature ``(self, X, y, params)``
    instead of ``(self, df, params)``. The wrapper:

    1. Calls ``unpack_pipeline_input(df)`` to get ``(X, y, is_tuple)``.
    2. Invokes the user's method with the unpacked ``X`` and ``y``.
    3. If the method returns a 2-tuple ``(X_out, y_out)``, that pair is
       packed; otherwise the result is treated as ``X_out`` and the
       original ``y`` is reused.
    4. Calls ``pack_pipeline_output`` to restore the original input shape.

    Useful for ~50 Appliers that share the same boilerplate. Skip it for
    splitters (which return ``SplitDataset`` directly) or analyzers that
    don't transform the frame.
    """

    @functools.wraps(fn)
    def wrapper(self: Any, df: Any, params: dict[str, Any]) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        result = fn(self, X, y, params)
        if isinstance(result, tuple) and len(result) == 2:
            X_out, y_out = result
        else:
            X_out, y_out = result, y
        return pack_pipeline_output(X_out, y_out, is_tuple)

    return wrapper


def fit_method[T: Mapping[str, Any]](fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator that handles unpack boilerplate around a Calculator's `fit`.

    The decorated method is written as ``(self, X, y, config)`` and may
    ignore ``y`` for X-only fits. No packing is done — `fit` returns a
    params dict, not a frame.

    The type parameter ``T`` preserves the specific TypedDict return type
    (see ``preprocessing._artifacts``) so callers see the concrete shape.
    """

    @functools.wraps(fn)
    def wrapper(self: Any, df: Any, config: dict[str, Any]) -> T:
        X, y, _ = unpack_pipeline_input(df)
        return fn(self, X, y, config)

    return wrapper  # type: ignore[return-value]


# from ..artifacts.store import ArtifactStore # Removed dependency on ArtifactStore for now


class BaseCalculator(ABC):
    @abstractmethod
    def fit(
        self, df: pd.DataFrame | SkyulfDataFrame | tuple, config: dict[str, Any]
    ) -> Mapping[str, Any]:
        """
        Calculates parameters from the training data.
        Returns a Mapping of fitted parameters (typically a TypedDict
        ``*Artifact`` declared in ``preprocessing._artifacts``). The return
        type is ``Mapping`` rather than ``Dict`` so concrete TypedDict
        subclasses are valid LSP-substitutable returns.
        """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        """Best-effort prediction of the output schema from config alone.

        Override this in concrete Calculators when the output columns/dtypes
        can be derived purely from ``input_schema`` and ``config`` (i.e.
        without seeing data). Examples:

        * Scalers — pass through (output == input).
        * Drop columns by name — drop the configured names.
        * One-hot — adds K columns per categorical (K is data-dependent →
          return ``None``).

        Default returns ``None`` to signal "unknown / data-dependent";
        callers should fall back to runtime introspection.
        """
        return None


class BaseApplier(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame | SkyulfDataFrame | tuple, params: dict[str, Any]) -> Any:
        """
        Applies the transformation using fitted parameters.

        The return type is intentionally `Any` because the concrete shape
        depends on the input: passing a `DataFrame` returns a `DataFrame`;
        passing an `(X, y)` tuple returns a tuple; splitters return
        `SplitDataset`. Encoding every case as a union forces callers to
        defensively narrow on every use, which is worse than `Any` here.
        """


class StatefulTransformer:
    """Fits + applies one pipeline step.

    Accepts anything satisfying :class:`~skyulf.core.protocols.CalculatorProtocol` /
    :class:`~skyulf.core.protocols.ApplierProtocol` (structural typing) — a
    ``BaseCalculator``/``BaseApplier`` subclass, or any duck-typed object
    exposing matching ``fit``/``apply`` methods, works without subclassing.
    """

    def __init__(
        self,
        calculator: CalculatorProtocol,
        applier: ApplierProtocol,
        node_id: str,
        apply_on_test: bool = True,
        apply_on_validation: bool = True,
    ):
        self.calculator = calculator
        self.applier = applier
        self.node_id = node_id
        self.apply_on_test = apply_on_test
        self.apply_on_validation = apply_on_validation
        self.params: dict[str, Any] = {}  # Store params in memory instead of ArtifactStore
        # Profiling metrics
        self.fit_time: float = 0.0
        self.peak_memory_bytes: int = 0
        self.rows_in: int = 0
        self.rows_out: int = 0

    def fit_transform(
        self,
        dataset: SplitDataset | pd.DataFrame | SkyulfDataFrame | tuple,
        config: dict[str, Any],
    ) -> SplitDataset | pd.DataFrame | SkyulfDataFrame | tuple:
        self.rows_in, _ = get_data_stats(dataset)
        tracemalloc.start()
        start = time.time()

        result = self._fit_transform_inner(dataset, config)

        self.fit_time = time.time() - start

        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            self.peak_memory_bytes = peak
            tracemalloc.stop()

        self.rows_out, _ = get_data_stats(result)
        return result

    def _fit_transform_inner(
        self,
        dataset: SplitDataset | pd.DataFrame | SkyulfDataFrame | tuple,
        config: dict[str, Any],
    ) -> SplitDataset | pd.DataFrame | SkyulfDataFrame | tuple:
        # Check for DataFrame-like (Pandas, Polars, Wrapper)
        if (
            hasattr(dataset, "shape")
            and hasattr(dataset, "columns")
            and not isinstance(dataset, tuple)
        ):
            # Fit on the whole dataframe (be careful about leakage!)
            # ty can't narrow a Union through hasattr — cast once for both calls.
            frame = cast(Any, dataset)
            # Calculator.fit returns Mapping (TypedDicts allowed); cast to Dict
            # for storage so Appliers continue to receive a concrete Dict.
            self.params = cast(dict[str, Any], self.calculator.fit(frame, config))
            return self.applier.apply(frame, self.params)

        # If dataset is a tuple (e.g. from FeatureTargetSplitter), pass it through.
        # This allows nodes like TrainTestSplitter to accept (X, y) tuples.
        if isinstance(dataset, tuple):
            self.params = cast(dict[str, Any], self.calculator.fit(dataset, config))
            return self.applier.apply(dataset, self.params)

        # 1. Calculate on Train
        self.params = cast(dict[str, Any], self.calculator.fit(dataset.train, config))

        # 2. Apply to all splits
        return self._apply_to_split_dataset(dataset, self.params)

    def _apply_guarded(self, data: Any, params: dict[str, Any]) -> Any:
        """Apply the applier to `data` and raise if it produces a nested SplitDataset."""
        result = self.applier.apply(data, params)
        if isinstance(result, SplitDataset):
            raise TypeError(
                "Applier returned SplitDataset inside StatefulTransformer, which is not supported."
            )
        return result

    def _apply_to_split_dataset(
        self, dataset: SplitDataset, params: dict[str, Any]
    ) -> SplitDataset:
        """Apply the applier to each split (train/test/validation) of a SplitDataset."""
        new_train = self._apply_guarded(dataset.train, params)

        new_test = dataset.test
        if self.apply_on_test:
            new_test = self._apply_guarded(dataset.test, params)

        new_val = dataset.validation
        if self.apply_on_validation and dataset.validation is not None:
            new_val = self._apply_guarded(dataset.validation, params)

        return SplitDataset(train=new_train, test=new_test, validation=new_val)

    def transform(
        self, dataset: SplitDataset | pd.DataFrame | SkyulfDataFrame | tuple
    ) -> SplitDataset | pd.DataFrame | SkyulfDataFrame | tuple:
        # Use stored params
        params = self.params

        if isinstance(dataset, pd.DataFrame):
            return self.applier.apply(dataset, params)

        if isinstance(dataset, tuple):
            return self.applier.apply(dataset, params)

        # 2. Apply
        # ty can't narrow SplitDataset out of the SkyulfDataFrame branch of this
        # Union via isinstance alone (mirrors the `frame = cast(Any, dataset)`
        # note in `_fit_transform_inner` above).
        return self._apply_to_split_dataset(cast(SplitDataset, dataset), params)
