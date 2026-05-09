import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar, Union, cast

import pandas as pd
import time
import tracemalloc
from ..utils import get_data_stats, pack_pipeline_output, unpack_pipeline_input

from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame
from ._schema import SkyulfSchema

# TypeVar lets the specific NodeArtifact TypedDict flow through fit_method
# so callers see the concrete return type. Bound to Mapping (not Dict) so
# TypedDicts — which are not LSP-substitutable for Dict — are valid returns.
_NodeParams = TypeVar("_NodeParams", bound=Mapping[str, Any])


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
    def wrapper(self: Any, df: Any, params: Dict[str, Any]) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        result = fn(self, X, y, params)
        if isinstance(result, tuple) and len(result) == 2:
            X_out, y_out = result
        else:
            X_out, y_out = result, y
        return pack_pipeline_output(X_out, y_out, is_tuple)

    return wrapper


def fit_method(fn: Callable[..., _NodeParams]) -> Callable[..., _NodeParams]:
    """Decorator that handles unpack boilerplate around a Calculator's `fit`.

    The decorated method is written as ``(self, X, y, config)`` and may
    ignore ``y`` for X-only fits. No packing is done — `fit` returns a
    params dict, not a frame.

    The TypeVar ``_NodeParams`` preserves the specific TypedDict return type
    (see ``preprocessing._artifacts``) so callers see the concrete shape.
    """

    @functools.wraps(fn)
    def wrapper(self: Any, df: Any, config: Dict[str, Any]) -> _NodeParams:
        X, y, _ = unpack_pipeline_input(df)
        return fn(self, X, y, config)

    return wrapper  # type: ignore[return-value]


# from ..artifacts.store import ArtifactStore # Removed dependency on ArtifactStore for now


class BaseCalculator(ABC):
    @abstractmethod
    def fit(
        self, df: Union[pd.DataFrame, SkyulfDataFrame, tuple], config: Dict[str, Any]
    ) -> Mapping[str, Any]:
        """
        Calculates parameters from the training data.
        Returns a Mapping of fitted parameters (typically a TypedDict
        ``*Artifact`` declared in ``preprocessing._artifacts``). The return
        type is ``Mapping`` rather than ``Dict`` so concrete TypedDict
        subclasses are valid LSP-substitutable returns.
        """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> Optional[SkyulfSchema]:
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
    def apply(self, df: Union[pd.DataFrame, SkyulfDataFrame, tuple], params: Dict[str, Any]) -> Any:
        """
        Applies the transformation using fitted parameters.

        The return type is intentionally `Any` because the concrete shape
        depends on the input: passing a `DataFrame` returns a `DataFrame`;
        passing an `(X, y)` tuple returns a tuple; splitters return
        `SplitDataset`. Encoding every case as a union forces callers to
        defensively narrow on every use, which is worse than `Any` here.
        """


class StatefulTransformer:
    def __init__(
        self,
        calculator: BaseCalculator,
        applier: BaseApplier,
        node_id: str,
        apply_on_test: bool = True,
        apply_on_validation: bool = True,
    ):
        self.calculator = calculator
        self.applier = applier
        self.node_id = node_id
        self.apply_on_test = apply_on_test
        self.apply_on_validation = apply_on_validation
        self.params: Dict[str, Any] = {}  # Store params in memory instead of ArtifactStore
        # Profiling metrics
        self.fit_time: float = 0.0
        self.peak_memory_bytes: int = 0
        self.rows_in: int = 0
        self.rows_out: int = 0

    def fit_transform(
        self,
        dataset: Union[SplitDataset, pd.DataFrame, SkyulfDataFrame, tuple],
        config: Dict[str, Any],
    ) -> Union[SplitDataset, pd.DataFrame, SkyulfDataFrame, tuple]:
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
        dataset: Union[SplitDataset, pd.DataFrame, SkyulfDataFrame, tuple],
        config: Dict[str, Any],
    ) -> Union[SplitDataset, pd.DataFrame, SkyulfDataFrame, tuple]:
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
            self.params = cast(Dict[str, Any], self.calculator.fit(frame, config))
            return self.applier.apply(frame, self.params)

        # If dataset is a tuple (e.g. from FeatureTargetSplitter), pass it through.
        # This allows nodes like TrainTestSplitter to accept (X, y) tuples.
        if isinstance(dataset, tuple):
            self.params = cast(Dict[str, Any], self.calculator.fit(dataset, config))
            return self.applier.apply(dataset, self.params)

        # 1. Calculate on Train
        self.params = cast(Dict[str, Any], self.calculator.fit(dataset.train, config))

        # 2. Apply to all splits
        new_train = self.applier.apply(dataset.train, self.params)
        if isinstance(new_train, SplitDataset):
            raise TypeError(
                "Applier returned SplitDataset inside StatefulTransformer, which is not supported."
            )

        new_test = dataset.test
        if self.apply_on_test:
            new_test_res = self.applier.apply(dataset.test, self.params)
            if isinstance(new_test_res, SplitDataset):
                raise TypeError(
                    "Applier returned SplitDataset inside StatefulTransformer, which is not supported."
                )
            new_test = new_test_res

        new_val = dataset.validation
        if self.apply_on_validation and dataset.validation is not None:
            new_val_res = self.applier.apply(dataset.validation, self.params)
            if isinstance(new_val_res, SplitDataset):
                raise TypeError(
                    "Applier returned SplitDataset inside StatefulTransformer, which is not supported."
                )
            new_val = new_val_res

        return SplitDataset(train=new_train, test=new_test, validation=new_val)

    def transform(
        self, dataset: Union[SplitDataset, pd.DataFrame, SkyulfDataFrame, tuple]
    ) -> Union[SplitDataset, pd.DataFrame, SkyulfDataFrame, tuple]:
        # Use stored params
        params = self.params

        if isinstance(dataset, pd.DataFrame):
            return self.applier.apply(dataset, params)

        if isinstance(dataset, tuple):
            return self.applier.apply(dataset, params)

        # 2. Apply
        new_train = self.applier.apply(dataset.train, params)
        if isinstance(new_train, SplitDataset):
            raise TypeError(
                "Applier returned SplitDataset inside StatefulTransformer, which is not supported."
            )

        new_test = dataset.test
        if self.apply_on_test:
            new_test_res = self.applier.apply(dataset.test, params)
            if isinstance(new_test_res, SplitDataset):
                raise TypeError(
                    "Applier returned SplitDataset inside StatefulTransformer, which is not supported."
                )
            new_test = new_test_res

        new_val = dataset.validation
        if self.apply_on_validation and dataset.validation is not None:
            new_val_res = self.applier.apply(dataset.validation, params)
            if isinstance(new_val_res, SplitDataset):
                raise TypeError(
                    "Applier returned SplitDataset inside StatefulTransformer, which is not supported."
                )
            new_val = new_val_res

        return SplitDataset(train=new_train, test=new_test, validation=new_val)
