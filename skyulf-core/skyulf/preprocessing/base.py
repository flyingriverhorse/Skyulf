from abc import ABC, abstractmethod
from typing import Any, Dict, Union, cast

import pandas as pd
import time
import tracemalloc
from ..utils import get_data_stats

from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame

# from ..artifacts.store import ArtifactStore # Removed dependency on ArtifactStore for now


class BaseCalculator(ABC):
    @abstractmethod
    def fit(
        self, df: Union[pd.DataFrame, SkyulfDataFrame, tuple], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates parameters from the training data.
        Returns a dictionary of fitted parameters (serializable).
        """


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
            self.params = self.calculator.fit(frame, config)
            return self.applier.apply(frame, self.params)

        # If dataset is a tuple (e.g. from FeatureTargetSplitter), pass it through.
        # This allows nodes like TrainTestSplitter to accept (X, y) tuples.
        if isinstance(dataset, tuple):
            self.params = self.calculator.fit(dataset, config)
            return self.applier.apply(dataset, self.params)

        # 1. Calculate on Train
        self.params = self.calculator.fit(dataset.train, config)

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
