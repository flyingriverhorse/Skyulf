from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import pandas as pd
from skyulf.engines import SkyulfDataFrame


@dataclass
class SplitDataset:
    train: Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]
    test: Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]
    validation: Optional[Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]] = None

    def copy(self) -> "SplitDataset":
        def copy_data(data):
            if isinstance(data, tuple):
                # Handle target copy safely (Series/Array/List)
                y = data[1]
                y_copy = y.copy() if hasattr(y, "copy") else (y.clone() if hasattr(y, "clone") else y)
                return (data[0].copy(), y_copy)
            return data.copy()

        return SplitDataset(
            train=copy_data(self.train),
            test=copy_data(self.test),
            validation=(
                copy_data(self.validation) if self.validation is not None else None
            ),
        )
