from dataclasses import dataclass
from typing import Optional, Union, Tuple
import pandas as pd

@dataclass
class SplitDataset:
    train: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]
    test: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]
    validation: Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]] = None
    
    def copy(self) -> 'SplitDataset':
        def copy_data(data):
            if isinstance(data, tuple):
                return (data[0].copy(), data[1].copy())
            return data.copy()

        return SplitDataset(
            train=copy_data(self.train),
            test=copy_data(self.test),
            validation=copy_data(self.validation) if self.validation is not None else None
        )
