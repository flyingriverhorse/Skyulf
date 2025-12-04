from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class SplitDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    validation: Optional[pd.DataFrame] = None
    
    def copy(self) -> 'SplitDataset':
        return SplitDataset(
            train=self.train.copy(),
            test=self.test.copy(),
            validation=self.validation.copy() if self.validation is not None else None
        )
