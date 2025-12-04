from typing import Optional, List, Tuple, Union, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from core.ml_pipeline.data.container import SplitDataset
from .base import BaseCalculator, BaseApplier

class TrainTestSplitterCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # No learning from data, just pass through config
        return config

class TrainTestSplitterApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> SplitDataset:
        splitter = DataSplitter(
            test_size=params.get("test_size", 0.2),
            validation_size=params.get("validation_size", 0.0),
            random_state=params.get("random_state", 42),
            shuffle=params.get("shuffle", True),
            stratify_col=params.get("target_column") if params.get("stratify", False) else None
        )
        return splitter.split(df)

class DataSplitter:
    """
    Splits a DataFrame into Train, Test, and optionally Validation sets.
    """
    def __init__(self, 
                 test_size: float = 0.2, 
                 validation_size: float = 0.0, 
                 random_state: int = 42, 
                 shuffle: bool = True, 
                 stratify_col: Optional[str] = None):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify_col = stratify_col

    def split(self, df: pd.DataFrame) -> SplitDataset:
        """
        Splits the input DataFrame.
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")

        stratify = None
        if self.stratify_col:
            if self.stratify_col not in df.columns:
                raise ValueError(f"Stratification column '{self.stratify_col}' not found in DataFrame")
            stratify = df[self.stratify_col]

        # First split: Train+Val vs Test
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify
        )

        validation = None
        if self.validation_size > 0:
            # Adjust validation size relative to the remaining train_val set
            # original_val_size = val / total
            # current_train_val_size = 1 - test
            # relative_val_size = val / (1 - test)
            relative_val_size = self.validation_size / (1 - self.test_size)
            
            stratify_val = None
            if self.stratify_col:
                stratify_val = train_val[self.stratify_col]

            train, validation = train_test_split(
                train_val,
                test_size=relative_val_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_val
            )
        else:
            train = train_val

        return SplitDataset(train=train, test=test, validation=validation)

class FeatureTargetSelector:
    """
    Separates features and target from a DataFrame.
    """
    def __init__(self, target_column: str, feature_columns: Optional[List[str]] = None):
        self.target_column = target_column
        self.feature_columns = feature_columns

    def select(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns (X, y).
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        y = df[self.target_column]
        
        if self.feature_columns:
            missing = [c for c in self.feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
            X = df[self.feature_columns]
        else:
            X = df.drop(columns=[self.target_column])

        return X, y
