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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[SplitDataset, Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
        stratify = params.get("stratify", False)
        target_col = params.get("target_column")
        
        # If stratify is requested but no target column is specified, 
        # we set a dummy value to enable stratification logic in split_xy (which uses y).
        # For DataFrame split, this will correctly raise an error if the column is missing.
        stratify_col = target_col if stratify else None
        if stratify and not target_col:
            stratify_col = "__implicit_target__"

        splitter = DataSplitter(
            test_size=params.get("test_size", 0.2),
            validation_size=params.get("validation_size", 0.0),
            random_state=params.get("random_state", 42),
            shuffle=params.get("shuffle", True),
            stratify_col=stratify_col
        )
        
        # Handle (X, y) tuple input
        if isinstance(df, tuple) and len(df) == 2:
            X, y = df
            return splitter.split_xy(X, y)
            
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

    def split_xy(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Splits X and y arrays.
        """
        stratify = y if self.stratify_col else None # If stratify is requested, use y
        
        # First split: Train+Val vs Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify
        )
        
        X_val, y_val = None, None
        if self.validation_size > 0:
            relative_val_size = self.validation_size / (1 - self.test_size)
            stratify_val = y_train_val if self.stratify_col else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=relative_val_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_val
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            
        result = {
            "train": (X_train, y_train),
            "test": (X_test, y_test)
        }
        if X_val is not None:
            result["validation"] = (X_val, y_val)
            
        return result

    def split(self, df: pd.DataFrame) -> SplitDataset:
        """
        Splits the input DataFrame.
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")

        stratify = None
        if self.stratify_col:
            if self.stratify_col == "__implicit_target__":
                raise ValueError("Stratification requested but no target column specified. Please select a target column in the node settings.")
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


class FeatureTargetSplitCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        return config

class FeatureTargetSplitApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, SplitDataset], params: Dict[str, Any]) -> Union[Tuple[pd.DataFrame, pd.Series], Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
        target_col = params.get("target_column")
        if not target_col:
            raise ValueError("Target column must be specified for Feature-Target Split")
            
        selector = FeatureTargetSelector(target_column=target_col)
        
        if isinstance(df, SplitDataset):
            # Apply to all splits
            X_train, y_train = selector.select(df.train)
            X_test, y_test = selector.select(df.test)
            
            result = {
                "train": (X_train, y_train),
                "test": (X_test, y_test)
            }
            
            if df.validation is not None:
                X_val, y_val = selector.select(df.validation)
                result["validation"] = (X_val, y_val)
                
            return result
            
        return selector.select(df)

