from typing import Optional, Tuple, Union, Dict, Any
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from ..data.dataset import SplitDataset
from .base import BaseCalculator, BaseApplier

logger = logging.getLogger(__name__)


class SplitCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[Any, ...]], config: Dict[str, Any]) -> Dict[str, Any]:
        # No learning from data, just pass through config
        return config


class SplitApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> SplitDataset:
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

    def split_xy(self, X: pd.DataFrame, y: pd.Series) -> SplitDataset:
        """
        Splits X and y arrays.
        """
        stratify = y if self.stratify_col else None  # If stratify is requested, use y

        if stratify is not None:
            class_counts = y.value_counts()
            if class_counts.min() < 2:
                logger.warning(
                    f"Stratified split requested but the least populated class has only {
                        class_counts.min()} member(s). Stratification will be disabled.")
                stratify = None

        # First split: Train+Val vs Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify
        )

        validation = None
        if self.validation_size > 0:
            relative_val_size = self.validation_size / (1 - self.test_size)
            stratify_val = y_train_val if self.stratify_col else None

            if stratify_val is not None:
                class_counts_val = y_train_val.value_counts()
                if class_counts_val.min() < 2:
                    logger.warning(
                        f"Stratified validation split requested but the least populated class has only {
                            class_counts_val.min()} member(s). Stratification will be disabled for validation split.")
                    stratify_val = None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=relative_val_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_val
            )
            validation = (X_val, y_val)
        else:
            X_train, y_train = X_train_val, y_train_val

        return SplitDataset(
            train=(X_train, y_train),
            test=(X_test, y_test),
            validation=validation
        )

    def split(self, df: pd.DataFrame) -> SplitDataset:
        """
        Splits a DataFrame.
        """
        stratify = None
        if self.stratify_col and self.stratify_col in df.columns:
            stratify = df[self.stratify_col]
            class_counts = stratify.value_counts()
            if class_counts.min() < 2:
                logger.warning(
                    f"Stratified split requested but the least populated class has only {
                        class_counts.min()} member(s). Stratification will be disabled.")
                stratify = None

        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify
        )

        validation = None
        if self.validation_size > 0:
            relative_val_size = self.validation_size / (1 - self.test_size)

            stratify_val = None
            if self.stratify_col and self.stratify_col in train_val.columns:
                stratify_val = train_val[self.stratify_col]
                class_counts_val = stratify_val.value_counts()
                if class_counts_val.min() < 2:
                    logger.warning(
                        f"Stratified validation split requested but the least populated class has only {
                            class_counts_val.min()} member(s). Stratification will be disabled for validation split.")
                    stratify_val = None

            train, val = train_test_split(
                train_val,
                test_size=relative_val_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_val
            )
            validation = val
        else:
            train = train_val

        return SplitDataset(train=train, test=test, validation=validation)


class FeatureTargetSplitCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, SplitDataset, Tuple[Any, ...]], config: Dict[str, Any]) -> Dict[str, Any]:
        return config


class FeatureTargetSplitApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, SplitDataset, Tuple[Any, ...]], params: Dict[str, Any]
              ) -> Union[Tuple[pd.DataFrame, pd.Series], SplitDataset]:
        target_col = params.get("target_column")
        if not target_col:
            raise ValueError("Target column must be specified for FeatureTargetSplitter")

        def split_one(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset")
            y = data[target_col]
            X = data.drop(columns=[target_col])
            return X, y

        if isinstance(df, SplitDataset):
            # Apply to all splits
            train = split_one(df.train) if isinstance(df.train, pd.DataFrame) else df.train
            test = split_one(df.test) if isinstance(df.test, pd.DataFrame) else df.test
            validation = None
            if df.validation is not None:
                validation = split_one(df.validation) if isinstance(df.validation, pd.DataFrame) else df.validation

            return SplitDataset(train=train, test=test, validation=validation)

        if isinstance(df, pd.DataFrame):
            return split_one(df)
        
        return df
