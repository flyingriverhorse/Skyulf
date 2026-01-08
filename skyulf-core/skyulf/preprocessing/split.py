import logging
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..data.dataset import SplitDataset
from .base import BaseApplier, BaseCalculator
from ..engines import SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)


class SplitApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> SplitDataset:
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
            stratify_col=stratify_col,
        )

        # Handle (X, y) tuple input
        if isinstance(df, tuple) and len(df) == 2:
            X, y = df
            return splitter.split_xy(X, y)

        return splitter.split(df)


@NodeRegistry.register("Split", SplitApplier)
@NodeRegistry.register("TrainTestSplitter", SplitApplier)
@node_meta(
    id="TrainTestSplitter",
    name="Train/Test Split",
    category="Data Operations",
    description="Split the dataset into training and testing sets.",
    params={"test_size": 0.2, "validation_size": 0.0, "random_state": 42, "shuffle": True, "stratify": False, "target_column": "target"}
)
class SplitCalculator(BaseCalculator):
    def fit(
        self, df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        # No learning from data, just pass through config
        return config


class DataSplitter:
    """
    Splits a DataFrame into Train, Test, and optionally Validation sets.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
        shuffle: bool = True,
        stratify_col: Optional[str] = None,
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify_col = stratify_col

    def split_xy(self, X: Union[pd.DataFrame, SkyulfDataFrame], y: Union[pd.Series, Any]) -> SplitDataset:
        """
        Splits X and y arrays.
        """
        engine = get_engine(X)
        is_polars = engine.name == "polars"

        if is_polars:
            # Convert to Pandas to preserve schema/metadata during split
            X_pd = X.to_pandas()
            y_pd = y.to_pandas() if y is not None else None
        else:
            X_pd = X
            y_pd = y

        stratify = y_pd if self.stratify_col else None  # If stratify is requested, use y

        if stratify is not None:
            # Check value counts
            class_counts = y_pd.value_counts()
            min_count = class_counts.min()
                
            if min_count < 2:
                logger.warning(
                    f"Stratified split requested but the least populated class has only {min_count} "
                    "member(s). Stratification will be disabled."
                )
                stratify = None

        # First split: Train+Val vs Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_pd,
            y_pd,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify,
        )

        validation = None
        if self.validation_size > 0:
            relative_val_size = self.validation_size / (1 - self.test_size)
            stratify_val = y_train_val if self.stratify_col else None

            if stratify_val is not None:
                class_counts_val = y_train_val.value_counts()
                min_count_val = class_counts_val.min()
                    
                if min_count_val < 2:
                    logger.warning(
                        "Stratified validation split requested but the least populated class has only "
                        f"{min_count_val} member(s). Stratification will be disabled for validation split."
                    )
                    stratify_val = None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=relative_val_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_val,
            )
            validation = (X_val, y_val)
        else:
            X_train, y_train = X_train_val, y_train_val

        # Convert back to Polars if needed
        if is_polars:
            import polars as pl
            
            def to_pl(df_or_series):
                if df_or_series is None: return None
                if isinstance(df_or_series, pd.DataFrame): return pl.from_pandas(df_or_series)
                if isinstance(df_or_series, pd.Series): return pl.from_pandas(df_or_series)
                return df_or_series

            X_train = to_pl(X_train)
            y_train = to_pl(y_train)
            X_test = to_pl(X_test)
            y_test = to_pl(y_test)
            
            if validation:
                validation = (to_pl(validation[0]), to_pl(validation[1]))

        return SplitDataset(
            train=(X_train, y_train), test=(X_test, y_test), validation=validation
        )

    def split(self, df: Union[pd.DataFrame, SkyulfDataFrame]) -> SplitDataset:
        """
        Splits a DataFrame.
        """
        engine = get_engine(df)
        is_polars = engine.name == "polars"

        if is_polars:
            # Convert to Pandas to preserve schema/metadata during split
            df_pd = df.to_pandas()
        else:
            df_pd = df
        
        stratify = None
        if self.stratify_col and self.stratify_col in df_pd.columns:
            stratify = df_pd[self.stratify_col]
            class_counts = stratify.value_counts()
            if class_counts.min() < 2:
                logger.warning(
                    f"Stratified split requested but the least populated class has only {class_counts.min()} "
                    "member(s). Stratification will be disabled."
                )
                stratify = None

        train_val, test = train_test_split(
            df_pd,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify,
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
                        "Stratified validation split requested but the least populated class has only "
                        f"{class_counts_val.min()} member(s). Stratification will be disabled for validation split."
                    )
                    stratify_val = None

            train, val = train_test_split(
                train_val,
                test_size=relative_val_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_val,
            )
            validation = val
        else:
            train = train_val

        # Convert back to Polars if needed
        if is_polars:
            import polars as pl
            train = pl.from_pandas(train)
            test = pl.from_pandas(test)
            if validation is not None:
                validation = pl.from_pandas(validation)

        return SplitDataset(
            train=(train, None),
            test=(test, None),
            validation=(validation, None) if validation is not None else None
        )


class FeatureTargetSplitApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, SplitDataset, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[Tuple[pd.DataFrame, pd.Series], SplitDataset]:
        target_col = params.get("target_column")
        if not target_col:
            raise ValueError(
                "Target column must be specified for FeatureTargetSplitter"
            )

        def split_one(data: Union[pd.DataFrame, SkyulfDataFrame, Any]) -> Tuple[Any, Any]:
            engine = get_engine(data)
            if engine.name == "polars":
                data_pl: Any = data
                if target_col not in data_pl.columns:
                    raise ValueError(f"Target column '{target_col}' not found in dataset")
                import polars as pl
                y = data_pl.select(pl.col(target_col)).to_series()
                X = data_pl.drop([target_col])
                return X, y
            
            # Pandas
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset")
            y = data[target_col]
            X = data.drop(columns=[target_col])
            return X, y

        if isinstance(df, SplitDataset):
            # Apply to all splits
            # We check if it's NOT a tuple (meaning it's a DataFrame-like object)
            train = (
                split_one(df.train) if not isinstance(df.train, tuple) else df.train
            )
            test = split_one(df.test) if not isinstance(df.test, tuple) else df.test
            validation = None
            if df.validation is not None:
                validation = (
                    split_one(df.validation)
                    if not isinstance(df.validation, tuple)
                    else df.validation
                )

            return SplitDataset(train=train, test=test, validation=validation)

        if isinstance(df, tuple):
            return df

        # Assume DataFrame-like
        return split_one(df)


@NodeRegistry.register("feature_target_split", FeatureTargetSplitApplier)
@node_meta(
    id="feature_target_split",
    name="Feature/Target Split",
    category="Data Operations",
    description="Split the dataset into features (X) and target (y).",
    params={"target_column": "target"}
)
class FeatureTargetSplitCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, SplitDataset, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return config
