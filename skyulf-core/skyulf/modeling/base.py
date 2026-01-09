import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

# Use relative imports assuming the structure is preserved
from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame, get_engine

# Evaluation imports - we will migrate these next
# from .evaluation.schemas import ModelEvaluationReport, ModelEvaluationSplitPayload
# from .evaluation.classification import build_classification_split_report
# from .evaluation.regression import build_regression_split_report

logger = logging.getLogger(__name__)


class BaseModelCalculator(ABC):
    @property
    @abstractmethod
    def problem_type(self) -> str:
        """Returns 'classification' or 'regression'."""
        pass

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default hyperparameters for the model."""
        return {}

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, SkyulfDataFrame],
        y: Union[pd.Series, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable[..., None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        validation_data: Optional[tuple[Union[pd.DataFrame, SkyulfDataFrame], Union[pd.Series, Any]]] = None,
    ) -> Any:
        """
        Trains the model. Returns the model object (serializable).
        """
        pass


class BaseModelApplier(ABC):
    @abstractmethod
    def predict(self, df: Union[pd.DataFrame, SkyulfDataFrame], model_artifact: Any) -> Union[pd.Series, Any]:
        """
        Generates predictions.
        """
        pass

    def predict_proba(
        self, df: Union[pd.DataFrame, SkyulfDataFrame], model_artifact: Any
    ) -> Optional[Union[pd.DataFrame, SkyulfDataFrame]]:
        """
        Generates prediction probabilities if supported.
        Returns DataFrame where columns are classes.
        """
        return None


class StatefulEstimator:
    def __init__(
        self, calculator: BaseModelCalculator, applier: BaseModelApplier, node_id: str
    ):
        self.calculator = calculator
        self.applier = applier
        self.node_id = node_id
        self.model = None  # In-memory model storage

    def _extract_xy(
        self, data: Any, target_column: str
    ) -> tuple[Any, Any]:
        """Helper to extract X and y from DataFrame or Tuple."""
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]

        engine = get_engine(data)

        if engine.name == "polars":
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            X = data.drop([target_column])
            y = data.select(target_column).to_series()
            return X, y

        # Pandas / Default
        # Check for DataFrame-like
        if hasattr(data, "columns"):
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            # Fallback for pure Pandas or Generic DataFrame
            # If we reached here without matching Polars explicitly, treat as generic/pandas
            # Try generic drop if available
            if hasattr(data, "drop"):
                # Handle pandas-like drop
                try:
                    return data.drop(columns=[target_column]), data[target_column]
                except TypeError:
                    # Maybe it doesn't support columns= kwarg, try position or list
                    pass
            
            # Simple attribute access fallback
            if hasattr(data, target_column):
                 return data, getattr(data, target_column)
                 
        raise ValueError(f"Unexpected data type: {type(data)}")

    def cross_validate(
        self,
        dataset: SplitDataset,
        target_column: str,
        config: Dict[str, Any],
        n_folds: int = 5,
        cv_type: str = "k_fold",
        shuffle: bool = True,
        random_state: int = 42,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Performs cross-validation on the training split.
        """
        # Import here to avoid circular dependency if any
        from .cross_validation import perform_cross_validation

        X_train, y_train = self._extract_xy(dataset.train, target_column)

        return perform_cross_validation(
            calculator=self.calculator,
            applier=self.applier,
            X=X_train,
            y=y_train,
            config=config,
            n_folds=n_folds,
            cv_type=cv_type,
            shuffle=shuffle,
            random_state=random_state,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    def fit_predict(
        self,
        dataset: Union[SplitDataset, pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        target_column: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        job_id: str = "unknown",
    ) -> Dict[str, pd.Series]:
        """
        Fits the model on training data and returns predictions for all splits.
        """
        # Handle raw DataFrame or Tuple input by wrapping it in a dummy SplitDataset
        if isinstance(dataset, pd.DataFrame):
            dataset = SplitDataset(train=dataset, test=pd.DataFrame(), validation=None)
        elif isinstance(dataset, tuple):
            # Check if it's (train_df, test_df) or (X, y)
            elem0 = dataset[0]
            if isinstance(elem0, pd.DataFrame) and target_column in elem0.columns:
                # It's (train_df, test_df)
                train_df, test_df = dataset
                dataset = SplitDataset(train=train_df, test=test_df, validation=None)  # type: ignore
            else:
                # Fallback: Treat input as training data (e.g. X, y tuple) and initialize empty test set.
                msg = (
                    "WARNING: No test set provided. Using entire input as training data. "
                    "Ensure data was split BEFORE preprocessing to avoid data leakage."
                )
                logger.warning(msg)
                if log_callback:
                    log_callback(msg)

                dataset = SplitDataset(
                    train=dataset, test=pd.DataFrame(), validation=None
                )

        # 1. Prepare Data
        X_train, y_train = self._extract_xy(dataset.train, target_column)

        validation_data = None
        if dataset.validation is not None:
            X_val, y_val = self._extract_xy(dataset.validation, target_column)
            validation_data = (X_val, y_val)

        # 2. Train Model
        self.model = self.calculator.fit(
            X_train,
            y_train,
            config,
            progress_callback=progress_callback,
            log_callback=log_callback,
            validation_data=validation_data,
        )

        # 3. Predict on all splits
        predictions = {}

        # Train Predictions
        predictions["train"] = self.applier.predict(X_train, self.model)

        # Test Predictions
        is_test_empty = False
        test_df = None
        if isinstance(dataset.test, tuple):
            test_df = dataset.test[0]
        else:
            test_df = dataset.test

        if hasattr(test_df, "empty"):
            is_test_empty = test_df.empty
        else:
            # Polars
            is_test_empty = test_df.is_empty()

        if not is_test_empty:
            if isinstance(dataset.test, tuple):
                X_test, _ = dataset.test
            else:
                if target_column in dataset.test.columns:
                    try:
                        X_test = dataset.test.drop(columns=[target_column])
                    except TypeError:
                        # Polars
                        X_test = dataset.test.drop([target_column])
                else:
                    X_test = dataset.test
            predictions["test"] = self.applier.predict(X_test, self.model)

        # Validation Predictions
        if dataset.validation is not None:
            if isinstance(dataset.validation, tuple):
                X_val, _ = dataset.validation
            else:
                if target_column in dataset.validation.columns:
                    X_val = dataset.validation.drop(columns=[target_column])
                else:
                    X_val = dataset.validation
            predictions["validation"] = self.applier.predict(X_val, self.model)

        return predictions

    def refit(
        self,
        dataset: SplitDataset,
        target_column: str,
        config: Dict[str, Any],
        job_id: str = "unknown",
    ) -> None:
        """
        Refits the model on Train + Validation data and updates the artifact.
        """
        if dataset.validation is None:
            # Fallback to normal fit if no validation set
            self.fit_predict(dataset, target_column, config, job_id=job_id)
            return

        # 1. Prepare Combined Data
        X_train, y_train = self._extract_xy(dataset.train, target_column)
        X_val, y_val = self._extract_xy(dataset.validation, target_column)

        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        # 2. Train Model
        self.model = self.calculator.fit(X_combined, y_combined, config)

    def evaluate(  # noqa: C901
        self, dataset: SplitDataset, target_column: str, job_id: str = "unknown"
    ) -> Any:
        """
        Evaluates the model on all splits and returns a detailed report.
        """
        # Import here to avoid circular dependency
        from .evaluation.classification import evaluate_classification_model
        from .evaluation.regression import evaluate_regression_model

        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call fit_predict() first."
            )

        problem_type = self.calculator.problem_type

        splits_payload = {}

        # Container for raw predictions
        evaluation_data = {
            "job_id": job_id,
            "node_id": self.node_id,
            "problem_type": problem_type,
            "splits": {},
        }

        # Helper to evaluate a single split
        def evaluate_split(split_name: str, data: Any):
            if isinstance(data, tuple):
                X, y = data
            elif isinstance(data, pd.DataFrame):
                if target_column not in data.columns:
                    return None  # Cannot evaluate without target
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                return None

            y_pred = self.applier.predict(X, self.model)

            # Try to get probabilities for classification
            y_proba = None
            if problem_type == "classification":
                y_proba_df = self.applier.predict_proba(X, self.model)
                if y_proba_df is not None:
                    y_proba = {
                        "classes": y_proba_df.columns.tolist(),
                        "values": y_proba_df.values.tolist(),
                    }

            split_data = {
                "y_true": y.tolist() if hasattr(y, "tolist") else list(y),
                "y_pred": (
                    y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
                ),
            }

            if y_proba:
                split_data["y_proba"] = y_proba

            evaluation_data["splits"][split_name] = split_data

            # Unpack model if it's a tuple (from Tuner)
            model_to_evaluate = self.model
            if isinstance(self.model, tuple) and len(self.model) == 2:
                # Check if first element looks like a model (has fit/predict)
                # or if it's just a convention from TuningCalculator
                model_to_evaluate = self.model[0]

            if problem_type == "classification":
                return evaluate_classification_model(
                    model=model_to_evaluate, dataset_name=split_name, X_test=X, y_test=y
                )
            elif problem_type == "regression":
                return evaluate_regression_model(
                    model=model_to_evaluate, dataset_name=split_name, X_test=X, y_test=y
                )
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")

        # 2. Evaluate Train
        splits_payload["train"] = evaluate_split("train", dataset.train)

        # 3. Evaluate Test
        has_test = False
        if isinstance(dataset.test, pd.DataFrame):
            has_test = not dataset.test.empty
        elif isinstance(dataset.test, tuple):
            has_test = len(dataset.test) == 2 and len(dataset.test[0]) > 0

        if has_test:
            splits_payload["test"] = evaluate_split("test", dataset.test)

        # 4. Evaluate Validation
        if dataset.validation is not None:
            has_val = False
            if isinstance(dataset.validation, pd.DataFrame):
                has_val = not dataset.validation.empty
            elif isinstance(dataset.validation, tuple):
                has_val = (
                    len(dataset.validation) == 2 and len(dataset.validation[0]) > 0
                )

            if has_val:
                splits_payload["validation"] = evaluate_split(
                    "validation", dataset.validation
                )

        # Return report object (simplified for now, assuming schema matches)
        return {
            "problem_type": problem_type,
            "splits": splits_payload,
            "raw_data": evaluation_data,
        }
