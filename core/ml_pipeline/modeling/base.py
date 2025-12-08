from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal, Callable, Union, Tuple
from datetime import datetime
import pandas as pd
from ..data.container import SplitDataset
from ..artifacts.store import ArtifactStore
from .evaluation.schemas import ModelEvaluationReport, ModelEvaluationSplitPayload
from .evaluation.classification import build_classification_split_report
from .evaluation.regression import build_regression_split_report
from .cross_validation import perform_cross_validation
import logging

logger = logging.getLogger(__name__)

class BaseModelCalculator(ABC):
    @property
    @abstractmethod
    def problem_type(self) -> str:
        """Returns 'classification' or 'regression'."""
        pass

    @abstractmethod
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any], 
        progress_callback: Optional[Callable[[int, int], None]] = None,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None
    ) -> Any:
        """
        Trains the model. Returns the model object (serializable).
        """
        pass

class BaseModelApplier(ABC):
    @abstractmethod
    def predict(self, df: pd.DataFrame, model_artifact: Any) -> pd.Series:
        """
        Generates predictions.
        """
        pass

    def predict_proba(self, df: pd.DataFrame, model_artifact: Any) -> Optional[pd.DataFrame]:
        """
        Generates prediction probabilities if supported.
        Returns DataFrame where columns are classes.
        """
        return None

class StatefulEstimator:
    def __init__(self, calculator: BaseModelCalculator, applier: BaseModelApplier, artifact_store: ArtifactStore, node_id: str):
        self.calculator = calculator
        self.applier = applier
        self.artifact_store = artifact_store
        self.node_id = node_id

    def _extract_xy(self, data: Any, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
        """Helper to extract X and y from DataFrame or Tuple."""
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, pd.DataFrame):
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            return data.drop(columns=[target_column]), data[target_column]
        else:
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
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Performs cross-validation on the training split.
        """
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
            progress_callback=progress_callback
        )

    def fit_predict(
        self, 
        dataset: Union[SplitDataset, pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], 
        target_column: str, 
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        job_id: str = "unknown"
    ) -> Dict[str, pd.Series]:
        """
        Fits the model on training data and returns predictions for all splits.
        """
        # Handle raw DataFrame or Tuple input by wrapping it in a dummy SplitDataset
        if isinstance(dataset, pd.DataFrame):
            dataset = SplitDataset(train=dataset, test=pd.DataFrame(), validation=None)
        elif isinstance(dataset, tuple):
            dataset = SplitDataset(train=dataset, test=pd.DataFrame(), validation=None)
        
        # 1. Prepare Data
        X_train, y_train = self._extract_xy(dataset.train, target_column)
        
        validation_data = None
        if dataset.validation is not None:
            X_val, y_val = self._extract_xy(dataset.validation, target_column)
            validation_data = (X_val, y_val)

        # 2. Train Model
        model_artifact = self.calculator.fit(
            X_train, 
            y_train, 
            config, 
            progress_callback=progress_callback,
            validation_data=validation_data
        )
        
        # 3. Save Artifact
        self.artifact_store.save(self.node_id, model_artifact)
        
        if job_id and job_id != "unknown":
             # Save with job_id to allow specific retrieval
             logger.info(f"Saving model artifact to job key: {job_id}")
             self.artifact_store.save(job_id, model_artifact)
        
        # 4. Predict on all splits
        predictions = {}
        
        # Train Predictions
        predictions['train'] = self.applier.predict(X_train, model_artifact)
        
        # Test Predictions
        # Check if dataset.test is empty. 
        # If it's a tuple, check if the first element (X) is empty.
        is_test_empty = False
        if isinstance(dataset.test, tuple):
            is_test_empty = dataset.test[0].empty
        else:
            is_test_empty = dataset.test.empty

        if not is_test_empty:
            if isinstance(dataset.test, tuple):
                X_test, _ = dataset.test
            else:
                if target_column in dataset.test.columns:
                    X_test = dataset.test.drop(columns=[target_column])
                else:
                    X_test = dataset.test
            predictions['test'] = self.applier.predict(X_test, model_artifact)
        
        # Validation Predictions
        if dataset.validation is not None:
            if isinstance(dataset.validation, tuple):
                X_val, _ = dataset.validation
            else:
                if target_column in dataset.validation.columns:
                    X_val = dataset.validation.drop(columns=[target_column])
                else:
                    X_val = dataset.validation
            predictions['validation'] = self.applier.predict(X_val, model_artifact)
            
        return predictions

    def refit(self, dataset: SplitDataset, target_column: str, config: Dict[str, Any], job_id: str = "unknown") -> None:
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
        model_artifact = self.calculator.fit(X_combined, y_combined, config)
        
        # 3. Save Artifact (Overwrites the previous one)
        self.artifact_store.save(self.node_id, model_artifact)
        
        if job_id and job_id != "unknown":
             # Save with job_id
             logger.info(f"Saving refitted model artifact to job key: {job_id}")
             self.artifact_store.save(job_id, model_artifact)

    def evaluate(self, dataset: SplitDataset, target_column: str, job_id: str = "unknown") -> ModelEvaluationReport:
        """
        Evaluates the model on all splits and returns a detailed report.
        Also saves raw predictions (y_true, y_pred) as an artifact for visualization.
        """
        # 1. Load Artifact
        model_artifact = self.artifact_store.load(self.node_id)
        problem_type = self.calculator.problem_type
        
        splits_payload: Dict[str, ModelEvaluationSplitPayload] = {}
        
        # Container for raw predictions to be saved as artifact
        evaluation_data = {
            "job_id": job_id,
            "node_id": self.node_id,
            "problem_type": problem_type,
            "splits": {}
        }
        
        # Helper to evaluate a single split
        def evaluate_split(split_name: str, data: Any):
            if isinstance(data, tuple):
                X, y = data
            elif isinstance(data, pd.DataFrame):
                if target_column not in data.columns:
                    return None # Cannot evaluate without target
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                return None

            # Generate predictions for saving
            # We need to predict here to save the raw values, even if build_*_report does it internally
            # Optimization: build_*_report could accept y_pred, but for now we re-predict or rely on the report builder
            # Actually, let's just let the report builder do the work, but we need the raw values for the artifact.
            # So we will predict here.
            y_pred = self.applier.predict(X, model_artifact)
            
            # Try to get probabilities for classification
            y_proba = None
            if problem_type == "classification":
                y_proba_df = self.applier.predict_proba(X, model_artifact)
                if y_proba_df is not None:
                    # Convert to list of dicts or list of lists?
                    # List of lists is more compact: [[p0, p1], [p0, p1]]
                    # But we need to know class order. DataFrame columns has it.
                    y_proba = {
                        "classes": y_proba_df.columns.tolist(),
                        "values": y_proba_df.values.tolist()
                    }

            # Save to evaluation_data
            # We convert to list to ensure JSON serializability if needed later, 
            # though joblib can handle numpy arrays. Lists are safer for generic consumption.
            split_data = {
                "y_true": y.tolist() if hasattr(y, "tolist") else list(y),
                "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
            }
            
            if y_proba:
                split_data["y_proba"] = y_proba
                
            evaluation_data["splits"][split_name] = split_data

            if problem_type == "classification":
                return build_classification_split_report(
                    model=model_artifact,
                    split_name=split_name,
                    features=X,
                    target=y
                )
            elif problem_type == "regression":
                return build_regression_split_report(
                    model=model_artifact,
                    split_name=split_name,
                    features=X,
                    target=y
                )
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")

        # 2. Evaluate Train
        splits_payload['train'] = evaluate_split('train', dataset.train)
        
        # 3. Evaluate Test
        has_test = False
        if isinstance(dataset.test, pd.DataFrame):
            has_test = not dataset.test.empty
        elif isinstance(dataset.test, tuple):
            has_test = len(dataset.test) == 2 and len(dataset.test[0]) > 0
            
        if has_test:
            splits_payload['test'] = evaluate_split('test', dataset.test)
        
        # 4. Evaluate Validation
        if dataset.validation is not None:
            splits_payload['validation'] = evaluate_split('validation', dataset.validation)
            
        # Save the raw evaluation data artifact
        # Key format: {node_id}_evaluation_data
        # We also save with job_id if available to avoid collisions or for easier lookup
        key = f"{self.node_id}_evaluation_data"
        logger.info(f"Saving evaluation artifact to key: {key}")
        self.artifact_store.save(key, evaluation_data)
        
        if job_id and job_id != "unknown":
             job_key = f"{job_id}_evaluation_data"
             logger.info(f"Saving evaluation artifact to job key: {job_key}")
             self.artifact_store.save(job_key, evaluation_data)
            
        # Feature cols for report metadata
        if isinstance(dataset.train, tuple):
            feature_cols = list(dataset.train[0].columns)
        else:
            feature_cols = [c for c in dataset.train.columns if c != target_column]

        return ModelEvaluationReport(
            job_id=job_id,
            node_id=self.node_id,
            generated_at=datetime.utcnow(),
            problem_type=problem_type, # type: ignore
            target_column=target_column,
            feature_columns=feature_cols,
            splits=splits_payload
        )

    def predict(self, dataset: SplitDataset, target_column: Optional[str] = None) -> Dict[str, pd.Series]:
        # 1. Load Artifact
        model_artifact = self.artifact_store.load(self.node_id)
        
        # 2. Predict
        predictions = {}
        
        # Train
        if target_column and target_column in dataset.train.columns:
             X_train = dataset.train.drop(columns=[target_column])
        else:
             X_train = dataset.train
        predictions['train'] = self.applier.predict(X_train, model_artifact)

        # Test
        if target_column and target_column in dataset.test.columns:
             X_test = dataset.test.drop(columns=[target_column])
        else:
             X_test = dataset.test
        predictions['test'] = self.applier.predict(X_test, model_artifact)
        
        # Validation
        if dataset.validation is not None:
            if target_column and target_column in dataset.validation.columns:
                X_val = dataset.validation.drop(columns=[target_column])
            else:
                X_val = dataset.validation
            predictions['validation'] = self.applier.predict(X_val, model_artifact)
            
        return predictions
