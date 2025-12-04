from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal, Callable
from datetime import datetime
import pandas as pd
from ..data.container import SplitDataset
from ..artifacts.store import ArtifactStore
from .evaluation.schemas import ModelEvaluationReport, ModelEvaluationSplitPayload
from .evaluation.classification import build_classification_split_report
from .evaluation.regression import build_regression_split_report
from .cross_validation import perform_cross_validation

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

class StatefulEstimator:
    def __init__(self, calculator: BaseModelCalculator, applier: BaseModelApplier, artifact_store: ArtifactStore, node_id: str):
        self.calculator = calculator
        self.applier = applier
        self.artifact_store = artifact_store
        self.node_id = node_id

    def cross_validate(
        self, 
        dataset: SplitDataset, 
        target_column: str, 
        config: Dict[str, Any], 
        n_folds: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Performs cross-validation on the training split.
        """
        X_train = dataset.train.drop(columns=[target_column])
        y_train = dataset.train[target_column]
        
        return perform_cross_validation(
            calculator=self.calculator,
            applier=self.applier,
            X=X_train,
            y=y_train,
            config=config,
            n_folds=n_folds,
            progress_callback=progress_callback
        )

    def fit_predict(
        self, 
        dataset: SplitDataset, 
        target_column: str, 
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, pd.Series]:
        """
        Fits the model on training data and returns predictions for all splits.
        """
        # 1. Prepare Data
        X_train = dataset.train.drop(columns=[target_column])
        y_train = dataset.train[target_column]
        
        validation_data = None
        if dataset.validation is not None:
            X_val = dataset.validation.drop(columns=[target_column])
            y_val = dataset.validation[target_column]
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
        
        # 4. Predict on all splits
        predictions = {}
        
        # Train Predictions
        predictions['train'] = self.applier.predict(X_train, model_artifact)
        
        # Test Predictions
        if target_column in dataset.test.columns:
             X_test = dataset.test.drop(columns=[target_column])
        else:
             X_test = dataset.test
        predictions['test'] = self.applier.predict(X_test, model_artifact)
        
        # Validation Predictions
        if dataset.validation is not None:
            if target_column in dataset.validation.columns:
                X_val = dataset.validation.drop(columns=[target_column])
            else:
                X_val = dataset.validation
            predictions['validation'] = self.applier.predict(X_val, model_artifact)
            
        return predictions

    def refit(self, dataset: SplitDataset, target_column: str, config: Dict[str, Any]) -> None:
        """
        Refits the model on Train + Validation data and updates the artifact.
        """
        if dataset.validation is None:
            # Fallback to normal fit if no validation set
            self.fit_predict(dataset, target_column, config)
            return

        # 1. Prepare Combined Data
        X_train = dataset.train.drop(columns=[target_column])
        y_train = dataset.train[target_column]
        
        X_val = dataset.validation.drop(columns=[target_column])
        y_val = dataset.validation[target_column]
        
        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)
        
        # 2. Train Model
        model_artifact = self.calculator.fit(X_combined, y_combined, config)
        
        # 3. Save Artifact (Overwrites the previous one)
        self.artifact_store.save(self.node_id, model_artifact)

    def evaluate(self, dataset: SplitDataset, target_column: str, job_id: str = "unknown") -> ModelEvaluationReport:
        """
        Evaluates the model on all splits and returns a detailed report.
        """
        # 1. Load Artifact
        model_artifact = self.artifact_store.load(self.node_id)
        problem_type = self.calculator.problem_type
        
        splits_payload: Dict[str, ModelEvaluationSplitPayload] = {}
        
        # Helper to evaluate a single split
        def evaluate_split(split_name: str, df: pd.DataFrame):
            if target_column not in df.columns:
                return # Cannot evaluate without target
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
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
        splits_payload['test'] = evaluate_split('test', dataset.test)
        
        # 4. Evaluate Validation
        if dataset.validation is not None:
            splits_payload['validation'] = evaluate_split('validation', dataset.validation)
            
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
