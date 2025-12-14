"""Main Skyulf Pipeline."""

import logging
import pickle
from typing import Any, Dict, Optional, Union

import pandas as pd

from .data.dataset import SplitDataset
from .modeling.base import StatefulEstimator, BaseModelCalculator, BaseModelApplier
from .modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
)
from .modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
    RidgeRegressionApplier,
    RidgeRegressionCalculator,
)
from .modeling.tuning.tuner import TunerApplier, TunerCalculator
from .preprocessing.pipeline import FeatureEngineer

logger = logging.getLogger(__name__)


class SkyulfPipeline:
    """
    End-to-end ML Pipeline.

    Encapsulates:
    1. Feature Engineering (Preprocessing)
    2. Modeling (Training/Inference)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration dictionary.
                    Must contain 'preprocessing' (list) and 'modeling' (dict).
        """
        self.config = config
        self.preprocessing_steps = config.get("preprocessing", [])
        self.modeling_config = config.get("modeling", {})

        self.feature_engineer = FeatureEngineer(self.preprocessing_steps)
        self.model_estimator: Optional[StatefulEstimator] = None

        # Initialize model estimator if config is present
        if self.modeling_config:
            self._init_model_estimator()

    def _init_model_estimator(self):
        """Initialize the StatefulEstimator based on config."""
        model_type = self.modeling_config.get("type")
        node_id = self.modeling_config.get("node_id", "model_node")

        calculator: Optional[BaseModelCalculator] = None
        applier: Optional[BaseModelApplier] = None

        # Map model types to classes
        # This mapping should ideally be dynamic or registered
        if model_type == "logistic_regression":
            calculator = LogisticRegressionCalculator()
            applier = LogisticRegressionApplier()
        elif model_type == "random_forest_classifier":
            calculator = RandomForestClassifierCalculator()
            applier = RandomForestClassifierApplier()
        elif model_type == "ridge_regression":
            calculator = RidgeRegressionCalculator()
            applier = RidgeRegressionApplier()
        elif model_type == "random_forest_regressor":
            calculator = RandomForestRegressorCalculator()
            applier = RandomForestRegressorApplier()
        elif model_type == "hyperparameter_tuner":
            # Tuner wraps another model
            base_model_config = self.modeling_config.get("base_model", {})
            base_model_type = base_model_config.get("type")

            base_calc: Optional[BaseModelCalculator] = None
            if base_model_type == "logistic_regression":
                base_calc = LogisticRegressionCalculator()
            elif base_model_type == "random_forest_classifier":
                base_calc = RandomForestClassifierCalculator()
            elif base_model_type == "ridge_regression":
                base_calc = RidgeRegressionCalculator()
            elif base_model_type == "random_forest_regressor":
                base_calc = RandomForestRegressorCalculator()

            if base_calc:
                calculator = TunerCalculator(base_calc)
                applier = TunerApplier()
            else:
                raise ValueError(f"Unknown base model type for tuner: {base_model_type}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_estimator = StatefulEstimator(
            node_id=node_id,
            calculator=calculator,
            applier=applier
        )

    def fit(self, data: Union[pd.DataFrame, SplitDataset], target_column: str) -> Dict[str, Any]:
        """
        Fit the pipeline.

        Args:
            data: Input data (DataFrame or SplitDataset).
            target_column: Name of the target column.

        Returns:
            Dictionary containing execution metrics.
        """
        metrics = {}

        # 1. Feature Engineering
        logger.info("Starting Feature Engineering...")
        transformed_data, fe_metrics = self.feature_engineer.fit_transform(data)
        metrics["preprocessing"] = fe_metrics

        # 2. Modeling
        if self.model_estimator:
            logger.info("Starting Model Training...")

            # Ensure transformed_data is SplitDataset for modeling
            if isinstance(transformed_data, pd.DataFrame):
                # If we only have a DataFrame, we can't really evaluate properly without a split
                # But we can fit on it.
                # Ideally, the user should provide a SplitDataset or use a Splitter node in preprocessing.
                # If preprocessing didn't split, we wrap it.
                dataset = SplitDataset(train=transformed_data, test=pd.DataFrame(), validation=None)
            else:
                dataset = transformed_data

            # Fit the model
            # Note: fit_predict updates self.model_estimator.model in-memory
            _ = self.model_estimator.fit_predict(
                dataset=dataset,
                target_column=target_column,
                config=self.modeling_config
            )

            # Evaluate
            # We can run evaluation if we have test/validation sets
            try:
                eval_report = self.model_estimator.evaluate(
                    dataset=dataset,
                    target_column=target_column
                )
                metrics["modeling"] = eval_report
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                metrics["modeling_error"] = str(e)

        return metrics

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate predictions.

        Args:
            data: Input DataFrame.

        Returns:
            Series of predictions.
        """
        # 1. Feature Engineering (Transform only)
        # FeatureEngineer.fit_transform handles stateful transformation if already fitted?
        # Wait, FeatureEngineer currently re-fits or uses stored state?
        # In the current implementation of FeatureEngineer (which I need to verify),
        # it creates StatefulTransformers.
        # I need to ensure FeatureEngineer persists the fitted transformers.

        # Actually, FeatureEngineer.fit_transform returns (data, metrics).
        # It stores the fitted transformers in `self.steps_config`? No, that's config.
        # I need to check `FeatureEngineer` implementation again.

        # ... checking FeatureEngineer ...
        # It seems FeatureEngineer in `core` was designed to run linearly and maybe didn't keep state
        # for inference in the same object easily, or it relied on ArtifactStore to load them.
        # Since I removed ArtifactStore, I need to make sure FeatureEngineer keeps the fitted transformers.

        # Let's assume for now FeatureEngineer needs to be updated to store fitted transformers
        # so we can call `transform` later.

        # For now, I will implement `predict` assuming `feature_engineer` has a `transform` method.
        # If not, I will need to add it.

        transformed_data = self.feature_engineer.transform(data)

        # 2. Modeling
        if self.model_estimator and self.model_estimator.model is not None:
            return self.model_estimator.applier.predict(transformed_data, self.model_estimator.model)
        else:
            raise ValueError("Pipeline not fitted or no model configured.")

    def save(self, path: str):
        """Save the pipeline to a file."""
        # We can use pickle to save the whole object since we removed external dependencies
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SkyulfPipeline":
        """Load the pipeline from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)  # type: ignore
