"""Main Skyulf Pipeline."""

import logging
import pickle
from typing import Any, Dict, Optional, Union, cast

import pandas as pd

from .types import PipelineConfig
from .data.dataset import SplitDataset
from .engines import SkyulfDataFrame, get_engine
from .modeling.base import BaseModelApplier, BaseModelCalculator, StatefulEstimator
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
from .modeling.tuning.engine import TuningApplier, TuningCalculator
from .preprocessing.pipeline import FeatureEngineer
from .registry import NodeRegistry

logger = logging.getLogger(__name__)


class SkyulfPipeline:
    """
    End-to-end ML Pipeline.

    Encapsulates:
    1. Feature Engineering (Preprocessing)
    2. Modeling (Training/Inference)
    """

    def __init__(self, config: PipelineConfig):
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
        if not model_type:
            return

        node_id = self.modeling_config.get("node_id", "model_node")

        calculator: Optional[BaseModelCalculator] = None
        applier: Optional[BaseModelApplier] = None

        # Try Registry first
        if model_type:
            try:
                calculator = NodeRegistry.get_calculator(model_type)()
                applier = NodeRegistry.get_applier(model_type)()
            except ValueError:
                pass

        if calculator is None:
            # Map model types to classes
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
                base_applier: Optional[BaseModelApplier] = None

                # Try Registry for base model
                if base_model_type:
                    try:
                        base_calc = NodeRegistry.get_calculator(base_model_type)()
                        base_applier = NodeRegistry.get_applier(base_model_type)()
                    except ValueError:
                        pass

                if base_calc is None:
                    if base_model_type == "logistic_regression":
                        base_calc = LogisticRegressionCalculator()
                        base_applier = LogisticRegressionApplier()
                    elif base_model_type == "random_forest_classifier":
                        base_calc = RandomForestClassifierCalculator()
                        base_applier = RandomForestClassifierApplier()
                    elif base_model_type == "ridge_regression":
                        base_calc = RidgeRegressionCalculator()
                        base_applier = RidgeRegressionApplier()
                    elif base_model_type == "random_forest_regressor":
                        base_calc = RandomForestRegressorCalculator()
                        base_applier = RandomForestRegressorApplier()

                if base_calc and base_applier:
                    calculator = TuningCalculator(base_calc)
                    applier = TuningApplier(base_applier)
                else:
                    raise ValueError(
                        f"Unknown base model type for tuner: {base_model_type}"
                    )

        if calculator is None or applier is None:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_estimator = StatefulEstimator(
            node_id=node_id, calculator=calculator, applier=applier
        )

    def fit(
        self, data: Union[pd.DataFrame, SkyulfDataFrame, SplitDataset], target_column: str
    ) -> Dict[str, Any]:
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
            if isinstance(transformed_data, SplitDataset):
                dataset = transformed_data
            else:
                # If we only have a DataFrame, we can't really evaluate properly without a split
                # But we can fit on it.
                # Ideally, the user should provide a SplitDataset or use a Splitter node in preprocessing.
                # If preprocessing didn't split, we wrap it.
                engine = get_engine(transformed_data)
                empty_df = engine.create_dataframe({})
                dataset = SplitDataset(
                    train=transformed_data, test=empty_df, validation=None
                )

            # Fit the model
            # Note: fit_predict updates self.model_estimator.model in-memory
            _ = self.model_estimator.fit_predict(
                dataset=dataset,
                target_column=target_column,
                config=cast(Dict[str, Any], self.modeling_config),
            )

            # Evaluate
            # We can run evaluation if we have test/validation sets
            try:
                eval_report = self.model_estimator.evaluate(
                    dataset=dataset, target_column=target_column
                )
                metrics["modeling"] = eval_report
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                metrics["modeling_error"] = str(e)

        return metrics

    def predict(self, data: Union[pd.DataFrame, SkyulfDataFrame]) -> Any:
        """
        Generate predictions.

        Args:
            data: Input DataFrame.

        Returns:
            Series of predictions.
        """
        # 1. Feature Engineering (Transform only)
        transformed_data = self.feature_engineer.transform(data)

        # 2. Modeling
        if self.model_estimator and self.model_estimator.model is not None:
            return self.model_estimator.applier.predict(
                transformed_data, self.model_estimator.model
            )
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
