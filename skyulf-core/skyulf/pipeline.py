"""Main Skyulf Pipeline."""

import hashlib
import json
import logging
import pickle  # nosec B403 - used only for internal pipeline serialization (see save/load below)
from typing import Any, cast

import pandas as pd

from .config_validation import validate_pipeline_config
from .data.dataset import SplitDataset
from .engines import SkyulfDataFrame, get_engine
from .leakage import validate_leakage_safety
from .modeling._tuning.engine import TuningApplier, TuningCalculator
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
from .preprocessing.pipeline import FeatureEngineer
from .registry import NodeRegistry
from .types import PipelineConfig

logger = logging.getLogger(__name__)

# Hardcoded fallback map from model_type -> (calculator_cls, applier_cls), used
# when NodeRegistry doesn't (yet) have a full calculator+applier registration
# for that type. Mirrors EngineRegistry.resolve()'s dispatch-table style.
_HARDCODED_MODEL_MAP: dict[str, tuple[type[BaseModelCalculator], type[BaseModelApplier]]] = {
    "logistic_regression": (LogisticRegressionCalculator, LogisticRegressionApplier),
    "random_forest_classifier": (RandomForestClassifierCalculator, RandomForestClassifierApplier),
    "ridge_regression": (RidgeRegressionCalculator, RidgeRegressionApplier),
    "random_forest_regressor": (RandomForestRegressorCalculator, RandomForestRegressorApplier),
}


def _mermaid_escape(text: str) -> str:
    """Escape characters that would break a Mermaid node label."""
    return text.replace('"', "'").replace("[", "(").replace("]", ")")


def _artifact_digest(obj: Any) -> bytes:
    """Stable digest of a fitted artifact.

    Pickle is deterministic for the same fitted estimator (same numpy arrays),
    which is what we want for a reproducibility seal. Falls back to ``repr`` for
    the rare object that refuses to pickle.
    """
    try:
        return hashlib.sha256(pickle.dumps(obj)).digest()  # nosec B301 nosemgrep: avoid-pickle -- trusted in-process artifact hashing, not attacker-controlled deserialization
    except Exception:
        return hashlib.sha256(repr(obj).encode("utf-8")).digest()


class SkyulfPipeline:
    """
    End-to-end ML Pipeline.

    Encapsulates:
    1. Feature Engineering (Preprocessing)
    2. Modeling (Training/Inference)

    Examples:
        >>> pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {}})
        >>> metrics = pipeline.fit(data, target_column="target")
    """

    def __init__(self, config: PipelineConfig | dict[str, Any]):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration dictionary.
                    Must contain 'preprocessing' (list) and 'modeling' (dict).
        """
        validate_pipeline_config(config)
        self.config = config
        self.preprocessing_steps = config.get("preprocessing", [])
        self.modeling_config = config.get("modeling", {})

        self.feature_engineer = FeatureEngineer(self.preprocessing_steps, _validated=True)
        self.model_estimator: StatefulEstimator | None = None
        self._fit_metrics: dict[str, Any] | None = None
        self._target_column: str | None = None

        # Initialize model estimator if config is present
        if self.modeling_config:
            self._init_model_estimator()

    @staticmethod
    def _resolve_from_registry(
        model_type: str | None,
    ) -> tuple[BaseModelCalculator | None, BaseModelApplier | None]:
        """Try resolving a calculator/applier pair for model_type from NodeRegistry.

        Returns (None, None) if model_type is falsy, or if the registry lookup
        fails (e.g. partial registration where only one of the two resolves).
        """
        if not model_type:
            return None, None
        try:
            calculator = NodeRegistry.get_calculator(model_type)()
            applier = NodeRegistry.get_applier(model_type)()
            return calculator, applier
        except ValueError as e:
            logger.debug(
                "Model type '%s' not found in NodeRegistry (%s); "
                "falling back to hardcoded type map.",
                model_type,
                e,
            )
            return None, None

    @staticmethod
    def _resolve_from_hardcoded_map(
        model_type: str | None,
    ) -> tuple[BaseModelCalculator | None, BaseModelApplier | None]:
        """Look up a plain (non-tuner) model_type in the hardcoded fallback map."""
        entry = _HARDCODED_MODEL_MAP.get(model_type) if model_type else None
        if entry is None:
            return None, None
        calculator_cls, applier_cls = entry
        return calculator_cls(), applier_cls()

    def _build_tuning_estimator(self) -> tuple[BaseModelCalculator, BaseModelApplier]:
        """Build the TuningCalculator/TuningApplier pair wrapping the configured base model."""
        base_model_config = self.modeling_config.get("base_model", {})
        base_model_type = base_model_config.get("type")

        base_calc, base_applier = self._resolve_from_registry(base_model_type)
        if base_calc is None or base_applier is None:
            base_calc, base_applier = self._resolve_from_hardcoded_map(base_model_type)

        if base_calc and base_applier:
            return TuningCalculator(base_calc), TuningApplier(base_applier)

        raise ValueError(f"Unknown base model type for tuner: {base_model_type}")

    def _init_model_estimator(self):
        """Initialize the StatefulEstimator based on config."""
        model_type = self.modeling_config.get("type")
        if not model_type:
            return

        node_id = self.modeling_config.get("node_id", "model_node")

        # Try Registry first
        calculator, applier = self._resolve_from_registry(model_type)

        if calculator is None or applier is None:
            if model_type == "hyperparameter_tuner":
                # Tuner wraps another model
                calculator, applier = self._build_tuning_estimator()
            else:
                calculator, applier = self._resolve_from_hardcoded_map(model_type)

        if calculator is None or applier is None:
            try:
                NodeRegistry.get_calculator(model_type)
            except ValueError as exc:
                raise ValueError(f"Unknown model type: {model_type}. {exc}") from exc
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_estimator = StatefulEstimator(
            node_id=node_id, calculator=calculator, applier=applier
        )

    def fit(
        self, data: pd.DataFrame | SkyulfDataFrame | SplitDataset, target_column: str
    ) -> dict[str, Any]:
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
                dataset = SplitDataset(train=transformed_data, test=empty_df, validation=None)

            # Fit the model
            # Note: fit_predict updates self.model_estimator.model in-memory
            _ = self.model_estimator.fit_predict(
                dataset=dataset,
                target_column=target_column,
                config=cast(dict[str, Any], self.modeling_config),
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

        self._fit_metrics = metrics
        self._target_column = target_column
        return metrics

    def predict(self, data: pd.DataFrame | SkyulfDataFrame) -> Any:
        """
        Generate predictions.

        Args:
            data: Input DataFrame.

        Returns:
            Series of predictions.

        Raises:
            ValueError: If the input still contains the target column used during fit.
        """
        if self._target_column is not None and self._target_column in data.columns:
            raise ValueError(
                f"predict() input still contains the target column '{self._target_column}' "
                "used during fit(); drop it before calling predict()."
            )

        # 1. Feature Engineering (Transform only)
        transformed_data = self.feature_engineer.transform(data)

        # 2. Modeling
        if self.model_estimator and self.model_estimator.model is not None:
            return self.model_estimator.applier.predict(
                transformed_data, self.model_estimator.model
            )
        else:
            raise ValueError("Pipeline not fitted or no model configured.")

    def describe(self) -> str:
        """Return a human-readable, multi-line summary of the pipeline.

        Renders the preprocessing chain (in order) and the model stage with
        their configured parameters. Pure read-only over ``self.config`` — safe
        to call before or after :meth:`fit`. Handy in notebooks and CI logs.
        """
        lines = ["SkyulfPipeline", "=" * 14]

        steps = list(self.preprocessing_steps)
        lines.append(f"Preprocessing ({len(steps)} step{'s' if len(steps) != 1 else ''}):")
        if steps:
            for i, step in enumerate(steps):
                name = step.get("name", f"step_{i}")
                transformer = step.get("transformer", "?")
                lines.append(f"  {i + 1}. {name} [{transformer}]")
                for key, value in step.get("params", {}).items():
                    lines.append(f"       - {key}: {value}")
        else:
            lines.append("  (none)")

        lines.append("Modeling:")
        if self.modeling_config:
            lines.append(f"  type: {self.modeling_config.get('type', '?')}")
            for key, value in self.modeling_config.items():
                if key != "type":
                    lines.append(f"    - {key}: {value}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)

    def validate_leakage_safety(self) -> list[str]:
        """Return warnings for preprocessing steps ordered before the train/test split."""
        return validate_leakage_safety(self.config)

    def to_mermaid(self) -> str:
        """Render the pipeline as a Mermaid ``flowchart`` string.

        Produces a top-down graph ``data -> [preprocessing steps] -> model``.
        Useful in docs and PR descriptions. Pure read-only over ``self.config``.
        """
        lines = ["flowchart TD", "    data[Input Data]"]
        prev = "data"

        for i, step in enumerate(self.preprocessing_steps):
            node = f"pp{i}"
            name = step.get("name", f"step_{i}")
            transformer = step.get("transformer", "?")
            label = _mermaid_escape(f"{name} ({transformer})")
            lines.append(f"    {node}[{label}]")
            lines.append(f"    {prev} --> {node}")
            prev = node

        if self.modeling_config:
            label = _mermaid_escape(str(self.modeling_config.get("type", "model")))
            lines.append(f"    model([{label}])")
            lines.append(f"    {prev} --> model")

        return "\n".join(lines)

    def is_fitted(self) -> bool:
        """True once preprocessing has been fit (or a model has been trained)."""
        if self.feature_engineer.fitted_steps:
            return True
        return self.model_estimator is not None and self.model_estimator.model is not None

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 over topology + fitted artifacts.

        The hash covers the pipeline graph (preprocessing + modeling config) and,
        once fitted, every fitted artifact and the trained model. Two pipelines
        with the same hash produce the same predictions, so callers can prove
        "this prediction came from exactly this pipeline". The digest changes
        across library versions by design (artifacts pickle differently).
        """
        hasher = hashlib.sha256()
        topology = {
            "preprocessing": self.preprocessing_steps,
            "modeling": self.modeling_config,
        }
        hasher.update(json.dumps(topology, sort_keys=True, default=str).encode("utf-8"))

        for step in self.feature_engineer.fitted_steps:
            hasher.update(_artifact_digest(step.get("artifact")))

        if self.model_estimator is not None and self.model_estimator.model is not None:
            hasher.update(_artifact_digest(self.model_estimator.model))

        return hasher.hexdigest()

    def export_model_card(self) -> dict[str, Any]:
        """Return a structured, JSON-friendly summary of the pipeline.

        Captures lineage (preprocessing chain), the model and its hyperparameters,
        the reproducibility fingerprint, and the metrics from the last :meth:`fit`
        (``None`` if never fitted). Intended for audit logs and model registries.
        """
        model: dict[str, Any] | None = None
        if self.modeling_config:
            model = {
                "type": self.modeling_config.get("type"),
                "params": {k: v for k, v in self.modeling_config.items() if k != "type"},
            }

        return {
            "schema_version": "1.0",
            "fitted": self.is_fitted(),
            "fingerprint": self.fingerprint(),
            "preprocessing": [
                {
                    "name": step.get("name"),
                    "transformer": step.get("transformer"),
                    "params": step.get("params", {}),
                }
                for step in self.preprocessing_steps
            ],
            "model": model,
            "metrics": self._fit_metrics,
        }

    def save(self, path: str):
        """Save the pipeline to a file."""
        # We can use pickle to save the whole object since we removed external dependencies
        with open(path, "wb") as f:
            pickle.dump(self, f)  # nosec B301 nosemgrep: avoid-pickle -- trusted local artifact save, not attacker-controlled

    @classmethod
    def load(cls, path: str) -> "SkyulfPipeline":
        """Load the pipeline from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)  # nosec B301 nosemgrep: avoid-pickle -- loads only artifacts previously saved by this same trusted process, not attacker-controlled input
