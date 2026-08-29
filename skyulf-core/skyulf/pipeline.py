"""Main Skyulf Pipeline."""

import hashlib
import json
import logging
import pickle  # nosec B403 - used only for internal pipeline serialization (see save/load below)
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from .config_validation import validate_pipeline_config
from .data.dataset import SplitDataset
from .engines import SkyulfDataFrame, get_engine
from .leakage import OnLeakage, validate_leakage_safety
from .modeling._evaluation.thresholds import apply_thresholds, optimize_thresholds
from .modeling._tuning.engine import TuningApplier, TuningCalculator
from .modeling.base import BaseModelApplier, BaseModelCalculator, StatefulEstimator, extract_xy
from .preprocessing.pipeline import FeatureEngineer
from .registry import NodeRegistry
from .types import PipelineConfig

logger = logging.getLogger(__name__)


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
    except Exception:  # noqa: BLE001 - objects that refuse pickle fall back to repr hashing
        return hashlib.sha256(repr(obj).encode("utf-8")).digest()


def _to_pandas(obj: Any) -> Any:
    """Convert a Polars DataFrame/Series (or any object exposing ``to_pandas()``)
    to its pandas equivalent; pass pandas objects (or ``None``) through unchanged."""
    if obj is None:
        return None
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return obj


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
        self._tuned_thresholds: dict[Any, float] | None = None

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
            logger.debug("Model type '%s' not resolvable from NodeRegistry: %s", model_type, e)
            return None, None

    def _build_tuning_estimator(self) -> tuple[BaseModelCalculator, BaseModelApplier]:
        """Build the TuningCalculator/TuningApplier pair wrapping the configured base model."""
        base_model_config = self.modeling_config.get("base_model", {})
        base_model_type = base_model_config.get("type")

        base_calc, base_applier = self._resolve_from_registry(base_model_type)
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

        if (calculator is None or applier is None) and model_type == "hyperparameter_tuner":
            # Tuner wraps another model
            calculator, applier = self._build_tuning_estimator()

        if calculator is None or applier is None:
            try:
                NodeRegistry.get_calculator(model_type)
            except ValueError as exc:
                raise ValueError(f"Unknown model type: {model_type}. {exc}") from exc
            raise ValueError(
                f"Model type '{model_type}' is only partially registered "
                "(calculator found, applier missing)."
            )

        self.model_estimator = StatefulEstimator(
            node_id=node_id, calculator=calculator, applier=applier
        )

    def fit(
        self,
        data: pd.DataFrame | pl.DataFrame | SkyulfDataFrame | SplitDataset,
        target_column: str,
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

        # Leakage structure check (advisory): the backend execution gate
        # hard-blocks data-dependent preprocessing before the split; in the
        # SDK the same verdict is surfaced as warnings before any fit
        # happens. Skipped when the caller supplies a SplitDataset — the
        # train/test boundary is then provided externally and enforced by
        # construction, and a flat config legitimately has no splitter node.
        if not isinstance(data, SplitDataset):
            for warning in validate_leakage_safety(self.config, on_leakage="warn"):
                logger.warning(warning)

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
            except Exception as e:  # noqa: BLE001 - evaluation failure is recorded as modeling_error; fit must continue
                logger.warning(f"Evaluation failed: {e}")
                metrics["modeling_error"] = str(e)

        self._fit_metrics = metrics
        self._target_column = target_column
        return metrics

    def get_fitted_split(
        self,
        data: pd.DataFrame | pl.DataFrame | SkyulfDataFrame | SplitDataset,
        target_column: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Run this pipeline's configured preprocessing chain and return the
        resulting train/test split as plain pandas objects.

        Runs ``self.feature_engineer.fit_transform(data)`` — the same
        preprocessing ``fit()`` uses internally — and extracts
        ``(X_train, y_train, X_test, y_test)`` from the resulting split using
        ``target_column``, converting any Polars/SkyulfDataFrame frames to
        pandas. Saves callers from re-implementing this split/convert step
        themselves for custom evaluation harnesses (e.g. comparing multiple
        raw sklearn-style estimators against the same preprocessed split).

        Args:
            data: Input data (DataFrame or SplitDataset).
            target_column: Name of the target column.

        Returns:
            ``(X_train, y_train, X_test, y_test)`` as pandas DataFrame/Series.

        Raises:
            ValueError: If the configured preprocessing steps don't produce a
                train/test split (e.g. no Splitter node configured).
        """
        transformed_data, _ = self.feature_engineer.fit_transform(data)

        if not isinstance(transformed_data, SplitDataset):
            raise ValueError(
                "get_fitted_split() requires the configured preprocessing steps "
                "to produce a train/test split (e.g. via a Splitter node); got "
                "a single, unsplit DataFrame instead."
            )

        X_train, y_train = extract_xy(transformed_data.train, target_column)
        X_test, y_test = extract_xy(transformed_data.test, target_column)

        return (
            _to_pandas(X_train),
            _to_pandas(y_train),
            _to_pandas(X_test),
            _to_pandas(y_test),
        )

    def _predict_proba_transformed(self, transformed_data: pd.DataFrame | SkyulfDataFrame) -> Any:
        """Run predict_proba on already-transformed data, raising if unsupported."""
        if self.model_estimator is None or self.model_estimator.model is None:
            raise ValueError("Pipeline not fitted or no model configured.")
        proba = self.model_estimator.applier.predict_proba(
            transformed_data, self.model_estimator.model
        )
        if proba is None:
            raise ValueError(
                "The configured model does not support predict_proba(); "
                "threshold tuning requires predicted class probabilities."
            )
        return proba

    def optimize_thresholds(
        self,
        X_val: pd.DataFrame | SkyulfDataFrame,
        y_val: pd.Series | Any,
        metric: Callable[[Any, Any], float],
        strategy: str | None = None,
        grid_points: int = 101,
    ) -> dict[Any, float]:
        """
        Search for per-class decision thresholds that maximize ``metric`` on
        caller-supplied validation data, and store the result for later use
        by ``predict(use_tuned_thresholds=True)``.

        Always uses the *explicit* ``(X_val, y_val)`` the caller passes in —
        never the pipeline's internal train/test split. Get a clean,
        independent holdout via ``get_fitted_split()`` (or your own split)
        before calling this, the same way you would for any other
        out-of-sample evaluation.

        Args:
            X_val: Validation features, *not* yet transformed (this method
                runs the pipeline's fitted preprocessing on it internally).
            y_val: Validation true labels.
            metric: Callable ``(y_true, y_pred) -> float`` to maximize.
            strategy: ``"grid"`` or ``"nelder-mead"``. If ``None``,
                auto-selects based on the number of classes (see
                ``skyulf.modeling.optimize_thresholds``).
            grid_points: Number of grid candidates for the ``"grid"``
                strategy.

        Returns:
            Dict mapping each class label to its tuned threshold. Also
            stored on ``self._tuned_thresholds`` for
            ``predict(use_tuned_thresholds=True)`` to use.

        Raises:
            ValueError: If the pipeline isn't fitted, or the underlying
                model doesn't support ``predict_proba``.
        """
        if self.model_estimator is None or self.model_estimator.model is None:
            raise ValueError(
                "Pipeline not fitted or no model configured. Call fit() before "
                "optimize_thresholds()."
            )

        model = self.model_estimator.model
        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            raise ValueError(
                "The fitted model does not expose class labels (classes_); "
                "threshold tuning requires a classifier."
            )

        transformed_val = self.feature_engineer.transform(X_val)
        proba_df = self._predict_proba_transformed(transformed_val)
        classes = np.asarray(model_classes)
        y_proba = np.asarray(proba_df)[:, : len(classes)]

        thresholds = optimize_thresholds(
            y_val,
            y_proba,
            metric=metric,
            classes=classes,
            strategy=strategy,
            grid_points=grid_points,
        )
        self._tuned_thresholds = thresholds
        return thresholds

    def predict(
        self,
        data: pd.DataFrame | SkyulfDataFrame,
        use_tuned_thresholds: bool = False,
    ) -> Any:
        """
        Generate predictions.

        Args:
            data: Input DataFrame.
            use_tuned_thresholds: If True, apply the decision thresholds
                stored by a prior ``optimize_thresholds()`` call instead of
                the model's default decision rule (argmax/0.5). Requires
                ``optimize_thresholds()`` to have been called on this
                pipeline instance first.

        Returns:
            Series (or array, when ``use_tuned_thresholds=True``) of
            predictions.

        Raises:
            ValueError: If the input still contains the target column used
                during fit(); if the pipeline isn't fitted; or if
                ``use_tuned_thresholds=True`` but ``optimize_thresholds()``
                was never called on this instance.
        """
        if self._target_column is not None and self._target_column in data.columns:
            raise ValueError(
                f"predict() input still contains the target column '{self._target_column}' "
                "used during fit(); drop it before calling predict()."
            )

        # 1. Feature Engineering (Transform only)
        transformed_data = self.feature_engineer.transform(data)

        # 2. Modeling
        if not (self.model_estimator and self.model_estimator.model is not None):
            raise ValueError("Pipeline not fitted or no model configured.")

        if not use_tuned_thresholds:
            return self.model_estimator.applier.predict(
                transformed_data, self.model_estimator.model
            )

        if self._tuned_thresholds is None:
            raise ValueError(
                "use_tuned_thresholds=True but optimize_thresholds() was never "
                "called on this pipeline instance. Call optimize_thresholds() first."
            )

        proba_df = self._predict_proba_transformed(transformed_data)
        classes = np.asarray(self.model_estimator.model.classes_)
        y_proba = np.asarray(proba_df)[:, : len(classes)]
        return apply_thresholds(y_proba, self._tuned_thresholds, classes=classes)

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

    def validate_leakage_safety(self, on_leakage: OnLeakage = "raise") -> list[str]:
        """Diagnose preprocessing steps ordered before the train/test split."""
        return validate_leakage_safety(self.config, on_leakage=on_leakage)

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
