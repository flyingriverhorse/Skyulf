"""Main Skyulf Pipeline."""

import hashlib
import json
import logging
import pickle  # nosec B403 - used only for internal pipeline serialization (see save/load below)
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from ..config_validation import validate_pipeline_config
from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame, get_engine
from ..leakage import OnLeakage, validate_leakage_safety
from ..modeling._evaluation.thresholds import apply_thresholds, optimize_thresholds
from ..modeling._tuning.engine import TuningApplier, TuningCalculator
from ..modeling._tuning.refit import tune_decision_thresholds
from ..modeling.base import BaseModelApplier, BaseModelCalculator, StatefulEstimator, extract_xy
from ..preprocessing.base import BaseApplier, apply_method
from ..preprocessing.fold_adapter import FeatureEngineerFoldAdapter
from ..preprocessing.pipeline import FeatureEngineer
from ..registry import NodeRegistry
from ..types import PipelineConfig
from .diagram import build_mermaid_diagram, mermaid_markdown
from .seal import artifact_digest

logger = logging.getLogger(__name__)


def _to_pandas(obj: Any) -> Any:
    """Convert a Polars DataFrame/Series to its pandas equivalent.

    Also converts any object exposing ``to_pandas()``; pandas objects (or
    ``None``) pass through unchanged.
    """
    if obj is None:
        return None
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return obj


class _PipelineTuningPreprocessor(FeatureEngineerFoldAdapter):
    """Retain the final fold fit's engineer, training representation, and metrics."""

    def __init__(self, steps_config: list[dict[str, Any]], target_column: str):
        """Validate the fold chain and initialize the final-fit observations."""
        super().__init__(steps_config, target_column)
        self.training_payload: Any = None
        self.metrics: dict[str, Any] = {}
        self.input_columns: list[str] = []

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Fit a fresh chain and retain its exact OOF or row-changing training output."""
        self._validate_payload(X)
        engineer = FeatureEngineer(self._steps_config)
        transformed, metrics = engineer.fit_transform((X, y), target_column=self._target_column)
        self._engineer = engineer
        self.training_payload = transformed
        self.metrics = metrics
        self.input_columns = list(X.columns)
        return transformed


class _TuningColumnDropApplier(BaseApplier):
    """Preserve the tuner's removal of its time-ordering column during serving."""

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Drop only the recorded sorting columns, preserving row and target alignment."""
        for column in params["columns"]:
            if column in X.columns:
                X = StatefulEstimator._drop_target_column(X, column)
        return X, _y


def _merge_preprocessing_metrics(
    prefix: dict[str, Any], suffix: dict[str, Any], prefix_length: int
) -> dict[str, Any]:
    """Combine the split prefix and final training fit without replaying preprocessing."""
    steps = dict(prefix["steps"])
    for key, value in suffix["steps"].items():
        index, _, name = key.partition(":")
        steps[f"{prefix_length + int(index)}:{name}"] = value
    before, after = prefix["summary"], suffix["summary"]
    summary = {
        "fit_time": before["fit_time"] + after["fit_time"],
        "peak_memory_bytes": max(before["peak_memory_bytes"], after["peak_memory_bytes"]),
        "rows_in": before["rows_in"] if prefix_length else after["rows_in"],
        "rows_out": after["rows_out"] if suffix["steps"] else before["rows_out"],
    }
    return {"summary": summary, "steps": steps, **summary}


class SkyulfPipeline:
    """End-to-end ML Pipeline.

    Encapsulates:
    1. Feature Engineering (Preprocessing)
    2. Modeling (Training/Inference)

    Examples:
        >>> pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {}})
        >>> metrics = pipeline.fit(data, target_column="target")
    """

    def __init__(self, config: PipelineConfig | dict[str, Any]):
        """Initialize the pipeline.

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

    def _fit_tuning_pipeline(
        self, data: Any, target_column: str
    ) -> tuple[SplitDataset, dict[str, Any]]:
        """Tune from raw outer partitions and adopt the final training preprocessor."""
        prefix_length = 0
        if not isinstance(data, SplitDataset):
            prefix_length = next(
                (
                    index + 1
                    for index, step in enumerate(self.preprocessing_steps)
                    if step["transformer"] in {"TrainTestSplitter", "Split"}
                ),
                0,
            )
        prefix = FeatureEngineer(self.preprocessing_steps[:prefix_length], _validated=True)
        raw_data, prefix_metrics = prefix.fit_transform(data, target_column=target_column)
        if isinstance(raw_data, SplitDataset):
            raw_dataset = raw_data
        else:
            raw_frame = raw_data[0] if isinstance(raw_data, tuple) else raw_data
            raw_dataset = SplitDataset(
                train=raw_data, test=get_engine(raw_frame).create_dataframe({}), validation=None
            )

        raw_train = extract_xy(raw_dataset.train, target_column)
        raw_validation = (
            extract_xy(raw_dataset.validation, target_column)
            if StatefulEstimator._is_non_empty_split(raw_dataset.validation)
            else None
        )
        adapter = _PipelineTuningPreprocessor(
            cast(list[dict[str, Any]], self.preprocessing_steps[prefix_length:]), target_column
        )
        estimator = self.model_estimator
        if estimator is None or not isinstance(estimator.calculator, TuningCalculator):
            raise RuntimeError("The tuning pipeline requires a tuning calculator.")
        calculator = estimator.calculator
        tuning_config = calculator._build_tuning_config(cast(dict[str, Any], self.modeling_config))
        estimator.model = calculator.fit(
            raw_train[0],
            raw_train[1],
            replace(tuning_config, tune_threshold=False),
            preprocessing=adapter,
            validation_data=raw_validation,
            validation_frames=raw_validation,
        )
        if adapter._engineer is None or adapter.training_payload is None:
            raise RuntimeError("The tuner did not produce fitted preprocessing.")

        # Time-series tuning removes its explicit or auto-detected time column
        # before fitting. Record that operation ahead of the fitted fold chain
        # so evaluation and serving see the same feature space, without sorting
        # new prediction requests or changing their row order.
        removed_columns = [c for c in raw_train[0].columns if c not in adapter.input_columns]
        if removed_columns:
            adapter._engineer.fitted_steps.insert(
                0,
                {
                    "name": "tuning_time_columns",
                    "type": "TuningColumnDrop",
                    "applier": _TuningColumnDropApplier(),
                    "artifact": {"columns": removed_columns},
                },
            )
        self.feature_engineer.fitted_steps = prefix.fitted_steps + adapter._engineer.fitted_steps
        transformed = SplitDataset(
            train=adapter.training_payload,
            test=self._transform_tuning_split(adapter, raw_dataset.test, target_column),
            validation=self._transform_tuning_split(adapter, raw_dataset.validation, target_column),
        )
        if tuning_config.tune_threshold:
            model, tuning_result = estimator.model
            tune_decision_thresholds(
                calculator.model_calculator,
                model,
                tuning_result,
                tuning_config,
                extract_xy(transformed.validation, target_column)
                if StatefulEstimator._is_non_empty_split(transformed.validation)
                else None,
                None,
            )
        return transformed, _merge_preprocessing_metrics(
            prefix_metrics, adapter.metrics, prefix_length
        )

    @staticmethod
    def _transform_tuning_split(
        adapter: _PipelineTuningPreprocessor, payload: Any, target_column: str
    ) -> Any:
        """Transform one raw held-out partition with the final fitted fold chain."""
        if not StatefulEstimator._is_non_empty_split(payload):
            return None if payload is None else payload
        X, y = extract_xy(payload, target_column)
        return adapter.transform(X, y)

    def fit(
        self,
        data: pd.DataFrame | pl.DataFrame | SkyulfDataFrame | SplitDataset,
        target_column: str,
        *,
        on_leakage: OnLeakage = "raise",
    ) -> dict[str, Any]:
        """Fit the pipeline.

        Once fitting starts, any failure invalidates the fitted model and
        preprocessing. Call ``fit()`` successfully again before predicting.

        Args:
            data: Input data (DataFrame or SplitDataset).
            target_column: Name of the target column.
            on_leakage: Reject definite leakage by default. Use "warn" or
                "ignore" only to explicitly allow unsafe preprocessing.

        Returns:
            Dictionary containing execution metrics.

        Raises:
            ValueError: If learned preprocessing precedes a train/test split
                with ``on_leakage="raise"``, or the mode is invalid.
        """
        # Check before any transformer or estimator can learn from the data.
        for warning in validate_leakage_safety(
            self.config,
            on_leakage=on_leakage,
            target_column=target_column,
            already_split=isinstance(data, SplitDataset),
        ):
            logger.warning(warning)

        self._fit_metrics = None
        self._target_column = None
        self._tuned_thresholds = None
        if self.model_estimator is not None:
            self.model_estimator.model = None
        try:
            return self._fit(data, target_column)
        except BaseException:
            # A failure can occur after model fitting, while transforming or
            # predicting held-out data. Never expose that partial replacement.
            self.feature_engineer.fitted_steps = []
            if self.model_estimator is not None:
                self.model_estimator.model = None
            raise

    def _fit(
        self,
        data: pd.DataFrame | pl.DataFrame | SkyulfDataFrame | SplitDataset,
        target_column: str,
    ) -> dict[str, Any]:
        """Fit validated data, publishing metadata only after all stages complete."""
        metrics = {}

        # 1. Feature Engineering
        logger.info("Starting Feature Engineering...")
        is_tuning = self.model_estimator is not None and isinstance(
            self.model_estimator.calculator, TuningCalculator
        )
        if is_tuning:
            transformed_data, fe_metrics = self._fit_tuning_pipeline(data, target_column)
        else:
            transformed_data, fe_metrics = self.feature_engineer.fit_transform(
                data, target_column=target_column
            )
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
            if not is_tuning:
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
        *,
        on_leakage: OnLeakage = "raise",
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Run this pipeline's configured preprocessing chain and return the split.

        The result is the train/test split as plain pandas objects. Runs a
        **throwaway** ``FeatureEngineer`` over the same configured steps — the
        same preprocessing ``fit()`` uses internally — and extracts
        ``(X_train, y_train, X_test, y_test)`` from the resulting split using
        ``target_column``, converting any Polars/SkyulfDataFrame frames to
        pandas. Saves callers from re-implementing this split/convert step
        themselves for custom evaluation harnesses (e.g. comparing multiple
        raw sklearn-style estimators against the same preprocessed split).

        This pipeline's own fitted preprocessing is deliberately left alone.
        Refitting it in place would swap the statistics a model already trained
        against, leaving ``predict()`` silently transforming inputs with a
        scaler the model never saw — so predictions must be identical before
        and after this call.

        Args:
            data: Input data (DataFrame or SplitDataset).
            target_column: Name of the target column.
            on_leakage: Reject definite leakage by default, as in ``fit()``.
                "warn" and "ignore" explicitly allow unsafe preprocessing.

        Returns:
            ``(X_train, y_train, X_test, y_test)`` as pandas DataFrame/Series.
            The frames are **already preprocessed** — hand them to a raw
            sklearn-style estimator, not back into this pipeline's
            ``optimize_thresholds()`` or ``predict()``, which run the fitted
            preprocessing on their own input and would transform these a
            second time.

        Raises:
            ValueError: If the configured preprocessing steps don't produce a
                train/test split, violate the selected leakage policy, or use
                an invalid leakage mode.
        """
        for warning in validate_leakage_safety(
            self.config,
            on_leakage=on_leakage,
            target_column=target_column,
            already_split=isinstance(data, SplitDataset),
        ):
            logger.warning(warning)

        transformed_data, _ = FeatureEngineer(
            self.preprocessing_steps, _validated=True
        ).fit_transform(data, target_column=target_column)

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
        """Search for per-class decision thresholds that maximize ``metric``.

        Runs on caller-supplied validation data and stores the result for later
        use by ``predict(use_tuned_thresholds=True)``. Always uses the
        *explicit* ``(X_val, y_val)`` the caller passes in — never the
        pipeline's internal train/test split. Carve an independent holdout out
        of the **raw** data before ``fit()`` and pass its rows here, the same
        way you would for any other out-of-sample evaluation.

        ``X_val`` must be raw. This method runs the pipeline's fitted
        preprocessing on it exactly once, which is what makes the probabilities
        it tunes against the ones ``predict()`` later reproduces.
        Validation labels follow the same row sorting and filtering as the
        features, so the metric scores matching observations.
        ``get_fitted_split()`` is therefore *not* a source for it: that helper
        returns already-preprocessed frames, so passing them here transforms
        the holdout a second time and fits the cutoffs against a distribution
        inference never sees.

        Args:
            X_val: Validation features, *not* yet transformed (this method
                runs the pipeline's fitted preprocessing on it internally).
            y_val: Validation true labels, paired positionally with ``X_val``.
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
                model doesn't support ``predict_proba``, or the raw validation
                features and labels have different row counts.
        """
        if self.model_estimator is None or self.model_estimator.model is None:
            raise ValueError(
                "Pipeline not fitted or no model configured. Call fit() before "
                "optimize_thresholds()."
            )

        model = self.model_estimator._unwrap_tuned_model()
        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            raise ValueError(
                "The fitted model does not expose class labels (classes_); "
                "threshold tuning requires a classifier."
            )

        if len(X_val) != len(y_val):
            raise ValueError(
                "X_val and y_val must contain the same number of rows before preprocessing."
            )

        # Array-like labels may come from a different engine than the features.
        transformed_val, transformed_y = self.feature_engineer.transform((X_val, np.asarray(y_val)))
        proba_df = self._predict_proba_transformed(transformed_val)
        classes = np.asarray(model_classes)
        y_proba = np.asarray(proba_df)[:, : len(classes)]

        thresholds = optimize_thresholds(
            transformed_y,
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
        """Generate predictions.

        Args:
            data: Input DataFrame.
            use_tuned_thresholds: If True, apply the decision thresholds
                stored by a prior ``optimize_thresholds()`` call instead of
                the model's default decision rule (argmax/0.5). Requires
                ``optimize_thresholds()`` to have been called on this
                pipeline instance since its most recent fit.

        Returns:
            Series (or array, when ``use_tuned_thresholds=True``) of
            predictions.

        Raises:
            ValueError: If the input still contains the target column used
                during fit(); if the pipeline isn't fitted; or if
                ``use_tuned_thresholds=True`` but ``optimize_thresholds()``
                has not been called since the most recent fit.
        """
        if not (self.model_estimator and self.model_estimator.model is not None):
            raise ValueError("Pipeline not fitted or no model configured.")

        if self._target_column is not None and self._target_column in data.columns:
            raise ValueError(
                f"predict() input still contains the target column '{self._target_column}' "
                "used during fit(); drop it before calling predict()."
            )

        # 1. Feature Engineering (Transform only)
        transformed_data = self.feature_engineer.transform(data)

        # 2. Modeling
        if not use_tuned_thresholds:
            return self.model_estimator.applier.predict(
                transformed_data, self.model_estimator.model
            )

        if self._tuned_thresholds is None:
            raise ValueError(
                "use_tuned_thresholds=True but no decision thresholds are available "
                "for the current fit. Call optimize_thresholds() first."
            )

        proba_df = self._predict_proba_transformed(transformed_data)
        model = self.model_estimator._unwrap_tuned_model()
        classes = np.asarray(model.classes_)
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

    def validate_leakage_safety(
        self, on_leakage: OnLeakage = "raise", *, target_column: str | None = None
    ) -> list[str]:
        """Diagnose preprocessing steps ordered before the train/test split."""
        return validate_leakage_safety(
            self.config, on_leakage=on_leakage, target_column=target_column
        )

    def to_mermaid(self) -> str:
        """Render the pipeline as a Mermaid ``flowchart`` string.

        Produces a top-down graph ``data -> [preprocessing steps] -> model``.
        Useful in docs and PR descriptions. Pure read-only over ``self.config``.
        """
        return build_mermaid_diagram(self.preprocessing_steps, self.modeling_config)

    def to_mermaid_markdown(self, heading: str | None = "Pipeline topology") -> str:
        """Return the diagram as a Markdown snippet with a ``mermaid`` fence.

        Includes a heading by default; pass ``heading=None`` for just the
        fenced block. Renders natively on GitHub, in VS Code previews, and
        in Jupyter markdown cells.
        """
        block = mermaid_markdown(self.to_mermaid())
        if heading is None:
            return block
        return f"# {heading}\n\n{block}"

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
        "this prediction came from exactly this pipeline". The digest is
        semantic (hyperparameters + fitted weights, not pickle bytes), so it is
        stable across library and pickle-protocol versions.
        """
        hasher = hashlib.sha256()
        topology = {
            "preprocessing": self.preprocessing_steps,
            "modeling": self.modeling_config,
        }
        hasher.update(json.dumps(topology, sort_keys=True, default=str).encode("utf-8"))

        for step in self.feature_engineer.fitted_steps:
            hasher.update(artifact_digest(step.get("artifact")))

        if self.model_estimator is not None and self.model_estimator.model is not None:
            hasher.update(artifact_digest(self.model_estimator.model))

        return hasher.hexdigest()

    def export_model_card(self) -> dict[str, Any]:
        """Return a structured, JSON-friendly summary of the pipeline.

        Captures lineage (preprocessing chain), the model and its hyperparameters,
        the reproducibility fingerprint, the metrics from the last :meth:`fit`
        (``None`` if never fitted), and a Mermaid ``flowchart`` of the topology
        under ``"diagram"``. Intended for audit logs and model registries.
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
            "diagram": self.to_mermaid(),
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
