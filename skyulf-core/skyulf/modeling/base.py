import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

# Use relative imports assuming the structure is preserved
from ..data.dataset import SplitDataset
from ..engines import EngineName, SkyulfDataFrame, get_engine

# Evaluation imports - we will migrate these next
# from ._evaluation.schemas import ModelEvaluationReport, ModelEvaluationSplitPayload
# from ._evaluation.classification import build_classification_split_report
# from ._evaluation.regression import build_regression_split_report

logger = logging.getLogger(__name__)


class BaseModelCalculator(ABC):
    @property
    @abstractmethod
    def problem_type(self) -> str:
        """Returns 'classification', 'regression', or 'clustering'."""

    @property
    def default_params(self) -> dict[str, Any]:
        """Default hyperparameters for the model."""
        return {}

    def prepare_tuning_params(self, config: dict[str, Any]) -> None:
        """Hook for structural models (e.g. ensembles) to absorb their
        sub-estimator selection before the tuner builds the base model.

        No-op for plain models. Ensembles override this to inject the resolved
        ``estimators`` (and ``final_estimator``) into :attr:`default_params` so
        the tuner can construct a valid meta-estimator.
        """
        return None

    def build_tuning_search_space(self, config: dict[str, Any], strategy: str) -> dict[str, Any]:
        """Hook: let a model auto-build its tuning search space.

        Returns an empty dict for plain models (the caller keeps the
        user-provided space). Ensembles override this to expand their base
        learners' parameter grids into nested ``<name>__<param>`` keys.
        """
        return {}

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | SkyulfDataFrame,
        y: pd.Series | Any,
        config: dict[str, Any],
        progress_callback: Callable[..., None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        validation_data: tuple[pd.DataFrame | SkyulfDataFrame, pd.Series | Any] | None = None,
    ) -> Any:
        """Trains the model and returns the fitted model artifact.

        The return type is intentionally `Any` rather than a narrower
        TypeVar/Protocol: most calculators (see `sklearn_wrapper.py`) return a
        single fitted estimator, but `TuningCalculator`
        (`_tuning/engine.py::fit`) returns a `(model, tuning_result)` tuple
        instead — the artifact shape is model-family-dependent, not just
        heterogeneous across libraries (sklearn estimator, xgboost booster,
        custom wrapper) but also heterogeneous *within* a single calculator
        depending on whether tuning was applied. Consumers already
        `isinstance(self.model, tuple)`-narrow where needed (see
        `StatefulEstimator.evaluate`); a forced union type here wouldn't
        remove that narrowing, so `Any` is the honest, pragmatic choice.
        """


class BaseModelApplier(ABC):
    @abstractmethod
    def predict(self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any) -> pd.Series | Any:
        """
        Generates predictions.
        """

    def predict_proba(
        self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any
    ) -> pd.DataFrame | SkyulfDataFrame | None:
        """
        Generates prediction probabilities if supported.
        Returns DataFrame where columns are classes.
        """
        return None


class StatefulEstimator:
    def __init__(self, calculator: BaseModelCalculator, applier: BaseModelApplier, node_id: str):
        self.calculator = calculator
        self.applier = applier
        self.node_id = node_id
        self.model = None  # In-memory model storage

    @staticmethod
    def _is_non_empty_split(data: Any) -> bool:
        """Engine-agnostic non-empty check for a dataset split.

        Handles pandas (`.empty`), polars/Skyulf wrappers (`.is_empty()`),
        and (X, y) tuples - previously only pandas DataFrames and tuples
        were recognized, so a bare polars DataFrame split (test/validation)
        was silently treated as absent.
        """
        if data is None:
            return False
        if isinstance(data, tuple):
            return len(data) == 2 and data[0] is not None and len(data[0]) > 0
        if hasattr(data, "empty"):
            return not data.empty
        if hasattr(data, "is_empty"):
            return not data.is_empty()
        try:
            return len(data) > 0
        except TypeError:
            return False

    def _extract_xy(self, data: Any, target_column: str) -> tuple[Any, Any]:
        """Helper to extract X and y from DataFrame or Tuple.

        An empty/falsy ``target_column`` is the established "no target"
        sentinel already used elsewhere in this codebase (see
        ``_node_runners.py``'s ``target_col=""`` for data-preview-only
        inputs). Unsupervised calculators (e.g. clustering) rely on this to
        get the whole frame back as ``X`` with ``y=None``, without touching
        the existing "raise on missing target" contract that classification/
        regression depend on below.
        """
        if not target_column:
            X = data[0] if isinstance(data, tuple) else data
            return X, None

        if isinstance(data, tuple) and len(data) == 2:
            return self._extract_xy_from_tuple(data, target_column)

        engine = get_engine(data)

        if engine.name == EngineName.POLARS:
            return self._extract_xy_polars(data, target_column)

        return self._extract_xy_pandas_like(data, target_column)

    def _extract_xy_from_tuple(self, data: tuple[Any, Any], target_column: str) -> tuple[Any, Any]:
        """Extracts X/y from a ``(X, y)`` tuple, pulling ``y`` out of ``X`` if it's missing."""
        X, y = data[0], data[1]
        # If y is None but X is a DataFrame containing the target, extract it
        if y is None and hasattr(X, "columns") and target_column in X.columns:
            return self._extract_xy(X, target_column)
        return X, y

    @staticmethod
    def _extract_xy_polars(data: Any, target_column: str) -> tuple[Any, Any]:
        """Extracts X/y from a Polars DataFrame by dropping/selecting ``target_column``."""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        X = data.drop([target_column])
        y = data.select(target_column).to_series()
        return X, y

    @staticmethod
    def _extract_xy_pandas_like(data: Any, target_column: str) -> tuple[Any, Any]:
        """Extracts X/y from a pandas or generic DataFrame-like object."""
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
        config: dict[str, Any],
        n_folds: int = 5,
        cv_type: str = "k_fold",
        shuffle: bool = True,
        random_state: int = 42,
        time_column: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
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
            time_column=time_column,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    @staticmethod
    def _drop_target_column(data: Any, target_column: str) -> Any:
        """Drop target_column from data, handling pandas (kwarg) and Polars (list-arg) APIs."""
        try:
            return data.drop(columns=[target_column])
        except TypeError:
            # Polars
            return data.drop([target_column])

    def _extract_split_features(self, split_data: Any, target_column: str) -> Any:
        """Extract the feature matrix from a test/validation split, dropping the target if present.

        Handles both the ``(X, y)`` tuple form and the plain DataFrame form
        (pandas or Polars), so the same logic can be reused for the test and
        validation splits of ``fit_predict``.
        """
        if isinstance(split_data, tuple):
            X, y_split = split_data
            X = cast(Any, X)
            # If y is None, the target may still be in X — drop it
            if y_split is None and hasattr(X, "columns") and target_column in X.columns:
                X = self._drop_target_column(X, target_column)
            return X

        if target_column in split_data.columns:
            return self._drop_target_column(split_data, target_column)
        return split_data

    def _normalize_fit_predict_dataset(
        self,
        dataset: SplitDataset
        | pd.DataFrame
        | tuple[pd.DataFrame, pd.Series]
        | tuple[pd.DataFrame, pd.DataFrame],
        target_column: str,
        log_callback: Callable[[str], None] | None,
    ) -> SplitDataset:
        """Wrap raw DataFrame/tuple ``fit_predict`` input into a SplitDataset."""
        if isinstance(dataset, pd.DataFrame):
            return SplitDataset(train=dataset, test=pd.DataFrame(), validation=None)

        if isinstance(dataset, tuple):
            # Check if it's (train_df, test_df) or (X, y)
            elem0 = dataset[0]
            if isinstance(elem0, pd.DataFrame) and target_column in elem0.columns:
                # It's (train_df, test_df)
                train_df, test_df = dataset
                return SplitDataset(train=train_df, test=test_df, validation=None)  # type: ignore

            # Fallback: Treat input as training data (e.g. X, y tuple) and initialize empty test set.
            msg = (
                "WARNING: No test set provided. Using entire input as training data. "
                "Ensure data was split BEFORE preprocessing to avoid data leakage."
            )
            logger.warning(msg)
            if log_callback:
                log_callback(msg)

            return SplitDataset(train=cast(Any, dataset), test=pd.DataFrame(), validation=None)

        return dataset

    def fit_predict(
        self,
        dataset: SplitDataset
        | pd.DataFrame
        | tuple[pd.DataFrame, pd.Series]
        | tuple[pd.DataFrame, pd.DataFrame],
        target_column: str,
        config: dict[str, Any],
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        job_id: str = "unknown",
    ) -> dict[str, pd.Series]:
        """
        Fits the model on training data and returns predictions for all splits.
        """
        # Handle raw DataFrame or Tuple input by wrapping it in a dummy SplitDataset
        dataset = self._normalize_fit_predict_dataset(dataset, target_column, log_callback)

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
        test_df = dataset.test[0] if isinstance(dataset.test, tuple) else dataset.test
        # is_test_empty: pandas uses `.empty`, Polars uses `.is_empty()`
        is_test_empty = test_df.empty if hasattr(test_df, "empty") else test_df.is_empty()

        if not is_test_empty:
            X_test = self._extract_split_features(dataset.test, target_column)
            predictions["test"] = self.applier.predict(X_test, self.model)

        # Validation Predictions
        if dataset.validation is not None:
            X_val = self._extract_split_features(dataset.validation, target_column)
            predictions["validation"] = self.applier.predict(X_val, self.model)

        return predictions

    def refit(
        self,
        dataset: SplitDataset,
        target_column: str,
        config: dict[str, Any],
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

        # y_train/y_val are None for unsupervised calculators (e.g. clustering,
        # see `_extract_xy`'s "no target" sentinel) — skip the y-concat rather
        # than crash trying to concatenate `None`.
        if get_engine(X_train).name == EngineName.POLARS:
            import polars as pl

            X_combined = pl.concat([X_train, X_val])
            y_combined = None if y_train is None else pl.concat([y_train, y_val])
        else:
            X_combined = pd.concat([X_train, X_val], axis=0)
            y_combined = None if y_train is None else pd.concat([y_train, y_val], axis=0)

        # 2. Train Model
        self.model = self.calculator.fit(X_combined, y_combined, config)

    def evaluate(
        self,
        dataset: SplitDataset,
        target_column: str,
        job_id: str = "unknown",
        reference_column: str = "",
    ) -> Any:
        """
        Evaluates the model on all splits and returns a detailed report.

        ``reference_column`` is clustering-only: an optional column (e.g. a
        known label like species name) excluded from training features but
        used here purely to build a post-hoc cluster/label breakdown.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit_predict() first.")

        problem_type = self.calculator.problem_type

        splits_payload = {}

        # Container for raw predictions
        evaluation_data: dict[str, Any] = {
            "job_id": job_id,
            "node_id": self.node_id,
            "problem_type": problem_type,
            "splits": {},
        }

        # 2. Evaluate Train
        splits_payload["train"] = self._evaluate_split(
            "train", dataset.train, target_column, problem_type, evaluation_data, reference_column
        )

        # 3. Evaluate Test
        has_test = self._is_non_empty_split(dataset.test)

        if has_test:
            splits_payload["test"] = self._evaluate_split(
                "test", dataset.test, target_column, problem_type, evaluation_data, reference_column
            )

        # 4. Evaluate Validation
        if dataset.validation is not None:
            has_val = self._is_non_empty_split(dataset.validation)

            if has_val:
                splits_payload["validation"] = self._evaluate_split(
                    "validation",
                    dataset.validation,
                    target_column,
                    problem_type,
                    evaluation_data,
                    reference_column,
                )

        # Return report object (simplified for now, assuming schema matches)
        return {
            "problem_type": problem_type,
            "splits": splits_payload,
            "raw_data": evaluation_data,
        }

    def _evaluate_split(
        self,
        split_name: str,
        data: Any,
        target_column: str,
        problem_type: str,
        evaluation_data: dict[str, Any],
        reference_column: str = "",
    ) -> Any:
        """Evaluates a single dataset split, recording raw predictions into ``evaluation_data``
        and returning the split's evaluation report (or ``None`` if it can't be evaluated).
        """
        # Delegate to the same engine-agnostic (pandas/polars/tuple) X/y
        # extraction used by fit_predict, instead of duplicating
        # ad-hoc pandas-only logic that silently dropped polars splits.
        try:
            X, y = self._extract_xy(data, target_column)
        except ValueError:
            return None  # Cannot evaluate without target
        if X is None:
            return None
        if problem_type != "clustering" and y is None:
            return None

        y_pred = self.applier.predict(X, self.model)
        model_to_evaluate = self._unwrap_tuned_model()

        if problem_type == "clustering":
            # Unsupervised: there is no y_true, only the cluster label
            # assigned to each row. KMeans genuinely supports out-of-sample
            # `.predict()`, so (unlike DBSCAN/Agglomerative) evaluating each
            # split independently with its own predicted labels is valid.
            split_report = self._evaluate_split_with_model(
                model_to_evaluate, split_name, X, y_pred, problem_type, reference_column
            )
            evaluation_data["splits"][split_name] = self._build_clustering_split_raw_data(
                y_pred, split_report
            )
            return split_report

        y_proba = self._predict_proba_payload(X, problem_type)
        evaluation_data["splits"][split_name] = self._build_split_raw_data(y, y_pred, y_proba)

        return self._evaluate_split_with_model(model_to_evaluate, split_name, X, y, problem_type)

    @staticmethod
    def _build_split_raw_data(
        y: Any, y_pred: Any, y_proba: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Builds the raw ``y_true``/``y_pred``/(optional) ``y_proba`` payload for a split."""
        split_data = {
            "y_true": y.tolist() if hasattr(y, "tolist") else list(y),
            "y_pred": (y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)),
        }
        if y_proba:
            split_data["y_proba"] = y_proba
        return split_data

    @staticmethod
    def _build_clustering_split_raw_data(labels: Any, split_report: Any = None) -> dict[str, Any]:
        """Builds the raw ``labels`` (+ clustering summary/metrics) payload for a clustering split.

        ``split_report`` is the ``ModelEvaluationReport`` for this split, if evaluation
        succeeded; its ``clustering`` field (cluster sizes/centroids) and quality
        ``metrics`` (silhouette/Calinski-Harabasz/Davies-Bouldin) are embedded so the
        API doesn't need a second round-trip to expose them.
        """
        raw: dict[str, Any] = {
            "labels": labels.tolist() if hasattr(labels, "tolist") else list(labels)
        }
        if split_report is not None:
            clustering = getattr(split_report, "clustering", None)
            if clustering is not None:
                raw["clustering"] = clustering.model_dump()
            metrics = getattr(split_report, "metrics", None)
            if metrics is not None:
                raw["metrics"] = dict(metrics)
        return raw

    def _unwrap_tuned_model(self) -> Any:
        """Unpacks ``self.model`` if it's a ``(model, ...)`` tuple, as produced by the Tuner."""
        # Check if first element looks like a model (has fit/predict)
        # or if it's just a convention from TuningCalculator
        if isinstance(self.model, tuple) and len(self.model) == 2:
            return self.model[0]
        return self.model

    def _predict_proba_payload(self, X: Any, problem_type: str) -> dict[str, Any] | None:
        """Returns the ``{"classes", "values"}`` probability payload for classification splits."""
        if problem_type != "classification":
            return None
        y_proba_df = self.applier.predict_proba(X, self.model)
        if y_proba_df is None:
            return None
        return {
            "classes": y_proba_df.columns.tolist(),
            "values": y_proba_df.values.tolist(),
        }

    @staticmethod
    def _evaluate_split_with_model(
        model_to_evaluate: Any,
        split_name: str,
        X: Any,
        y: Any,
        problem_type: str,
        reference_column: str = "",
    ) -> Any:
        """Dispatches to the classification, regression, or clustering evaluator.

        For clustering, ``y`` is the *predicted* cluster labels for this split
        (there is no ground-truth target), computed by the caller via
        ``self.applier.predict(X, self.model)``.
        """
        # Import here to avoid circular dependency
        from ._evaluation.classification import evaluate_classification_model
        from ._evaluation.clustering import evaluate_clustering_model
        from ._evaluation.regression import evaluate_regression_model

        if problem_type == "classification":
            return evaluate_classification_model(
                model=model_to_evaluate, dataset_name=split_name, X_test=X, y_test=y
            )
        elif problem_type == "regression":
            return evaluate_regression_model(
                model=model_to_evaluate, dataset_name=split_name, X_test=X, y_test=y
            )
        elif problem_type == "clustering":
            return evaluate_clustering_model(
                model=model_to_evaluate,
                X=X,
                labels=y,
                dataset_name=split_name,
                reference_column=reference_column,
            )
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
