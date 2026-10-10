"""Contracts shared by every modeling node.

Defines :class:`BaseModelCalculator` and :class:`BaseModelApplier`, the
fit/predict halves each model node implements; :func:`extract_xy` and its
engine-specific helpers, the pandas/Polars/tuple ``(X, y)`` extraction every
model node shares; and :class:`StatefulEstimator`, which binds a
calculator/applier pair and owns the in-memory fitted model across fit,
predict, cross-validation and evaluation.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

import pandas as pd
import polars as pl

# Use relative imports assuming the structure is preserved
from .._validation import raise_invalid_choice
from ..data.coverage import record_coverage
from ..data.dataset import SplitDataset
from ..engines import SkyulfDataFrame, SkyulfPolarsWrapper, get_engine
from ._evaluation.classification import evaluate_classification_model
from ._evaluation.clustering import evaluate_clustering_model
from ._evaluation.regression import evaluate_regression_model
from ._evaluation.schemas import ModelEvaluationReport
from ._sample_weights import validate_sample_weight
from .cross_validation import perform_cross_validation
from .fold_preprocessing import FoldPreprocessor

logger = logging.getLogger(__name__)


def extract_xy(data: Any, target_column: str) -> tuple[Any, Any]:
    """Extract ``(X, y)`` from a DataFrame or an ``(X, y)`` tuple.

    The frame may be pandas or Polars, and ``target_column`` names the column to
    lift into ``y``.

    An empty/falsy ``target_column`` is the established "no target" sentinel
    (see ``_node_runners.py``'s ``target_col=""`` for data-preview-only
    inputs): unsupervised calculators (e.g. clustering) rely on this to get
    the whole frame back as ``X`` with ``y=None``.
    """
    if not target_column:
        X = data[0] if isinstance(data, tuple) else data
        return X, None

    if isinstance(data, tuple) and len(data) == 2:
        return _extract_xy_from_tuple(data, target_column)

    # A fallback engine selection does not establish the input's frame type.
    if isinstance(data, pl.DataFrame | SkyulfPolarsWrapper):
        return _extract_xy_polars(data, target_column)

    return _extract_xy_pandas_like(data, target_column)


def _extract_xy_from_tuple(data: tuple[Any, Any], target_column: str) -> tuple[Any, Any]:
    """Extracts X/y from a ``(X, y)`` tuple, pulling ``y`` out of ``X`` if it's missing."""
    X, y = data[0], data[1]
    if hasattr(X, "columns") and target_column in X.columns:
        features, embedded_y = extract_xy(X, target_column)
        return features, embedded_y if y is None else y
    return X, y


def _extract_xy_polars(data: Any, target_column: str) -> tuple[Any, Any]:
    """Extracts X/y from a Polars DataFrame by dropping/selecting ``target_column``."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    X = data.drop([target_column])
    y = data.select(target_column).to_series()
    return X, y


def _extract_xy_pandas_like(data: Any, target_column: str) -> tuple[Any, Any]:
    """Extracts X/y from a pandas or generic DataFrame-like object."""
    if hasattr(data, "columns"):
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        if hasattr(data, "drop"):
            try:
                return data.drop(columns=[target_column]), data[target_column]
            except TypeError:
                pass

        if hasattr(data, target_column):
            return data, getattr(data, target_column)

    raise ValueError(f"Unexpected data type: {type(data)}")


def _evaluation_split_coverage(
    dataset: SplitDataset, name: str, report: ModelEvaluationReport | None
) -> dict[str, Any] | None:
    """Preserve saved exclusions and infer row counts only for evaluated splits."""
    coverage = dataset.evaluation_coverage.get(name)
    if coverage is not None:
        return deepcopy(coverage)
    if report is None:
        return None
    payload = getattr(dataset, name)
    frame = payload[0] if isinstance(payload, tuple) else payload
    return record_coverage(len(frame), len(frame))


class BaseModelCalculator(ABC):
    """Fitting half of a model node: declares the problem type and trains the estimator.

    Subclasses must supply :attr:`problem_type` and :meth:`fit`. The
    :attr:`default_params` and tuning hooks carry plain-model defaults that
    structural models such as ensembles override.
    """

    @property
    @abstractmethod
    def problem_type(self) -> str:
        """Returns 'classification', 'regression', or 'clustering'."""

    #: Config keys that a model's ``prepare_tuning_params`` absorbs into its
    #: own structural state (e.g. an ensemble's resolved ``estimators``)
    #: rather than treating as a literal single-item search-space candidate.
    #: Empty for plain models. See ``_BaseEnsembleCalculator`` in
    #: ``ensemble.py`` for the non-trivial override.
    STRUCTURAL_TUNING_KEYS: tuple[str, ...] = ()

    @property
    def default_params(self) -> dict[str, Any]:
        """Default hyperparameters for the model."""
        return {}

    def prepare_tuning_params(self, config: dict[str, Any]) -> None:
        """Hook for structural models (e.g. ensembles) to absorb their sub-estimator selection.

        Runs before the tuner builds the base model. No-op for plain models.
        Ensembles override this to inject the resolved
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

    def _boosting_fit_kwargs(
        self,
        model: Any,
        X_np: Any,
        y_np: Any,
        iteration_callback: Callable[..., None] | None,
    ) -> dict[str, Any]:
        """Hook: extra kwargs for the underlying ``model.fit(...)`` call.

        Boosting calculators (XGBoost/LightGBM) override this to attach an
        eval set + iteration callback when one is supplied; every other model
        keeps the plain fit. May also mutate ``model`` (XGBoost 3.x carries
        callbacks on the estimator itself); returning ``"_detach_callbacks":
        True`` tells the caller to clear ``model.callbacks`` after fit so the
        saved artifact doesn't pickle live callback closures.
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
        iteration_callback: Callable[..., None] | None = None,
        *,
        sample_weight: Any = None,
    ) -> Any:
        """Trains the model and returns the fitted model artifact.

        ``class_weight`` is a model setting in ``config["params"]`` for
        supported classifiers. ``sample_weight`` is a separate vector aligned
        with the rows of ``X`` and ``y``; it can also weight regression fits.
        A tuner keeps class weights in ``base_model.params`` (or searches them
        in ``search_space``) and slices row weights for each training fold.

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
    """Predicting half of a model node: turns a fitted artifact into predictions.

    :meth:`predict_proba` is optional. The base implementation returns ``None``,
    which is how callers detect an estimator that cannot produce class
    probabilities.
    """

    @abstractmethod
    def predict(self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any) -> pd.Series | Any:
        """Generates predictions."""

    def predict_proba(
        self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any
    ) -> pd.DataFrame | SkyulfDataFrame | None:
        """Generates prediction probabilities if supported.

        Returns DataFrame where columns are classes.
        """
        return None


def _fit_predict_weights(
    dataset: SplitDataset, preprocessing: Any, raw_train: Any, raw_weights: Any
) -> Any:
    """Keep raw preprocessing weights separate from the processed dataset row axis."""
    if preprocessing is None:
        if raw_weights is not None:
            raise ValueError("preprocessing_sample_weight requires preprocessing")
        return dataset.train_sample_weight
    if raw_train is None:
        raise ValueError("preprocessing requires the preprocessing_train (X, y) payload")
    if raw_weights is None and dataset.train_sample_weight is not None:
        raise ValueError(
            "preprocessing_sample_weight must supply original weights aligned with "
            "preprocessing_train; processed dataset weights cannot be reused"
        )
    return validate_sample_weight(raw_weights, len(raw_train[0]))


class StatefulEstimator:
    """Drive a model node end to end, holding the fitted model between calls.

    Wraps a calculator/applier pair and owns the in-memory ``model`` artifact,
    so fitting, cross-validation, prediction and evaluation all share one
    fitted state instead of retraining.
    """

    def __init__(self, calculator: BaseModelCalculator, applier: BaseModelApplier, node_id: str):
        """Store the node pair and start with no fitted model.

        ``node_id`` is recorded into the evaluation payload alongside
        ``job_id``; ``model`` stays ``None`` until the first fit.
        """
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
        """Instance-method wrapper around the module-level ``extract_xy()``.

        Kept for backward compatibility with existing call sites/tests.
        """
        return extract_xy(data, target_column)

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
        preprocessing: FoldPreprocessor | None = None,
        *,
        cv_nested_type: str = "auto",
        group_column: str | None = None,
        gap: int = 0,
        test_size: int | None = None,
        max_train_size: int | None = None,
        inner_folds: int | None = None,
    ) -> dict[str, Any]:
        """Performs cross-validation on the training split."""
        X_train, y_train = self._extract_xy(dataset.train, target_column)
        from ._policy_cv import (  # noqa: PLC0415 - avoid tuning import cycle
            policy_config,
        )
        from ._tuning.cv_policy import (  # noqa: PLC0415 - avoid tuning import cycle
            validate_holdout_metadata,
        )

        policy = policy_config(
            cv_type,
            n_folds,
            shuffle,
            random_state,
            time_column,
            {
                "cv_nested_type": cv_nested_type,
                "group_column": group_column,
                "gap": gap,
                "test_size": test_size,
                "max_train_size": max_train_size,
                "inner_folds": inner_folds,
            },
        )
        for heldout in (dataset.test, dataset.validation):
            if self._is_non_empty_split(heldout):
                validate_holdout_metadata(
                    X_train,
                    self._extract_xy(heldout, target_column)[0],
                    policy,
                    self.calculator.problem_type,
                )

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
            preprocessing=preprocessing,
            cv_nested_type=cv_nested_type,
            group_column=group_column,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
            inner_folds=inner_folds,
            **(
                {"sample_weight": dataset.train_sample_weight}
                if dataset.train_sample_weight is not None
                else {}
            ),
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
            return self._extract_xy(split_data, target_column)[0]

        if target_column in split_data.columns:
            return self._drop_target_column(split_data, target_column)
        return split_data

    def _normalize_fit_predict_dataset(
        self,
        dataset: SplitDataset
        | pd.DataFrame
        | pl.DataFrame
        | SkyulfDataFrame
        | tuple[pd.DataFrame, pd.Series]
        | tuple[pd.DataFrame, pd.DataFrame],
        target_column: str,
        log_callback: Callable[[str], None] | None,
    ) -> SplitDataset:
        """Wrap raw DataFrame/tuple ``fit_predict`` input into a SplitDataset.

        Handles pandas, raw (unwrapped) Polars, and wrapped ``SkyulfDataFrame``
        input alike -- checking only ``isinstance(dataset, pd.DataFrame)``
        would silently misroute a raw ``pl.DataFrame`` (e.g. the no-splitter
        fallback in ``pipeline.py``'s ``fit()``, which hands the modeling
        layer a bare frame of whatever engine produced it) into the
        ``SplitDataset``-shaped branch below, crashing with
        ``AttributeError: 'DataFrame' object has no attribute 'train'``.
        """
        if isinstance(dataset, tuple):
            # Check if it's (train_df, test_df) or (X, y)
            elem0 = dataset[0]
            elem1 = dataset[1]
            if (
                isinstance(elem0, pd.DataFrame)
                and isinstance(elem1, pd.DataFrame)
                and target_column in elem0.columns
            ):
                # It's (train_df, test_df)
                return SplitDataset(train=elem0, test=elem1, validation=None)

            # Fallback: Treat input as training data (e.g. X, y tuple) and initialize empty test set.
            msg = (
                "WARNING: No test set provided. Using entire input as training data. "
                "Ensure data was split BEFORE preprocessing to avoid data leakage."
            )
            logger.warning(msg)
            if log_callback:
                log_callback(msg)

            return SplitDataset(train=cast(Any, dataset), test=pd.DataFrame(), validation=None)

        if hasattr(dataset, "shape") and hasattr(dataset, "columns"):
            # A single frame-like object (pandas, raw Polars, or wrapper) --
            # build the empty "test" placeholder with a same-engine empty
            # frame rather than always defaulting to pandas.
            empty_test = get_engine(dataset).create_dataframe({})
            return SplitDataset(train=cast(Any, dataset), test=empty_test, validation=None)

        return dataset

    def fit_predict(
        self,
        dataset: SplitDataset
        | pd.DataFrame
        | pl.DataFrame
        | SkyulfDataFrame
        | tuple[pd.DataFrame, pd.Series]
        | tuple[pd.DataFrame, pd.DataFrame],
        target_column: str,
        config: dict[str, Any],
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        job_id: str = "unknown",
        preprocessing: FoldPreprocessor | None = None,
        preprocessing_train: tuple[Any, Any] | None = None,
        preprocessing_validation: tuple[Any, Any] | None = None,
        iteration_callback: Callable[..., None] | None = None,
        *,
        preprocessing_sample_weight: Any = None,
    ) -> dict[str, pd.Series]:
        """Fits the model on training data and returns predictions for all splits.

        ``preprocessing`` (F-15): forwarded to calculators that
        support per-fold refit (``TuningCalculator``). When set,
        ``preprocessing_train`` must carry the pre-transform ``(X, y)``
        payload the calculator should fit/tune on, so fold slicing stays
        aligned with the preprocessor; predictions still run on this
        dataset's (post-transform) splits. ``preprocessing_validation`` is
        the matching pre-transform validation payload for holdout tuning —
        ``dataset.validation`` is post-transform, so the refit cannot score
        against it directly. ``preprocessing_sample_weight`` is the original
        positional weight vector aligned with ``preprocessing_train``. It is
        required when the processed dataset carries weights: processed rows may
        have been reordered or filtered, so their weights cannot be reused.
        """
        # Handle raw DataFrame or Tuple input by wrapping it in a dummy SplitDataset
        dataset = self._normalize_fit_predict_dataset(dataset, target_column, log_callback)

        fit_weight = _fit_predict_weights(
            dataset, preprocessing, preprocessing_train, preprocessing_sample_weight
        )

        # 1. Prepare Data
        X_train, y_train = self._extract_xy(dataset.train, target_column)

        validation_data = None
        if dataset.validation is not None:
            X_val, y_val = self._extract_xy(dataset.validation, target_column)
            validation_data = (X_val, y_val)

        # 2. Train Model
        if preprocessing is not None:
            if preprocessing_train is None:
                raise ValueError("preprocessing requires the preprocessing_train (X, y) payload")
            # Only TuningCalculator accepts the hook today; the backend only
            # passes it when wrapping one, so a narrow cast keeps the generic
            # calculator interface clean.
            self.model = cast(Any, self.calculator).fit(
                preprocessing_train[0],
                preprocessing_train[1],
                config,
                progress_callback=progress_callback,
                log_callback=log_callback,
                validation_data=validation_data,
                preprocessing=preprocessing,
                validation_frames=preprocessing_validation,
                iteration_callback=iteration_callback,
                **({"sample_weight": fit_weight} if fit_weight is not None else {}),
            )
        else:
            self.model = self.calculator.fit(
                X_train,
                y_train,
                config,
                progress_callback=progress_callback,
                log_callback=log_callback,
                validation_data=validation_data,
                iteration_callback=iteration_callback,
                **({"sample_weight": fit_weight} if fit_weight is not None else {}),
            )

        # 3. Predict on all splits
        predictions = {}

        # Train Predictions
        predictions["train"] = self.applier.predict(X_train, self.model)

        # Test Predictions
        test_df = dataset.test[0] if isinstance(dataset.test, tuple) else dataset.test
        is_test_empty = len(test_df) == 0

        if not is_test_empty:
            X_test = self._extract_split_features(dataset.test, target_column)
            predictions["test"] = self.applier.predict(X_test, self.model)

        # Validation Predictions
        if self._is_non_empty_split(dataset.validation):
            X_val = self._extract_split_features(dataset.validation, target_column)
            predictions["validation"] = self.applier.predict(X_val, self.model)

        return predictions

    def evaluate(
        self,
        dataset: SplitDataset,
        target_column: str,
        job_id: str = "unknown",
        reference_column: str = "",
    ) -> Any:
        """Evaluates the model on all splits and returns a detailed report.

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

        self._attach_evaluation_coverage(dataset, splits_payload, evaluation_data)

        # Return report object (simplified for now, assuming schema matches)
        return {
            "problem_type": problem_type,
            "splits": splits_payload,
            "raw_data": evaluation_data,
        }

    @staticmethod
    def _attach_evaluation_coverage(
        dataset: SplitDataset, reports: dict[str, Any], raw_data: dict[str, Any]
    ) -> None:
        """Expose eligible-row denominators, including splits completely excluded by filters."""
        for name in ("train", "test", "validation"):
            payload = getattr(dataset, name)
            if payload is None:
                continue
            report = reports.get(name)
            coverage = _evaluation_split_coverage(dataset, name, report)
            if coverage is None:
                continue
            if report is None and coverage["excluded_rows"] != 0 and coverage["scored_rows"] == 0:
                report = ModelEvaluationReport(
                    dataset_name=name,
                    metrics={},
                    omitted_metrics={"evaluation": "No eligible rows remain after preprocessing."},
                )
                reports[name] = report
                raw_data["splits"][name] = (
                    {"labels": []}
                    if raw_data["problem_type"] == "clustering"
                    else {"y_true": [], "y_pred": []}
                )
            if report is not None:
                report.coverage = coverage
                raw_data["splits"].setdefault(name, {})["coverage"] = deepcopy(coverage)

    def _evaluate_split(
        self,
        split_name: str,
        data: Any,
        target_column: str,
        problem_type: str,
        evaluation_data: dict[str, Any],
        reference_column: str = "",
    ) -> Any:
        """Evaluates a single dataset split, recording raw predictions into ``evaluation_data``.

        Returns the split's evaluation report, or ``None`` if it can't be
        evaluated.
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

        X = self._evaluation_features(X)
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

        return self._evaluate_split_with_model(
            model_to_evaluate, split_name, X, y, problem_type, predictions=y_pred
        )

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

    def _evaluation_features(self, X: Any) -> Any:
        """Exclude persisted split metadata from both predictions and metric evaluation."""
        from ._tuning.cv_policy import (  # noqa: PLC0415 - avoid tuning import cycle
            prediction_features,
        )

        if isinstance(self.model, tuple) and len(self.model) == 2:
            return prediction_features(X, self.model[1])
        return X

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
        y_proba_df = cast(pd.DataFrame, y_proba_df)
        return {
            "classes": y_proba_df.columns.tolist(),
            "values": y_proba_df.to_numpy().tolist(),
        }

    @staticmethod
    def _evaluate_split_with_model(
        model_to_evaluate: Any,
        split_name: str,
        X: Any,
        y: Any,
        problem_type: str,
        reference_column: str = "",
        *,
        predictions: Any | None = None,
    ) -> Any:
        """Dispatches to the classification, regression, or clustering evaluator.

        For clustering, ``y`` is the *predicted* cluster labels for this split
        (there is no ground-truth target), computed by the caller via
        ``self.applier.predict(X, self.model)``.
        """
        if problem_type == "classification":
            return evaluate_classification_model(
                model=model_to_evaluate,
                dataset_name=split_name,
                X_test=X,
                y_test=y,
                predictions=predictions,
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
            raise_invalid_choice(
                problem_type, ("classification", "regression", "clustering"), "problem type"
            )
