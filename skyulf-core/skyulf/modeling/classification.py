"""Classification models."""

import logging
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier  # ty: ignore[unresolved-import]

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as _lgb  # ty: ignore[unresolved-import]
    from lightgbm import LGBMClassifier  # ty: ignore[unresolved-import]

    LIGHTGBM_AVAILABLE = True

    # LightGBM 4.x emits C++ stderr warnings ("No further splits with positive
    # gain", auto col-wise info, etc.) that the `verbose=-1` constructor param
    # does not always silence. Register a no-op logger so all native messages
    # are intercepted by Python and dropped. Safe to call multiple times.
    class _SilentLgbmLogger:
        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

    _lgb.register_logger(_SilentLgbmLogger())  # ty: ignore[unresolved-attribute]
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..types import DEFAULT_RANDOM_STATE
from ._boosting_progress import LightGBMIterationAdapter, XgboostIterationAdapter
from ._sklearn_compat import normalize_logistic_regression_params
from .sklearn_wrapper import SklearnApplier, SklearnCalculator

logger = logging.getLogger(__name__)


# --- Logistic Regression ---
class LogisticRegressionApplier(SklearnApplier):
    """Logistic Regression Applier."""


@NodeRegistry.register("logistic_regression", LogisticRegressionApplier)
@node_meta(
    id="logistic_regression",
    name="Logistic Regression",
    category="Modeling",
    description="Linear model for classification.",
    params={"max_iter": 1000, "solver": "lbfgs"},
    # "text": logistic regression on TF-IDF/vectorized features is a common,
    # well-performing baseline for text classification (alongside Naive Bayes
    # and the SGD-based linear SVM approximation below).
    tags=["requires_scaling", "classification", "text", "nlp"],
    learns_from_data=True,
)
class LogisticRegressionCalculator(SklearnCalculator):
    """Logistic Regression Calculator."""

    # sklearn solver -> penalties it actually supports. Manual/UI configuration
    # allows selecting solver and penalty independently (unlike the tuner's own
    # search space, which restricts solver to "saga" whenever penalty is
    # varied), so an incompatible combination reaches `fit()` unchecked and
    # would otherwise surface as an opaque sklearn ValueError at model-fit time.
    _SOLVER_PENALTIES: ClassVar[dict[str, set[Any]]] = {
        "lbfgs": {"l2", None},
        "liblinear": {"l1", "l2"},
        "newton-cg": {"l2", None},
        "newton-cholesky": {"l2", None},
        "sag": {"l2", None},
        "saga": {"l1", "l2", "elasticnet", None},
    }

    def __init__(self):
        """Bind ``LogisticRegression`` with ``max_iter=1000`` and ``solver="lbfgs"``.

        Both are defaults, not fixtures: ``fit``'s config may override either,
        and the resulting pair is then checked against ``_SOLVER_PENALTIES``.
        """
        super().__init__(
            model_class=LogisticRegression,
            default_params={
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            problem_type="classification",
        )

    def fit(
        self,
        X: Any,
        y: Any,
        config: dict[str, Any],
        progress_callback: Callable[..., Any] | None = None,
        log_callback: Callable[..., Any] | None = None,
        validation_data: Any = None,
        iteration_callback: Callable[..., Any] | None = None,
    ) -> Any:
        """Reject an unsupported solver/penalty pair, then delegate to the base fit."""
        self._validate_solver_penalty(config)
        return super().fit(
            X,
            y,
            config,
            progress_callback,
            log_callback,
            validation_data,
            iteration_callback=iteration_callback,
        )

    def _resolve_fit_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merges fit params, then normalizes ``penalty`` for sklearn >=1.8.

        sklearn >=1.8 deprecates the ``penalty`` constructor arg entirely (in
        favor of ``l1_ratio``/``C``) and will remove it in sklearn 1.10. We
        keep ``penalty`` ("l1"/"l2"/"elasticnet"/None) as our own public
        config/UI field unchanged — it's translated to the newer kwargs here,
        right before the sklearn estimator is constructed, so we never pass a
        bare ``penalty=`` to sklearn regardless of installed sklearn version.
        """
        params = super()._resolve_fit_params(config)
        return normalize_logistic_regression_params(params)

    @classmethod
    def _extract_solver_penalty_params(cls, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """Returns the params dict from config, or None if unavailable/not a dict."""
        if not config:
            return None
        params = config.get("params", config)
        if not isinstance(params, dict):
            return None
        return params

    @classmethod
    def _raise_incompatible_solver_penalty(cls, solver: Any, penalty: Any) -> None:
        """Raises a ValueError listing solvers compatible with the requested penalty."""
        compatible_solvers = sorted(
            s for s, penalties in cls._SOLVER_PENALTIES.items() if penalty in penalties
        )
        raise ValueError(
            f"Logistic Regression: solver={solver!r} does not support "
            f"penalty={penalty!r}. Solvers compatible with this penalty: "
            f"{compatible_solvers or 'none'}."
        )

    def _validate_solver_penalty(self, config: dict[str, Any] | None) -> None:
        """Raise a clear, actionable error for an invalid solver/penalty pair.

        sklearn's own error for this (e.g. "Solver lbfgs supports only 'l2' or
        None penalties") is only raised deep inside `LogisticRegression.fit`,
        after data has already been split/validated upstream. Failing fast
        here with the full list of compatible solvers is more actionable.

        Validates against the *merged* effective params (``default_params``
        overlaid with the config's overrides), not just the raw config —
        otherwise overriding only ``penalty`` (very common; ``solver``
        defaults to ``"lbfgs"``) would skip validation entirely since
        ``solver`` never appears in the raw override dict, letting an
        incompatible combo reach sklearn's own opaque error at fit time.
        """
        overrides = self._extract_solver_penalty_params(config) or {}
        params = {**self.default_params, **overrides}
        solver = params.get("solver")
        if solver is None or "penalty" not in params:
            return
        penalty = params.get("penalty")
        compatible = self._SOLVER_PENALTIES.get(solver)
        if compatible is not None and penalty not in compatible:
            self._raise_incompatible_solver_penalty(solver, penalty)


# --- Calibrated Classifier ---
_CALIBRATED_BASE_ESTIMATORS: dict[str, Callable[[], BaseEstimator]] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=DEFAULT_RANDOM_STATE
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE),
    "gaussian_nb": GaussianNB,
    "svc": lambda: SVC(probability=True, random_state=DEFAULT_RANDOM_STATE),
}


def _make_calibrated_base_estimator(key: str) -> BaseEstimator:
    """Resolve a symbolic calibration base consistently for direct fits and search trials."""
    factory = _CALIBRATED_BASE_ESTIMATORS.get(key)
    if factory is None:
        logger.warning("Unknown base_estimator '%s'; falling back to logistic_regression.", key)
        factory = _CALIBRATED_BASE_ESTIMATORS["logistic_regression"]
    return factory()


class _SeededCalibratedClassifierCV(CalibratedClassifierCV):
    """Resolve a cloneable base selection and seed for each calibration fit.

    Tuning constructs and clones ``model_class`` directly, bypassing the
    calculator's ``fit``. Both ``base_estimator`` and ``random_state`` must
    survive searcher ``set_params`` calls, including fold pipeline prefixes.
    An explicit symbolic base overrides ``estimator`` for that candidate;
    without one, the supplied estimator or sklearn default is preserved.
    Integer CV folds retain sklearn's unshuffled splitting semantics.
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        method: str = "sigmoid",
        cv: Any = None,
        n_jobs: int | None = None,
        ensemble: bool | str = "auto",
        random_state: int | None = DEFAULT_RANDOM_STATE,
        base_estimator: str | None = None,
    ) -> None:
        """Keep constructor parameters intact for sklearn cloning and ``set_params``."""
        super().__init__(
            estimator=estimator, method=method, cv=cv, n_jobs=n_jobs, ensemble=ensemble
        )
        self.random_state = random_state
        self.base_estimator = base_estimator

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **fit_params: Any) -> Any:
        """Resolve and seed this candidate without modifying a caller-supplied estimator."""
        if self.base_estimator is not None:
            self.estimator = _make_calibrated_base_estimator(self.base_estimator)
        estimator = self._get_estimator()
        if "random_state" in estimator.get_params(deep=False):
            self.estimator = clone(estimator).set_params(random_state=self.random_state)
        return super().fit(X, y, sample_weight=sample_weight, **fit_params)


class CalibratedClassifierApplier(SklearnApplier):
    """Calibrated Classifier Applier (well-calibrated predict_proba)."""


@NodeRegistry.register("calibrated_classifier", CalibratedClassifierApplier)
@node_meta(
    id="calibrated_classifier",
    name="Calibrated Classifier",
    category="Modeling",
    description=(
        "Wraps a base classifier with CalibratedClassifierCV so predicted "
        "probabilities are well-calibrated (Platt/sigmoid or isotonic)."
    ),
    params={"base_estimator": "logistic_regression", "method": "sigmoid", "cv": 5},
    tags=["requires_scaling", "classification"],
    learns_from_data=True,
)
class CalibratedClassifierCalculator(SklearnCalculator):
    """Calibrated Classifier Calculator with a selectable base estimator.

    The frontend sends ``base_estimator`` as a string key (e.g.
    ``"random_forest"``); it is resolved here into a fresh estimator instance
    before ``CalibratedClassifierCV`` is constructed. Defaults to logistic
    regression for backward compatibility. ``random_state`` seeds supported
    base estimators, defaults to 42, and accepts ``None`` for unseeded fits.
    It does not shuffle calibration CV folds or affect deterministic bases.
    A tuning search space can select one or more ``base_estimator`` keys;
    each candidate and the final refit use the selected classifier family.
    """

    # Map of selectable base estimators → factory. Each must support
    # ``predict_proba`` (or ``decision_function``) so calibration is meaningful.
    BASE_ESTIMATORS: ClassVar[dict[str, Callable[[], BaseEstimator]]] = _CALIBRATED_BASE_ESTIMATORS

    STRUCTURAL_TUNING_KEYS: tuple[str, ...] = ("base_estimator",)

    def __init__(self):
        """Bind ``CalibratedClassifierCV`` over a logistic-regression base estimator.

        Fixes ``method="sigmoid"`` and ``cv=5`` alongside that base estimator as
        overridable defaults, and starts ``_tuning_base_config`` empty so
        :attr:`default_params` yields the bound estimator until
        :meth:`prepare_tuning_params` records a different selection.
        """
        super().__init__(
            model_class=_SeededCalibratedClassifierCV,
            default_params={
                "estimator": LogisticRegression(max_iter=1000),
                "method": "sigmoid",
                "cv": 5,
            },
            problem_type="classification",
        )
        self._tuning_base_config: dict[str, Any] = {}

    @property
    def default_params(self) -> dict[str, Any]:
        """Return the defaults with any tuning-selected base estimator resolved in.

        An unrecognized ``base_estimator`` key warns and falls back to logistic
        regression rather than raising. The factory runs on every access, so
        separate tuning trials never share one estimator instance.
        """
        params = dict(self._default_params)
        if self._tuning_base_config:
            key = self._tuning_base_config.get("base_estimator")
            if isinstance(key, str):
                params["estimator"] = _make_calibrated_base_estimator(key)
        return params

    def prepare_tuning_params(self, config: dict[str, Any]) -> None:
        """Absorb ``base_estimator`` so it selects an estimator instead of being tuned.

        Only keys named in :attr:`STRUCTURAL_TUNING_KEYS` are taken, read from
        either the nested ``params`` dict or the flat config.
        """
        src = config.get("params") if isinstance(config.get("params"), dict) else config
        src = src or {}
        self._tuning_base_config = {k: src[k] for k in self.STRUCTURAL_TUNING_KEYS if k in src}

    def fit(
        self,
        X: Any,
        y: Any,
        config: dict[str, Any],
        progress_callback: Callable[..., Any] | None = None,
        log_callback: Callable[..., Any] | None = None,
        validation_data: Any = None,
        iteration_callback: Callable[..., Any] | None = None,
    ) -> Any:
        """Resolve the base estimator in ``config``, then delegate to the base fit."""
        config = self._resolve_base_estimator(config)
        return super().fit(
            X,
            y,
            config,
            progress_callback,
            log_callback,
            validation_data,
            iteration_callback=iteration_callback,
        )

    @classmethod
    def _resolve_base_estimator(cls, config: dict[str, Any] | None) -> dict[str, Any]:
        """Translate a ``base_estimator`` string key into an estimator instance.

        Supports both the flat config shape and the nested ``{"params": {...}}``
        shape used by the model-training payload. Unknown keys fall back to
        logistic regression with a warning.
        """
        if not config:
            return config or {}
        resolved = dict(config)
        nested = isinstance(resolved.get("params"), dict)
        bucket = dict(resolved["params"]) if nested else resolved
        key = bucket.pop("base_estimator", None)
        if isinstance(key, str):
            bucket["estimator"] = _make_calibrated_base_estimator(key)
        if nested:
            resolved["params"] = bucket
            return resolved
        return bucket


# --- Random Forest Classifier ---
class RandomForestClassifierApplier(SklearnApplier):
    """Random Forest Classifier Applier."""


@NodeRegistry.register("random_forest_classifier", RandomForestClassifierApplier)
@node_meta(
    id="random_forest_classifier",
    name="Random Forest Classifier",
    category="Modeling",
    description="Ensemble of decision trees.",
    params={"n_estimators": 50, "max_depth": 10, "min_samples_split": 5},
    tags=["classification"],
    learns_from_data=True,
)
class RandomForestClassifierCalculator(SklearnCalculator):
    """Random Forest Classifier Calculator."""

    def __init__(self):
        """Bind ``RandomForestClassifier`` with 50 shallow, constrained trees.

        ``max_depth=10``, ``min_samples_split=5`` and ``min_samples_leaf=2``
        restrict tree growth that sklearn leaves open by default, and the forest
        is half sklearn's default size. ``n_jobs=-1`` fits across every core.
        All are overridable from ``fit``'s config.
        """
        super().__init__(
            model_class=RandomForestClassifier,
            default_params={
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "n_jobs": -1,
            },
            problem_type="classification",
        )


# --- SVC ---
class SVCApplier(SklearnApplier):
    """SVC Applier."""


@NodeRegistry.register("svc", SVCApplier)
@node_meta(
    id="svc",
    name="Support Vector Classifier",
    category="Modeling",
    description="C-Support Vector Classification.",
    params={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    tags=["requires_scaling", "classification"],
    learns_from_data=True,
)
class SVCCalculator(SklearnCalculator):
    """SVC Calculator."""

    def __init__(self):
        """Bind ``SVC`` with an RBF kernel and probability estimates switched on.

        ``probability=True`` is the load-bearing default — without it the
        applier would have no ``predict_proba`` to call — and it costs extra
        cross-validation at fit time, because Platt scaling is fitted on top of
        the margins. ``C=1.0``, ``kernel="rbf"`` and ``gamma="scale"`` are
        sklearn's own defaults.
        """
        super().__init__(
            model_class=SVC,
            default_params={
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "probability": True,
            },
            problem_type="classification",
        )


# --- K-Neighbors ---
class KNeighborsClassifierApplier(SklearnApplier):
    """K-Neighbors Classifier Applier."""


@NodeRegistry.register("k_neighbors_classifier", KNeighborsClassifierApplier)
@node_meta(
    id="k_neighbors_classifier",
    name="K-Neighbors Classifier",
    category="Modeling",
    description="Classifier implementing the k-nearest neighbors vote.",
    params={"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
    tags=["requires_scaling", "classification"],
    learns_from_data=True,
)
class KNeighborsClassifierCalculator(SklearnCalculator):
    """K-Neighbors Classifier Calculator."""

    def __init__(self):
        """Bind ``KNeighborsClassifier`` with an unweighted 5-neighbour vote.

        ``algorithm="auto"`` leaves sklearn to pick the neighbour-search
        structure from the data's shape and size, and ``n_jobs=-1`` parallelizes
        the queries.
        """
        super().__init__(
            model_class=KNeighborsClassifier,
            default_params={
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto",
                "n_jobs": -1,
            },
            problem_type="classification",
        )


# --- Decision Tree ---
class DecisionTreeClassifierApplier(SklearnApplier):
    """Decision Tree Classifier Applier."""


@NodeRegistry.register("decision_tree_classifier", DecisionTreeClassifierApplier)
@node_meta(
    id="decision_tree_classifier",
    name="Decision Tree Classifier",
    category="Modeling",
    description="A non-parametric supervised learning method used for classification.",
    params={"max_depth": None, "min_samples_split": 2, "criterion": "gini"},
    tags=["classification"],
    learns_from_data=True,
)
class DecisionTreeClassifierCalculator(SklearnCalculator):
    """Decision Tree Classifier Calculator."""

    def __init__(self):
        """Bind ``DecisionTreeClassifier`` with sklearn's own unpruned defaults.

        ``max_depth=None`` grows the tree until its leaves are pure or fall
        below ``min_samples_split=2``, so unlike the forest and boosting nodes
        in this module this one ships no regularization of its own.
        """
        super().__init__(
            model_class=DecisionTreeClassifier,
            default_params={
                "max_depth": None,
                "min_samples_split": 2,
                "criterion": "gini",
            },
            problem_type="classification",
        )


# --- Gradient Boosting ---
class GradientBoostingClassifierApplier(SklearnApplier):
    """Gradient Boosting Classifier Applier."""


@NodeRegistry.register("gradient_boosting_classifier", GradientBoostingClassifierApplier)
@node_meta(
    id="gradient_boosting_classifier",
    name="Gradient Boosting Classifier",
    category="Modeling",
    description="Gradient Boosting for classification.",
    params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    tags=["classification"],
    learns_from_data=True,
)
class GradientBoostingClassifierCalculator(SklearnCalculator):
    """Gradient Boosting Classifier Calculator."""

    def __init__(self):
        """Bind ``GradientBoostingClassifier`` with 100 depth-3 trees at rate 0.1.

        All three are exactly sklearn's own defaults, restated so the node's
        tunable surface is explicit; ``fit``'s config overrides any of them.
        """
        super().__init__(
            model_class=GradientBoostingClassifier,
            default_params={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
            },
            problem_type="classification",
        )


# --- AdaBoost ---
class AdaBoostClassifierApplier(SklearnApplier):
    """AdaBoost Classifier Applier."""


@NodeRegistry.register("adaboost_classifier", AdaBoostClassifierApplier)
@node_meta(
    id="adaboost_classifier",
    name="AdaBoost Classifier",
    category="Modeling",
    description="An AdaBoost classifier.",
    params={"n_estimators": 50, "learning_rate": 1.0},
    tags=["classification"],
    learns_from_data=True,
)
class AdaBoostClassifierCalculator(SklearnCalculator):
    """AdaBoost Classifier Calculator."""

    def __init__(self):
        """Bind ``AdaBoostClassifier`` with 50 weak learners at rate 1.0.

        Both are sklearn's defaults. No base estimator is pinned here, so
        ``fit``'s config may supply one; sklearn's depth-1 decision stump is
        used when it does not.
        """
        super().__init__(
            model_class=AdaBoostClassifier,
            default_params={
                "n_estimators": 50,
                "learning_rate": 1.0,
            },
            problem_type="classification",
        )


# --- XGBoost ---
if XGBOOST_AVAILABLE:

    class XGBClassifierApplier(SklearnApplier):
        """XGBoost Classifier Applier."""

    @NodeRegistry.register("xgboost_classifier", XGBClassifierApplier)
    @node_meta(
        id="xgboost_classifier",
        name="XGBoost Classifier",
        category="Modeling",
        description="Extreme Gradient Boosting classifier.",
        params={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3},
        tags=["classification"],
        learns_from_data=True,
    )
    class XGBClassifierCalculator(SklearnCalculator):
        """XGBoost Classifier Calculator."""

        def __init__(self):
            """Bind ``XGBClassifier`` with XGBoost's defaults plus row/column subsampling.

            ``n_estimators=100``, ``max_depth=6`` and ``learning_rate=0.3`` are
            XGBoost's own. ``subsample`` and ``colsample_bytree`` are pulled
            down from XGBoost's 1.0 to 0.8, and neither appears in the node's
            ``params`` metadata, so they are only reachable through ``fit``'s
            config.
            """
            super().__init__(
                model_class=XGBClassifier,
                default_params={
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_jobs": -1,
                },
                problem_type="classification",
            )

        def _boosting_fit_kwargs(self, model, X_np, y_np, iteration_callback):
            if iteration_callback is None or XgboostIterationAdapter is None:
                return {}
            # XGBoost 3.x reads callbacks from the estimator itself (they were
            # removed from fit()). eval_set is display-only (no early
            # stopping), so the trained model is identical to a plain fit — it
            # just streams per-round training loss for the live chart.
            model.callbacks = [
                XgboostIterationAdapter(iteration_callback, total=int(model.n_estimators))
            ]
            return {
                "eval_set": [(X_np, y_np)],
                "verbose": False,
                "_detach_callbacks": True,
            }


# --- Extra Trees Classifier ---
class ExtraTreesClassifierApplier(SklearnApplier):
    """Extra Trees Classifier Applier."""


@NodeRegistry.register("extra_trees_classifier", ExtraTreesClassifierApplier)
@node_meta(
    id="extra_trees_classifier",
    name="Extra Trees Classifier",
    category="Modeling",
    description="Extremely randomised trees — faster than Random Forest, often comparably accurate.",
    params={"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
    tags=["classification"],
    learns_from_data=True,
)
class ExtraTreesClassifierCalculator(SklearnCalculator):
    """Extra Trees Classifier Calculator."""

    def __init__(self):
        """Bind ``ExtraTreesClassifier`` with 100 unpruned, non-bootstrapped trees.

        ``bootstrap=False`` is the distinguishing default: extra-trees randomizes
        split thresholds instead of resampling rows, so every tree sees the whole
        sample. The remaining settings match sklearn's defaults, and
        ``n_jobs=-1`` fits across every core.
        """
        super().__init__(
            model_class=ExtraTreesClassifier,
            default_params={
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "criterion": "gini",
                "bootstrap": False,
                "n_jobs": -1,
            },
            problem_type="classification",
        )


# --- HistGradientBoosting Classifier ---
class HistGradientBoostingClassifierApplier(SklearnApplier):
    """HistGradientBoosting Classifier Applier."""


@NodeRegistry.register("hist_gradient_boosting_classifier", HistGradientBoostingClassifierApplier)
@node_meta(
    id="hist_gradient_boosting_classifier",
    name="Hist Gradient Boosting Classifier",
    category="Modeling",
    description="Histogram-based gradient boosting — sklearn's fast LightGBM-style implementation.",
    params={"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
    tags=["classification"],
    learns_from_data=True,
)
class HistGradientBoostingClassifierCalculator(SklearnCalculator):
    """HistGradientBoosting Classifier Calculator."""

    def __init__(self):
        """Bind ``HistGradientBoostingClassifier`` with sklearn's own defaults.

        Tree shape is governed leaf-wise — ``max_leaf_nodes=31`` with
        ``max_depth=None`` — because this estimator bins features into
        ``max_bins=255`` buckets and grows by best leaf rather than level by
        level. That is the opposite of the level-wise boosting nodes elsewhere
        in this module.
        """
        super().__init__(
            model_class=HistGradientBoostingClassifier,
            default_params={
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
                "max_depth": None,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
                "max_bins": 255,
            },
            problem_type="classification",
        )


# --- LightGBM Classifier (optional) ---
if LIGHTGBM_AVAILABLE:

    class LGBMClassifierApplier(SklearnApplier):
        """LightGBM Classifier Applier.

        LightGBM 4.x sets ``feature_names_in_`` to auto-generated names
        (``Column_0``, ``Column_1``...) even when fit with numpy arrays, and the
        property's deleter is intentionally a no-op (see upstream source). That
        triggers sklearn's ``UserWarning: X does not have valid feature names``
        on every predict call. We suppress it locally here so the warning never
        leaks out of the applier boundary.
        """

        def predict(self, df, model_artifact):
            """Delegate to the base predict with the feature-name warning suppressed."""
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().predict(df, model_artifact)

        def predict_proba(self, df, model_artifact):
            """Delegate to the base predict_proba with the same warning suppressed."""
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().predict_proba(df, model_artifact)

    @NodeRegistry.register("lgbm_classifier", LGBMClassifierApplier)
    @node_meta(
        id="lgbm_classifier",
        name="LightGBM Classifier",
        category="Modeling",
        description="LightGBM: leaf-wise gradient boosting, fast and memory-efficient with categorical support.",
        params={"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
        tags=["classification"],
        learns_from_data=True,
    )
    class LGBMClassifierCalculator(SklearnCalculator):
        """LightGBM Classifier Calculator."""

        def __init__(self):
            """Bind ``LGBMClassifier`` with an unregularized leaf-wise configuration.

            ``max_depth=-1`` leaves depth unbounded, so ``num_leaves=31`` alone
            governs tree shape, and both subsampling rates and both L1/L2
            penalties start at their neutral values. ``verbose`` and
            ``verbosity`` are pinned to -1 to quiet LightGBM's native logging
            alongside the no-op logger registered at import time.
            """
            super().__init__(
                model_class=LGBMClassifier,
                default_params={
                    "n_estimators": 100,
                    "num_leaves": 31,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "min_child_samples": 20,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.0,
                    "boosting_type": "gbdt",
                    "n_jobs": -1,
                    "verbose": -1,
                    "verbosity": -1,
                },
                problem_type="classification",
            )

        def fit(
            self,
            X,
            y,
            config,
            progress_callback=None,
            log_callback=None,
            validation_data=None,
            iteration_callback=None,
        ):
            """Delegate to the base fit with the feature-name warning suppressed."""
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().fit(
                    X,
                    y,
                    config,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    validation_data=validation_data,
                    iteration_callback=iteration_callback,
                )

        def _boosting_fit_kwargs(self, model, X_np, y_np, iteration_callback):
            if iteration_callback is None:
                return {}
            return {
                "eval_set": [(X_np, y_np)],
                "callbacks": [LightGBMIterationAdapter(iteration_callback)],
            }


# --- Gaussian NB ---
class GaussianNBApplier(SklearnApplier):
    """Gaussian Naive Bayes Applier."""


@NodeRegistry.register("gaussian_nb", GaussianNBApplier)
@node_meta(
    id="gaussian_nb",
    name="Gaussian Naive Bayes",
    category="Modeling",
    description="Gaussian Naive Bayes (GaussianNB).",
    params={"var_smoothing": 1e-9},
    tags=["classification"],
    learns_from_data=True,
)
class GaussianNBCalculator(SklearnCalculator):
    """Gaussian Naive Bayes Calculator."""

    def __init__(self):
        """Bind ``GaussianNB``, whose only knob is ``var_smoothing=1e-9``.

        Naive Bayes estimates each feature's variance from the training data, so
        there is little to configure up front. ``var_smoothing`` is sklearn's own
        default: the portion of the largest variance of all features added to
        every variance for numerical stability.
        """
        super().__init__(
            model_class=GaussianNB,
            default_params={"var_smoothing": 1e-9},
            problem_type="classification",
        )


# --- SGD Classifier ---
class SGDClassifierApplier(SklearnApplier):
    """Stochastic Gradient Descent Classifier Applier."""


@NodeRegistry.register("sgd_classifier", SGDClassifierApplier)
@node_meta(
    id="sgd_classifier",
    name="SGD Classifier (text / linear)",
    category="Modeling",
    description=(
        "Linear classifiers (SVM, logistic regression, etc.) with SGD training. "
        "Highly efficient for high-dimensional sparse/dense text representations "
        "and large datasets."
    ),
    params={
        "loss": "log_loss",
        "penalty": "l2",
        "alpha": 0.0001,
        "l1_ratio": 0.15,
        "max_iter": 1000,
    },
    # Text-classification-scoped only (no "classification" tag): SGD with
    # hinge/log loss is a fast linear-SVM/logistic-regression approximation
    # that excels on sparse high-dimensional text features (TF-IDF/counts),
    # so it's offered via the Text Classification node rather than the
    # general Classification node, which already has logistic_regression and
    # other dense-feature-friendly linear models covering that role.
    tags=["text", "nlp", "linear", "requires_scaling"],
    learns_from_data=True,
)
class SGDClassifierCalculator(SklearnCalculator):
    """SGD Classifier Calculator."""

    def __init__(self):
        """Bind ``SGDClassifier`` as a logistic-regression approximation.

        ``loss="log_loss"`` replaces sklearn's default ``"hinge"``, and that is
        what makes ``predict_proba`` available to the applier — a hinge loss
        yields a linear SVM with no probability output. ``l1_ratio=0.15`` stays
        inert unless ``penalty`` is switched to ``"elasticnet"``.
        """
        super().__init__(
            model_class=SGDClassifier,
            default_params={
                "loss": "log_loss",
                "penalty": "l2",
                "alpha": 0.0001,
                "l1_ratio": 0.15,
                "max_iter": 1000,
            },
            problem_type="classification",
        )
