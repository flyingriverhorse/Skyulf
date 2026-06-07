"""Ensemble meta-models — Voting and Stacking (classification + regression).

Each node combines several *base* learners into one estimator. The frontend
sends ``base_estimators`` as a list of string keys (e.g.
``["random_forest", "logistic_regression"]``); they are resolved here into fresh
estimator instances before the sklearn meta-estimator is constructed — mirroring
the ``base_estimator`` pattern in :class:`CalibratedClassifierCalculator`.

Strategies:

* **Voting** — fit each base model once, then average (soft = mean proba,
  hard = majority vote) or average predictions (regression). No internal CV.
* **Stacking** — fit base models, then train a *final estimator* on their
  out-of-fold predictions. The internal ``cv`` (default 5) generates the OOF
  predictions; without it the meta-learner leaks/over-fits. Same shape as
  ``CalibratedClassifierCV``'s internal ``cv``.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Optional gradient-boosting libraries. Mirrors the guarded imports in
# ``modeling.classification`` / ``modeling.regression`` — when the wheel is not
# installed the base learner is simply absent from the selectable factories.
try:
    from xgboost import XGBClassifier, XGBRegressor  # ty: ignore[unresolved-import]

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import (  # ty: ignore[unresolved-import]
        LGBMClassifier,
        LGBMRegressor,
    )

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator

logger = logging.getLogger(__name__)


# Selectable base learners. Each value is a zero-arg factory so every fit builds
# fresh, unfitted instances (sklearn clones them internally, but rebuilding keeps
# repeated fits independent of any shared state).
BASE_ESTIMATORS_CLF: Dict[str, Callable[[], BaseEstimator]] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=42),
    "adaboost": lambda: AdaBoostClassifier(random_state=42),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "gaussian_nb": lambda: GaussianNB(),
    "svc": lambda: SVC(probability=True, random_state=42),
    "knn": lambda: KNeighborsClassifier(),
}

BASE_ESTIMATORS_REG: Dict[str, Callable[[], BaseEstimator]] = {
    "linear_regression": lambda: LinearRegression(),
    "ridge": lambda: Ridge(),
    "lasso": lambda: Lasso(),
    "elasticnet": lambda: ElasticNet(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=42),
    "adaboost": lambda: AdaBoostRegressor(random_state=42),
    "decision_tree": lambda: DecisionTreeRegressor(random_state=42),
    "svr": lambda: SVR(),
    "knn": lambda: KNeighborsRegressor(),
}

# Append the optional boosters only when their library is importable so the
# selectable set always matches what can actually be constructed at fit time.
if XGBOOST_AVAILABLE:
    BASE_ESTIMATORS_CLF["xgboost"] = lambda: XGBClassifier(random_state=42)
    BASE_ESTIMATORS_REG["xgboost"] = lambda: XGBRegressor(random_state=42)

if LIGHTGBM_AVAILABLE:
    BASE_ESTIMATORS_CLF["lightgbm"] = lambda: LGBMClassifier(random_state=42, verbose=-1)
    BASE_ESTIMATORS_REG["lightgbm"] = lambda: LGBMRegressor(random_state=42, verbose=-1)


class _BaseEnsembleCalculator(SklearnCalculator):
    """Shared resolver: string keys → ``estimators=[(name, instance), ...]``.

    Subclasses set the factory map, default base/final keys, and whether the
    node is a stacking (has a ``final_estimator`` + ``cv``) or voting model.

    Per-base-model hyperparameters are supported via ``base_estimator_params``
    (a ``{name: {param: value}}`` map) and ``final_estimator_params``. Nested
    ``<name>__<param>`` / ``final_estimator__<param>`` keys (produced by the
    hyperparameter tuner) are absorbed into the same maps so tuned values are
    applied during basic training and post-tuning cross-validation alike.
    """

    BASE_ESTIMATORS: Dict[str, Callable[[], BaseEstimator]] = {}
    DEFAULT_KEYS: Tuple[str, ...] = ()
    DEFAULT_FINAL_KEY: str = ""
    MODEL_KEY: str = ""  # Registry id, e.g. "voting_classifier".
    IS_STACKING: bool = False
    HAS_VOTING: bool = False  # Only VotingClassifier exposes the ``voting`` param.

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Structural config remembered when tuning so the meta-estimator can be
        # rebuilt with the user's base learners during post-tuning CV refits.
        self._tuning_base_config: Dict[str, Any] = {}

    @property
    def default_params(self) -> Dict[str, Any]:
        """Base defaults, plus resolved ``estimators`` while tuning.

        The tuner builds the meta-estimator from ``default_params``; without the
        resolved ``estimators`` a bare ``VotingClassifier()`` would raise. When
        tuning, the structural selection captured by :meth:`prepare_tuning_params`
        is merged in here.
        """
        params = dict(self._default_params)
        if self._tuning_base_config:
            resolved = self._resolve_estimators(dict(self._tuning_base_config))
            for key in ("estimators", "final_estimator", "voting", "cv", "passthrough", "weights", "n_jobs"):
                if key in resolved:
                    params[key] = resolved[key]
        return params

    def fit(
        self,
        X: Any,
        y: Any,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[..., Any]] = None,
        log_callback: Optional[Callable[..., Any]] = None,
        validation_data: Any = None,
    ) -> Any:
        config = self._inject_tuning_base_config(config)
        config = self._resolve_estimators(config)
        return super().fit(X, y, config, progress_callback, log_callback, validation_data)

    # --- tuning hooks -------------------------------------------------------

    def prepare_tuning_params(self, config: Dict[str, Any]) -> None:
        """Remember the structural selection so the tuner can build the model."""
        src = config.get("params") if isinstance(config.get("params"), dict) else config
        src = src or {}
        keep = (
            "base_estimators",
            "final_estimator",
            "voting",
            "cv",
            "passthrough",
            "weights",
            "n_jobs",
            "base_estimator_params",
            "final_estimator_params",
        )
        self._tuning_base_config = {k: src[k] for k in keep if k in src}

    def build_tuning_search_space(self, config: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Auto-build the ensemble's tuning space.

        Always includes the meta-params (``voting`` / ``cv``). When
        ``tune_base_models`` is set, each chosen base learner's grid is expanded
        into nested ``<name>__<param>`` keys (plus ``final_estimator__<param>``).
        """
        # Lazy import avoids a circular dependency (the hyperparameters package
        # imports the ensemble PARAM definitions at module load).
        from .hyperparameters import build_ensemble_search_space, get_default_search_space

        src = config.get("params") if isinstance(config.get("params"), dict) else config
        src = src or {}
        if not src.get("tune_base_models"):
            return dict(get_default_search_space(self.MODEL_KEY, strategy))
        final_est = src.get("final_estimator") or ""
        return build_ensemble_search_space(
            self.MODEL_KEY,
            src.get("base_estimators") or list(self.DEFAULT_KEYS),
            final_estimator=final_est if self.IS_STACKING else "",
            strategy=strategy,
            problem_type=self.problem_type,
        )

    def _inject_tuning_base_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge the captured structural config when it is missing.

        During post-tuning CV the config carries only ``best_params`` (no base
        selection); this restores the user's base learners so the refit evaluates
        the same ensemble that was tuned.
        """
        if not self._tuning_base_config:
            return config
        resolved = dict(config) if config else {}
        nested = isinstance(resolved.get("params"), dict)
        bucket = resolved["params"] if nested else resolved
        if "base_estimators" in bucket:
            return config
        merged = {**self._tuning_base_config, **bucket}
        if nested:
            return {**resolved, "params": merged}
        return merged

    # --- structural helpers -------------------------------------------------

    def _build_estimators(
        self, keys: Any, params_map: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, BaseEstimator]]:
        """Translate a list of string keys into ``(name, instance)`` tuples.

        Unknown keys are skipped with a warning; an empty/invalid selection
        falls back to ``DEFAULT_KEYS`` so the node always has something to fit.
        Per-key params (if any) are applied via ``set_params``.
        """
        if not isinstance(keys, (list, tuple)) or not keys:
            keys = self.DEFAULT_KEYS
        params_map = params_map if isinstance(params_map, dict) else {}
        estimators: List[Tuple[str, BaseEstimator]] = []
        seen: set = set()
        for key in keys:
            if not isinstance(key, str) or key in seen:
                continue
            factory = self.BASE_ESTIMATORS.get(key)
            if factory is None:
                logger.warning("Unknown base estimator '%s'; skipping.", key)
                continue
            estimators.append((key, self._apply_params(factory(), params_map.get(key), key)))
            seen.add(key)
        if not estimators:
            estimators = [(k, self.BASE_ESTIMATORS[k]()) for k in self.DEFAULT_KEYS]
        return estimators

    @staticmethod
    def _apply_params(estimator: BaseEstimator, params: Any, name: str) -> BaseEstimator:
        """Apply a ``{param: value}`` map to *estimator*, ignoring bad params."""
        if isinstance(params, dict) and params:
            try:
                estimator.set_params(**params)
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid params for estimator '%s': %s", name, exc)
        return estimator

    def _resolve_final_estimator(self, key: Any, params: Any = None) -> BaseEstimator:
        """Resolve the stacking meta-learner key, defaulting on unknown/missing."""
        factory = self.BASE_ESTIMATORS.get(key) if isinstance(key, str) else None
        if factory is None:
            if isinstance(key, str):
                logger.warning("Unknown final_estimator '%s'; using default.", key)
            factory = self.BASE_ESTIMATORS[self.DEFAULT_FINAL_KEY]
            key = self.DEFAULT_FINAL_KEY
        return self._apply_params(factory(), params, str(key))

    def _clean_meta_keys(self, bucket: Dict[str, Any], final_params: Any = None) -> None:
        """Keep only the meta-keys valid for this estimator family."""
        # ``n_jobs`` (parallel base-model fitting) is valid for every family;
        # coerce to int and drop anything malformed so sklearn never sees junk.
        if "n_jobs" in bucket:
            try:
                bucket["n_jobs"] = int(bucket["n_jobs"])
            except (TypeError, ValueError):
                bucket.pop("n_jobs", None)
        if self.IS_STACKING:
            bucket["final_estimator"] = self._resolve_final_estimator(
                bucket.pop("final_estimator", None), final_params
            )
            bucket.pop("voting", None)
            bucket.pop("weights", None)
            # ``passthrough`` (let the meta-learner also see the raw features) is
            # a valid Stacking-only param; coerce to bool and keep it.
            if "passthrough" in bucket:
                bucket["passthrough"] = bool(bucket["passthrough"])
        else:
            bucket.pop("final_estimator", None)
            bucket.pop("cv", None)
            # ``passthrough`` is meaningless for Voting — drop it so the sklearn
            # constructor does not reject an unexpected keyword.
            bucket.pop("passthrough", None)
            if not self.HAS_VOTING:
                bucket.pop("voting", None)
            # ``weights`` (per-base-model relative weight) is Voting-only. Keep it
            # only when it is a list/tuple matching the estimator count; otherwise
            # drop it so sklearn falls back to equal weighting instead of raising.
            weights = bucket.get("weights")
            estimators = bucket.get("estimators") or []
            if not (
                isinstance(weights, (list, tuple))
                and len(weights) == len(estimators)
                and len(weights) > 0
            ):
                bucket.pop("weights", None)

    @staticmethod
    def _absorb_nested_keys(
        bucket: Dict[str, Any],
        base_params: Dict[str, Any],
        final_params: Dict[str, Any],
    ) -> None:
        """Fold ``<name>__<param>`` keys into the per-estimator param maps."""
        nested = [k for k in bucket if isinstance(k, str) and "__" in k]
        for key in nested:
            prefix, _, param = key.partition("__")
            value = bucket.pop(key)
            if prefix == "final_estimator":
                final_params[param] = value
            else:
                base_params.setdefault(prefix, {})[param] = value

    def _resolve_estimators(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Inject ``estimators`` (+ ``final_estimator`` for stacking) into config.

        Supports both the flat config shape and the nested ``{"params": {...}}``
        shape used by the model-training payload.
        """
        resolved = dict(config) if config else {}
        nested = isinstance(resolved.get("params"), dict)
        bucket = dict(resolved["params"]) if nested else resolved
        base_params: Dict[str, Any] = dict(bucket.pop("base_estimator_params", None) or {})
        final_params: Dict[str, Any] = dict(bucket.pop("final_estimator_params", None) or {})
        self._absorb_nested_keys(bucket, base_params, final_params)
        bucket["estimators"] = self._build_estimators(
            bucket.pop("base_estimators", None), base_params
        )
        self._clean_meta_keys(bucket, final_params or None)
        if nested:
            resolved["params"] = bucket
            return resolved
        return bucket


# --- Voting Classifier ---
class VotingClassifierApplier(SklearnApplier):
    """Voting Classifier Applier (hard/soft vote over base classifiers)."""


@NodeRegistry.register("voting_classifier", VotingClassifierApplier)
@node_meta(
    id="voting_classifier",
    name="Voting Classifier",
    category="Ensemble",
    description=(
        "Combines several classifiers by majority vote (hard) or averaged "
        "probabilities (soft). Fits each base model once; no internal CV."
    ),
    params={
        "base_estimators": ["random_forest", "logistic_regression", "gradient_boosting"],
        "voting": "soft",
    },
    tags=["requires_scaling"],
)
class VotingClassifierCalculator(_BaseEnsembleCalculator):
    """Voting Classifier Calculator with selectable base learners."""

    BASE_ESTIMATORS = BASE_ESTIMATORS_CLF
    DEFAULT_KEYS = ("random_forest", "logistic_regression", "gradient_boosting")
    MODEL_KEY = "voting_classifier"
    HAS_VOTING = True

    def __init__(self):
        super().__init__(
            model_class=VotingClassifier,
            default_params={"voting": "soft"},
            problem_type="classification",
        )


# --- Stacking Classifier ---
class StackingClassifierApplier(SklearnApplier):
    """Stacking Classifier Applier (meta-learner over base classifiers)."""


@NodeRegistry.register("stacking_classifier", StackingClassifierApplier)
@node_meta(
    id="stacking_classifier",
    name="Stacking Classifier",
    category="Ensemble",
    description=(
        "Trains a final classifier on the out-of-fold predictions of several "
        "base classifiers. Uses internal CV folds to avoid leakage."
    ),
    params={
        "base_estimators": ["random_forest", "gradient_boosting", "svc"],
        "final_estimator": "logistic_regression",
        "cv": 5,
    },
    tags=["requires_scaling"],
)
class StackingClassifierCalculator(_BaseEnsembleCalculator):
    """Stacking Classifier Calculator with selectable base + final learners."""

    BASE_ESTIMATORS = BASE_ESTIMATORS_CLF
    DEFAULT_KEYS = ("random_forest", "gradient_boosting", "svc")
    DEFAULT_FINAL_KEY = "logistic_regression"
    MODEL_KEY = "stacking_classifier"
    IS_STACKING = True

    def __init__(self):
        super().__init__(
            model_class=StackingClassifier,
            default_params={"cv": 5},
            problem_type="classification",
        )


# --- Voting Regressor ---
class VotingRegressorApplier(SklearnApplier):
    """Voting Regressor Applier (averaged predictions over base regressors)."""


@NodeRegistry.register("voting_regressor", VotingRegressorApplier)
@node_meta(
    id="voting_regressor",
    name="Voting Regressor",
    category="Ensemble",
    description=(
        "Averages the predictions of several regressors (optionally weighted). "
        "Fits each base model once; no internal CV."
    ),
    params={
        "base_estimators": ["linear_regression", "random_forest", "gradient_boosting"],
    },
    tags=["requires_scaling"],
)
class VotingRegressorCalculator(_BaseEnsembleCalculator):
    """Voting Regressor Calculator with selectable base learners."""

    BASE_ESTIMATORS = BASE_ESTIMATORS_REG
    DEFAULT_KEYS = ("linear_regression", "random_forest", "gradient_boosting")
    MODEL_KEY = "voting_regressor"

    def __init__(self):
        super().__init__(
            model_class=VotingRegressor,
            default_params={},
            problem_type="regression",
        )


# --- Stacking Regressor ---
class StackingRegressorApplier(SklearnApplier):
    """Stacking Regressor Applier (meta-learner over base regressors)."""


@NodeRegistry.register("stacking_regressor", StackingRegressorApplier)
@node_meta(
    id="stacking_regressor",
    name="Stacking Regressor",
    category="Ensemble",
    description=(
        "Trains a final regressor on the out-of-fold predictions of several "
        "base regressors. Uses internal CV folds to avoid leakage."
    ),
    params={
        "base_estimators": ["random_forest", "gradient_boosting", "ridge"],
        "final_estimator": "ridge",
        "cv": 5,
    },
    tags=["requires_scaling"],
)
class StackingRegressorCalculator(_BaseEnsembleCalculator):
    """Stacking Regressor Calculator with selectable base + final learners."""

    BASE_ESTIMATORS = BASE_ESTIMATORS_REG
    DEFAULT_KEYS = ("random_forest", "gradient_boosting", "ridge")
    DEFAULT_FINAL_KEY = "ridge"
    MODEL_KEY = "stacking_regressor"
    IS_STACKING = True

    def __init__(self):
        super().__init__(
            model_class=StackingRegressor,
            default_params={"cv": 5},
            problem_type="regression",
        )
