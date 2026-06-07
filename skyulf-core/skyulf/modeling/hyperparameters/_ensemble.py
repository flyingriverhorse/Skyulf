"""Ensemble meta-model params (Voting / Stacking, classification + regression).

``base_estimators`` is a ``multiselect`` field — the list of base-learner keys
the node combines. ``final_estimator`` (stacking only) and ``voting`` (voting
classifier only) round out the per-family options. The string keys map to the
factory tables in :mod:`skyulf.modeling.ensemble`.
"""

from ._field import HyperparameterField

# Selectable base learners, mirrored from ``ensemble.BASE_ESTIMATORS_*``.
# Optional boosters (xgboost / lightgbm) are intentionally omitted from this
# manual picker because they depend on optional wheels; they are still picked up
# automatically when a trained XGBoost/LightGBM node is wired into the ensemble.
_CLF_OPTIONS = [
    {"label": "Logistic Regression", "value": "logistic_regression"},
    {"label": "Random Forest", "value": "random_forest"},
    {"label": "Extra Trees", "value": "extra_trees"},
    {"label": "Gradient Boosting", "value": "gradient_boosting"},
    {"label": "Hist Gradient Boosting", "value": "hist_gradient_boosting"},
    {"label": "AdaBoost", "value": "adaboost"},
    {"label": "Decision Tree", "value": "decision_tree"},
    {"label": "Gaussian Naive Bayes", "value": "gaussian_nb"},
    {"label": "Support Vector Classifier", "value": "svc"},
    {"label": "K-Nearest Neighbors", "value": "knn"},
]

_REG_OPTIONS = [
    {"label": "Linear Regression", "value": "linear_regression"},
    {"label": "Ridge", "value": "ridge"},
    {"label": "Lasso", "value": "lasso"},
    {"label": "ElasticNet", "value": "elasticnet"},
    {"label": "Random Forest", "value": "random_forest"},
    {"label": "Extra Trees", "value": "extra_trees"},
    {"label": "Gradient Boosting", "value": "gradient_boosting"},
    {"label": "Hist Gradient Boosting", "value": "hist_gradient_boosting"},
    {"label": "AdaBoost", "value": "adaboost"},
    {"label": "Decision Tree", "value": "decision_tree"},
    {"label": "Support Vector Regressor", "value": "svr"},
    {"label": "K-Nearest Neighbors", "value": "knn"},
]

_VOTING_FIELD = HyperparameterField(
    name="voting",
    label="Voting Type",
    type="select",
    default="soft",
    options=[
        {"label": "Soft (average probabilities)", "value": "soft"},
        {"label": "Hard (majority vote)", "value": "hard"},
    ],
    description=(
        "Soft averages predicted probabilities (needs predict_proba on every "
        "base model); hard takes the majority class vote."
    ),
)

_CV_FIELD = HyperparameterField(
    name="cv",
    label="CV Folds (stacking)",
    type="number",
    default=5,
    min=2,
    max=10,
    description=(
        "Cross-validation folds used to generate out-of-fold predictions for "
        "the final estimator. Keep small (e.g. 3) when also running an outer "
        "hyperparameter search to limit nested-CV cost."
    ),
)


def _base_estimators_field(options, default):
    return HyperparameterField(
        name="base_estimators",
        label="Base Models",
        type="multiselect",
        default=default,
        options=options,
        description="Base learners combined by the ensemble.",
    )


VOTING_CLASSIFIER_PARAMS = [
    _base_estimators_field(
        _CLF_OPTIONS, ["random_forest", "logistic_regression", "gradient_boosting"]
    ),
    _VOTING_FIELD,
]

STACKING_CLASSIFIER_PARAMS = [
    _base_estimators_field(_CLF_OPTIONS, ["random_forest", "gradient_boosting", "svc"]),
    HyperparameterField(
        name="final_estimator",
        label="Final Estimator",
        type="select",
        default="logistic_regression",
        options=_CLF_OPTIONS,
        description="Meta-learner trained on the base models' out-of-fold predictions.",
    ),
    _CV_FIELD,
]

VOTING_REGRESSOR_PARAMS = [
    _base_estimators_field(
        _REG_OPTIONS, ["linear_regression", "random_forest", "gradient_boosting"]
    ),
]

STACKING_REGRESSOR_PARAMS = [
    _base_estimators_field(_REG_OPTIONS, ["random_forest", "gradient_boosting", "ridge"]),
    HyperparameterField(
        name="final_estimator",
        label="Final Estimator",
        type="select",
        default="ridge",
        options=_REG_OPTIONS,
        description="Meta-learner trained on the base models' out-of-fold predictions.",
    ),
    _CV_FIELD,
]
