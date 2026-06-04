"""Probability-calibration model params (CalibratedClassifierCV)."""

from ._field import HyperparameterField

# The calibration method and CV folds are tunable; ``base_estimator`` selects
# which classifier gets calibrated (resolved to an instance in the node).
CALIBRATED_CLASSIFIER_PARAMS = [
    HyperparameterField(
        name="base_estimator",
        label="Base Estimator",
        type="select",
        default="logistic_regression",
        options=[
            {"label": "Logistic Regression", "value": "logistic_regression"},
            {"label": "Random Forest", "value": "random_forest"},
            {"label": "Gradient Boosting", "value": "gradient_boosting"},
            {"label": "Decision Tree", "value": "decision_tree"},
            {"label": "Gaussian Naive Bayes", "value": "gaussian_nb"},
            {"label": "Support Vector Classifier", "value": "svc"},
        ],
        description=(
            "Classifier whose probabilities are calibrated. Uses sensible "
            "defaults for the chosen estimator."
        ),
    ),
    HyperparameterField(
        name="method",
        label="Calibration Method",
        type="select",
        default="sigmoid",
        options=[
            {"label": "Sigmoid (Platt)", "value": "sigmoid"},
            {"label": "Isotonic", "value": "isotonic"},
        ],
        description=(
            "How to map raw scores to calibrated probabilities. "
            "Sigmoid (Platt) suits small datasets; isotonic is non-parametric "
            "and needs more data."
        ),
    ),
    HyperparameterField(
        name="cv",
        label="CV Folds",
        type="number",
        default=5,
        min=2,
        max=10,
        description="Number of cross-validation folds used to fit the calibrator.",
    ),
]
