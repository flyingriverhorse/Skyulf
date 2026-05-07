"""Support Vector Machines (SVC and SVR share the same hyperparameter set)."""

from ._field import HyperparameterField

SVM_PARAMS = [
    HyperparameterField(
        name="C",
        label="C (Regularization)",
        type="number",
        default=1.0,
        min=0.01,
        max=1000.0,
        description=(
            "Regularization parameter. The strength of the regularization is "
            "inversely proportional to C."
        ),
    ),
    HyperparameterField(
        name="kernel",
        label="Kernel",
        type="select",
        default="rbf",
        options=[
            {"label": "Linear", "value": "linear"},
            {"label": "Poly", "value": "poly"},
            {"label": "RBF", "value": "rbf"},
            {"label": "Sigmoid", "value": "sigmoid"},
        ],
        description="Specifies the kernel type to be used in the algorithm.",
    ),
    HyperparameterField(
        name="gamma",
        label="Gamma",
        type="select",
        default="scale",
        options=[{"label": "Scale", "value": "scale"}, {"label": "Auto", "value": "auto"}],
        description="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.",
    ),
]
