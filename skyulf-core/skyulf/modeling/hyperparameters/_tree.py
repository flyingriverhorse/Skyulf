"""Tree-based ensemble models: Random Forest, Decision Tree, Extra Trees,
Gradient Boosting, AdaBoost, XGBoost, HistGradientBoosting, LightGBM.
"""

from ._field import HyperparameterField

# --- Random Forest (Classifier & Regressor share base set) ---
RANDOM_FOREST_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="The number of trees in the forest.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=100,
        description=(
            "The maximum depth of the tree. If None, nodes are expanded until "
            "all leaves are pure."
        ),
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="The minimum number of samples required to split an internal node.",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        min=1,
        max=20,
        description="The minimum number of samples required to be at a leaf node.",
    ),
    HyperparameterField(
        name="bootstrap",
        label="Bootstrap",
        type="select",
        default=True,
        options=[
            {"label": "True", "value": True},
            {"label": "False", "value": False},
        ],
        description="Whether bootstrap samples are used when building trees.",
    ),
]

# Classifier-only addition: criterion for split quality.
RANDOM_FOREST_CLASSIFIER_PARAMS = RANDOM_FOREST_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="gini",
        options=[
            {"label": "Gini", "value": "gini"},
            {"label": "Entropy", "value": "entropy"},
            {"label": "Log Loss", "value": "log_loss"},
        ],
        description="The function to measure the quality of a split.",
    )
]

# --- Decision Tree ---
DECISION_TREE_PARAMS = [
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=100,
        description="The maximum depth of the tree.",
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="The minimum number of samples required to split an internal node.",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        min=1,
        max=20,
        description="The minimum number of samples required to be at a leaf node.",
    ),
]
DECISION_TREE_CLASSIFIER_PARAMS = DECISION_TREE_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="gini",
        options=[
            {"label": "Gini", "value": "gini"},
            {"label": "Entropy", "value": "entropy"},
            {"label": "Log Loss", "value": "log_loss"},
        ],
        description="The function to measure the quality of a split.",
    )
]
DECISION_TREE_REGRESSOR_PARAMS = DECISION_TREE_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="squared_error",
        options=[
            {"label": "Squared Error", "value": "squared_error"},
            {"label": "Friedman MSE", "value": "friedman_mse"},
            {"label": "Absolute Error", "value": "absolute_error"},
            {"label": "Poisson", "value": "poisson"},
        ],
        description="The function to measure the quality of a split.",
    )
]

# --- Gradient Boosting (Sklearn) ---
GRADIENT_BOOSTING_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="The number of boosting stages to perform.",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.1,
        min=0.001,
        max=1.0,
        description="Shrinks the contribution of each tree by learning_rate.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=3,
        min=1,
        max=20,
        description="Maximum depth of the individual regression estimators.",
    ),
    HyperparameterField(
        name="subsample",
        label="Subsample",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="The fraction of samples to be used for fitting the individual base learners.",
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="Minimum samples required to split an internal node.",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        min=1,
        max=20,
        description="Minimum samples required at a leaf node.",
    ),
]

# --- AdaBoost ---
ADABOOST_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Estimators",
        type="number",
        default=50,
        min=10,
        max=1000,
        step=10,
        description="The maximum number of estimators at which boosting is terminated.",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=1.0,
        min=0.001,
        max=5.0,
        description="Weight applied to each classifier at each boosting iteration.",
    ),
]

# --- XGBoost ---
XGBOOST_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Estimators",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="Number of gradient boosted trees.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=6,
        min=1,
        max=20,
        description="Maximum tree depth for base learners.",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.3,
        min=0.001,
        max=1.0,
        description="Boosting learning rate (eta).",
    ),
    HyperparameterField(
        name="subsample",
        label="Subsample",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Subsample ratio of the training instances.",
    ),
    HyperparameterField(
        name="colsample_bytree",
        label="Colsample By Tree",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Subsample ratio of columns when constructing each tree.",
    ),
    HyperparameterField(
        name="min_child_weight",
        label="Min Child Weight",
        type="number",
        default=1,
        min=0,
        max=50,
        step=1,
        description=(
            "Minimum sum of instance weights in a child. "
            "Higher values regularize against overfitting."
        ),
    ),
    HyperparameterField(
        name="gamma",
        label="Gamma (min_split_loss)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description=(
            "Minimum loss reduction required to make a split. "
            "Higher = more conservative tree growth."
        ),
    ),
    HyperparameterField(
        name="reg_alpha",
        label="L1 Regularization (reg_alpha)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L1 regularization term on leaf weights.",
    ),
    HyperparameterField(
        name="reg_lambda",
        label="L2 Regularization (reg_lambda)",
        type="number",
        default=1.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L2 regularization term on leaf weights. Default sklearn XGBoost is 1.",
    ),
]

# --- Extra Trees (Classifier & Regressor) ---
EXTRA_TREES_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="Number of trees in the forest.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=100,
        description="Maximum depth of the tree. None = unlimited.",
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="Minimum samples required to split an internal node.",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        min=1,
        max=20,
        description="Minimum samples required at a leaf node.",
    ),
    HyperparameterField(
        name="bootstrap",
        label="Bootstrap",
        type="select",
        default=False,
        options=[
            {"label": "False (default)", "value": False},
            {"label": "True", "value": True},
        ],
        description="Whether bootstrap samples are used (False = use full dataset).",
    ),
]

EXTRA_TREES_CLASSIFIER_PARAMS = EXTRA_TREES_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="gini",
        options=[
            {"label": "Gini", "value": "gini"},
            {"label": "Entropy", "value": "entropy"},
            {"label": "Log Loss", "value": "log_loss"},
        ],
        description="Function to measure the quality of a split.",
    )
]

EXTRA_TREES_REGRESSOR_PARAMS = EXTRA_TREES_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="squared_error",
        options=[
            {"label": "Squared Error", "value": "squared_error"},
            {"label": "Absolute Error", "value": "absolute_error"},
            {"label": "Friedman MSE", "value": "friedman_mse"},
        ],
        description="Function to measure the quality of a split.",
    )
]

# --- HistGradientBoosting (Classifier & Regressor) ---
HIST_GRADIENT_BOOSTING_PARAMS = [
    HyperparameterField(
        name="max_iter",
        label="Max Iterations (Trees)",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="Maximum number of iterations (boosting rounds).",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.1,
        min=0.001,
        max=1.0,
        description="Shrinks the contribution of each tree.",
    ),
    HyperparameterField(
        name="max_leaf_nodes",
        label="Max Leaf Nodes",
        type="number",
        default=31,
        min=2,
        max=255,
        description="Maximum number of leaves per tree.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=20,
        description=(
            "Maximum depth per tree. None = unlimited " "(max_leaf_nodes is the effective limit)."
        ),
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=20,
        min=1,
        max=200,
        description="Minimum samples at a leaf node.",
    ),
    HyperparameterField(
        name="l2_regularization",
        label="L2 Regularization",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L2 penalty applied to the leaves' values.",
    ),
    HyperparameterField(
        name="max_bins",
        label="Max Bins",
        type="number",
        default=255,
        min=10,
        max=255,
        step=5,
        description=(
            "Maximum number of bins for feature discretisation. "
            "Higher = more precise but slower."
        ),
    ),
]

# --- LightGBM (Classifier & Regressor) ---
LGBM_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Estimators",
        type="number",
        default=100,
        min=10,
        max=2000,
        step=10,
        description="Number of boosting rounds.",
    ),
    HyperparameterField(
        name="num_leaves",
        label="Num Leaves",
        type="number",
        default=31,
        min=2,
        max=512,
        step=1,
        description=(
            "Maximum number of leaves per tree. "
            "Controls model complexity; increase for more accuracy."
        ),
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.1,
        min=0.001,
        max=1.0,
        description="Shrinks the contribution of each tree.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=-1,
        min=-1,
        max=50,
        description="Maximum tree depth. -1 = unlimited (controlled by num_leaves).",
    ),
    HyperparameterField(
        name="min_child_samples",
        label="Min Child Samples",
        type="number",
        default=20,
        min=1,
        max=200,
        description="Minimum samples required in a leaf. Regularises against overfitting.",
    ),
    HyperparameterField(
        name="subsample",
        label="Subsample (Bagging Fraction)",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Fraction of training data sampled per iteration.",
    ),
    HyperparameterField(
        name="colsample_bytree",
        label="Colsample By Tree (Feature Fraction)",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Fraction of features sampled per tree.",
    ),
    HyperparameterField(
        name="reg_alpha",
        label="L1 Regularization (reg_alpha)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L1 regularization term on leaf weights.",
    ),
    HyperparameterField(
        name="reg_lambda",
        label="L2 Regularization (reg_lambda)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L2 regularization term on leaf weights.",
    ),
    HyperparameterField(
        name="boosting_type",
        label="Boosting Type",
        type="select",
        default="gbdt",
        options=[
            {"label": "GBDT (default)", "value": "gbdt"},
            {"label": "DART (dropout)", "value": "dart"},
            {"label": "GOSS (gradient-based sampling)", "value": "goss"},
        ],
        description=(
            "GBDT is standard gradient boosting. DART adds dropout. "
            "GOSS samples by gradient magnitude."
        ),
    ),
]
