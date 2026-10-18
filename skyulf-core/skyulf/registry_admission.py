"""Spark and serving-endpoint admission declared for every built-in node.

Spark pandas-UDF scoring and serving endpoints accept a fitted pipeline only
through the review in ``skyulf.inference.partition_safety``. This table records,
for each built-in node, whether that review admits it and, if not, why, so a
new node cannot be registered without that decision being written down. Tests
enforce it (``tests/unit/test_registry_admission.py``); nothing reads it at
runtime, and it never widens what the review admits.

A node whose calculator declares a pandas ``python_batch`` apply capability is
admitted through its ``execution_capabilities`` and needs no entry here.
"""

from .registry import NodeRegistry

ADMITTED_BY_CAPABILITY = "admitted: pandas python_batch apply capability"
ADMITTED_MODEL = "admitted: reviewed exact model applier"
INFERENCE_SKIP = "admitted: training-only row filter, skipped at inference"
PROBED = "admitted: only with sealed training probe evidence"
PROBED_BUILTIN = "admitted: row-preserving built-in state with sealed training probe evidence"
PROBED_EXCLUSION = "admitted: fitted row exclusions with sealed training probe evidence"
SPLIT_MARKER = "admitted: split marker, absent from the fitted history"

UNREVIEWED = "not admitted: no reviewed partition-safe applier yet"
WINDOW = "not admitted: reads neighbouring rows (window context)"
RESAMPLING = "not admitted: training-only resampling changes the row count"
INSPECTION = "not admitted: inspection node without a reviewed apply"
TEXT = "not admitted: text vectorizer or embedder without a reviewed applier"
ROW_FILTER = "not admitted: fitted preprocessing removes prediction rows"
GLOBAL_STATE = "not admitted: active fitted transformation can depend on the whole request"
UNREVIEWED_MODEL = "not admitted: model without a reviewed exact applier"
PROBED_MODEL = "admitted: exact model and nested members with sealed training probe evidence"
CLUSTERING = "not admitted: clustering is not a Databricks task type"
ENSEMBLE = "not admitted: stacking and voting composition is not reviewed"

DECLARATIONS: dict[str, str] = {
    **dict.fromkeys(
        (
            "linear_regression",
            "logistic_regression",
            "decision_tree_regressor",
            "decision_tree_classifier",
            "random_forest_regressor",
            "random_forest_classifier",
            "extra_trees_regressor",
            "extra_trees_classifier",
            "xgboost_regressor",
        ),
        ADMITTED_MODEL,
    ),
    **dict.fromkeys(("Deduplicate", "DropMissingRows", "RowFilterFunction"), INFERENCE_SKIP),
    **dict.fromkeys(("ColumnFunction", "FittedFunction"), PROBED),
    **dict.fromkeys(("TrainTestSplitter", "Split", "feature_target_split"), SPLIT_MARKER),
    **dict.fromkeys(
        (
            "AliasReplacement",
            "Casting",
            "CorrelationThreshold",
            "CustomBinning",
            "DateFeatures",
            "DropMissingColumns",
            "DummyEncoder",
            "FeatureGeneration",
            "FeatureGenerationNode",
            "FeatureMath",
            "FrequencyEncoder",
            "GeneralBinning",
            "GeneralTransformation",
            "GeoDistance",
            "H3Index",
            "HashEncoder",
            "InvalidValueReplacement",
            "IterativeImputer",
            "KBinsDiscretizer",
            "KNNImputer",
            "LabelEncoder",
            "MaxAbsScaler",
            "MissingIndicator",
            "ModelBasedSelection",
            "Normalizer",
            "OrdinalEncoder",
            "PolynomialFeatures",
            "PolynomialFeaturesNode",
            "QuantileTransformer",
            "RareCategoryGrouping",
            "RobustScaler",
            "SimpleTransformation",
            "TargetEncoder",
            "TextCleaning",
            "UnivariateSelection",
            "ValueReplacement",
            "VarianceThreshold",
            "WOEEncoder",
            "Winsorize",
            "feature_selection",
        ),
        PROBED_BUILTIN,
    ),
    **dict.fromkeys(
        (
            "IQR",
            "EllipticEnvelope",
            "IsolationForest",
            "LocalOutlierFactor",
            "ManualBounds",
            "ZScore",
            "PowerTransformer",
        ),
        PROBED_EXCLUSION,
    ),
    **dict.fromkeys(("LagFeatures", "RollingAggregate"), WINDOW),
    **dict.fromkeys(("Oversampling", "Undersampling"), RESAMPLING),
    **dict.fromkeys(("DataSnapshot", "DatasetProfile"), INSPECTION),
    **dict.fromkeys(
        (
            "count_vectorizer",
            "hashing_vectorizer",
            "tfidf_vectorizer",
            "tokenizer",
        ),
        PROBED_BUILTIN,
    ),
    "sentence_embedder": TEXT,
    **dict.fromkeys(
        (
            "adaboost_classifier",
            "adaboost_regressor",
            "bernoulli_nb",
            "calibrated_classifier",
            "elasticnet_regression",
            "gaussian_nb",
            "gradient_boosting_classifier",
            "gradient_boosting_regressor",
            "hist_gradient_boosting_classifier",
            "hist_gradient_boosting_regressor",
            "k_neighbors_classifier",
            "k_neighbors_regressor",
            "lasso_regression",
            "lgbm_classifier",
            "lgbm_regressor",
            "multinomial_nb",
            "ridge_regression",
            "sgd_classifier",
            "svc",
            "svr",
            "xgboost_classifier",
        ),
        PROBED_MODEL,
    ),
    **dict.fromkeys(("birch", "gaussian_mixture", "kmeans", "minibatch_kmeans"), CLUSTERING),
    **dict.fromkeys(
        ("stacking_classifier", "stacking_regressor", "voting_classifier", "voting_regressor"),
        PROBED_MODEL,
    ),
}


def spark_admission(node_type: str) -> str | None:
    """Return the node's declared Spark/endpoint admission, or None when it declares none.

    Args:
        node_type: A registered node ID.

    Returns:
        A string starting with ``admitted:`` or ``not admitted:``, or None for an
        undeclared node.

    Raises:
        ValueError: The node is not registered.
    """
    capabilities = vars(NodeRegistry.get_calculator(node_type)).get(
        "__execution_capabilities__", ()
    )
    if any(
        capability.engine == "pandas"
        and capability.operation == "apply"
        and capability.execution_kind == "python_batch"
        for capability in capabilities
    ):
        return ADMITTED_BY_CAPABILITY
    return DECLARATIONS.get(node_type)
