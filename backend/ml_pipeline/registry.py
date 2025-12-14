from typing import List, Dict, Any
from pydantic import BaseModel


class RegistryItem(BaseModel):
    id: str
    name: str
    category: str
    description: str
    params: Dict[str, Any] = {}


class NodeRegistry:
    @staticmethod
    def get_all_nodes() -> List[RegistryItem]:
        return [
            # --- Data Loading ---
            RegistryItem(
                id="data_loader",
                name="Data Loader",
                category="Data Source",
                description="Loads data from a source.",
                params={
                    "source_id": "string",
                    "date_range": "optional[dict]"
                }
            ),

            # --- Preprocessing (Transformers) ---
            RegistryItem(
                id="TrainTestSplitter",
                name="Train-Test Split",
                category="Preprocessing",
                description="Splits data into training and testing sets.",
                params={"test_size": 0.2, "random_state": 42, "stratify": None}
            ),
            RegistryItem(
                id="feature_target_split",
                name="Feature-Target Split",
                category="Preprocessing",
                description="Separate features (X) from target (y).",
                params={"target_column": "string"}
            ),
            RegistryItem(
                id="SimpleImputer",
                name="Simple Imputer",
                category="Preprocessing",
                description="Imputes missing values using mean, median, or constant.",
                params={"strategy": "mean", "fill_value": None}
            ),
            RegistryItem(
                id="KNNImputer",
                name="KNN Imputer",
                category="Preprocessing",
                description="Imputes missing values using k-Nearest Neighbors.",
                params={"n_neighbors": 5}
            ),
            RegistryItem(
                id="IterativeImputer",
                name="Iterative Imputer (MICE)",
                category="Preprocessing",
                description="Multivariate imputation by chained equations.",
                params={"max_iter": 10, "random_state": 0}
            ),
            RegistryItem(
                id="OneHotEncoder",
                name="One-Hot Encoder",
                category="Preprocessing",
                description="Encodes categorical features as a one-hot numeric array.",
                params={"handle_unknown": "ignore"}
            ),
            RegistryItem(
                id="OrdinalEncoder",
                name="Ordinal Encoder",
                category="Preprocessing",
                description="Encodes categorical features as an integer array.",
                params={}
            ),
            RegistryItem(
                id="LabelEncoder",
                name="Label Encoder",
                category="Preprocessing",
                description="Encode target labels with value between 0 and n_classes-1.",
                params={}
            ),
            RegistryItem(
                id="TargetEncoder",
                name="Target Encoder",
                category="Preprocessing",
                description="Encode categorical features using target statistics.",
                params={"smoothing": 10.0}
            ),
            RegistryItem(
                id="HashEncoder",
                name="Hash Encoder",
                category="Preprocessing",
                description="Encode categorical features using hashing.",
                params={"n_components": 8}
            ),
            RegistryItem(
                id="StandardScaler",
                name="Standard Scaler",
                category="Preprocessing",
                description="Standardize features by removing the mean and scaling to unit variance.",
                params={}
            ),
            RegistryItem(
                id="MinMaxScaler",
                name="Min-Max Scaler",
                category="Preprocessing",
                description="Transform features by scaling each feature to a given range.",
                params={"feature_range": [0, 1]}
            ),
            RegistryItem(
                id="RobustScaler",
                name="Robust Scaler",
                category="Preprocessing",
                description="Scale features using statistics that are robust to outliers.",
                params={"quantile_range": [25.0, 75.0]}
            ),
            RegistryItem(
                id="MaxAbsScaler",
                name="MaxAbs Scaler",
                category="Preprocessing",
                description="Scale each feature by its maximum absolute value.",
                params={}
            ),
            RegistryItem(
                id="IQR",
                name="IQR Outlier Removal",
                category="Preprocessing",
                description="Remove outliers using Interquartile Range.",
                params={"factor": 1.5}
            ),
            RegistryItem(
                id="ZScore",
                name="Z-Score Outlier Removal",
                category="Preprocessing",
                description="Remove outliers using Z-Score.",
                params={"threshold": 3.0}
            ),
            RegistryItem(
                id="Winsorize",
                name="Winsorization",
                category="Preprocessing",
                description="Limit extreme values in the data.",
                params={"limits": [0.05, 0.05]}
            ),
            RegistryItem(
                id="PowerTransformer",
                name="Power Transformer",
                category="Preprocessing",
                description="Apply a power transform featurewise to make data more Gaussian-like.",
                params={"method": "yeo-johnson"}
            ),
            RegistryItem(
                id="SimpleTransformation",
                name="Simple Transformation",
                category="Preprocessing",
                description="Apply simple mathematical transformations (log, sqrt, etc.).",
                params={"func": "log"}
            ),
            RegistryItem(
                id="GeneralBinning",
                name="General Binning",
                category="Preprocessing",
                description="Bin continuous data into intervals.",
                params={"n_bins": 5, "strategy": "uniform"}
            ),
            RegistryItem(
                id="CustomBinning",
                name="Custom Binning",
                category="Preprocessing",
                description="Bin data using custom edges.",
                params={"bins": []}
            ),
            RegistryItem(
                id="KBinsDiscretizer",
                name="K-Bins Discretizer",
                category="Preprocessing",
                description="Bin continuous data into k intervals.",
                params={"n_bins": 5, "encode": "ordinal", "strategy": "uniform"}
            ),
            RegistryItem(
                id="VarianceThreshold",
                name="Variance Threshold",
                category="Preprocessing",
                description="Feature selector that removes all low-variance features.",
                params={"threshold": 0.0}
            ),
            RegistryItem(
                id="CorrelationThreshold",
                name="Correlation Threshold",
                category="Preprocessing",
                description="Remove features highly correlated with others.",
                params={"threshold": 0.9}
            ),
            RegistryItem(
                id="UnivariateSelection",
                name="Univariate Selection",
                category="Preprocessing",
                description="Select features based on univariate statistical tests.",
                params={"k": 10, "score_func": "f_classif"}
            ),
            RegistryItem(
                id="ModelBasedSelection",
                name="Model-Based Selection",
                category="Preprocessing",
                description="Select features using a model (e.g. Random Forest importance).",
                params={"estimator": "RandomForest"}
            ),
            RegistryItem(
                id="feature_selection",
                name="Feature Selection",
                category="Preprocessing",
                description="Select features using various methods (Variance, Correlation, Univariate, Model-based).",
                params={"method": "select_k_best"}
            ),
            RegistryItem(
                id="Casting",
                name="Type Casting",
                category="Preprocessing",
                description="Cast columns to specific data types.",
                params={"target_type": "float"}
            ),
            RegistryItem(
                id="PolynomialFeatures",
                name="Polynomial Features",
                category="Preprocessing",
                description="Generate polynomial and interaction features.",
                params={"degree": 2, "interaction_only": False}
            ),
            RegistryItem(
                id="FeatureGenerationNode",
                name="Feature Generation",
                category="Preprocessing",
                description="Create new features using mathematical expressions.",
                params={"expression": ""}
            ),
            RegistryItem(
                id="Oversampling",
                name="Oversampling",
                category="Preprocessing",
                description="Oversample the minority class.",
                params={"strategy": "auto"}
            ),
            RegistryItem(
                id="Undersampling",
                name="Undersampling",
                category="Preprocessing",
                description="Undersample the majority class.",
                params={"strategy": "auto"}
            ),
            RegistryItem(
                id="DatasetProfile",
                name="Dataset Profile",
                category="Inspection",
                description="Generate a statistical profile of the dataset.",
                params={}
            ),
            RegistryItem(
                id="DataSnapshot",
                name="Data Snapshot",
                category="Inspection",
                description="Save a snapshot of the data at this point.",
                params={}
            ),
            RegistryItem(
                id="TextCleaning",
                name="Text Cleaning",
                category="Preprocessing",
                description="Basic text cleaning operations.",
                params={"lowercase": True, "remove_punctuation": True}
            ),
            RegistryItem(
                id="ValueReplacement",
                name="Value Replacement",
                category="Preprocessing",
                description="Replaces specific values in columns.",
                params={"to_replace": {}, "value": None}
            ),
            RegistryItem(
                id="AliasReplacement",
                name="Alias Replacement",
                category="Preprocessing",
                description="Replace aliases (e.g. 'USA', 'U.S.A.') with a canonical value.",
                params={"mode": "custom"}
            ),
            RegistryItem(
                id="InvalidValueReplacement",
                name="Invalid Value Replacement",
                category="Preprocessing",
                description="Replace invalid values (e.g. negative ages) with NaN.",
                params={"mode": "negative_to_nan"}
            ),
            RegistryItem(
                id="Deduplicate",
                name="Deduplicate",
                category="Preprocessing",
                description="Remove duplicate rows.",
                params={"subset": None}
            ),
            RegistryItem(
                id="DropMissingColumns",
                name="Drop Missing Columns",
                category="Preprocessing",
                description="Drop columns with too many missing values.",
                params={"threshold": 0.5}
            ),
            RegistryItem(
                id="DropMissingRows",
                name="Drop Missing Rows",
                category="Preprocessing",
                description="Drop rows with missing values.",
                params={"threshold": None}
            ),
            RegistryItem(
                id="MissingIndicator",
                name="Missing Indicator",
                category="Preprocessing",
                description="Create binary indicators for missing values.",
                params={}
            ),

            # --- Modeling ---
            RegistryItem(
                id="logistic_regression",
                name="Logistic Regression",
                category="Model",
                description="Logistic Regression classifier.",
                params={"C": 1.0, "penalty": "l2"}
            ),
            RegistryItem(
                id="random_forest_classifier",
                name="Random Forest Classifier",
                category="Model",
                description="A random forest classifier.",
                params={"n_estimators": 100, "max_depth": None}
            ),
            RegistryItem(
                id="ridge_regression",
                name="Ridge Regression",
                category="Model",
                description="Linear least squares with l2 regularization.",
                params={"alpha": 1.0}
            ),
            RegistryItem(
                id="random_forest_regressor",
                name="Random Forest Regressor",
                category="Model",
                description="A random forest regressor.",
                params={"n_estimators": 100, "max_depth": None}
            ),
        ]
