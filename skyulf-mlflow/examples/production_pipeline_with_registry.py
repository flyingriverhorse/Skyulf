"""
Production Pipeline Example with Model Registry
================================================

This example demonstrates a COMPLETE end-to-end production-ready ML pipeline:

0. **EDA Phase**: Exploratory Data Analysis
   - Data quality reports (missing values, duplicates, data types)
   - Domain analysis (detect domain characteristics)
   - Domain-specific recommendations

1. **Data Loading**: Load and prepare dataset
   
2. **Feature Engineering**: 
   - FeatureMath (ratio operations, arithmetic operations)
   - SmartBinning (custom bins with labels)
   
3. **Preprocessing**:
   - OneHotEncoder for categorical variables
   - StandardScaler for numerical features
   
4. **Model Training**: 
   - RandomForestClassifier with class balancing
   
5. **Evaluation**: 
   - Multiple metrics (accuracy, f1, roc_auc, precision, recall)
   - Classification report
   
6. **Model Registry**: 
   - Save ENTIRE pipeline (feature engineering + preprocessing + model)
   - Version tracking and metadata
   
7. **Production Inference**: 
   - Load pipeline from registry
   - Make predictions with train-inference parity
   
8. **Version Comparison**: 
   - Compare multiple model versions
   - Track performance metrics over time

Key Principle: EVERYTHING is inside the model artifact for production safety.
Custom ProductionPipeline class encapsulates all transformations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from skyulf_mlflow_library.features.transform import FeatureMath, SmartBinning
from skyulf_mlflow_library.features.encoding import OneHotEncoder
from skyulf_mlflow_library.modeling.registry import ModelRegistry
from skyulf_mlflow_library.modeling import MetricsCalculator
from skyulf_mlflow_library.eda.quality import generate_quality_report
from skyulf_mlflow_library.eda.domain import infer_domain


# ============================================================================
# STEP 0: EDA - Understand Your Data First
# ============================================================================

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset.
    Shows quality issues and domain insights before modeling.
    """
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Step 1: Data Quality Report
    print("\n📊 Generating Data Quality Report...")
    quality_report = generate_quality_report(df)
    
    summary = quality_report['summary']
    print(f"\nDataset Shape: {summary['rows']} rows × {summary['columns']} columns")
    print(f"\nData Completeness: {summary['completeness_pct']:.2f}%")
    print(f"Quality Score: {quality_report['quality_score']:.2f}/100")
    
    if summary.get('duplicate_rows', 0) > 0:
        print(f"\n⚠️  Duplicate Rows: {summary['duplicate_rows']}")
    
    # Show quality checks
    print(f"\n✅ Quality Checks:")
    checks = quality_report['quality_checks']
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name.replace('_', ' ').title()}")
    
    # Show insights
    if quality_report.get('insights'):
        print(f"\n💡 Key Insights:")
        for insight in quality_report['insights'][:3]:
            print(f"  • {insight}")
    
    # Step 2: Domain Analysis
    print("\n🔍 Analyzing Domain Characteristics...")
    domain_result = infer_domain(df)
    domain_dict = domain_result.to_dict()
    
    print(f"\nDetected Domain: {domain_dict['primary_domain']}")
    print(f"Confidence: {domain_dict['primary_confidence']:.2%}")
    
    if domain_dict.get('secondary_domains'):
        secondary = ', '.join([d['domain'] for d in domain_dict['secondary_domains'][:3]])
        print(f"Secondary Domains: {secondary}")
    
    print("\n💡 Domain-Specific Recommendations:")
    for i, rec in enumerate(domain_dict.get('recommendations', [])[:5], 1):
        print(f"  {i}. {rec}")
    
    return quality_report, domain_dict


# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

def load_data():
    """Load sample customer churn data."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1000, 1000 + n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 60, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples),
        'payment_method': np.random.choice(['Electronic Check', 'Credit Card', 'Bank Transfer'], n_samples),
        'num_services': np.random.randint(1, 8, n_samples),
        'support_tickets': np.random.randint(0, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target (churn) with some logic
    churn_prob = (
        0.1 +
        0.2 * (df['tenure_months'] < 12) +
        0.15 * (df['contract_type'] == 'Month-to-Month') +
        0.1 * (df['support_tickets'] > 5) +
        0.05 * (df['monthly_charges'] > 100)
    )
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df


# ============================================================================
# STEP 2: Define Production Pipeline
# ============================================================================

class ProductionPipeline:
    """
    Production-ready pipeline that encapsulates ALL preprocessing and modeling.
    
    This ensures train-inference parity - the same transformations are applied
    at training and prediction time.
    """
    
    def __init__(self):
        self.feature_math = None
        self.encoder = None
        self.scaler = None
        self.model = None
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
        
    def fit(self, X, y):
        """
        Fit the entire pipeline: feature engineering → encoding → scaling → model.
        """
        X_copy = X.copy()
        
        # Store column types for reproducibility
        self.feature_columns = X_copy.columns.tolist()
        self.categorical_columns = X_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = X_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove ID columns if present
        if 'customer_id' in self.numerical_columns:
            self.numerical_columns.remove('customer_id')
            X_copy = X_copy.drop(columns=['customer_id'])
        
        # STEP 1: Feature Engineering
        print("Step 1: Feature Engineering...")
        
        # 1a. Mathematical operations
        feature_operations = [
            {
                'type': 'ratio',
                'numerator': ['total_charges'],
                'denominator': ['tenure_months'],
                'output': 'avg_monthly_spend'
            },
            {
                'type': 'arithmetic',
                'method': 'multiply',
                'columns': ['monthly_charges', 'num_services'],
                'output': 'service_value'
            }
        ]
        
        self.feature_math = FeatureMath(operations=feature_operations)
        X_engineered = self.feature_math.fit_transform(X_copy)
        
        # 1b. Binning (separate transformer)
        self.binner = SmartBinning(
            strategy='custom',
            columns=['age'],
            bins={'age': [18, 30, 45, 60, 70]},
            labels={'age': ['Young', 'Adult', 'Middle', 'Senior']},
            suffix='_group'
        )
        X_engineered = self.binner.fit_transform(X_engineered)
        
        # Update categorical columns if new ones were created (including binned columns)
        new_categorical = [col for col in X_engineered.columns if col not in X_copy.columns 
                          and X_engineered[col].dtype in ['object', 'category']]
        if new_categorical:
            self.categorical_columns.extend(new_categorical)
        
        created_features = [op['output'] for op in feature_operations] + [col for col in X_engineered.columns if '_group' in col]
        print(f"  Created features: {created_features}")
        
        # STEP 2: Encoding
        print("Step 2: Encoding categorical variables...")
        self.encoder = OneHotEncoder(columns=self.categorical_columns)
        X_encoded = self.encoder.fit_transform(X_engineered)
        
        print(f"  Encoded columns: {self.categorical_columns}")
        
        # STEP 3: Scaling
        print("Step 3: Scaling numerical features...")
        numerical_cols_present = [col for col in self.numerical_columns if col in X_encoded.columns]
        
        self.scaler = StandardScaler()
        X_scaled = X_encoded.copy()
        X_scaled[numerical_cols_present] = self.scaler.fit_transform(X_encoded[numerical_cols_present])
        
        print(f"  Scaled columns: {numerical_cols_present}")
        
        # STEP 4: Train Model
        print("Step 4: Training model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        print("  Model trained successfully!")
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the full pipeline.
        Ensures train-inference parity.
        """
        X_copy = X.copy()
        
        # Remove ID columns if present
        if 'customer_id' in X_copy.columns:
            X_copy = X_copy.drop(columns=['customer_id'])
        
        # Apply the same transformations as training
        X_engineered = self.feature_math.transform(X_copy)
        X_engineered = self.binner.transform(X_engineered)
        X_encoded = self.encoder.transform(X_engineered)
        
        # Scale numerical features
        numerical_cols_present = [col for col in self.numerical_columns if col in X_encoded.columns]
        X_scaled = X_encoded.copy()
        X_scaled[numerical_cols_present] = self.scaler.transform(X_encoded[numerical_cols_present])
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        """
        X_copy = X.copy()
        
        if 'customer_id' in X_copy.columns:
            X_copy = X_copy.drop(columns=['customer_id'])
        
        X_engineered = self.feature_math.transform(X_copy)
        X_engineered = self.binner.transform(X_engineered)
        X_encoded = self.encoder.transform(X_engineered)
        
        numerical_cols_present = [col for col in self.numerical_columns if col in X_encoded.columns]
        X_scaled = X_encoded.copy()
        X_scaled[numerical_cols_present] = self.scaler.transform(X_encoded[numerical_cols_present])
        
        return self.model.predict_proba(X_scaled)


# ============================================================================
# STEP 3: Train and Evaluate
# ============================================================================

def train_and_evaluate(df):
    """
    Complete training workflow with evaluation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target column 'churn'.
    """
    print("\n" + "=" * 80)
    print("TRAINING PHASE")
    print("=" * 80)
    
    print(f"\nDataset: {len(df)} samples with {df['churn'].sum()} churned customers ({df['churn'].mean():.2%})")
    
    # Split data
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train pipeline
    print("\nTraining production pipeline...")
    pipeline = ProductionPipeline()
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION PHASE")
    print("=" * 80)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'precision': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
        'recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    }
    
    print("\nPerformance Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))
    
    return pipeline, metrics, X_test, y_test


# ============================================================================
# STEP 4: Save to Model Registry
# ============================================================================

def save_to_registry(pipeline, metrics):
    """
    Save the complete pipeline to the Model Registry.
    """
    print("\n" + "=" * 80)
    print("SAVING TO MODEL REGISTRY")
    print("=" * 80)
    
    registry = ModelRegistry(registry_path='./model_registry')
    
    # Save the ENTIRE pipeline (not just the model!)
    model_id = registry.save_model(
        model=pipeline,  # Save the full pipeline, not just pipeline.model
        name='customer_churn_production_pipeline',
        problem_type='classification',
        metrics=metrics,
        description=(
            'Production-ready customer churn pipeline. '
            'Includes feature engineering (ratios, binning), one-hot encoding, '
            'standard scaling, and Random Forest classifier. '
            'Ensures train-inference parity by bundling all transformations.'
        ),
        tags=['production', 'churn-prediction', 'random-forest']
    )
    
    print(f"\n✅ Pipeline saved with model ID: {model_id}")
    print(f"   Model name: customer_churn_production_pipeline")
    print(f"   Includes: Feature engineering, encoding, scaling, and trained model")
    
    return model_id


# ============================================================================
# STEP 5: Load and Predict (Simulating Production)
# ============================================================================

def load_and_predict():
    """
    Load the saved pipeline and make predictions on new data.
    This simulates production inference.
    """
    print("\n" + "=" * 80)
    print("PRODUCTION INFERENCE PHASE")
    print("=" * 80)
    
    # Initialize registry
    registry = ModelRegistry(registry_path='./model_registry')
    
    # Load the latest production pipeline
    print("\nLoading production pipeline from registry...")
    loaded_pipeline = registry.load_model('customer_churn_production_pipeline')
    print("✅ Pipeline loaded successfully!")
    
    # Simulate new customer data (same structure as training data)
    print("\nSimulating new customer data...")
    new_customers = pd.DataFrame({
        'customer_id': [2000, 2001, 2002],
        'age': [25, 55, 40],
        'tenure_months': [3, 48, 24],
        'monthly_charges': [89.99, 45.50, 120.00],
        'total_charges': [269.97, 2184.00, 2880.00],
        'contract_type': ['Month-to-Month', 'Two Year', 'One Year'],
        'payment_method': ['Electronic Check', 'Bank Transfer', 'Credit Card'],
        'num_services': [5, 2, 7],
        'support_tickets': [2, 0, 8],
    })
    
    print(f"\nNew customers to score: {len(new_customers)}")
    print(new_customers[['customer_id', 'age', 'tenure_months', 'contract_type']].to_string(index=False))
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = loaded_pipeline.predict(new_customers)
    probabilities = loaded_pipeline.predict_proba(new_customers)[:, 1]
    
    # Display results
    results = new_customers[['customer_id', 'age', 'tenure_months', 'contract_type']].copy()
    results['churn_prediction'] = predictions
    results['churn_probability'] = probabilities
    results['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))
    
    return results


# ============================================================================
# STEP 6: Compare Versions (Optional)
# ============================================================================

def compare_model_versions():
    """
    Compare different versions of the pipeline (if multiple exist).
    """
    registry = ModelRegistry(registry_path='./model_registry')
    
    models_df = registry.list_models(name='customer_churn_production_pipeline')
    
    if len(models_df) < 2:
        print("\nOnly one version exists. Train and save more models to compare versions.")
        return
    
    print("\n" + "=" * 80)
    print("MODEL VERSION COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    for _, row in models_df.iterrows():
        metrics = row['metrics'] if row['metrics'] else {}
        comparison_data.append({
            'Version': row['version'],
            'Accuracy': metrics.get('accuracy', 'N/A'),
            'F1 Score': metrics.get('f1_score', 'N/A'),
            'ROC AUC': metrics.get('roc_auc', 'N/A'),
            'Created': row['created_at']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run the complete production pipeline workflow.
    """
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PRODUCTION ML PIPELINE EXAMPLE" + " " * 28 + "║")
    print("║" + " " * 15 + "with Model Registry Integration" + " " * 32 + "║")
    print("║" + " " * 10 + "Includes: EDA → Feature Engineering → Modeling → Registry" + " " * 11 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Step 0: Load data and perform EDA
    print("\n📂 Loading data...")
    df = load_data()
    quality_report, domain_info = perform_eda(df)
    
    # Step 1: Train and evaluate
    pipeline, metrics, X_test, y_test = train_and_evaluate(df)
    
    # Step 2: Save to registry
    version = save_to_registry(pipeline, metrics)
    
    # Step 3: Load and predict (simulating production)
    results = load_and_predict()
    
    # Step 4: Compare versions (if applicable)
    compare_model_versions()
    
    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETE!")
    print("=" * 80)
    print("\n📋 Key Takeaways:")
    print("  1. ✅ EDA revealed data quality and domain characteristics")
    print("  2. ✅ Entire pipeline (feature engineering + encoding + scaling + model) is saved")
    print("  3. ✅ Train-inference parity is guaranteed")
    print("  4. ✅ Model versioning and metadata tracking are handled automatically")
    print("  5. ✅ Production predictions are as simple as: load_model → predict")
    print("\n🚀 Next Steps:")
    print("  - Deploy the pipeline to a REST API (FastAPI, Flask)")
    print("  - Set up model monitoring and retraining triggers")
    print("  - Implement A/B testing with multiple versions")
    print("  - Add model explainability (SHAP, LIME)")


if __name__ == '__main__':
    main()
