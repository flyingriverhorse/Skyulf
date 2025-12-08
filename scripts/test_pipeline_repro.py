import pandas as pd
import numpy as np
import logging
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from core.ml_pipeline.preprocessing.drop_and_missing import DropMissingColumnsCalculator, DropMissingRowsCalculator, DropMissingColumnsApplier, DropMissingRowsApplier
from core.ml_pipeline.preprocessing.encoding import TargetEncoderCalculator, LabelEncoderCalculator, TargetEncoderApplier, LabelEncoderApplier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    # 1. Load Data
    data_path = "uploads/data/066c7d7a_tr.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # 2. Drop Columns (capital, admin_name, population_proper)
    drop_cols = ["capital", "admin_name", "population_proper"]
    # Verify they exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    print(f"Dropping columns: {drop_cols}")
    
    drop_calc = DropMissingColumnsCalculator()
    drop_config = {"columns": drop_cols}
    drop_res = drop_calc.fit(df, drop_config)
    
    drop_applier = DropMissingColumnsApplier()
    df = drop_applier.apply(df, drop_res)
    print(f"After drop cols: {df.shape}")

    # 3. Drop Rows > 15% missing
    row_calc = DropMissingRowsCalculator()
    row_config = {"missing_threshold": 15.0}
    row_res = row_calc.fit(df, row_config)
    
    row_applier = DropMissingRowsApplier()
    df = row_applier.apply(df, row_res)
    print(f"After drop rows: {df.shape}")

    # 4. Feature Target Split
    target = "population"
    if target not in df.columns:
        print(f"Target {target} not found")
        return
        
    X = df.drop(columns=[target])
    y = df[target]
    
    # 5. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Encoding Target on 'city'
    print("Fitting TargetEncoder on 'city'...")
    te_calc = TargetEncoderCalculator()
    te_config = {"columns": ["city"], "target_column": target}
    
    te_res = te_calc.fit((X_train, y_train), te_config)
    te_applier = TargetEncoderApplier()
    
    # Apply to Train
    X_train_enc = te_applier.apply((X_train, y_train), te_res)[0]
    
    # 7. Encoding Label on 'country', 'iso2'
    print("Fitting LabelEncoder on 'country', 'iso2'...")
    le_calc = LabelEncoderCalculator()
    le_config = {"columns": ["country", "iso2"]}
    
    le_res = le_calc.fit((X_train_enc, y_train), le_config)
    le_applier = LabelEncoderApplier()
    
    X_train_final = le_applier.apply((X_train_enc, y_train), le_res)[0]
    
    print(f"Final Train Shape: {X_train_final.shape}")
    print(f"Final Train Columns: {X_train_final.columns.tolist()}")
    
    # 8. Hyperparameter Tuning (Random Search)
    print("Starting Hyperparameter Tuning (Random Search)...")
    
    param_dist = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "bootstrap": [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_mean_squared_error",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train_final, y_train)
    
    print(f"Best Parameters: {search.best_params_}")
    print(f"Best MSE: {-search.best_score_}")
    
    best_model = search.best_estimator_
    print("Model fitted (Best Estimator).")

    # --- SIMULATE INFERENCE ---
    
    # Construct Artifact Dictionary
    artifact = {
        "model": best_model,
        "transformers": [], # Not strictly needed if metadata has everything
        "transformer_plan": [
            {
                "node_id": "n1",
                "transformers": [{"transformer_name": "drop_cols", "column_name": None, "metadata": drop_res}]
            },
            {
                "node_id": "n2",
                "transformers": [{"transformer_name": "drop_rows", "column_name": None, "metadata": row_res}]
            },
            {
                "node_id": "n3",
                "transformers": [{"transformer_name": "target_enc", "column_name": "city", "metadata": te_res}]
            },
            {
                "node_id": "n4",
                "transformers": [{"transformer_name": "label_enc", "column_name": "country", "metadata": le_res}]
            }
        ]
    }
    
    # Input Data
    input_data = [
      {
        "city": "Istanbul",
        "lat": 41.0136,
        "lng": 28.955,
        "country": "Turkey",
        "iso2": "TR"
      }
    ]
    
    print("\n--- Running Inference Simulation ---")
    
    from core.ml_pipeline.deployment.service import APPLIER_MAP
    
    df_pred = pd.DataFrame(input_data)
    print(f"Input DF: \n{df_pred}")
    
    model = artifact["model"]
    plan = artifact.get("transformer_plan", [])
    
    for step in plan:
        for t_spec in step.get("transformers", []):
            metadata = t_spec.get("metadata") or {}
            t_type = metadata.get("type")
            
            print(f"Applying {t_type}...")
            
            ApplierCls = APPLIER_MAP.get(t_type)
            if ApplierCls:
                applier = ApplierCls()
                params = metadata.copy()
                
                res = applier.apply(df_pred, params)
                if isinstance(res, tuple):
                    df_pred = res[0]
                else:
                    df_pred = res
                    
    print(f"Pre-prediction DF columns: {df_pred.columns.tolist()}")
    print(f"Pre-prediction DF values: \n{df_pred.values}")
    
    # Model Prediction Logic (from Service)
    if hasattr(model, "set_output"):
        try:
            model.set_output(transform="pandas")
        except:
            pass
            
    if hasattr(model, "feature_names_in_"):
        model_cols = model.feature_names_in_.tolist()
        missing = set(model_cols) - set(df_pred.columns)
        if missing:
            print(f"Missing columns: {missing}")
            for c in missing:
                df_pred[c] = 0
        df_pred = df_pred[model_cols]
        
    try:
        preds = model.predict(df_pred)
        print(f"Prediction: {preds}")
    except Exception as e:
        print(f"Prediction Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
