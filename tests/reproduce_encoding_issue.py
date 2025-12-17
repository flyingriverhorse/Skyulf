import pandas as pd
import numpy as np
from skyulf.preprocessing.encoding import LabelEncoderCalculator, LabelEncoderApplier

def test_label_encoding_flow():
    # 1. Setup Data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'species': ['A', 'B', 'A', 'C', 'B']
    })
    
    # 2. Simulate Feature Target Split (X, y)
    X = df[['feature1']]
    y = df['species']
    y.name = 'species' # Ensure name is set
    
    print(f"Original y:\n{y}")
    
    # 3. Configure Encoding
    # User selects 'species' in the UI, so config has columns=['species']
    config = {
        'method': 'label',
        'columns': ['species']
    }
    
    # 4. Fit
    calculator = LabelEncoderCalculator()
    # Input to fit is (X, y) tuple
    fit_result = calculator.fit((X, y), config)
    
    print("\nFit Result Keys:", fit_result.keys())
    if 'encoders' in fit_result:
        print("Encoders:", fit_result['encoders'].keys())
        
    # 5. Apply
    applier = LabelEncoderApplier()
    # Input to apply is (X, y) tuple
    # unpack_pipeline_input handles tuples
    result = applier.apply((X, y), fit_result)
    
    if isinstance(result, tuple):
        X_out, y_out = result
    else:
        X_out = result
        y_out = None
    
    print(f"\nTransformed y:\n{y_out}")
    
    # Check if y_out is numeric
    if y_out is not None and pd.api.types.is_numeric_dtype(y_out):
        print("\nSUCCESS: y is numeric.")
    else:
        print("\nFAILURE: y is still object/string or None.")

if __name__ == "__main__":
    test_label_encoding_flow()
