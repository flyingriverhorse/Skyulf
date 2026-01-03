import polars as pl
import numpy as np
from typing import List, Optional, Tuple
from .schemas import CorrelationMatrix

def calculate_correlations(df: pl.LazyFrame, numeric_cols: List[str]) -> Optional[CorrelationMatrix]:
    """
    Calculates Pearson correlation matrix for numeric columns.
    """
    if len(numeric_cols) < 2:
        return None
        
    try:
        # Polars `corr` is efficient.
        # We need to collect to compute correlation matrix usually.
        # df.select(pl.corr(col1, col2)) for all pairs is one way.
        # Or collect and use .corr() on DataFrame.
        
        # Since correlation matrix is N*N and N is usually small (number of columns),
        # collecting the numeric columns (if not too huge) or computing pair-wise is the way.
        
        # For very large datasets, we should use lazy expressions for covariance and std dev.
        # But `df.corr()` in Polars is optimized.
        
        # Let's try to use the built-in `corr` on the collected DataFrame of just numeric cols.
        # Collecting just numeric cols is usually fine unless we have 10k columns.
        
        # Limit to reasonable number of columns to avoid OOM on huge column sets?
        # Let's assume < 1000 columns for now.
        
        # HARD LIMIT: Top 20 numeric columns to prevent backend/frontend crash
        # Reduced from 50 to 20 as per user report of crashes
        if len(numeric_cols) > 20:
            numeric_cols = numeric_cols[:20]

        subset = df.select(numeric_cols).collect()
        
        # Handle constant columns to avoid RuntimeWarning: invalid value encountered in divide
        # Filter out columns with 0 std dev
        valid_cols = []
        for col in numeric_cols:
            std_val = subset[col].std()
            # Check for None (all nulls or single value) and 0 variance
            if std_val is not None and std_val > 1e-9:
                valid_cols.append(col)
        
        if len(valid_cols) < 2:
            return None
            
        # Re-select only valid columns
        subset = subset.select(valid_cols)
        
        # Suppress numpy warnings that might occur during correlation calculation
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr_df = subset.corr()
        
        # Convert to our schema
        matrix = []
        for row in corr_df.iter_rows():
            # Handle NaN/None/Inf
            cleaned_row = []
            for x in row:
                if x is None or np.isnan(x) or np.isinf(x):
                    cleaned_row.append(0.0)
                else:
                    cleaned_row.append(float(x))
            matrix.append(cleaned_row)
            
        return CorrelationMatrix(
            columns=valid_cols,
            values=matrix
        )
        
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None
