import polars as pl
from typing import List, Optional
from .schemas import HistogramBin
import numpy as np

def calculate_histogram(df: pl.LazyFrame, col_name: str, bins: int = 20) -> Optional[List[HistogramBin]]:
    """
    Calculates histogram bins for a numeric column using Polars.
    """
    # We need to execute to get min/max for binning, or use an approximation.
    # For accurate bins, we need min/max.
    try:
        # print(f"Calculating histogram for {col_name}")
        stats = df.select([
            pl.col(col_name).min().alias("min"),
            pl.col(col_name).max().alias("max")
        ]).collect()
        
        min_val = stats["min"][0]
        max_val = stats["max"][0]
        
        if min_val is None or max_val is None or min_val == max_val:
            return None

        # Create bins using numpy (fastest way to get edges)
        edges = np.linspace(min_val, max_val, bins + 1)
        
        hist_df = (
            df.select(pl.col(col_name))
            .with_columns(
                pl.col(col_name).cut(breaks=list(edges[1:-1]), labels=[str(i) for i in range(bins)])
                .alias("bin")
            )
            .group_by("bin")
            .len()
            .collect()
        )
        
        # Map back to HistogramBin
        result = []
        
        counts = {}
        for row in hist_df.iter_rows(named=True):
            try:
                # The bin column is Categorical, but iter_rows might return string or int depending on version
                # We used labels "0", "1", etc.
                bin_val = row['bin']
                # print(f"Bin val: {bin_val} type: {type(bin_val)}")
                bin_idx = int(bin_val)
                # The count column is named "len" by default in Polars group_by().len()
                counts[bin_idx] = row.get('len', row.get('count'))
            except Exception as e:
                # print(f"Error parsing bin: {e}")
                continue
                
        histogram = []
        for i in range(bins):
            start = edges[i]
            end = edges[i+1]
            count = counts.get(i, 0)
            histogram.append(HistogramBin(start=start, end=end, count=count))
            
        return histogram

    except Exception as e:
        print(f"Error calculating histogram for {col_name}: {e}")
        return None
