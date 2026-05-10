"""
Skyulf EDA Example: Time Series & Geospatial Analysis
===================================================

This script demonstrates how to use Skyulf's EDA module for:
- Time Series Analysis (Trend, Seasonality)
- Geospatial Analysis (Lat/Lon distribution)
- Manual column specification for analysis

Requirements:
    pip install skyulf-core[viz] polars numpy
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.visualizer import EDAVisualizer

console = Console()

def generate_synthetic_data(n_rows=1000):
    """Generates a synthetic dataset with time series and geospatial patterns."""
    np.random.seed(42)
    
    # Time Series: 1 year of data
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_rows)]
    dates.sort()
    
    # Geospatial: Clusters around NYC and LA
    # NYC: 40.7128° N, 74.0060° W
    # LA: 34.0522° N, 118.2437° W
    lats = []
    lons = []
    cities = []
    
    for _ in range(n_rows):
        if np.random.random() > 0.5:
            # NYC
            lats.append(40.7128 + np.random.normal(0, 0.1))
            lons.append(-74.0060 + np.random.normal(0, 0.1))
            cities.append("NYC")
        else:
            # LA
            lats.append(34.0522 + np.random.normal(0, 0.1))
            lons.append(-118.2437 + np.random.normal(0, 0.1))
            cities.append("LA")
            
    # Numeric Value with Trend + Seasonality
    # Value increases over time + sine wave
    values = []
    for d in dates:
        day_of_year = d.timetuple().tm_yday
        trend = day_of_year * 0.1
        seasonality = 10 * np.sin(2 * np.pi * day_of_year / 365)
        noise = np.random.normal(0, 5)
        values.append(trend + seasonality + noise)

    return pl.DataFrame({
        "timestamp": dates,
        "latitude": lats,
        "longitude": lons,
        "city": cities,
        "sales": values,
        "category": np.random.choice(["A", "B", "C"], n_rows)
    })

def main():
    console.print(Panel.fit("Skyulf Advanced EDA: Time Series & Geospatial", style="bold purple"))

    # 1. Generate Data
    with console.status("[bold green]Generating synthetic data..."):
        df = generate_synthetic_data(2000)
    
    console.print(f"Generated dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    console.print(df.head())

    # 2. Run Analysis
    # We explicitly specify the columns to ensure they are picked up correctly,
    # although Skyulf can often auto-detect them.
    with console.status("[bold green]Running analysis..."):
        analyzer = EDAAnalyzer(df)
        profile = analyzer.analyze(
            target_col="sales",
            date_col="timestamp",
            lat_col="latitude",
            lon_col="longitude"
        )

    # 3. Visualize Results
    viz = EDAVisualizer(profile, df)
    
    # Print rich terminal dashboard
    viz.summary()
    
    # Show plots (Time Series Trend, Geospatial Map, etc.)
    viz.plot()

if __name__ == "__main__":
    main()
