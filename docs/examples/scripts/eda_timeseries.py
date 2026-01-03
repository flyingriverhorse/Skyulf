"""
Skyulf EDA Example: Time Series Analysis
========================================

This script demonstrates how to use Skyulf's EDAAnalyzer to automatically detect
and analyze time series data.

Requirements:
    pip install skyulf-core matplotlib rich
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from skyulf.profiling.analyzer import EDAAnalyzer

console = Console()

def generate_timeseries_data(n_days=365):
    """Generates synthetic daily sales data with trend and seasonality."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Trend: Linear growth
    trend = np.linspace(100, 200, n_days)
    
    # Seasonality: Weekly pattern (sine wave)
    seasonality = 20 * np.sin(np.array(range(n_days)) * (2 * np.pi / 7))
    
    # Noise
    noise = np.random.normal(0, 5, n_days)
    
    values = trend + seasonality + noise
    return pl.DataFrame({"date": dates, "sales": values})

def visualize_decomposition(profile):
    """Plots the original data vs the extracted trend."""
    if not profile.timeseries:
        return

    dates = [p.date for p in profile.timeseries.trend]
    
    # point.values is a dict {col_name: value}
    # We'll plot a line for each numeric column found in the trend
    if not profile.timeseries.trend:
        return
        
    keys = profile.timeseries.trend[0].values.keys()
    
    plt.figure(figsize=(10, 6))
    
    for key in keys:
        trend_values = [p.values[key] for p in profile.timeseries.trend]
        plt.plot(dates, trend_values, label=f'Trend ({key})', linewidth=2)

    plt.title(f"Time Series Trend Analysis (Date Column: {profile.timeseries.date_col})")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    print("Displaying plot...")
    plt.show()

def main():
    console.print(Panel.fit("Skyulf Time Series Analysis Example", style="bold blue"))

    # 1. Generate Data
    with console.status("[bold green]Generating synthetic data..."):
        df = generate_timeseries_data()
    console.print(f"Generated {len(df)} rows of data.")

    # 2. Run Analysis
    with console.status("[bold green]Running Skyulf EDA..."):
        analyzer = EDAAnalyzer(df)
        profile = analyzer.analyze()

    # 3. Report Results
    if profile.timeseries:
        console.print(f"\n[bold green]✓ Detected Time Column:[/bold green] {profile.timeseries.date_col}")
        
        # Create a table for the trend
        table = Table(title="Trend Analysis (First 5 Days)")
        table.add_column("Date", style="cyan")
        table.add_column("Trend Values", justify="right", style="magenta")
        
        for point in profile.timeseries.trend[:5]:
            # Format dict values
            val_str = ", ".join([f"{k}={v:.2f}" for k, v in point.values.items()])
            table.add_row(str(point.date), val_str)
            
        console.print(table)
        
        # Seasonality
        if profile.timeseries.seasonality:
            console.print("\n[bold]Seasonality Detected:[/bold]")
            console.print(profile.timeseries.seasonality.day_of_week)

        # 4. Visualize
        visualize_decomposition(profile)
        console.print("\n[bold blue]Visualization displayed.[/bold blue]")
    else:
        console.print("[bold red]No time series detected![/bold red]")

if __name__ == "__main__":
    main()
