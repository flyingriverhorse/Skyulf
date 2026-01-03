"""
Skyulf EDA Example: Automated Analysis (The Easy Way)
===================================================

This script demonstrates the simplified EDA workflow using the new `EDAVisualizer`.
Instead of writing manual plotting code, you can generate a full report in 3 lines.

Requirements:
    pip install skyulf-core[viz]
"""

import polars as pl
from sklearn.datasets import load_iris
from rich.console import Console
from rich.panel import Panel

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.visualizer import EDAVisualizer

console = Console()

def main():
    console.print(Panel.fit("Skyulf Automated EDA", style="bold purple"))

    # 1. Load Data
    iris = load_iris()
    df = pl.DataFrame(iris.data, schema=iris.feature_names)
    df = df.with_columns(pl.Series("target", iris.target))
    
    console.print(f"Loaded Iris dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Run Analysis
    with console.status("[bold green]Running analysis..."):
        analyzer = EDAAnalyzer(df)
        profile = analyzer.analyze(target_col="target")

    # 3. Visualize Results (The Easy Way)
    # This single class handles all the rich terminal output and matplotlib plots
    viz = EDAVisualizer(profile, df)
    
    # Print the dashboard
    viz.summary()
    
    # Show the plots
    viz.plot()

if __name__ == "__main__":
    main()
