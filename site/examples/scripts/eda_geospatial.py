"""
Skyulf EDA Example: Geospatial Analysis
=======================================

This script demonstrates how Skyulf automatically detects latitude/longitude columns
and computes spatial statistics.

Requirements:
    pip install skyulf-core matplotlib rich
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from skyulf.profiling.analyzer import EDAAnalyzer

console = Console()

def generate_geo_data(n_points=100):
    """Generates synthetic store locations clustered around NYC."""
    # NYC Center approx: 40.7128° N, 74.0060° W
    lats = np.random.normal(40.7128, 0.05, n_points)
    lons = np.random.normal(-74.0060, 0.05, n_points)
    revenue = np.random.randint(1000, 5000, n_points)
    
    return pl.DataFrame({
        "store_id": range(n_points),
        "latitude": lats,
        "longitude": lons,
        "revenue": revenue
    })

def visualize_map(df, profile):
    """Plots the points and the calculated centroid."""
    if not profile.geospatial:
        return

    plt.figure(figsize=(8, 8))
    
    # Plot all points
    plt.scatter(
        df["longitude"], 
        df["latitude"], 
        c=df["revenue"], 
        cmap='viridis', 
        alpha=0.7, 
        label='Stores'
    )
    plt.colorbar(label='Revenue')
    
    # Plot Centroid
    plt.scatter(
        profile.geospatial.centroid_lon, 
        profile.geospatial.centroid_lat, 
        color='red', 
        s=200, 
        marker='*', 
        label='Centroid'
    )
    
    # Plot Bounding Box
    min_lat = profile.geospatial.min_lat
    max_lat = profile.geospatial.max_lat
    min_lon = profile.geospatial.min_lon
    max_lon = profile.geospatial.max_lon
    
    plt.plot([min_lon, max_lon, max_lon, min_lon, min_lon], 
             [min_lat, min_lat, max_lat, max_lat, min_lat], 
             color='black', linestyle='--', label='Bounding Box')

    plt.title("Geospatial Analysis: Store Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Displaying plot...")
    plt.show()

def main():
    console.print(Panel.fit("Skyulf Geospatial Analysis Example", style="bold green"))

    # 1. Generate Data
    df = generate_geo_data()
    console.print(f"Generated {len(df)} locations around NYC.")

    # 2. Run Analysis
    with console.status("[bold green]Running Skyulf EDA..."):
        analyzer = EDAAnalyzer(df)
        profile = analyzer.analyze(target_col="revenue")

    # 3. Report Results
    if profile.geospatial:
        console.print("\n[bold]Geospatial Statistics:[/bold]")
        
        table = Table(show_header=False)
        table.add_row("Latitude Range", f"{profile.geospatial.min_lat:.4f} to {profile.geospatial.max_lat:.4f}")
        table.add_row("Longitude Range", f"{profile.geospatial.min_lon:.4f} to {profile.geospatial.max_lon:.4f}")
        table.add_row("Centroid", f"{profile.geospatial.centroid_lat:.4f}, {profile.geospatial.centroid_lon:.4f}")
        
        console.print(table)

        # 4. Visualize
        visualize_map(df, profile)
        console.print("\n[bold blue]Visualization displayed.[/bold blue]")
    else:
        console.print("[bold red]No geospatial data detected![/bold red]")

if __name__ == "__main__":
    main()
