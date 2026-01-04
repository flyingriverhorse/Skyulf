"""
Skyulf EDA Example: Comprehensive Analysis (Iris Dataset)
=======================================================

This script demonstrates a full EDA workflow including:
- Data Quality Checks
- Outlier Detection (Isolation Forest)
- PCA (Dimensionality Reduction)
- Causal Discovery (PC Algorithm)
- Smart Alerts

Requirements:
    pip install skyulf-core matplotlib rich scikit-learn
"""

import polars as pl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import print as rprint

from skyulf.profiling.analyzer import EDAAnalyzer

console = Console()

def visualize_pca(profile):
    """Plots the first two principal components."""
    if not profile.pca_data:
        return

    x = [p.x for p in profile.pca_data]
    y = [p.y for p in profile.pca_data]
    labels = [p.label for p in profile.pca_data]
    
    # Convert labels to numeric for coloring
    try:
        c_values = [float(l) for l in labels]
    except (ValueError, TypeError):
        # If labels are text (e.g. "setosa") or None, map them to numbers
        unique_labels = list(set([l for l in labels if l is not None]))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        c_values = [label_map.get(l, -1) for l in labels]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=c_values, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Target Class')
    plt.title("PCA Projection (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    
    print("Displaying plot...")
    plt.show()

def visualize_distributions(profile):
    """Plots histograms for numeric columns using pre-computed bins."""
    # Filter for numeric columns that have histogram data
    numeric_cols = [
        (name, col) for name, col in profile.columns.items() 
        if col.dtype == "Numeric" and col.histogram
    ]
    
    if not numeric_cols:
        return

    # Limit to first 4 for readability
    display_cols = numeric_cols[:4]
    n_cols = len(display_cols)
    
    plt.figure(figsize=(5 * n_cols, 4))
    
    for i, (name, col) in enumerate(display_cols):
        plt.subplot(1, n_cols, i+1)
        
        # Skyulf returns pre-binned data (start, end, count)
        # We use plt.bar to visualize this
        widths = [b.end - b.start for b in col.histogram]
        centers = [(b.start + b.end)/2 for b in col.histogram]
        counts = [b.count for b in col.histogram]
        
        plt.bar(centers, counts, width=widths, align='center', alpha=0.7, edgecolor='black', color='skyblue')
        plt.title(f"Distribution: {name}")
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    print("Displaying distributions...")
    plt.show()

def visualize_correlations(profile):
    """Plots the correlation matrix as a heatmap."""
    if not profile.correlations:
        return
        
    cols = profile.correlations.columns
    matrix = profile.correlations.values
    
    plt.figure(figsize=(8, 8))
    im = plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    
    # Add labels
    plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
    plt.yticks(range(len(cols)), cols)
    
    # Add text annotations
    for i in range(len(cols)):
        for j in range(len(cols)):
            text = plt.text(j, i, f"{matrix[i][j]:.2f}",
                           ha="center", va="center", color="black", fontsize=8)

    plt.title("Correlation Matrix")
    plt.tight_layout()
    
    print("Displaying correlation matrix...")
    plt.show()

def visualize_target_interactions(profile):
    """Plots boxplots for features vs categorical target."""
    if not profile.target_interactions:
        return
        
    # Filter for boxplot types
    boxplots = [i for i in profile.target_interactions if i.plot_type == "boxplot"]
    
    if not boxplots:
        return
        
    # Limit to top 3 features (sorted by p-value if available, else just first 3)
    # Lower p-value means more significant difference
    boxplots.sort(key=lambda x: x.p_value if x.p_value is not None else 1.0)
    display_items = boxplots[:3]
    
    n_plots = len(display_items)
    plt.figure(figsize=(5 * n_plots, 5))
    
    for i, interaction in enumerate(display_items):
        plt.subplot(1, n_plots, i+1)
        
        # Prepare data for plt.boxplot
        # We have pre-computed stats (min, q1, median, q3, max)
        # Matplotlib's bxp takes a list of dictionaries
        
        bxp_stats = []
        for cat_data in interaction.data:
            bxp_stats.append({
                'label': cat_data.name,
                'whislo': cat_data.stats.min,    # Bottom whisker
                'q1': cat_data.stats.q1,         # First quartile
                'med': cat_data.stats.median,    # Median
                'q3': cat_data.stats.q3,         # Third quartile
                'whishi': cat_data.stats.max,    # Top whisker
                'fliers': []                     # Outliers not stored in this summary
            })
            
        plt.bxp(bxp_stats, showfliers=False)
        
        title = f"{interaction.feature} by Target"
        if interaction.p_value is not None:
            title += f"\n(ANOVA p={interaction.p_value:.4f})"
            
        plt.title(title)
        plt.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    print("Displaying target interactions (Boxplots)...")
    plt.show()

def visualize_scatter_matrix(df, target_col="target"):
    """Creates a scatter matrix (pair plot) using the raw dataframe."""
    # Select numeric columns + target
    numeric_cols = [col for col, dtype in df.schema.items() if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    
    # Limit to 4 columns for readability
    if len(numeric_cols) > 5:
        numeric_cols = numeric_cols[:5]
        
    # Convert to pandas for easy plotting with pandas/seaborn style logic
    # or just use matplotlib manually
    pdf = df.select(numeric_cols).to_pandas()
    
    # Simple scatter matrix using pandas plotting
    from pandas.plotting import scatter_matrix
    
    colors = None
    if target_col in pdf.columns:
        # Map target to colors
        unique_targets = pdf[target_col].unique()
        color_map = {val: i for i, val in enumerate(unique_targets)}
        colors = pdf[target_col].map(color_map)
    
    plt.figure(figsize=(10, 10))
    scatter_matrix(pdf, alpha=0.8, figsize=(10, 10), diagonal='kde', c=colors, cmap='viridis')
    
    plt.suptitle("Scatter Matrix (Pair Plot)")
    print("Displaying scatter matrix...")
    plt.show()

def main():
    console.print(Panel.fit("Skyulf Comprehensive EDA (Iris Dataset)", style="bold purple"))

    # 1. Load Data
    iris = load_iris()
    df = pl.DataFrame(iris.data, schema=iris.feature_names)
    df = df.with_columns(pl.Series("target", iris.target))
    
    console.print(f"Loaded Iris dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Run Analysis
    with console.status("[bold green]Running full analysis pipeline..."):
        analyzer = EDAAnalyzer(df)
        profile = analyzer.analyze(target_col="target")

    # 3. Report: Data Quality
    console.print("\n[bold]1. Data Quality[/bold]")
    dq_table = Table(show_header=True, header_style="bold magenta")
    dq_table.add_column("Metric")
    dq_table.add_column("Value")
    dq_table.add_row("Missing Cells", f"{profile.missing_cells_percentage}%")
    dq_table.add_row("Duplicate Rows", str(profile.duplicate_rows))
    console.print(dq_table)

    # 3.5 Report: Numeric Statistics (Skewness, Kurtosis, Normality)
    console.print("\n[bold]1.5 Numeric Statistics[/bold]")
    stats_table = Table(show_header=True, header_style="bold cyan")
    stats_table.add_column("Column")
    stats_table.add_column("Skewness", justify="right")
    stats_table.add_column("Kurtosis", justify="right")
    stats_table.add_column("Normality (p-value)", justify="right")
    stats_table.add_column("Is Normal?", justify="center")

    for col_name, col_profile in profile.columns.items():
        if col_profile.dtype == "Numeric" and col_profile.numeric_stats:
            stats = col_profile.numeric_stats
            
            skew = f"{stats.skewness:.2f}" if stats.skewness is not None else "N/A"
            kurt = f"{stats.kurtosis:.2f}" if stats.kurtosis is not None else "N/A"
            
            norm_p = "N/A"
            is_normal = "N/A"
            if col_profile.normality_test:
                norm_p = f"{col_profile.normality_test.p_value:.4f}"
                is_normal = "[green]Yes[/green]" if col_profile.normality_test.is_normal else "[red]No[/red]"
            
            stats_table.add_row(col_name, skew, kurt, norm_p, is_normal)
            
    console.print(stats_table)

    # 4. Report: Outliers
    if profile.outliers:
        console.print("\n[bold]2. Outlier Detection[/bold]")
        console.print(f"Detected [red]{profile.outliers.total_outliers}[/red] outliers ({profile.outliers.outlier_percentage}%)")
        
        outlier_table = Table(title="Top Anomalies")
        outlier_table.add_column("Row Index", justify="right")
        outlier_table.add_column("Anomaly Score", justify="right")
        outlier_table.add_column("Explanation", style="italic")
        
        for outlier in profile.outliers.top_outliers[:3]:
            explanation = outlier.explanation if outlier.explanation else "N/A"
            outlier_table.add_row(str(outlier.index), f"{outlier.score:.4f}", str(explanation))
        
        console.print(outlier_table)

    # 5. Report: Causal Graph
    if profile.causal_graph:
        console.print("\n[bold]3. Causal Discovery[/bold]")
        console.print(f"Graph: {len(profile.causal_graph.nodes)} nodes, {len(profile.causal_graph.edges)} edges")
        
        edge_table = Table(show_header=False)
        for edge in profile.causal_graph.edges:
            arrow = "->" if edge.type == "directed" else "--"
            edge_table.add_row(f"{edge.source} {arrow} {edge.target}")
        console.print(edge_table)

    # 6. Report: Alerts
    if profile.alerts:
        console.print("\n[bold]4. Smart Alerts[/bold]")
        for alert in profile.alerts:
            color = "red" if alert.severity == "high" else "yellow"
            console.print(f"[{color}]• {alert.message}[/{color}]")

    # 7. Visualize
    visualize_distributions(profile)
    visualize_correlations(profile)
    visualize_target_interactions(profile)
    visualize_scatter_matrix(df, target_col="target")
    visualize_pca(profile)
    console.print("\n[bold blue]Visualizations displayed.[/bold blue]")

if __name__ == "__main__":
    main()
