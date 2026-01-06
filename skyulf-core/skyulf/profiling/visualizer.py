from typing import Optional, List, Any
import polars as pl
from .schemas import DatasetProfile

class EDAVisualizer:
    """
    Helper class to visualize Skyulf EDA results using Rich (terminal) and Matplotlib (plots).
    """
    
    def __init__(self, profile: DatasetProfile, df: Optional[pl.DataFrame] = None):
        self.profile = profile
        self.df = df
        
    def summary(self):
        """Prints a rich terminal dashboard summary."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
        except ImportError:
            print("Please install 'rich' to use the terminal summary: pip install rich")
            return

        console = Console()
        console.print(Panel.fit("Skyulf EDA Summary", style="bold blue"))

        # 1. Data Quality
        console.print("\n[bold]1. Data Quality[/bold]")
        dq_table = Table(show_header=True, header_style="bold magenta")
        dq_table.add_column("Metric")
        dq_table.add_column("Value")
        dq_table.add_row("Rows", str(self.profile.row_count))
        dq_table.add_row("Columns", str(self.profile.column_count))
        dq_table.add_row("Missing Cells", f"{self.profile.missing_cells_percentage}%")
        dq_table.add_row("Duplicate Rows", str(self.profile.duplicate_rows))
        if self.profile.target_col:
            dq_table.add_row("Target Column", self.profile.target_col)
            if self.profile.task_type:
                dq_table.add_row("Task Type", self.profile.task_type)
        console.print(dq_table)

        # 2. Numeric Stats
        console.print("\n[bold]2. Numeric Statistics[/bold]")
        stats_table = Table(show_header=True, header_style="bold cyan")
        stats_table.add_column("Column")
        stats_table.add_column("Mean", justify="right")
        stats_table.add_column("Std", justify="right")
        stats_table.add_column("Min", justify="right")
        stats_table.add_column("Max", justify="right")
        stats_table.add_column("Skew", justify="right")
        stats_table.add_column("Kurt", justify="right")
        stats_table.add_column("Normality", justify="center")

        has_numeric = False
        for col_name, col_profile in self.profile.columns.items():
            if col_profile.dtype == "Numeric" and col_profile.numeric_stats:
                has_numeric = True
                stats = col_profile.numeric_stats
                
                mean = f"{stats.mean:.2f}" if stats.mean is not None else "-"
                std = f"{stats.std:.2f}" if stats.std is not None else "-"
                min_val = f"{stats.min:.2f}" if stats.min is not None else "-"
                max_val = f"{stats.max:.2f}" if stats.max is not None else "-"
                skew = f"{stats.skewness:.2f}" if stats.skewness is not None else "-"
                kurt = f"{stats.kurtosis:.2f}" if stats.kurtosis is not None else "-"
                
                is_normal = "-"
                if col_profile.normality_test:
                    is_normal = "[green]Yes[/green]" if col_profile.normality_test.is_normal else "[red]No[/red]"
                
                stats_table.add_row(col_name, mean, std, min_val, max_val, skew, kurt, is_normal)
        
        if has_numeric:
            console.print(stats_table)
        else:
            console.print("[italic]No numeric columns found.[/italic]")

        # 2.1 VIF (Multicollinearity)
        if self.profile.vif:
            console.print("\n[bold]2.1 Multicollinearity (VIF)[/bold]")
            vif_table = Table(show_header=True, header_style="bold red")
            vif_table.add_column("Feature")
            vif_table.add_column("VIF Score", justify="right")
            vif_table.add_column("Status", justify="center")
            
            # Sort by VIF descending
            sorted_vif = sorted(self.profile.vif.items(), key=lambda x: x[1], reverse=True)
            
            for col, val in sorted_vif:
                status = "[green]OK[/green]"
                if val > 10:
                    status = "[red]Severe[/red]"
                elif val > 5:
                    status = "[yellow]High[/yellow]"
                
                vif_table.add_row(col, f"{val:.2f}", status)
            
            console.print(vif_table)

        # 3. Categorical Stats
        console.print("\n[bold]3. Categorical Statistics[/bold]")
        cat_table = Table(show_header=True, header_style="bold yellow")
        cat_table.add_column("Column")
        cat_table.add_column("Unique", justify="right")
        cat_table.add_column("Top Categories (Count)", style="dim")
        
        has_cat = False
        for col_name, col_profile in self.profile.columns.items():
            if col_profile.dtype == "Categorical" and col_profile.categorical_stats:
                has_cat = True
                stats = col_profile.categorical_stats
                
                top_cats = []
                for item in stats.top_k[:3]:
                    val = str(item.get("value", "N/A"))
                    count = item.get("count", 0)
                    top_cats.append(f"{val} ({count})")
                
                top_str = ", ".join(top_cats)
                cat_table.add_row(col_name, str(stats.unique_count), top_str)

        if has_cat:
            console.print(cat_table)
        else:
            console.print("[italic]No categorical columns found.[/italic]")

        # 4. Text Stats
        console.print("\n[bold]4. Text Statistics[/bold]")
        text_table = Table(show_header=True, header_style="bold white")
        text_table.add_column("Column")
        text_table.add_column("Avg Len", justify="right")
        text_table.add_column("Min/Max Len", justify="right")
        text_table.add_column("Sentiment (Pos/Neu/Neg)", justify="center")
        
        has_text = False
        for col_name, col_profile in self.profile.columns.items():
            if col_profile.dtype == "Text" and col_profile.text_stats:
                has_text = True
                stats = col_profile.text_stats
                
                avg_len = f"{stats.avg_length:.1f}" if stats.avg_length else "-"
                min_max = f"{stats.min_length}/{stats.max_length}" if stats.min_length is not None else "-"
                
                sentiment_str = "-"
                if stats.sentiment_distribution:
                    pos = stats.sentiment_distribution.get("positive", 0) * 100
                    neu = stats.sentiment_distribution.get("neutral", 0) * 100
                    neg = stats.sentiment_distribution.get("negative", 0) * 100
                    sentiment_str = f"[green]{pos:.0f}%[/green] / [grey]{neu:.0f}%[/grey] / [red]{neg:.0f}%[/red]"

                text_table.add_row(col_name, avg_len, min_max, sentiment_str)

        if has_text:
            console.print(text_table)
        else:
            console.print("[italic]No text columns found.[/italic]")

        # 5. Outliers
        if self.profile.outliers:
            console.print("\n[bold]5. Outlier Detection[/bold]")
            console.print(f"Detected [red]{self.profile.outliers.total_outliers}[/red] outliers ({self.profile.outliers.outlier_percentage:.2f}%)")
            
            outlier_table = Table(title="Top Anomalies")
            outlier_table.add_column("Index", justify="right")
            outlier_table.add_column("Score", justify="right")
            outlier_table.add_column("Explanation", style="italic")
            
            for outlier in self.profile.outliers.top_outliers[:3]:
                explanation = str(outlier.explanation) if outlier.explanation else "-"
                outlier_table.add_row(str(outlier.index), f"{outlier.score:.4f}", explanation)
            
            console.print(outlier_table)

        # 6. Causal Graph
        if self.profile.causal_graph:
            console.print("\n[bold]6. Causal Discovery[/bold]")
            console.print(f"Graph: {len(self.profile.causal_graph.nodes)} nodes, {len(self.profile.causal_graph.edges)} edges")
            
            edge_table = Table(show_header=False)
            for edge in self.profile.causal_graph.edges:
                arrow = "->" if edge.type == "directed" else "--"
                edge_table.add_row(f"{edge.source} {arrow} {edge.target}")
            console.print(edge_table)

        # 7. Geospatial Analysis
        if self.profile.geospatial:
            console.print("\n[bold]7. Geospatial Analysis[/bold]")
            console.print(f"Detected Lat/Lon: {self.profile.geospatial.lat_col}, {self.profile.geospatial.lon_col}")
            console.print(f"Bounds: ({self.profile.geospatial.min_lat:.4f}, {self.profile.geospatial.min_lon:.4f}) to ({self.profile.geospatial.max_lat:.4f}, {self.profile.geospatial.max_lon:.4f})")

        # 8. Time Series Analysis
        if self.profile.timeseries and self.profile.timeseries.trend:
            console.print("\n[bold]8. Time Series Analysis[/bold]")
            console.print(f"Detected Date Column: {self.profile.timeseries.date_col}")
            
            min_date = self.profile.timeseries.trend[0].date
            max_date = self.profile.timeseries.trend[-1].date
            console.print(f"Range: {min_date} to {max_date}")
            
            if self.profile.timeseries.seasonality:
                # Seasonality object doesn't have 'period' or 'strength' directly on it based on schema
                # It has day_of_week and month_of_year lists.
                # Let's just print that seasonality analysis is available.
                console.print("Seasonality Analysis: Available (Day of Week, Month of Year)")

        # 9. Target Analysis
        if self.profile.target_col:
            console.print(f"\n[bold]9. Target Analysis (Target: {self.profile.target_col})[/bold]")
            
            # Numeric Target: Correlations
            if self.profile.target_correlations:
                corr_table = Table(show_header=True, header_style="bold green", title="Top Correlations")
                corr_table.add_column("Feature")
                corr_table.add_column("Correlation", justify="right")
                
                # Sort by absolute correlation
                sorted_corrs = sorted(self.profile.target_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                for col, corr in sorted_corrs[:5]:
                    corr_table.add_row(col, f"{corr:.4f}")
                console.print(corr_table)

            # Categorical Target: Interactions (ANOVA)
            if self.profile.target_interactions:
                # Filter for boxplots (which imply categorical target vs numeric feature)
                interactions = [i for i in self.profile.target_interactions if i.plot_type == "boxplot"]
                if interactions:
                    int_table = Table(show_header=True, header_style="bold green", title="Top Feature Associations (ANOVA)")
                    int_table.add_column("Feature")
                    int_table.add_column("p-value", justify="right")
                    int_table.add_column("Significance", justify="center")
                    
                    # Sort by p-value
                    interactions.sort(key=lambda x: x.p_value if x.p_value is not None else 1.0)
                    
                    for interaction in interactions[:5]:
                        p_val = interaction.p_value
                        p_str = f"{p_val:.4e}" if p_val is not None else "N/A"
                        sig = "[green]High[/green]" if p_val is not None and p_val < 0.05 else "[yellow]Low[/yellow]"
                        int_table.add_row(interaction.feature, p_str, sig)
                    console.print(int_table)

        # 10. Decision Tree Rules
        if self.profile.rule_tree:
            # Check if regression (accuracy might be R2 or similar, but let's check rules format)
            # Or check if class_name looks like a number
            is_regression = False
            if self.profile.rule_tree.nodes and self.profile.rule_tree.nodes[0].class_name:
                try:
                    float(self.profile.rule_tree.nodes[0].class_name)
                    # If root has a numeric class name (mean value), it's likely regression
                    # But wait, root is not a leaf, so it might not have class_name set correctly in all implementations
                    # Let's check the first leaf
                    for node in self.profile.rule_tree.nodes:
                        if node.is_leaf:
                            float(node.class_name)
                            is_regression = True
                            break
                except ValueError:
                    pass

            metric_name = "R²" if is_regression else "Accuracy"
            acc_str = f"{self.profile.rule_tree.accuracy:.2f}" if is_regression else f"{self.profile.rule_tree.accuracy:.1%}"
            
            console.print(f"\n[bold]10. Decision Tree Rules ({metric_name}: {acc_str})[/bold]")
            
            from rich.tree import Tree
            
            nodes_map = {n.id: n for n in self.profile.rule_tree.nodes}
            
            def add_nodes(node_id, tree):
                node = nodes_map.get(node_id)
                if not node: return
                
                if node.is_leaf:
                    if is_regression:
                        # Regression Leaf
                        val = float(node.class_name)
                        tree.add(f"[green]➜ Value = {val:.2f}[/green] [dim]n={node.samples}[/dim]")
                    else:
                        # Classification Leaf
                        total = sum(node.value)
                        conf = (max(node.value) / total * 100) if total > 0 else 0
                        tree.add(f"[green]➜ {node.class_name}[/green] ({conf:.1f}%) [dim]n={node.samples}[/dim]")
                else:
                    # Left (True)
                    if len(node.children) > 0:
                        left_child = nodes_map.get(node.children[0])
                        if left_child:
                            branch = tree.add(f"[blue]{node.feature} <= {node.threshold:.2f}[/blue]")
                            add_nodes(node.children[0], branch)
                    
                    # Right (False)
                    if len(node.children) > 1:
                        right_child = nodes_map.get(node.children[1])
                        if right_child:
                            branch = tree.add(f"[magenta]{node.feature} > {node.threshold:.2f}[/magenta]")
                            add_nodes(node.children[1], branch)

            if 0 in nodes_map:
                root = Tree("Root")
                add_nodes(0, root)
                console.print(root)

            # Print Text Rules if available
            if self.profile.rule_tree.rules:
                console.print("\n[italic]Extracted Rules:[/italic]")
                for rule in self.profile.rule_tree.rules:
                    console.print(f"[dim]• {rule}[/dim]")

            # Feature Importance
            if self.profile.rule_tree.feature_importances:
                console.print("\n[bold]Feature Importance (Surrogate Model)[/bold]")
                fi_table = Table(show_header=True, header_style="bold green")
                fi_table.add_column("Feature")
                fi_table.add_column("Importance", justify="right")
                fi_table.add_column("Bar", style="dim")

                for item in self.profile.rule_tree.feature_importances[:10]: # Top 10
                    feature = item.get("feature", "Unknown")
                    importance = item.get("importance", 0.0)
                    bar_len = int(importance * 20)
                    bar = "█" * bar_len
                    fi_table.add_row(feature, f"{importance:.4f}", bar)
                
                console.print(fi_table)

        # 11. PCA Structure (Latent Features)
        if self.profile.pca_components:
             console.print(f"\n[bold]11. PCA Latent Structure[/bold]")
             pca_table = Table(show_header=True, header_style="bold magenta")
             pca_table.add_column("Component")
             pca_table.add_column("Variance")
             pca_table.add_column("Top Loading Features")
             
             for comp in self.profile.pca_components[:3]:
                 feats = []
                 for k, v in comp.top_features.items():
                     sign = "+" if v > 0 else ""
                     feats.append(f"{k} ({sign}{v:.2f})")
                 feat_str = ", ".join(feats)
                 pca_table.add_row(comp.component, f"{comp.explained_variance_ratio:.1%}", feat_str)
             console.print(pca_table)

        # 12. Clustering Analysis
        if self.profile.clustering:
            console.print(f"\n[bold]12. Clustering Structure ({self.profile.clustering.method})[/bold]")
            console.print(f"Clusters: {self.profile.clustering.n_clusters} | Inertia: {self.profile.clustering.inertia:.2f}")

            cluster_table = Table(show_header=True, header_style="bold cyan")
            cluster_table.add_column("ID", justify="right")
            cluster_table.add_column("Size", justify="right")
            cluster_table.add_column("Size %", justify="right")
            cluster_table.add_column("Key Characteristics (Centroids)", style="italic")

            for cluster in self.profile.clustering.clusters:
                # Format centroids
                features = []
                # Sort by absolute magnitude or variance? 
                # Ideally we show top features that deviate from global mean, 
                # but 'center' here is just the coordinate. 
                # For now, let's just show top 3 features by order (assuming important ones are first or unsorted)
                # Or show all if few.
                
                # Let's show up to 3 features
                items = list(cluster.center.items())[:3] 
                feature_str = ", ".join([f"{k}={v:.2f}" for k, v in items])
                if len(cluster.center) > 3:
                     feature_str += "..."
                
                cluster_table.add_row(
                    str(cluster.cluster_id), 
                    str(cluster.size), 
                    f"{cluster.percentage:.1f}%",
                    feature_str
                )
            console.print(cluster_table)

        # 12. Smart Alerts
        if self.profile.alerts:
            console.print("\n[bold]12. Smart Alerts[/bold]")
            for alert in self.profile.alerts:
                color = "red" if alert.severity == "high" else "yellow"
                console.print(f"[{color}]• {alert.message}[/{color}]")

    def plot(self):
        """Generates and shows all available plots using Matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Please install 'matplotlib' to use plotting: pip install matplotlib")
            return

        self._plot_distributions()
        self._plot_correlations()
        self._plot_correlations_with_target()
        self._plot_target_interactions()
        self._plot_scatter_matrix()
        self._plot_pca()
        self._plot_geospatial()
        self._plot_timeseries()
        
        print("Displaying plots...")
        plt.show()

    def _plot_distributions(self):
        import matplotlib.pyplot as plt
        
        numeric_cols = [
            (name, col) for name, col in self.profile.columns.items() 
            if col.dtype == "Numeric" and col.histogram
        ]
        
        if not numeric_cols:
            return

        display_cols = numeric_cols[:4] # Limit to 4
        n_cols = len(display_cols)
        
        plt.figure(figsize=(5 * n_cols, 4))
        for i, (name, col) in enumerate(display_cols):
            plt.subplot(1, n_cols, i+1)
            widths = [b.end - b.start for b in col.histogram]
            centers = [(b.start + b.end)/2 for b in col.histogram]
            counts = [b.count for b in col.histogram]
            
            plt.bar(centers, counts, width=widths, align='center', alpha=0.7, edgecolor='black', color='skyblue')
            plt.title(f"Distribution: {name}")
            plt.xlabel(name)
            plt.ylabel("Count")
            plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

    def _plot_correlations(self):
        if not self.profile.correlations:
            return
        
        import matplotlib.pyplot as plt
        cols = self.profile.correlations.columns
        matrix = self.profile.correlations.values
        
        plt.figure(figsize=(8, 8))
        im = plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation')
        
        plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
        plt.yticks(range(len(cols)), cols)
        
        for i in range(len(cols)):
            for j in range(len(cols)):
                plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)

        plt.title("Correlation Matrix")
        plt.tight_layout()

    def _plot_correlations_with_target(self):
        if not self.profile.correlations_with_target:
            return
        
        import matplotlib.pyplot as plt
        cols = self.profile.correlations_with_target.columns
        matrix = self.profile.correlations_with_target.values
        
        plt.figure(figsize=(8, 8))
        im = plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation')
        
        plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
        plt.yticks(range(len(cols)), cols)
        
        for i in range(len(cols)):
            for j in range(len(cols)):
                plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)

        plt.title("Correlation Matrix (With Target)")
        plt.tight_layout()

    def _plot_target_interactions(self):
        if not self.profile.target_interactions:
            return
            
        import matplotlib.pyplot as plt
        boxplots = [i for i in self.profile.target_interactions if i.plot_type == "boxplot"]
        if not boxplots:
            return
            
        boxplots.sort(key=lambda x: x.p_value if x.p_value is not None else 1.0)
        display_items = boxplots[:6]
        
        n_plots = len(display_items)
        plt.figure(figsize=(5 * n_plots, 5))
        
        for i, interaction in enumerate(display_items):
            plt.subplot(1, n_plots, i+1)
            bxp_stats = []
            for cat_data in interaction.data:
                bxp_stats.append({
                    'label': cat_data.name,
                    'whislo': cat_data.stats.min,
                    'q1': cat_data.stats.q1,
                    'med': cat_data.stats.median,
                    'q3': cat_data.stats.q3,
                    'whishi': cat_data.stats.max,
                    'fliers': []
                })
            ax = plt.gca()
            ax.bxp(bxp_stats, showfliers=False)
            title = f"{interaction.feature} by Target"
            if interaction.p_value is not None:
                title += f"\n(ANOVA p={interaction.p_value:.4f})"
            plt.title(title)
            plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

    def _plot_scatter_matrix(self):
        if self.df is None:
            return
            
        import matplotlib.pyplot as plt
        try:
            from pandas.plotting import scatter_matrix
        except ImportError:
            return

        numeric_cols = [col for col, dtype in self.df.schema.items() if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
        if len(numeric_cols) > 5:
            numeric_cols = numeric_cols[:5]
            
        pdf = self.df.select(numeric_cols).to_pandas()
        
        colors = None
        target_col = self.profile.target_col
        if target_col and target_col in pdf.columns:
            unique_targets = pdf[target_col].unique()
            color_map = {val: i for i, val in enumerate(unique_targets)}
            colors = pdf[target_col].map(color_map)
        
        plt.figure(figsize=(10, 10))
        scatter_matrix(pdf, alpha=0.8, figsize=(10, 10), diagonal='kde', c=colors, cmap='viridis')
        plt.suptitle("Scatter Matrix")

    def _plot_pca(self):
        if not self.profile.pca_data:
            return
            
        import matplotlib.pyplot as plt
        x = [p.x for p in self.profile.pca_data]
        y = [p.y for p in self.profile.pca_data]
        labels = [p.label for p in self.profile.pca_data]
        
        try:
            c_values = [float(l) for l in labels]
        except (ValueError, TypeError):
            unique_labels = list(set([l for l in labels if l is not None]))
            label_map = {l: i for i, l in enumerate(unique_labels)}
            c_values = [label_map.get(l, -1) for l in labels]

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=c_values, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Target')
        plt.title("PCA Projection (2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)

    def _plot_geospatial(self):
        if not self.profile.geospatial or not self.profile.geospatial.sample_points:
            return
            
        import matplotlib.pyplot as plt
        
        lats = [p.lat for p in self.profile.geospatial.sample_points]
        lons = [p.lon for p in self.profile.geospatial.sample_points]
        labels = [p.label for p in self.profile.geospatial.sample_points]
        
        # Color by label if available
        c_values = None
        if any(labels):
            try:
                c_values = [float(l) if l is not None else -1 for l in labels]
            except (ValueError, TypeError):
                unique_labels = list(set([l for l in labels if l is not None]))
                label_map = {l: i for i, l in enumerate(unique_labels)}
                c_values = [label_map.get(l, -1) for l in labels]
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(lons, lats, c=c_values, cmap='viridis', alpha=0.6, s=10)
        if c_values:
            plt.colorbar(scatter, label='Target')
            
        plt.title(f"Geospatial Distribution ({len(lats)} points)")
        plt.xlabel(f"Longitude ({self.profile.geospatial.lon_col})")
        plt.ylabel(f"Latitude ({self.profile.geospatial.lat_col})")
        plt.grid(True, alpha=0.3)

    def _plot_timeseries(self):
        if not self.profile.timeseries or not self.profile.timeseries.trend:
            return
            
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        dates = []
        values_map = {} # col -> list of values
        
        # Initialize lists for each column found in the first point
        first_point = self.profile.timeseries.trend[0]
        for col in first_point.values.keys():
            values_map[col] = []
            
        for point in self.profile.timeseries.trend:
            try:
                # Parse date string back to datetime for plotting
                dt = datetime.fromisoformat(point.date)
                dates.append(dt)
                
                for col, val in point.values.items():
                    if col in values_map:
                        values_map[col].append(val)
            except ValueError:
                continue
                
        if not dates:
            return
            
        plt.figure(figsize=(12, 6))
        for col, values in values_map.items():
            plt.plot(dates, values, label=col)
            
        plt.title(f"Time Series Trend (Daily Aggregation)")
        plt.xlabel(f"Date ({self.profile.timeseries.date_col})")
        plt.ylabel("Value (Mean)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

