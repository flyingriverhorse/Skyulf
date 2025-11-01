"""
Async File-Based Context Service for FastAPI

Service for gathering data context from files and data manager.
Migrated from Flask with async patterns.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .eda_context_formatter import format_eda_context_summary

logger = logging.getLogger(__name__)


class AsyncFileBasedContextService:
    """Async service for gathering comprehensive data context from files and data manager"""

    def __init__(self, upload_folder: Optional[str] = None, cache_folder: Optional[str] = None):
        # Use file-based approach instead of database
        self.upload_folder = upload_folder or "uploads/data"
        self.cache_folder = cache_folder or "data/cache"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def get_data_context(self, source_id: str) -> Dict[str, Any]:
        """
        Gather comprehensive data context for a given source ID from files and data manager

        Args:
            source_id: The data source identifier

        Returns:
            Dictionary with data context information
        """
        context = {
            "source_id": source_id,
            "file_info": await self._get_file_info(source_id),
            "data_preview": await self._get_data_preview_from_manager(source_id),
            "source_metadata": await self._get_source_metadata_from_manager(source_id),
        }

        return context

    async def _get_file_info(self, source_id: str) -> Dict[str, Any]:
        """Get file information using the data manager (async version)"""
        try:
            # Mock implementation - in real version this would integrate with data manager
            self.logger.debug(f"Getting file info for source {source_id}")
            
            # Simulate async file operations
            await asyncio.sleep(0.001)  # Minimal async delay
            
            return {
                "file_found": False,
                "reason": "Data manager integration not implemented in async version",
                "source_id": source_id
            }

        except Exception as e:
            self.logger.error(f"Error getting file info for {source_id}: {e}")
            return {"error": str(e), "source_id": source_id}

    async def _get_data_preview_from_manager(self, source_id: str) -> Dict[str, Any]:
        """Get data preview using the data manager (async version)"""
        try:
            # Mock implementation
            self.logger.debug(f"Getting data preview for source {source_id}")
            
            # Simulate async operation
            await asyncio.sleep(0.001)
            
            return {
                "preview_available": False,
                "reason": "Data manager integration not implemented in async version"
            }

        except Exception as e:
            self.logger.error(f"Error getting data preview for {source_id}: {e}")
            return {"error": str(e)}

    async def _get_source_metadata_from_manager(self, source_id: str) -> Dict[str, Any]:
        """Get source metadata using the data manager (async version)"""
        try:
            # Mock implementation
            self.logger.debug(f"Getting source metadata for source {source_id}")
            
            # Simulate async operation
            await asyncio.sleep(0.001)
            
            return {
                "metadata_available": False,
                "reason": "Data manager integration not implemented in async version"
            }

        except Exception as e:
            self.logger.error(f"Error getting source metadata for {source_id}: {e}")
            return {"error": str(e)}

    async def format_data_context(
        self, 
        source_id: str, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format data context into a comprehensive string for LLM consumption
        Enhanced to extract rich information from user's current view
        
        Args:
            source_id: The data source identifier
            user_context: Additional context from user's current view (from frontend)
            
        Returns:
            Formatted context string with detailed data information
        """
        try:
            def safe_cell_text(value: Any, limit: int = 120) -> str:
                if value is None:
                    return ""
                if isinstance(value, bool):
                    return "true" if value else "false"
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return str(value)
                if isinstance(value, (list, tuple, set)):
                    joined = ", ".join(str(item) for item in list(value)[:5])
                    if len(value) > 5:
                        joined += ", …"
                    return joined[: limit - 1] + "…" if len(joined) > limit else joined
                text = str(value)
                return text[: limit - 1] + "…" if len(text) > limit else text

            def build_markdown_table(headers: List[str], rows: List[List[Any]]) -> Optional[str]:
                limited_headers = headers[:12]
                if not limited_headers or not rows:
                    return None
                rendered_rows: List[str] = []
                for row in rows[:25]:
                    if isinstance(row, list):
                        rendered = [safe_cell_text(cell) for cell in row[: len(limited_headers)]]
                    elif isinstance(row, dict):
                        rendered = [safe_cell_text(row.get(col)) for col in limited_headers]
                    else:
                        rendered = [safe_cell_text(row)]
                        if len(limited_headers) > 1:
                            rendered.extend("" for _ in range(len(limited_headers) - 1))
                    rendered_rows.append("| " + " | ".join(rendered) + " |")
                header_line = "| " + " | ".join(limited_headers) + " |"
                separator = "| " + " | ".join(["---"] * len(limited_headers)) + " |"
                table_lines = [header_line, separator] + rendered_rows
                return "\n".join(table_lines)

            def render_preview_sample(sample: Dict[str, Any]) -> List[str]:
                if not isinstance(sample, dict):
                    return []
                columns = sample.get("columns") or sample.get("headers")
                rows = sample.get("rows") or sample.get("data")
                if not isinstance(columns, list) or not columns or not isinstance(rows, list) or not rows:
                    return []

                table = build_markdown_table(columns, rows)
                lines: List[str] = []
                total_rows = sample.get("total_rows") or sample.get("totalRows")
                row_count = sample.get("row_count") or len(rows)
                if total_rows:
                    try:
                        total_rows_int = int(total_rows)
                        lines.append(f"Rows included: {min(row_count, 25)} of ~{total_rows_int:,}")
                    except (ValueError, TypeError):
                        lines.append(f"Rows included: {min(row_count, 25)} (total rows unavailable)")
                else:
                    lines.append(f"Rows included: {min(row_count, 25)}")

                column_count = sample.get("column_count") or len(columns)
                truncated_columns = sample.get("truncated_columns") or 0
                if truncated_columns:
                    lines.append(
                        f"Columns displayed: {len(columns[:12])} of {column_count} (remaining {truncated_columns} hidden for brevity)"
                    )
                else:
                    lines.append(f"Columns displayed: {len(columns[:12])} of {column_count}")

                if table:
                    lines.append(table)

                type_summary = sample.get("type_summary")
                if isinstance(type_summary, list) and type_summary:
                    lines.append("Column type summary:")
                    lines.extend(f"- {entry}" for entry in type_summary[:6])
                else:
                    dtypes = sample.get("dtypes")
                    if isinstance(dtypes, dict) and dtypes:
                        preview = list(dtypes.items())[:6]
                        lines.append("Column types:")
                        lines.extend(f"- {col}: {dtype}" for col, dtype in preview)

                numeric_cols = sample.get("numeric_columns")
                if isinstance(numeric_cols, list) and numeric_cols:
                    preview_numeric = ", ".join(numeric_cols[:6])
                    suffix = f" (+{len(numeric_cols) - 6})" if len(numeric_cols) > 6 else ""
                    lines.append(f"Numeric focus: {preview_numeric}{suffix}")

                categorical_cols = sample.get("categorical_columns")
                if isinstance(categorical_cols, list) and categorical_cols:
                    preview_cats = ", ".join(categorical_cols[:6])
                    suffix = f" (+{len(categorical_cols) - 6})" if len(categorical_cols) > 6 else ""
                    lines.append(f"Categorical focus: {preview_cats}{suffix}")

                if sample.get("column_limit") and sample.get("column_count"):
                    limit = sample.get("column_limit")
                    if isinstance(limit, int) and isinstance(sample.get("column_count"), int) and sample["column_count"] > limit:
                        lines.append(f"Only the first {limit} columns are displayed to keep the preview concise.")

                return lines

            # Build comprehensive formatted context
            formatted_parts = []
            
            # Header with source information
            formatted_parts.append(f"📊 **DATASET ANALYSIS CONTEXT**")
            formatted_parts.append(f"Source ID: {source_id}")
            formatted_parts.append("")
            
            # Extract and format current view context (from frontend JavaScript)
            if user_context:
                # Process current tab information
                current_tab = user_context.get("currentTab", "unknown")
                formatted_parts.append(f"**Current View:** {current_tab} tab")

                # Process data structures from frontend
                data_structures = user_context.get("dataStructures", {})
                preview_sample = data_structures.get("previewSample")
                sample_lines = render_preview_sample(preview_sample) if preview_sample else []
                if sample_lines:
                    formatted_parts.append("**📋 DATA PREVIEW SNAPSHOT:**")
                    formatted_parts.extend(sample_lines)
                    formatted_parts.append("")
                if data_structures.get("currentQualityReport"):
                    formatted_parts.append("**✅ Quality Report Data Available**")

                # Extract visible data from current tab
                visible_data = user_context.get("visibleData", {})
                if visible_data:
                    formatted_parts.append("**Current Page Content:**")

                    if "qualityTabContent" in visible_data:
                        formatted_parts.append("- Quality analysis is active and loaded")
                    if "textTabContent" in visible_data:
                        formatted_parts.append("- Text analysis data available")
                    if "previewTabContent" in visible_data:
                        formatted_parts.append("- Data preview is loaded")
                    if "recommendationsTabContent" in visible_data:
                        formatted_parts.append("- ML recommendations available")

                extracted_data = user_context.get("extractedData", {})
                if extracted_data:
                    quality_metrics = extracted_data.get("qualityMetrics", [])
                    if quality_metrics:
                        formatted_parts.append("")
                        formatted_parts.append("**📈 QUALITY METRICS:**")
                        for metric in quality_metrics[:10]:
                            label = metric.get("label", "Unknown")
                            value = metric.get("value", "N/A")
                            formatted_parts.append(f"- {label}: {value}")

                formatted_parts.append("")

                eda_context = user_context.get("edaNotebook") if isinstance(user_context.get("edaNotebook"), dict) else None
                if eda_context:
                    formatted_parts.append("**🧠 ACTIVE EDA NOTEBOOK CONTEXT:**")
                    formatted_parts.append(format_eda_context_summary(eda_context))
                    formatted_parts.append("")

            # Add guidance for comprehensive analysis capabilities
            formatted_parts.append("**🧠 COMPREHENSIVE ANALYSIS CAPABILITIES:**")
            formatted_parts.append("**Data Quality & Profiling:**")
            formatted_parts.append("  • Missing value patterns and systematic gaps analysis")
            formatted_parts.append("  • Data type consistency and validation errors")
            formatted_parts.append("  • Duplicate records detection and handling strategies")
            formatted_parts.append("  • Outlier identification using statistical methods (IQR, Z-score)")
            formatted_parts.append("  • Data distribution analysis (normal, skewed, multimodal)")
            formatted_parts.append("  • Column-level quality scoring and ranking")
            formatted_parts.append("")
            
            formatted_parts.append("**Statistical & Exploratory Analysis:**")
            formatted_parts.append("  • Descriptive statistics with business interpretation")
            formatted_parts.append("  • Correlation analysis and multicollinearity detection")
            formatted_parts.append("  • Feature importance and information gain analysis")
            formatted_parts.append("  • Categorical variable analysis and cardinality assessment")
            formatted_parts.append("  • Time series patterns and seasonality (if applicable)")
            formatted_parts.append("  • Clustering tendency and segmentation opportunities")
            formatted_parts.append("")
            
            formatted_parts.append("**Machine Learning & Modeling:**")
            formatted_parts.append("  • Algorithm recommendations based on data characteristics")
            formatted_parts.append("  • Feature engineering strategies and transformations")
            formatted_parts.append("  • Preprocessing pipeline design and optimization")
            formatted_parts.append("  • Cross-validation strategy and evaluation metrics")
            formatted_parts.append("  • Model performance predictions and accuracy estimates")
            formatted_parts.append("  • Overfitting risk assessment and mitigation strategies")
            formatted_parts.append("")

            # Enhanced example questions with more depth
            formatted_parts.append("**❓ ADVANCED ANALYSIS QUESTIONS:**")
            formatted_parts.append("**Data Quality & Preprocessing:**")
            formatted_parts.append("  • 'Create a comprehensive data quality report with actionable insights'")
            formatted_parts.append("  • 'Which columns should I drop vs. impute for missing values?'")
            formatted_parts.append("  • 'What's the best strategy for handling high-cardinality features?'")
            formatted_parts.append("  • 'Are there systematic biases or data collection issues?'")
            formatted_parts.append("")
            
            formatted_parts.append("**Statistical Insights:**")
            formatted_parts.append("  • 'Show me the statistical distribution of key variables'")
            formatted_parts.append("  • 'Which features are most correlated and why does it matter?'")
            formatted_parts.append("  • 'Identify outliers and explain their business impact'")
            formatted_parts.append("  • 'What patterns suggest seasonal or temporal effects?'")
            formatted_parts.append("")
            
            formatted_parts.append("**ML Strategy & Implementation:**")
            formatted_parts.append("  • 'Recommend the best ML algorithms for my use case'")
            formatted_parts.append("  • 'Design a feature engineering pipeline for this data'")
            formatted_parts.append("  • 'What evaluation metrics should I use and why?'")
            formatted_parts.append("  • 'How can I prevent overfitting with this dataset?'")
            formatted_parts.append("  • 'Create a complete preprocessing and modeling roadmap'")
            formatted_parts.append("")
            
            formatted_parts.append("**Business Impact & Insights:**")
            formatted_parts.append("  • 'Translate these data patterns into business value'")
            formatted_parts.append("  • 'What are the ROI implications of fixing data quality issues?'")
            formatted_parts.append("  • 'How reliable are predictions likely to be with this data?'")
            formatted_parts.append("  • 'What additional data would significantly improve model performance?'")
            formatted_parts.append("")

            result = "\n".join(formatted_parts)
            self.logger.debug(f"Enhanced context formatted for {source_id} ({len(result)} characters)")
            
            return result

        except Exception as e:
            self.logger.error(f"Error formatting enhanced context for {source_id}: {e}")
            return f"Error gathering data context for {source_id}: {str(e)}. Please try asking about general data analysis or upload a new dataset."

    async def format_eda_context(
        self,
        source_id: Optional[str],
        user_context: Optional[Dict[str, Any]] = None,
        cell_id: Optional[str] = None,
        cell_scope: Optional[str] = None,
    ) -> str:
        """Create an EDA-focused context summary scoped to a specific notebook cell when provided."""
        try:
            parts: List[str] = []

            def truncate(value: Optional[str], limit: int = 280) -> Optional[str]:
                if not value or not isinstance(value, str):
                    return None
                return value if len(value) <= limit else value[: limit - 1].strip() + "…"

            def render_metrics(metrics: Any, limit: int = 6) -> Optional[str]:
                if not isinstance(metrics, list):
                    return None
                rendered: List[str] = []
                for metric in metrics[:limit]:
                    if not isinstance(metric, dict):
                        continue
                    label = metric.get("label") or metric.get("name") or "Metric"
                    value = metric.get("value")
                    if value is None:
                        continue
                    unit = metric.get("unit")
                    if isinstance(value, float):
                        value_text = f"{value:0.3g}"
                    else:
                        value_text = str(value)
                    if unit:
                        value_text = f"{value_text} {unit}"
                    rendered.append(f"{label}: {value_text}")
                if not rendered:
                    return None
                return "; ".join(rendered)

            def render_table(table: Dict[str, Any], row_limit: int = 5) -> Optional[str]:
                if not isinstance(table, dict):
                    return None
                headers = table.get("headers") or table.get("columns")
                rows = table.get("rows") or table.get("data")
                if not isinstance(headers, list) or not isinstance(rows, list) or not headers or not rows:
                    return None
                safe_rows: List[List[str]] = []
                for row in rows[:row_limit]:
                    if isinstance(row, dict):
                        safe_rows.append([str(row.get(col, "")) for col in headers])
                    elif isinstance(row, list):
                        safe_rows.append([str(item) for item in row[: len(headers)]])
                    else:
                        safe_rows.append([str(row)])
                header_line = "| " + " | ".join(headers) + " |"
                separator = "| " + " | ".join(["---"] * len(headers)) + " |"
                row_lines = ["| " + " | ".join(row) + " |" for row in safe_rows]
                if len(rows) > row_limit:
                    row_lines.append(f"| … ({len(rows) - row_limit} more) |")
                lines = [header_line, separator] + row_lines
                title = table.get("title")
                if title:
                    lines.insert(0, f"Table: {title}")
                return "\n".join(lines)

            def render_insights(insights: Any, limit: int = 4) -> List[str]:
                if not isinstance(insights, list):
                    return []
                rendered: List[str] = []
                for insight in insights[:limit]:
                    if isinstance(insight, str):
                        rendered.append(insight)
                        continue
                    if not isinstance(insight, dict):
                        continue
                    prefix = insight.get("level")
                    text = insight.get("text") or insight.get("summary")
                    if not text:
                        continue
                    rendered.append(f"{prefix.title()}: {text}" if prefix else text)
                return rendered

            def next_step_code(columns: List[str]) -> Optional[str]:
                usable = [col for col in columns if isinstance(col, str)]
                if not usable:
                    return None
                focus = usable[:3]
                quoted = ", ".join(f"'{col}'" for col in focus)
                code_lines = ["import pandas as pd", f"df_subset = df[[{quoted}]].dropna()"]
                if len(focus) >= 2:
                    code_lines.append(f"df_subset.groupby('{focus[0]}')['{focus[1]}'].describe()")
                code_lines.append("df_subset.corr(numeric_only=True)")
                return "```python\n" + "\n".join(code_lines) + "\n```"

            def describe_analysis_cell(cell: Dict[str, Any], label: str, cached_columns: List[str]) -> List[str]:
                lines: List[str] = []
                name = cell.get("analysisName") or cell.get("analysisType") or label
                status = cell.get("status", "unknown")
                run_count = cell.get("runCount") or 0
                lines.append(f"{label}: {name} (status={status}, runs={run_count})")

                structured = cell.get("structuredResults") or []
                numeric_metrics: List[Tuple[str, float]] = []
                if structured and isinstance(structured, list):
                    first = structured[0] or {}
                    metrics_summary = render_metrics(first.get("metrics"))
                    if metrics_summary:
                        lines.append(f"Metrics: {metrics_summary}")
                    raw_metrics = first.get("metrics") or []
                    for metric in raw_metrics:
                        if isinstance(metric, dict):
                            label_text = metric.get("label") or metric.get("name")
                            value = metric.get("value")
                            if label_text and isinstance(value, (int, float)):
                                numeric_metrics.append((label_text.lower(), float(value)))
                    insights = render_insights(first.get("insights"))
                    for insight in insights:
                        lines.append(f"Insight: {insight}")
                    tables = first.get("tables") or []
                    if isinstance(tables, list) and tables:
                        table_text = render_table(tables[0])
                        if table_text:
                            lines.append(table_text)
                    preview_text = truncate(first.get("responsePreview") or first.get("metaSummary"), 360)
                    if preview_text:
                        lines.append(f"Summary: {preview_text}")

                legacy_output = cell.get("legacyOutput") or {}
                stdout = truncate(legacy_output.get("stdout"), 220)
                if stdout:
                    lines.append(f"Console sample: {stdout}")
                if legacy_output.get("stderr"):
                    warning_text = truncate(legacy_output.get("stderr"), 180)
                    if warning_text:
                        lines.append(f"Warnings: {warning_text}")

                request = cell.get("request") or {}
                requested_columns = request.get("selected_columns") or request.get("columns") or []
                column_pool = [col for col in requested_columns if isinstance(col, str)] or cached_columns

                recommendations: List[str] = []
                analysis_type = (cell.get("analysisType") or "").lower()
                if "correlation" in analysis_type or "corr" in analysis_type:
                    recommendations.append("Explore feature interactions by creating polynomial or interaction terms for highly correlated pairs.")
                    recommendations.append("Check for multicollinearity before modeling—consider dimensionality reduction (PCA, feature selection).")
                elif "distribution" in analysis_type or "histogram" in analysis_type:
                    recommendations.append("Identify skewed features and apply transformations (log, Box-Cox, Yeo-Johnson) to normalize distributions.")
                    recommendations.append("Detect outliers using IQR or Z-score methods and decide on capping, removal, or robust scaling.")
                elif "missing" in analysis_type or "quality" in analysis_type:
                    recommendations.append("Evaluate imputation strategies: simple (mean/median/mode), advanced (KNN, iterative), or create 'missing' indicator features.")
                    recommendations.append("Check if missingness is systematic (MCAR, MAR, MNAR)—may require domain-specific handling.")
                elif "categorical" in analysis_type or "cardinality" in analysis_type:
                    recommendations.append("Plan encoding: one-hot for low-cardinality, target/frequency encoding for high-cardinality, embeddings for very high dimensions.")
                    recommendations.append("Group rare categories into 'Other' to reduce noise and improve model generalization.")
                elif "time" in analysis_type or "temporal" in analysis_type or "series" in analysis_type:
                    recommendations.append("Engineer time-based features: hour, day of week, month, quarter, is_weekend, is_holiday, time since last event.")
                    recommendations.append("Check for seasonality and trends—consider decomposition, lag features, rolling statistics.")
                elif "text" in analysis_type or "nlp" in analysis_type:
                    recommendations.append("Extract text features: length, word count, unique word ratio, sentiment scores, TF-IDF, or embeddings (Word2Vec, BERT).")
                    recommendations.append("Clean text data: lowercasing, removing punctuation/stopwords, stemming/lemmatization before feature extraction.")

                for metric_name, metric_value in numeric_metrics:
                    if "missing" in metric_name and metric_value > 5:
                        recommendations.append(f"High missing rate ({metric_value:.1f}%)—consider domain-driven imputation or creating a missingness indicator feature.")
                    if "correlation" in metric_name and abs(metric_value) > 0.8:
                        recommendations.append(f"Strong correlation detected ({metric_value:.2f})—investigate for feature redundancy or potential interaction effects.")
                    if "skew" in metric_name and abs(metric_value) > 1:
                        recommendations.append(f"High skewness ({metric_value:.2f})—apply log or Box-Cox transformation to improve normality for modeling.")
                    if "cardinality" in metric_name and metric_value > 50:
                        recommendations.append(f"Very high cardinality ({int(metric_value)} unique)—consider grouping, hashing, or embeddings to reduce dimensionality.")

                if column_pool:
                    if len(column_pool) >= 2:
                        recommendations.append(f"Create interaction features between {column_pool[0]} and {column_pool[1]} to capture non-linear relationships.")
                    if any(token in col.lower() for col in column_pool for token in ['amount', 'price', 'value', 'count']):
                        recommendations.append("Numeric features detected—consider binning, scaling (StandardScaler, MinMaxScaler), or ratio features.")

                if recommendations:
                    lines.append("Recommended next steps:")
                    lines.extend(f"- {rec}" for rec in recommendations[:5])

                code_block = next_step_code(column_pool)
                if code_block:
                    lines.append("Suggested code to extend analysis:")
                    lines.append(code_block)

                return lines

            def describe_custom_cell(cell: Dict[str, Any], label: str, cached_columns: List[str]) -> List[str]:
                lines: List[str] = []
                status = cell.get("status", "unknown")
                lines.append(f"{label}: custom cell {cell.get('cellId')} (status={status})")
                code = truncate(cell.get("code"), 320)
                code_lower = (code or "").lower()
                if code:
                    lines.append("Code excerpt:")
                    lines.append("```python\n" + code + "\n```")
                text_output = truncate(cell.get("textOutput"), 320)
                if text_output:
                    lines.append(f"Output: {text_output}")
                error = truncate(cell.get("error"), 240)
                if error:
                    lines.append(f"Error: {error}")
                plot_count = cell.get("plotCount") or cell.get("plotsPreview")
                if plot_count:
                    lines.append(f"Plots generated: {plot_count}")

                custom_suggestions: List[str] = []
                if error:
                    custom_suggestions.append("Debug the error by checking variable types, ensuring required imports, and validating data shapes/dtypes.")
                    custom_suggestions.append("Add print/logging statements to trace execution flow and inspect intermediate results.")
                elif status == "success":
                    if "plot" in code_lower or "plt." in code_lower or "sns." in code_lower:
                        custom_suggestions.append("Enhance visualizations: add titles, labels, legends, and annotations for clarity.")
                        custom_suggestions.append("Explore alternative plot types (violin, box, pair plots) to reveal hidden patterns.")
                    if "groupby" in code_lower or "agg" in code_lower:
                        custom_suggestions.append("Turn aggregated results into new features—ratios, rolling averages, or deviation from group mean.")
                    if "merge" in code_lower or "join" in code_lower:
                        custom_suggestions.append("Validate merge cardinality (1:1, 1:many)—check for duplicates or unexpected row count changes.")
                    if "transform" in code_lower or "apply" in code_lower:
                        custom_suggestions.append("Profile transformation performance—consider vectorized operations or Numba/Cython for speed-ups.")
                    custom_suggestions.append("Experiment with parameter variations (window sizes, thresholds, sample splits) to test robustness.")
                    custom_suggestions.append("Export intermediate results for downstream tasks or save useful transformations as reusable functions.")

                if code and not error:
                    if "sklearn" in code_lower or "model" in code_lower:
                        custom_suggestions.append("Evaluate model with cross-validation, compare multiple algorithms, and tune hyperparameters (GridSearchCV, Optuna).")
                    if "feature" in code_lower or "engineer" in code_lower:
                        custom_suggestions.append("Validate new features: check correlation with target, feature importance, and impact on model performance.")

                if custom_suggestions:
                    lines.append("Improvement ideas:")
                    lines.extend(f"- {sug}" for sug in custom_suggestions[:5])

                snippet_columns = cached_columns[:3]
                code_block = next_step_code(snippet_columns)
                if code_block:
                    lines.append("Next experiment snippet:")
                    lines.append(code_block)

                return lines

            def summarise_other_cells(cells: List[Dict[str, Any]], skip_id: Optional[str], limit: int = 4) -> List[str]:
                summary: List[str] = []
                for cell in cells:
                    if not isinstance(cell, dict):
                        continue
                    cell_identifier = cell.get("cellId")
                    if not cell_identifier or (skip_id and cell_identifier == skip_id):
                        continue
                    name = cell.get("analysisName") or cell.get("analysisType") or cell.get("cellId") or "Cell"
                    status = cell.get("status", "unknown")
                    summary.append(f"- {cell_identifier}: {name} ({status})")
                    if len(summary) >= limit:
                        break
                return summary

            if source_id:
                parts.append(f"Source ID: {source_id}")

            if not user_context:
                parts.append("No EDA session data provided. Encourage the user to share which analyses they have run and propose next diagnostic steps.")
                return "\n".join(parts)

            notebook_summary = user_context.get("edaNotebookSummary")
            eda_notebook = user_context.get("edaNotebook") or {}
            eda_highlights = user_context.get("edaHighlights") or {}

            dataset = eda_notebook.get("dataset") or {}
            if dataset:
                name = dataset.get("name") or dataset.get("sourceId")
                if name:
                    parts.append(f"Dataset: {name}")
                info = dataset.get("info")
                if isinstance(info, dict) and info:
                    snippet = [f"{k}={v}" for k, v in list(info.items())[:6]]
                    if snippet:
                        parts.append("Dataset info: " + "; ".join(snippet))

            selected_columns = eda_notebook.get("selectedColumns")
            cached_columns: List[str] = []
            if isinstance(selected_columns, list) and selected_columns:
                cached_columns = [col for col in selected_columns if isinstance(col, str)]
                preview = ", ".join(cached_columns[:12])
                if len(cached_columns) > 12:
                    preview += f" (+{len(cached_columns) - 12} more)"
                parts.append("Selected columns: " + preview)

            analysis_cells = eda_notebook.get("analysisCells") or []
            analysis_map = {cell.get("cellId"): cell for cell in analysis_cells if isinstance(cell, dict)}
            custom_cells = eda_notebook.get("customAnalyses") or []
            custom_map = {cell.get("cellId"): cell for cell in custom_cells if isinstance(cell, dict)}

            requested_cell_id = cell_id or user_context.get("targetCellId")
            requested_scope = cell_scope or user_context.get("targetCellScope")

            active_context = eda_notebook.get("activeContext") or {}
            if not requested_cell_id and isinstance(active_context, dict):
                requested_cell_id = active_context.get("cellId")
            if not requested_scope and isinstance(active_context, dict):
                requested_scope = active_context.get("scope")

            scope_normalised = (requested_scope or "analysis").lower()
            target_cell = None
            if requested_cell_id:
                if scope_normalised == "custom":
                    target_cell = custom_map.get(requested_cell_id)
                if target_cell is None:
                    target_cell = analysis_map.get(requested_cell_id)
                    if target_cell is not None:
                        scope_normalised = "analysis"
                elif target_cell is not None:
                    scope_normalised = "custom"

            if target_cell is None:
                if scope_normalised == "custom" and custom_cells:
                    target_cell = custom_cells[0]
                    requested_cell_id = target_cell.get("cellId")
                elif analysis_cells:
                    target_cell = analysis_cells[0]
                    requested_cell_id = target_cell.get("cellId")
                    scope_normalised = "analysis"

            if requested_cell_id:
                parts.append(f"Focused cell: {scope_normalised} {requested_cell_id}")

            if target_cell is not None:
                if scope_normalised == "custom":
                    parts.extend(describe_custom_cell(target_cell, "Active custom analysis", cached_columns))
                else:
                    parts.extend(describe_analysis_cell(target_cell, "Active analysis", cached_columns))
            else:
                parts.append("No executed EDA cells were found. Encourage running an analysis or custom code cell to generate results before chatting.")

            other_analysis_lines = summarise_other_cells(analysis_cells, requested_cell_id, limit=4)
            if other_analysis_lines:
                parts.append("Other analysis cells available (not included in this chat):")
                parts.extend(other_analysis_lines)

            other_custom_lines = summarise_other_cells(custom_cells, requested_cell_id, limit=3)
            if other_custom_lines:
                parts.append("Other custom cells available (not included in this chat):")
                parts.extend(other_custom_lines)

            def filter_highlights(highlights: Any) -> List[str]:
                if not isinstance(highlights, list):
                    return []
                filtered: List[str] = []
                for item in highlights:
                    if isinstance(item, dict):
                        cell_ref = item.get("cellId") or item.get("cell_id")
                        if requested_cell_id and cell_ref and cell_ref != requested_cell_id:
                            continue
                        text = item.get("summary") or item.get("text")
                        if text:
                            filtered.append(text)
                    elif not requested_cell_id:
                        filtered.append(str(item))
                return filtered[:5]

            analysis_highlights = filter_highlights(eda_highlights.get("analyses"))
            if analysis_highlights:
                parts.append("Cell highlights:")
                parts.extend(f"- {item}" for item in analysis_highlights)

            custom_highlights = filter_highlights(eda_highlights.get("custom"))
            if custom_highlights:
                parts.append("Custom cell notes:")
                parts.extend(f"- {item}" for item in custom_highlights)

            extracted = user_context.get("extractedData") or {}
            quality_metrics = extracted.get("qualityMetrics") or []
            if quality_metrics:
                parts.append("Quality metrics from preview panel:")
                for metric in quality_metrics[:6]:
                    label_text = metric.get("label") or "Metric"
                    value_text = metric.get("value") or "N/A"
                    parts.append(f"- {label_text}: {value_text}")

            visible_quality = (user_context.get("visibleData") or {}).get("qualityTabContent")
            if isinstance(visible_quality, str) and visible_quality not in ("Content not yet loaded", ""):
                snippet = truncate(visible_quality, 600)
                if snippet:
                    parts.append("Quality tab narrative:")
                    parts.append(snippet)

            if not requested_cell_id and notebook_summary:
                parts.append("Notebook summary:")
                parts.append(notebook_summary)

            if not parts:
                parts.append("No EDA session data provided. Encourage the user to share which analyses they have run and propose next diagnostic steps.")

            return "\n".join(parts)

        except Exception as e:
            self.logger.error(f"Error formatting EDA context for {source_id}: {e}")
            return "EDA context unavailable; offer a structured plan covering descriptive stats, data quality review, correlations, and modeling readiness."

    async def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        try:
            # Mock implementation - would integrate with actual data manager
            await asyncio.sleep(0.001)
            
            return []  # Return empty list for now

        except Exception as e:
            self.logger.error(f"Error getting available sources: {e}")
            return []