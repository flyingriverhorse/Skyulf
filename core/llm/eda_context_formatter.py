"""Utilities for preparing enriched EDA notebook context for LLM prompts."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional

ISO_FORMATS: Iterable[str] = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
)


def _human_ts(value: Optional[str]) -> str:
    """Return a short human readable timestamp if possible."""
    if not value or not isinstance(value, str):
        return "recently"

    for fmt in ISO_FORMATS:
        try:
            parsed = _dt.datetime.strptime(value, fmt)
            break
        except ValueError:
            continue
    else:
        return "recently"

    # Normalise to local timezone-less output for consistency
    if parsed.tzinfo:
        parsed = parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    return parsed.strftime("%b %d • %H:%M")


def _truncate(text: Optional[str], limit: int = 320) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    if len(text) <= limit:
        return text.strip()
    return text[: limit - 1].strip() + "…"


def _summarise_metrics(metrics: List[Dict[str, Any]], limit: int = 3) -> str:
    highlights: List[str] = []
    for metric in metrics[:limit]:
        label = metric.get("label") or metric.get("name") or "Metric"
        value = metric.get("value")
        unit = metric.get("unit")
        if value is None:
            continue
        if isinstance(value, float):
            value_fmt = f"{value:0.3g}"
        else:
            value_fmt = str(value)
        if unit:
            value_fmt = f"{value_fmt} {unit}"
        highlights.append(f"{label}: {value_fmt}")
    return "; ".join(highlights)


def _summarise_insights(insights: List[Dict[str, Any]], limit: int = 2) -> str:
    summaries: List[str] = []
    for insight in insights[:limit]:
        text = insight.get("text")
        if not text:
            continue
        level = insight.get("level")
        if level:
            summaries.append(f"{level.title()}: {text}")
        else:
            summaries.append(text)
    return " | ".join(summaries)


def _build_analysis_summary(cell: Dict[str, Any]) -> str:
    name = cell.get("analysisName") or cell.get("analysisType") or "Analysis"
    status = cell.get("status", "unknown").title()
    when = _human_ts(cell.get("completedAt") or cell.get("startedAt"))

    # Prefer explicitly pre-computed LLM summaries when available
    llm_summary = _truncate(cell.get("llmSummary"), 360)
    if llm_summary:
        return f"{name} — {status} ({when})\n{llm_summary}"

    parts: List[str] = []
    structured = cell.get("structuredResults") or []
    if structured:
        first = structured[0] or {}
        metrics = first.get("metrics") or []
        if metrics:
            metric_summary = _summarise_metrics(metrics)
            if metric_summary:
                parts.append(metric_summary)
        insights = first.get("insights") or []
        insight_summary = _summarise_insights(insights)
        if insight_summary:
            parts.append(insight_summary)
    meta_summary = _truncate(cell.get("metaSummary"), 280)
    if meta_summary:
        parts.append(meta_summary)
    legacy = cell.get("legacyOutput") or {}
    stdout = _truncate(legacy.get("stdout"), 220)
    if stdout:
        parts.append(f"Console: {stdout}")

    if not parts:
        request = cell.get("request") or {}
        selected = request.get("selected_columns") or cell.get("selectedColumnsSnapshot")
        if selected:
            cols = ", ".join(selected[:5])
            if len(selected) > 5:
                cols += f" (+{len(selected) - 5} more)"
            parts.append(f"Columns: {cols}")

    description = "\n".join(parts) if parts else "No detailed output captured."
    return f"{name} — {status} ({when})\n{description}"


def _build_custom_summary(cell: Dict[str, Any]) -> str:
    status = cell.get("status", "unknown").title()
    when = _human_ts(cell.get("completedAt") or cell.get("startedAt"))
    snippet = _truncate(cell.get("textOutput"))
    if not snippet:
        error = _truncate(cell.get("error"))
        snippet = f"Error: {error}" if error else "No textual output captured."
    plot_count = cell.get("plotCount") or cell.get("plotsPreview")
    plot_suffix = f" • Plots: {plot_count}" if plot_count else ""
    return f"Custom cell {cell.get('cellId', '?')} — {status} ({when}){plot_suffix}\n{snippet}"


def format_eda_context_summary(eda_context: Dict[str, Any]) -> str:
    """Create a rich textual summary from the EDA notebook context."""
    sections: List[str] = []

    dataset = eda_context.get("dataset") or {}
    if dataset:
        header = ["🗂️ DATASET"]
        name = dataset.get("name") or "Unknown dataset"
        header.append(f"Name: {name}")
        if dataset.get("sourceId"):
            header.append(f"Source ID: {dataset['sourceId']}")
        info = dataset.get("info") or {}
        if isinstance(info, dict) and info:
            details = []
            for key, value in list(info.items())[:6]:
                details.append(f"- {key}: {value}")
            if details:
                header.append("Metadata:")
                header.extend(details)
        sections.append("\n".join(header))

    preprocessing = eda_context.get("preprocessing") or {}
    if preprocessing:
        prep_lines = ["🧹 PREPROCESSING"]
        applied = preprocessing.get("applied")
        if applied is not None:
            prep_lines.append(f"Applied: {bool(applied)}")
        pending = preprocessing.get("pendingChanges")
        if pending is not None:
            prep_lines.append(f"Pending changes: {bool(pending)}")
        summary = _truncate(preprocessing.get("summary"), 400)
        if summary:
            prep_lines.append(f"Summary: {summary}")
        last_report = preprocessing.get("lastReport") or {}
        dropped = last_report.get("dropped_columns") or []
        if dropped:
            cols = ", ".join(dropped[:8])
            if len(dropped) > 8:
                cols += f" (+{len(dropped) - 8} more)"
            prep_lines.append(f"Dropped columns: {cols}")
        steps = last_report.get("applied_steps") or []
        if steps:
            prep_lines.append("Steps: " + "; ".join(map(str, steps[:6])))
        sections.append("\n".join(prep_lines))

    selected = eda_context.get("selectedColumns")
    if isinstance(selected, list) and selected:
        snippet = ", ".join(selected[:12])
        if len(selected) > 12:
            snippet += f" (+{len(selected) - 12} more)"
        sections.append("🔍 Selected columns: " + snippet)

    active = eda_context.get("activeContext") or {}
    if active:
        sections.append(
            "🎯 Active focus: "
            f"{active.get('scope', 'analysis')} cell {active.get('cellId')} "
            f"(set { _human_ts(active.get('timestamp')) })"
        )

    analysis_cells = eda_context.get("analysisCells") or []
    if analysis_cells:
        sorted_cells = sorted(
            analysis_cells,
            key=lambda c: c.get("completedAt") or c.get("startedAt") or "",
            reverse=True,
        )
        highlights = sorted_cells[:4]
        lines = ["📈 RECENT ANALYSES"]
        for cell in highlights:
            lines.append(_build_analysis_summary(cell))
        sections.append("\n".join(lines))

    custom_cells = eda_context.get("customAnalyses") or []
    if custom_cells:
        sorted_custom = sorted(
            custom_cells,
            key=lambda c: c.get("completedAt") or c.get("startedAt") or "",
            reverse=True,
        )
        highlights = sorted_custom[:3]
        lines = ["💡 CUSTOM CODE RUNS"]
        for cell in highlights:
            lines.append(_build_custom_summary(cell))
        sections.append("\n".join(lines))

    summary = eda_context.get("summary")
    if summary:
        sections.append("🧾 NOTEBOOK SUMMARY\n" + summary)

    final_text = "\n\n".join(section.strip() for section in sections if section)
    return final_text.strip()
