# Data Drift — Improvement Roadmap

Ideas for future iterations of the Data Drift Analysis page, ordered by estimated impact.

---

## High Impact

### 4. Auto-Scheduled Drift Checks
Periodic Celery task that automatically runs drift calculation against the latest production data upload.

- New "Schedule" option per job (daily / weekly / manual).
- Celery beat task picks up scheduled jobs, runs `DriftCalculator`, stores results in `DriftHistory`.
- Pairs with Drift History for continuous monitoring.

### 5. Multi-File / Batch Comparison
Upload multiple production CSV files and compare drift trends across data batches in one view.

- Batch upload UI (drag-and-drop multiple files).
- Overlay distribution charts from different batches.
- Side-by-side or stacked timeline of drift scores per batch.

---

## Medium Impact

*All medium impact items completed in v0.2.0:*

### ~~6. Dropdown Metric Overflow~~ ✅
Truncate or show only the primary metric in dropdown items and display the full metric set in the metadata bar only.

### ~~7. Empty State Guidance~~ ✅
Show a helpful placeholder ("Select a model and upload data to compare") when no job is selected and no report exists.

### ~~8. Export Drift Report~~ ✅
Download the full drift analysis as CSV. Export button in the filter bar serializes all columns, metrics, and risk levels.

### ~~9. Custom Drift Thresholds~~ ✅
Collapsible threshold settings panel (PSI, KS p-value, Wasserstein, KL Divergence) with numeric inputs and reset-to-defaults. Thresholds are passed to the backend `DriftCalculator`.

### ~~10. Alert Badges on Sidebar~~ ✅
Red dot badge on the "Data Drift" nav item when the most recent drift check found significant drift. Lightweight `GET /monitoring/drift/status` endpoint.

---

## Nice-to-Have

### 11. Categorical Drift Detection
Expand drift metrics to handle categorical features (chi-square test, Jensen-Shannon divergence, category frequency comparison).

### 12. Drift Explanation / Root Cause Hints
Use SHAP or simple heuristics to suggest *why* a column drifted. Planned approach: lightweight stats comparison (mean, std, min, max, skewness) for reference vs current in the expanded row detail — no SHAP dependency.

### 13. Inline Data Preview
Show a mini data preview (first 5 rows) of both reference and current data side-by-side when a column is expanded.

### ~~14. Per-Column Historical Sparklines~~ ✅
Tiny inline SVG sparklines in each table row showing that column's PSI trend over past drift checks. Color-coded (green/amber/red) based on latest value.

### ~~15. Threshold Customization Sliders~~ ✅
Client-side threshold re-evaluation via `useMemo` — changing thresholds instantly re-classifies drift status (drifted/stable) without re-uploading data. Summary cards, filter counts, and export all reflect the updated thresholds.
