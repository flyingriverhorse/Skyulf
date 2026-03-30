# Data Drift — Improvement Roadmap

Ideas for future iterations of the Data Drift Analysis page, ordered by estimated impact.

---

## High Impact

### 1. Drift History & Timeline
Store each drift calculation result in the database and show a time-series chart of drift scores per column. Lets you answer: *"Is drift getting worse week over week?"*

- Save `DriftReport` snapshots with timestamps to a new `DriftHistory` table.
- Timeline chart (Recharts line chart) on the drift page showing PSI / Wasserstein over time.
- Filter by column, date range, and job.

### 2. Feature Importance × Drift Cross-Reference
Cross-reference drifted columns with the model's feature importances to highlight **high-risk drifts** — columns that drifted AND are top predictors.

- Read feature importances from the trained model artifact (`.coef_`, `.feature_importances_`).
- Add a "Risk" column to the drift table: `High` if drifted + top-10 feature, `Low` otherwise.
- Sort by risk by default.

### 3. Auto-Scheduled Drift Checks
Periodic Celery task that automatically runs drift calculation against the latest production data upload.

- New "Schedule" option per job (daily / weekly / manual).
- Celery beat task picks up scheduled jobs, runs `DriftCalculator`, stores results in `DriftHistory`.
- Pairs with Drift History for continuous monitoring.

### 4. Multi-File / Batch Comparison
Upload multiple production CSV files and compare drift trends across data batches in one view.

- Batch upload UI (drag-and-drop multiple files).
- Overlay distribution charts from different batches.
- Side-by-side or stacked timeline of drift scores per batch.

---

## Medium Impact

### 5. Export Drift Report
Download the full drift analysis as PDF or CSV for sharing with stakeholders.

- PDF: Use html-to-canvas or a server-side template (WeasyPrint / ReportLab).
- CSV: Column-level metrics table export.
- Add a "Download Report" button next to the "Run Analysis" button.

### 6. Custom Drift Thresholds
Let users configure per-column PSI / KS / Wasserstein thresholds instead of using global defaults.

- Threshold config stored in `job_metadata` JSON.
- Settings panel or inline editing on the drift results table.
- Re-evaluate drift status on threshold change without re-uploading data.

### 7. Alert Badges on Sidebar
Show a warning badge on the "Data Drift" navigation item if the most recent drift check found significant drift.

- Store latest drift status per job in DB (or in-memory cache).
- Badge component on sidebar nav: red dot if any high-drift columns, green if all stable.
- Clears once the user views the drift page.

---

## Nice-to-Have

### 8. Categorical Drift Detection
Expand drift metrics to handle categorical features (chi-square test, Jensen-Shannon divergence, category frequency comparison).

### 9. Drift Explanation / Root Cause Hints
Use SHAP or simple heuristics to suggest *why* a column drifted (e.g., "mean shifted from 45.2 → 52.8, likely a seasonal effect").

### 10. Inline Data Preview
Show a mini data preview (first 5 rows) of both reference and current data side-by-side when a column is expanded.
