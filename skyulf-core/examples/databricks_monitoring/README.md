# Shared monitoring moved into the existing project template

The canonical [dashboard and operator guide](../../templates/databricks/template/%7B%7B.project_name%7D%7D/src/monitoring/README.md)
now live in `templates/databricks/template/{{.project_name}}/src/monitoring/`.
There is no separate monitoring template or Bundle to deploy from this directory.

All single, competition and model-set projects reuse one shared dashboard and
monitoring catalog/schema. Missing owned store objects are created on first use;
existing objects and history are reused. Existing deployments should preserve
their dashboard ID and state, and configure project refresh with its published
URL. The new owner option is only for creating the first shared dashboard.

Use **Overview → Issues to investigate** for recorded health, quality, drift and
active-policy problems. Native table pagination retains the complete list and the counter shows the
full filtered total. Follow **Evidence run** to monitoring evidence, or open
**Training and lineage** with the same model/version/context retained by the
shared filter panel. Every matching identity remains attributed in all lineage
tables; an incompatible selection leaves empty tables. The three shared controls
are Model (fully qualified name), Monitored version and Context. Overview and
lineage, Drift and Model performance show every matching model/version/context.
The latest observation and policy result are selected separately for each identity;
disabled policies and identities without measurements remain visible. Online
serving shows matching endpoint contexts, with monitored and served versions kept
separate. Execution retains its
own Workspace/Job inputs; copy IDs from its Recent accessible jobs table.
Model filters do not attribute job costs. Use All models / All versions /
All contexts to restore the portfolio. Pin Model/Version/Context to narrow that
scope while new observations arrive. Drift and Performance charts keep models
and observation windows separate, without combining their scores or counts.

Page controls are limited to history dates, features, metrics, targets and jobs.
Drift and Performance time charts and date filters use the UTC data-window end;
calculation time is identified separately. The latest policy table lists each
identity's metric, baseline, current score, allowed limit, direction and result.
Accuracy, F1 and RMSE policies keep their own time-chart panels and scales.
**Chart metric** changes only the latest observed-metric table and its time chart.
Missing scores remain unavailable.
The prediction-coverage table replaces three shared counters with a row per
identity, and the confusion-matrix pivot separates identities before class counts.
Serving request, error and latency charts appear before the detailed tables.
Serving observations can overlap and must not be summed as unique endpoint traffic.
Inspect the owning training run and recorded code evidence. Alias metadata is
labeled with its capture time; it is a snapshot and never a historical identity.

New observations capture a bounded metadata snapshot in the existing report.
Dashboard refresh rereads it; no live registry lookup occurs per table row.
Historical missing metadata and optional permission failures remain explicit,
and disabled policies create no new issue. The linked operator guide explains
refresh, missing evidence and store upgrades.
