# Shared monitoring moved into the existing project template

The canonical [dashboard and operator guide](../../templates/databricks/template/%7B%7B.project_name%7D%7D/src/monitoring/README.md)
now live in `templates/databricks/template/{{.project_name}}/src/monitoring/`.
There is no separate monitoring template or Bundle to deploy from this directory.

All single, competition and model-set projects reuse one shared dashboard and
monitoring catalog/schema. Missing owned store objects are created on first use;
existing objects and history are reused. Existing deployments should preserve
their dashboard ID and state, and configure project refresh with its published
URL. The new owner option is only for creating the first shared dashboard.
