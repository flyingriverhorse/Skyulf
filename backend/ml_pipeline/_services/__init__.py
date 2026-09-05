# Internal services sub-package for ml_pipeline.
"""Business logic for the ML pipeline, kept out of the router and task layers.

``job_service`` loads ``TrainingJob`` rows, ``pipeline_execution_service`` runs
a pipeline job outside a Celery worker context, ``evaluation_service`` fetches
the y_true/y_pred pairs behind the metrics views, ``threshold_tuning_service``
owns decision-threshold tuning, ``pipeline_versions_service`` is the
version-history CRUD, and ``prediction_utils`` holds the shared prediction
decode helpers. Routers, Celery tasks and tests all import these by path —
nothing is re-exported here.
"""
