"""Exploratory data analysis: the ``/api/eda`` routes and the job behind them.

``router`` serves the endpoints — trigger, cancel, list, history, latest, and the
on-demand decomposition splits. ``tasks`` holds the actual work: it runs
``skyulf``'s ``EDAAnalyzer`` off the request path, through Celery when
``USE_CELERY`` is set and FastAPI ``BackgroundTasks`` otherwise, then persists the
profile onto the ``EDAReport`` row the router created.

There are no re-exports here; import from the submodules directly.
"""
