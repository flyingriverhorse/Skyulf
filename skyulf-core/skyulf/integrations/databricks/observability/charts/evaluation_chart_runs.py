"""Isolate native image grids from the training run's parameter header."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def chart_run(client: Any, source_run_id: str) -> Iterator[str]:
    """Publish images in a parameter-free child without reopening the training run.

    Explicit client calls avoid inheriting an active fluent run or its parameters.
    Failed chart attempts remain inspectable as failed children; the source run's
    latest successful chart link changes only after publication completes.
    """
    source = client.get_run(source_run_id)
    child = client.create_run(
        source.info.experiment_id,
        run_name="Charts",
        tags={
            "mlflow.parentRunId": source_run_id,
            "skyulf.run_kind": "evaluation_charts",
        },
    )
    destination = child.info.run_id
    try:
        yield destination
    except Exception:  # noqa: BLE001 - finalize only the failed reporting child
        client.set_terminated(destination, status="FAILED")
        raise
    else:
        client.set_terminated(destination, status="FINISHED")
        client.set_tag(source_run_id, "skyulf.charts.run_id", destination)
