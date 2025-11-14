"""Command-line helper to purge background training and tuning jobs from the database."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Type

from core.database.engine import close_db, init_db
from core.database.models import get_database_session
from core.feature_engineering.modeling.training.jobs import purge_training_jobs
from core.feature_engineering.modeling.hyperparameter_tuning.jobs import purge_tuning_jobs
from core.feature_engineering.schemas import TrainingJobStatus
from core.feature_engineering.schemas import HyperparameterTuningJobStatus

DEFAULT_STATUS_FILTER: List[str] = [
    TrainingJobStatus.SUCCEEDED.value,
    TrainingJobStatus.FAILED.value,
    TrainingJobStatus.CANCELLED.value,
]

JOB_TYPE_CHOICES = ("training", "tuning", "both")


def _parse_statuses(
    raw_statuses: Optional[Iterable[str]],
    include_all: bool,
    *,
    status_enum: Type[Enum],
    default_statuses: Sequence[str],
) -> Optional[List[str]]:
    if include_all:
        return None

    if not raw_statuses:
        return list(default_statuses)

    parsed: List[str] = []
    for value in raw_statuses:
        try:
            parsed.append(status_enum(value.lower()).value)  # type: ignore[arg-type]
        except ValueError as exc:  # pragma: no cover - CLI validation safeguard
            raise SystemExit(f"Unsupported status: {value}") from exc
    return parsed


def _parse_timestamp(older_than: Optional[str], older_than_days: Optional[int]) -> Optional[datetime]:
    if older_than:
        try:
            return datetime.fromisoformat(older_than)
        except ValueError as exc:  # pragma: no cover - CLI validation safeguard
            raise SystemExit(
                "--older-than must be an ISO-8601 timestamp, e.g. 2024-01-31T12:00:00"
            ) from exc

    if older_than_days is not None:
        if older_than_days < 0:
            raise SystemExit("--older-than-days must be zero or a positive integer")
        return datetime.utcnow() - timedelta(days=older_than_days)

    return None


def _format_job_count(count: int, label: str) -> str:
    noun = "job" if count == 1 else "jobs"
    return f"{count} {label} {noun}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Purge background model training or hyperparameter tuning jobs from the database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--status",
        dest="statuses",
        action="append",
        help=(
            "Limit deletions to a specific job status. May be supplied multiple times. "
            "Defaults to succeeded, failed, and cancelled."
        ),
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Ignore status filtering and target every lifecycle state.",
    )
    parser.add_argument(
        "--older-than",
        metavar="TIMESTAMP",
        help="ISO-8601 timestamp; only jobs created at or before this time are removed.",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        metavar="DAYS",
        help="Alternative to --older-than; subtract the given number of days from now.",
    )
    parser.add_argument(
        "--dataset-source-id",
        help="Optionally restrict deletions to jobs for a specific dataset source.",
    )
    parser.add_argument(
        "--pipeline-id",
        help="Optionally restrict deletions to jobs for a specific pipeline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of jobs to delete in this invocation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete anything; report how many rows would be removed.",
    )
    parser.add_argument(
        "--job-type",
        choices=JOB_TYPE_CHOICES,
        default="training",
        help="Choose which job table(s) to target.",
    )
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    target_training = args.job_type in {"training", "both"}
    target_tuning = args.job_type in {"tuning", "both"}

    training_statuses = (
        _parse_statuses(
            args.statuses,
            args.all_statuses,
            status_enum=TrainingJobStatus,
            default_statuses=DEFAULT_STATUS_FILTER,
        )
        if target_training
        else None
    )
    tuning_statuses = (
        _parse_statuses(
            args.statuses,
            args.all_statuses,
            status_enum=HyperparameterTuningJobStatus,
            default_statuses=DEFAULT_STATUS_FILTER,
        )
        if target_tuning
        else None
    )
    older_than = _parse_timestamp(args.older_than, args.older_than_days)

    def _has_status_filter(statuses: Optional[List[str]]) -> bool:
        return statuses is None or (isinstance(statuses, list) and len(statuses) > 0)

    has_status_filters = False
    if target_training:
        has_status_filters = has_status_filters or _has_status_filter(training_statuses)
    if target_tuning:
        has_status_filters = has_status_filters or _has_status_filter(tuning_statuses)

    if not any([
        has_status_filters,
        older_than,
        args.dataset_source_id,
        args.pipeline_id,
        args.limit,
        args.dry_run,
    ]):
        raise SystemExit(
            "Refusing to purge without any filters; specify --status/--all-statuses, "
            "--older-than, --older-than-days, --dataset-source-id, --pipeline-id, --limit, or use --dry-run."
        )

    await init_db()
    try:
        async with get_database_session(expire_on_commit=False) as session:
            results: List[tuple[str, int]] = []

            if target_training:
                deleted_training = await purge_training_jobs(
                    session,
                    statuses=training_statuses,
                    older_than=older_than,
                    dataset_source_id=args.dataset_source_id,
                    pipeline_id=args.pipeline_id,
                    limit=args.limit,
                    dry_run=args.dry_run,
                )
                results.append(("training", deleted_training))

            if target_tuning:
                deleted_tuning = await purge_tuning_jobs(
                    session,
                    statuses=tuning_statuses,
                    older_than=older_than,
                    dataset_source_id=args.dataset_source_id,
                    pipeline_id=args.pipeline_id,
                    limit=args.limit,
                    dry_run=args.dry_run,
                )
                results.append(("tuning", deleted_tuning))
    finally:
        await close_db()

    if not results:
        print("No job types selected; nothing to do.")
        return 0

    summary_parts = [_format_job_count(count, label) for label, count in results]
    if len(summary_parts) == 1:
        summary_text = summary_parts[0]
    else:
        summary_text = " and ".join(summary_parts)

    if args.dry_run:
        print(f"Dry run complete - {summary_text} would be deleted.")
    else:
        print(f"Deleted {summary_text}.")

    return 0


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(_run_cli(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

