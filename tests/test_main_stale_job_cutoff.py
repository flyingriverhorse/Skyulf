"""Regression test for the config-driven orphan-job staleness cutoff.

`backend.main._reset_stale_jobs` used to hardcode a 2-hour staleness window;
it now reads `Settings.JOB_ORPHAN_STALE_HOURS`. This test verifies the
setting actually drives the cutoff used when marking orphaned jobs failed.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine

from backend.database.models import Base, TrainingJob


def _make_engine(db_path):
    """Create a fresh sqlite engine with the ML job tables provisioned."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(
        engine,
        tables=[TrainingJob.__table__],
    )
    return engine


def _insert_running_job(engine, job_id: str, started_at: datetime) -> None:
    """Insert a minimal 'running' TrainingJob row (run_mode='fixed') started at the given time."""
    with engine.begin() as conn:
        conn.execute(
            TrainingJob.__table__.insert().values(
                id=job_id,
                pipeline_id="p1",
                node_id="n1",
                dataset_source_id="d1",
                status="running",
                model_type="classifier",
                run_mode="fixed",
                graph={},
                started_at=started_at,
            )
        )


def test_reset_stale_jobs_uses_configured_orphan_stale_hours(tmp_path, monkeypatch):
    """Changing JOB_ORPHAN_STALE_HOURS changes which jobs are treated as orphaned."""
    import backend.main as main_mod

    db_path = tmp_path / "stale_jobs.db"
    engine = _make_engine(db_path)

    # Job started 90 minutes ago: stale under a 1-hour cutoff, not stale under a 3-hour one.
    started_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=90)
    _insert_running_job(engine, "job-90m", started_at)

    class _FakeSettings:
        JOB_ORPHAN_STALE_HOURS = 3

    monkeypatch.setattr(main_mod.settings, "DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr(main_mod, "get_settings", lambda: _FakeSettings())

    main_mod._reset_stale_jobs()

    with engine.begin() as conn:
        row = conn.execute(
            TrainingJob.__table__.select().where(TrainingJob.__table__.c.id == "job-90m")
        ).one()
    # 90 minutes < 3-hour cutoff => job should NOT have been marked failed.
    assert row.status == "running"

    class _FakeSettingsShort:
        JOB_ORPHAN_STALE_HOURS = 1

    monkeypatch.setattr(main_mod, "get_settings", lambda: _FakeSettingsShort())

    main_mod._reset_stale_jobs()

    with engine.begin() as conn:
        row = conn.execute(
            TrainingJob.__table__.select().where(TrainingJob.__table__.c.id == "job-90m")
        ).one()
    # 90 minutes > 1-hour cutoff => job should now be marked failed.
    assert row.status == "failed"
