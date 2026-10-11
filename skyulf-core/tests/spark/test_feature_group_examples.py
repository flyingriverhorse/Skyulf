"""Execute the shipped group functions and their documented merge on real Spark."""

import runpy
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from skyulf.integrations.databricks.features.config import parse_feature_config
from skyulf.integrations.databricks.features.joins import join_feature_groups

PROJECT = Path(__file__).resolve().parents[2] / "templates/databricks/template/{{.project_name}}"


def test_shipped_groups_preserve_observations_and_do_not_join_future_activity(spark):
    """The copyable config must produce real totals without duplicating or leaking labels."""
    text = (PROJECT / "config/features.yml").read_text(encoding="utf-8")
    _, marker, example = text.partition("# version: 1\n")
    assert marker
    config = yaml.safe_load("version: 1\n" + "\n".join(line[2:] for line in example.splitlines()))
    plan = parse_feature_config(config)
    assert plan is not None
    base = spark.sql("""
        SELECT * FROM VALUES
          (1, DATE'2026-02-01', 0), (1, DATE'2026-03-01', 1), (2, DATE'2026-03-01', 0)
        AS rows(company_id, observed_at, churn)
    """)
    sources = {
        "company": spark.sql("""
            SELECT * FROM VALUES
              (1, DATE'2026-02-01', 10), (1, DATE'2026-03-01', 12),
              (2, DATE'2026-03-01', CAST(NULL AS INT))
            AS rows(company_id, observed_at, employee_count)
        """),
        "activity": spark.sql("""
            SELECT * FROM VALUES
              (1, DATE'2026-01-31', 10D), (1, DATE'2026-01-31', 20D),
              (1, DATE'2026-02-28', 40D), (1, DATE'2026-03-31', 9999D),
              (2, DATE'2026-02-28', CAST(NULL AS DOUBLE))
            AS rows(company_id, observed_at, amount)
        """),
    }
    frames = {}
    for group in plan.groups:
        relative, function = group.transform.split(":")
        transform = runpy.run_path(str(PROJECT / relative))[function]
        frame = transform(sources[group.name])
        assert set(frame.columns) == {*plan.record_keys, *group.columns}
        frames[group.name] = frame
    result = join_feature_groups(base, frames, plan)
    values = result.orderBy("company_id", "observed_at").select(
        "company_id", "churn", "employee_count", "monthly_amount", "transaction_count"
    )
    assert [tuple(row) for row in values.collect()] == [
        (1, 0, 10, 30.0, 2),
        (1, 1, 12, 40.0, 1),
        (2, 0, None, None, 1),
    ]
