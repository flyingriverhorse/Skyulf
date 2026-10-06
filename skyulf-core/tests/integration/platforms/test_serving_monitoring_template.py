"""Generated monitoring jobs expose an explicit serving enrollment action."""

from pathlib import Path


def test_monitoring_template_passes_json_enrollments_to_spark_job():
    """An operator run must deliver its configured batch to the independent monitor task."""
    root = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
    template = (root / "resources/monitoring.job.yml.tmpl").read_text(encoding="utf-8")
    assert "- name: monitoring_serving_enrollments\n          default: '[]'" in template
    assert (
        "monitoring_serving_enrollments: '{{\"{{job.parameters.monitoring_serving_enrollments}}\"}}'"
        in template
    )
    tasks = [line for line in template.splitlines() if line.startswith("        - task_key:")]
    assert tasks == [
        "        - task_key: monitor_model",
        "        - task_key: monitoring_report",
        "        - task_key: evaluate_retraining",
    ]
