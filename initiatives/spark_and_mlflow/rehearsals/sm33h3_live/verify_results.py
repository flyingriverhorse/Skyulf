"""Accept only successful remote tasks and matching bounded H3 results."""

import json
import math
from pathlib import Path

folder = Path(__file__).resolve().parent


def read(name):
    """Read CLI evidence written by Windows PowerShell with its UTF-8 BOM."""
    return json.loads((folder / name).read_text(encoding="utf-8-sig"))


def output(task):
    """Require a real notebook result rather than accepting task status alone."""
    record = read(f"{task}-output.json")
    assert not record.get("error"), record.get("error")
    assert not record["notebook_output"].get("truncated", False)
    return json.loads(record["notebook_output"]["result"])


status = read("run-status.json")
assert status["run_id"] == 848857785722024
assert status["state"]["life_cycle_state"] == "TERMINATED", status["state"]
assert status["state"]["result_state"] == "SUCCESS", status["state"]
assert len(status["tasks"]) == 3
for task in status["tasks"]:
    assert task["state"]["result_state"] == "SUCCESS", task

score = output("score")
summary = {"run_id": status["run_id"], "duration_seconds": status["run_duration"] / 1000}
for engine in ("pandas", "polars"):
    training = output(f"{engine}_train")
    evidence = training["evidence"]
    assert evidence["source_rows"] == evidence["pre_filter_rows"] == 240
    assert evidence["survivor_rows"] == 47
    assert evidence["training_rows"] == 35 and evidence["holdout_rows"] == 12
    assert len(training["metrics"]) == 30
    assert all(math.isfinite(value) for value in training["metrics"].values())
    assert score["approvals"][engine]["new_version"] == "1"
    initial, appended, noop = (score[phase][engine] for phase in ("initial", "appended", "noops"))
    assert initial["input_count"] == initial["output_count"] == 240
    assert appended["input_count"] == appended["output_count"] == 3
    assert noop["noop"] and noop["input_count"] == noop["output_count"] == 0
    assert appended["commit_version"] == noop["commit_version"]
    summary[engine] = {
        "mlflow_run_id": training["run_id"],
        "metrics": training["metrics"],
        "training_rows": 35,
        "holdout_rows": 12,
        "initial_predictions": 240,
        "appended_predictions": 3,
        "champion_version": "1",
        "noop_commit_version": noop["commit_version"],
    }
assert summary["pandas"]["metrics"] == summary["polars"]["metrics"]
(folder / "acceptance-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Accepted run {summary['run_id']} in {summary['duration_seconds']} seconds.")
print(
    "Both engines: 35 train + 12 holdout, 30 equal metrics, champion v1, 240 + 3 predictions, unchanged no-op commit."
)
