"""Check project configuration, syntax and saved package inputs without running hooks.

With --probe SAMPLE, also fit the selected recipe on a local labeled sample and
run the Spark/endpoint admission lint and probe (this runs your project code).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from skyulf.integrations.databricks.lifecycle.workflow import resolve_target_config
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.projects.project_checks import check_project
from skyulf.integrations.databricks.projects.yaml_config import (
    project_config_path,
    read_workflow_config,
)
from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
    local_admission_probe,
)

BINDINGS = ("catalog", "input_schema", "output_schema", "metadata_schema", "resource_suffix")


def _probe(root, args):
    """Run training's admission lint and probe on a CSV or Parquet sample."""
    config = read_workflow_config(project_config_path(root))
    if config.get("training_layout", "single_model") != "single_model":
        return "--probe supports single-model projects."
    sample = Path(args.probe)
    frame = pd.read_parquet(sample) if sample.suffix == ".parquet" else pd.read_csv(sample)
    loaded = load_project_workflow(config, root / "src/features")
    bindings = {key: getattr(args, key) for key in BINDINGS}
    return local_admission_probe(resolve_target_config(loaded, bindings), frame)


def main():
    """Run credential-free static checks and print their explicitly bounded scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--input-schema", default="air_quality_no2")
    parser.add_argument("--output-schema", default="air_quality_no2")
    parser.add_argument("--metadata-schema", default="air_quality_no2")
    parser.add_argument("--resource-suffix", default="")
    parser.add_argument(
        "--probe",
        metavar="SAMPLE",
        help="CSV or Parquet with input and target columns (16+ rows, e.g. a LIMIT 256 "
        "export of the training table): fit locally and run the admission probe.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        result = check_project(root, {k: v for k, v in vars(args).items() if k != "probe"})
        if args.probe:
            result["admission_probe"] = _probe(root, args)
    except (OSError, ValueError, SyntaxError) as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "project_hooks_executed": bool(args.probe),
            "remote_operations": False,
        }
        if isinstance(exc, SyntaxError):
            failure.update(file=exc.filename, line=exc.lineno, column=exc.offset)
        print(json.dumps(failure, indent=2))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
