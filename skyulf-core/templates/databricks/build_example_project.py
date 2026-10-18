"""Regenerate the prepared example projects from the real Bundle template.

Each example is ordinary CLI output for its answer file in examples/,
without the template's sample code it does not use, plus the files its author
wrote (recipes, scoring rules, custom steps) and a few text edits to the
generated training.yml that keep its comments. Everything else must match
ordinary CLI output bytewise.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples/nyctaxi"
# Online serving renders empty lookup declarations; the online example fills them
# with the zone features that load_nyctaxi.sql creates.
LOOKUP_FILL = (
    '      - table_name: ""\n'
    "        lookup_key: []\n"
    "        feature_names: []\n"
    "  online_features:\n"
    "    required_features: []\n"
    "    freshness:\n"
    '      - column: ""\n'
    "        max_age_seconds: null\n",
    '      - table_name: "{catalog}.{input_schema}.zone_features"\n'
    "        lookup_key: [pickup_zip]\n"
    "        feature_names: [zone_avg_distance, zone_trip_count, zone_updated_at]\n"
    "  online_features:\n"
    "    required_features: [zone_avg_distance, zone_trip_count, zone_updated_at]\n"
    "    freshness:\n"
    "      - column: zone_updated_at\n"
    "        max_age_seconds: 604800.0 # Refresh zone_features at least weekly.\n",
)
# Template samples that only teach; the examples keep the code they run.
SAMPLES = ("src/features/groups",)
OWN_CODE = (
    "config/pre_split.yml",
    "config/preprocessing.yml",
    "src/features/pre_split.py",
    "src/features/preprocessing.py",
    "src/features/scoring.py",
    "src/features/trip_rules.py",
)
PROJECTS = {
    "taxi_fare": {
        "sources": ("config/pre_split.yml", "config/preprocessing.yml", "src/features/scoring.py"),
        # Built-in transformers only: no custom filter or step modules.
        "removed": (*SAMPLES, "src/features/pre_split.py", "src/features/preprocessing.py"),
        "training_edits": (
            (
                "  quality_threshold: 6.0\n",
                "  quality_threshold: 6.0\n  preprocessing_recipe: spark_safe\n",
            ),
        ),
    },
    "taxi_fare_online": {
        "sources": (
            "config/pre_split.yml",
            "config/preprocessing.yml",
            "src/features/pre_split.py",
            "src/features/scoring.py",
        ),
        # Built-in preprocessing only; pre_split.py keeps the value_range filter.
        "removed": (*SAMPLES, "src/features/preprocessing.py"),
        "training_edits": (LOOKUP_FILL,),
    },
    "taxi_fare_robust": {
        "readme_note": (
            "\n## Report WAPE performance changes\n\n"
            "`deployment/targets.yml` enables a report-only WAPE policy for this model.\n"
            "It compares completed daily windows with version 1's saved training holdout,\n"
            "using a 20% relative tolerance and two consecutive windows. It never requests\n"
            "training. Before deployment, replace the baseline version with the concrete\n"
            "version you will monitor; older Spark reference caches without WAPE remain unavailable.\n"
            "See [SERVE_EXAMPLE.md](SERVE_EXAMPLE.md#report-performance-loss-without-retraining)\n"
            "for label coverage, timing, baseline changes and disabling the policy.\n"
        ),
        "sources": (
            "config/pre_split.yml",
            "config/preprocessing.yml",
            "config/serving.yml",
            "deployment/targets.yml",
            "src/features/pre_split.py",
            "src/features/scoring.py",
            "src/serve.py",
            "SERVE_EXAMPLE.md",
        ),
        "removed": (*SAMPLES, "src/features/preprocessing.py"),
        "training_edits": (),
    },
    "taxi_high_fare": {"sources": OWN_CODE, "removed": SAMPLES, "training_edits": ()},
    "taxi_trips": {"sources": OWN_CODE, "removed": SAMPLES, "training_edits": ()},
    "diagnostic_single": {
        "group": "real_data",
        "sources": (
            "config/pre_split.yml",
            "config/preprocessing.yml",
            "src/features/pre_split.py",
            "src/features/preprocessing.py",
            "src/features/scoring.py",
            "src/features/numeric.py",
        ),
        "removed": SAMPLES,
        "training_edits": (
            (
                "  quality_threshold: 0.85\n",
                "  quality_threshold: 0.85\n  preprocessing_recipe: measured_shape\n  pre_split_recipe: valid_measurements\n",
            ),
        ),
    },
    "bike_demand_set": {
        "group": "real_data",
        "sources": (
            "config/pre_split.yml",
            "config/preprocessing.yml",
            "src/features/pre_split.py",
            "src/features/preprocessing.py",
            "src/features/scoring.py",
            "src/features/numeric.py",
        ),
        "removed": SAMPLES,
        "training_edits": (),
    },
    "air_quality_no2": {
        "group": "real_data",
        "sources": (
            "config/pre_split.yml",
            "config/preprocessing.yml",
            "deployment/targets.yml",
            "src/features/pre_split.py",
            "src/features/preprocessing.py",
            "src/features/scoring.py",
            "src/features/sensors.py",
        ),
        "removed": SAMPLES,
        "training_edits": (),
    },
}


def example_root(name):
    """Locate the source group without changing the generated project layout."""
    return ROOT / "examples" / PROJECTS[name].get("group", "nyctaxi")


def project_files(project):
    """Read project payloads, excluding local caches produced by operator commands."""
    return {
        path.relative_to(project).as_posix(): path.read_bytes()
        for path in sorted(project.rglob("*"))
        if path.is_file()
        and not {"__pycache__", ".databricks"}.intersection(path.parts)
        and path.suffix != ".pyc"
    }


@contextmanager
def offline_cli_environment():
    """Answer CLI host discovery without network retries or workspace credentials.

    Only the metadata document is available; workspace lookups return HTTP 404.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            """Keep local host discovery out of generated example diagnostics."""

        def do_GET(self):
            """Expose only the empty metadata document needed by offline bundle init."""
            self.send_response(200 if self.path == "/.well-known/databricks-config" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("DATABRICKS_")
    }
    environment.update(
        DATABRICKS_HOST=f"http://127.0.0.1:{server.server_port}",
        DATABRICKS_TOKEN="local-render-only",
        DATABRICKS_AUTH_TYPE="pat",
        DATABRICKS_CONFIG_FILE=os.devnull,
        DATABRICKS_CLI_DISABLE_TELEMETRY="1",
    )
    try:
        yield environment
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def generate_example(directory, name):
    """Render offline with the example answers, then apply only the documented example edits."""
    spec = PROJECTS[name]
    with offline_cli_environment() as environment:
        result = subprocess.run(
            [
                "databricks",
                "bundle",
                "init",
                str(ROOT),
                "--config-file",
                str(example_root(name) / f"{name}.json"),
                "--output-dir",
                str(directory),
            ],
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    project = directory / name
    for removed in spec["removed"]:
        target = project / removed
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    for source in spec["sources"]:
        destination = project / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example_root(name) / name / source, destination)
    training_path = project / "config/training.yml"
    text = training_path.read_text(encoding="utf-8")
    # Edit text, not parsed YAML, so the generated explanatory comments survive.
    for old, new in spec["training_edits"]:
        if text.count(old) != 1:
            raise RuntimeError(f"{name}: generated training.yml no longer contains: {old!r}")
        text = text.replace(old, new)
    training_path.write_text(text, encoding="utf-8", newline="\n")
    if note := spec.get("readme_note"):
        readme = project / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8") + note, encoding="utf-8", newline="\n")
    return project


def changed_files(name, generated):
    """List paths whose stored bytes differ from the regenerated example."""
    stored = project_files(example_root(name) / name)
    return sorted(
        path for path in generated.keys() | stored.keys() if generated.get(path) != stored.get(path)
    )


def main():
    """Check all generated bytes, or refresh the stored snapshots while preserving their sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for name in PROJECTS:
        with tempfile.TemporaryDirectory(prefix="skyulf-example-") as directory:
            generated = project_files(generate_example(Path(directory), name))
        changed = changed_files(name, generated)
        if args.check:
            stale += [f"{name}/{path}" for path in changed]
            continue
        obsolete = project_files(example_root(name) / name).keys() - generated.keys()
        if obsolete:
            parser.error(f"Review obsolete {name} files first: " + ", ".join(sorted(obsolete)))
        for path, content in generated.items():
            destination = example_root(name) / name / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        print(f"Updated {name} ({len(generated)} files, {len(changed)} changed).")
    if args.check:
        if stale:
            parser.exit(1, "Examples differ: " + ", ".join(stale) + "\n")
        print(f"Prepared examples are current ({len(PROJECTS)} projects).")


if __name__ == "__main__":
    main()
