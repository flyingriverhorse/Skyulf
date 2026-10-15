"""Regenerate the prepared NYC taxi example projects from the real Bundle template.

Each example is ordinary CLI output for its answer file in examples/nyctaxi/,
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples/nyctaxi"
PROBE = (
    "  preprocessing_probe: false",
    "  preprocessing_probe: true",
)
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
            PROBE,
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
        "training_edits": (LOOKUP_FILL, PROBE),
    },
    "taxi_high_fare": {"sources": OWN_CODE, "removed": SAMPLES, "training_edits": (PROBE,)},
    "taxi_trips": {"sources": OWN_CODE, "removed": SAMPLES, "training_edits": (PROBE,)},
}


def project_files(project):
    """Read project payloads, excluding local caches produced by operator commands."""
    return {
        path.relative_to(project).as_posix(): path.read_bytes()
        for path in sorted(project.rglob("*"))
        if path.is_file()
        and not {"__pycache__", ".databricks"}.intersection(path.parts)
        and path.suffix != ".pyc"
    }


def generate_example(directory, name):
    """Render offline with the example answers, then apply only the documented example edits."""
    spec = PROJECTS[name]
    environment = os.environ | {
        "DATABRICKS_HOST": "https://127.0.0.1",
        "DATABRICKS_TOKEN": "local-render-only",
        "DATABRICKS_AUTH_TYPE": "pat",
    }
    result = subprocess.run(
        [
            "databricks",
            "bundle",
            "init",
            str(ROOT),
            "--config-file",
            str(EXAMPLES / f"{name}.json"),
            "--output-dir",
            str(directory),
        ],
        env=environment,
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
        shutil.copyfile(EXAMPLES / name / source, destination)
    training_path = project / "config/training.yml"
    text = training_path.read_text(encoding="utf-8")
    # Edit text, not parsed YAML, so the generated explanatory comments survive.
    for old, new in spec["training_edits"]:
        if text.count(old) != 1:
            raise RuntimeError(f"{name}: generated training.yml no longer contains: {old!r}")
        text = text.replace(old, new)
    training_path.write_text(text, encoding="utf-8", newline="\n")
    return project


def changed_files(name, generated):
    """List paths whose stored bytes differ from the regenerated example."""
    stored = project_files(EXAMPLES / name)
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
        obsolete = project_files(EXAMPLES / name).keys() - generated.keys()
        if obsolete:
            parser.error(f"Review obsolete {name} files first: " + ", ".join(sorted(obsolete)))
        for path, content in generated.items():
            destination = EXAMPLES / name / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        print(f"Updated {name} ({len(generated)} files, {len(changed)} changed).")
    if args.check:
        if stale:
            parser.exit(1, "Examples differ: " + ", ".join(stale) + "\n")
        print(f"NYC taxi examples are current ({len(PROJECTS)} projects).")


if __name__ == "__main__":
    main()
