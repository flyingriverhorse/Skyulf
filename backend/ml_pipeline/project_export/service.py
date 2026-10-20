"""Validate fitted artifacts and package independent prediction projects without deployment."""

import hashlib
import io
import json
import pickle
import re
import sys
import zipfile
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any

from sklearn.base import is_classifier, is_regressor
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.database.models import Deployment
from backend.ml_pipeline._services.job_service import JobService
from backend.ml_pipeline._services.prediction_utils import extract_target_label_encoder
from backend.ml_pipeline.deployment.service import DeploymentService
from skyulf.preprocessing._exclusion import is_exclusion_step
from skyulf.preprocessing.function_steps import COLUMN_STEP, FILTER_STEP, FITTED_STEP
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.preprocessing.pipeline import FeatureEngineer

from .core_wheel import bundled_core_wheel
from .templates import project_files

_PACKAGES = {
    "skyulf": "skyulf-core",
    "sklearn": "scikit-learn",
    "_loss": "scikit-learn",  # sklearn's compiled loss objects report this module.
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "category_encoders": "category-encoders",
    "dateutil": "python-dateutil",
    "pytz": "pytz",
}


class ProjectJobNotFoundError(LookupError):
    """Identify an absent database job separately from malformed artifact keys."""


class _PortablePickler(pickle.Pickler):
    """Reject dependencies on application/custom modules and collect retained package pins."""

    def __init__(self, stream: io.BytesIO):
        """Use standard pickle so the generated project needs no custom unpickler."""
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.packages = {
            "skyulf-core",
            "pandas",
            "numpy",
            "scikit-learn",
            "scipy",
            "polars",
            "pyarrow",
        }

    def persistent_id(self, obj: Any) -> None:
        """Inspect serialized identities without replacing their normal pickle representation."""
        owner = obj if isinstance(obj, type) or callable(obj) else type(obj)
        module = getattr(owner, "__module__", "").split(".")[0]
        if module in _PACKAGES:
            self.packages.add(_PACKAGES[module])
        elif module not in sys.stdlib_module_names:
            raise ValueError(
                f"Standalone export does not support the custom dependency {module!r}. "
                "Use notebook export and package your custom code explicitly."
            )
        return None


def _admitted(step: dict, capability: Any) -> bool:
    """Admit per-row steps that keep rows, plus built-in filters scored as exclusions."""
    if capability is None or capability.context != "row":
        return False
    # Outlier/invalid filters run per row; scoring keeps their rows with a reason.
    return capability.row_effect == "preserve" or (
        capability.row_effect == "filter" and is_exclusion_step(step)
    )


def _prediction_step(step: dict) -> dict:
    """Inspect one saved apply contract and retain only executable fitted fields."""
    if step["type"] == "sentence_embedder":
        raise ValueError(
            "sentence_embedder snapshots require separate neural dependency packaging. "
            "Use notebook export for this pipeline."
        )
    if step["type"] in {COLUMN_STEP, FITTED_STEP, FILTER_STEP}:
        # The step stores only a module reference; that project code is not packaged.
        raise ValueError(
            f"Step {step['name']!r} ({step['type']}) calls project code that is not "
            "packaged in a standalone export. Use notebook export for this pipeline."
        )
    state = step["artifact"]
    if step["type"] in {"LabelEncoder", "OrdinalEncoder"}:
        # Pipeline-owned target provenance is not part of the applier's state schema.
        # Retain it in the payload; inspect only the state consumed by the applier.
        state = {key: value for key, value in state.items() if key != "target_label_map"}
    capability = get_inference_capability(
        step["type"], step.get("params", {}), state, engine="pandas", applier=step["applier"]
    )
    if not _admitted(step, capability):
        raise ValueError(
            f"Step {step['name']!r} ({step['type']}) is not declared row-independent and "
            "row-preserving or a built-in exclusion filter. Use notebook export for "
            "contextual/batch workflows."
        )
    kept = {key: step[key] for key in ("name", "type", "applier", "artifact")}
    if kept["name"] == "step":
        # Canvas runs name every step "step"; exclusion reasons need the filter type.
        kept["name"] = kept["type"]
    return kept


def _portable_engineer(artifact: dict) -> FeatureEngineer:
    """Keep fitted prediction steps only, rejecting unknown or context-dependent behavior."""
    original = artifact.get("feature_engineer")
    if original is None and artifact.get("transformers") == []:
        return FeatureEngineer([])
    if type(original) is not FeatureEngineer:
        raise ValueError("Export requires a built-in fitted FeatureEngineer bundle.")
    engineer = FeatureEngineer([])
    engineer.fitted_steps = [
        _prediction_step(step) for step in original._transform_steps(preserve_rows=True)
    ]
    return engineer


def _schema_column(item: Any) -> dict[str, str]:
    """Validate one column before using its name in request and feature-order contracts."""
    if (
        not isinstance(item, dict)
        or not {"name", "dtype"} <= item.keys()
        or not isinstance(item["dtype"], str)
    ):
        raise ValueError("Raw input schema must record a name and dtype for every column.")
    if not isinstance(item["name"], str) or not item["name"]:
        raise ValueError("Raw input schema must contain unique nonempty column names.")
    return {"name": item["name"], "dtype": item["dtype"]}


def _raw_schema(artifact: dict) -> list[dict[str, str]]:
    """Require recorded raw inputs rather than guessing from post-transform model features."""
    schema = artifact.get("input_schema")
    if not isinstance(schema, list) or not schema:
        raise ValueError(
            "This artifact has no saved raw input schema. Retrain it in this Kayra version "
            "using a linear pipeline, or use notebook export."
        )
    result = [_schema_column(item) for item in schema]
    names = [item["name"] for item in result]
    if len(set(names)) != len(names):
        raise ValueError("Raw input schema must contain unique nonempty column names.")
    if artifact.get("target_column") in names:
        raise ValueError("Raw input schema must not require the prediction target.")
    return result


def _fitted_payload(artifact: dict, job: Any) -> dict:
    """Select only state needed for supervised inference, freezing the enabled thresholds."""
    engineer = _portable_engineer(artifact)
    model = DeploymentService._unwrap_tuple_estimator(artifact.get("model"))
    DeploymentService._validate_predict_artifact({"model": model, "feature_engineer": engineer})
    if not (is_classifier(model) or is_regressor(model)):
        raise ValueError(
            "Standalone export currently supports supervised classification/regression."
        )
    columns = artifact.get("feature_columns")
    if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
        raise ValueError("The artifact must record its ordered model feature columns.")
    thresholds = DeploymentService._resolve_thresholds_for_predict(
        None, job, getattr(model, "classes_", None)
    )
    return {
        "model": model,
        "feature_engineer": engineer,
        "feature_columns": columns,
        "thresholds": thresholds,
        "target_encoder": extract_target_label_encoder(engineer, artifact.get("target_column")),
    }


def _example_value(dtype: str) -> Any:
    """Return clearly documented placeholders without copying learned training values."""
    dtype = dtype.lower()
    if "bool" in dtype:
        return False
    if any(part in dtype for part in ("int", "float", "double", "decimal")):
        return 0
    if "date" in dtype or "time" in dtype:
        return "2000-01-01"
    return "REPLACE_ME"


def _json(value: Any) -> str:
    """Write strict readable UTF-8 JSON without executable interpolation into source files."""
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _runtime_requirements(packages: set[str], state: dict, mode: str) -> list[str]:
    """Pin saved object dependencies plus optional imports invoked lazily by apply."""
    packages = set(packages)
    if mode != "batch":
        packages.update({"fastapi", "uvicorn"})
    if any(step["type"] == "H3Index" for step in state["feature_engineer"].fitted_steps):
        packages.add("h3")
    return [f"{name}=={version(name)}" for name in sorted(packages)]


def _core_installation() -> dict[str, Any]:
    """Describe install provenance without exporting its path or contacting a package index."""
    installed = distribution("skyulf-core")
    direct = json.loads(installed.read_text("direct_url.json") or "{}")
    source = "direct" if direct else "index_or_unknown"
    if direct.get("dir_info", {}).get("editable"):
        source = "editable"
    return {
        "version": installed.version,
        "source": source,
        "local_version": "+" in installed.version,
        "package_index_availability": "not_checked",
        "matching_wheel_recommended": bool(direct) or "+" in installed.version,
    }


def build_project(artifact: dict, job: Any, mode: str) -> tuple[bytes, str]:
    """Create a ZIP of the fitted model, one scorer and the requested API/batch entry points."""
    if mode not in {"api", "batch", "both"}:
        raise ValueError("Export mode must be api, batch or both.")
    if not isinstance(artifact, dict):
        raise ValueError("Export requires a fitted Kayra model bundle.")
    schema = _raw_schema(artifact)
    state = _fitted_payload(artifact, job)
    stream = io.BytesIO()
    pickler = _PortablePickler(stream)
    pickler.dump(state)
    payload = stream.getvalue()
    requirements = _runtime_requirements(pickler.packages, state, mode)
    core_wheel = bundled_core_wheel()
    wheel_path = f"wheels/{core_wheel.filename}"
    core_installation = _core_installation() | {
        "bundled_wheel": wheel_path,
        "wheel_sha256": hashlib.sha256(core_wheel.content).hexdigest(),
        "source_sha256": core_wheel.source_sha256,
        "matching_wheel_recommended": False,
    }
    manifest = {
        "format_version": 1,
        "job_id": job.id,
        "model_version": job.version,
        "model_type": job.model_type,
        "mode": mode,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "requirements": requirements,
        "example_is_synthetic": True,
        "core_installation": core_installation,
        "privacy": "No source dataset copied. Fitted state may retain training examples or categories.",
    }
    files: dict[str, str | bytes] = dict(project_files(mode, manifest["core_installation"]))
    files.update(
        {
            "model.pkl": payload,
            "scoring.py": Path(__file__).with_name("runtime.py").read_text(encoding="utf-8"),
            "schema.json": _json(schema),
            "manifest.json": _json(manifest),
            "requirements.txt": (
                f"--find-links ./wheels\n./{wheel_path}\n" + "\n".join(requirements) + "\n"
            ),
            wheel_path: core_wheel.content,
            "example_request.json": _json(
                {"data": [{col["name"]: _example_value(col["dtype"]) for col in schema}]}
            ),
        }
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    filename = "kayra_model_" + re.sub(r"[^a-zA-Z0-9_-]", "_", job.id) + ".zip"
    return output.getvalue(), filename


def _load_and_build(job: Any, mode: str) -> tuple[bytes, str]:
    """Reuse trusted artifact-root enforcement without persisting a Deployment record."""
    uri = DeploymentService._resolve_final_deployment_uri(job.artifact_uri, job.id, job.pipeline_id)
    if not uri:
        raise ValueError("This job has no saved model artifact.")
    try:
        artifact = DeploymentService._load_predict_artifact(
            Deployment(job_id=job.id, model_type=job.model_type, artifact_uri=uri)
        )
    except ValueError as exc:
        raise ValueError(
            "The saved model artifact cannot be loaded. Check the training job."
        ) from exc
    return build_project(artifact, job, mode)


async def export_model_project(session: AsyncSession, job_id: str, mode: str) -> tuple[bytes, str]:
    """Read a completed job and package its fitted artifact off the async request loop."""
    job = await JobService.get_job_by_id(session, job_id)
    if job is None:
        raise ProjectJobNotFoundError("Training job not found.")
    DeploymentService._validate_job_for_deployment(job, job_id)
    return await run_in_threadpool(_load_and_build, job, mode)
