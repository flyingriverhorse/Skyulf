import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import sklearn
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database.models import Deployment, TrainingJob
from backend.ml_pipeline._services.job_service import JobService
from backend.ml_pipeline._services.prediction_utils import extract_target_label_encoder
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore
from backend.utils import sanitize_for_log

logger = logging.getLogger(__name__)


class OverrideThresholdMismatch(ValueError):
    """Raised when override_thresholds keys don't match the model's classes."""


def _maybe_decode_predictions(
    predictions: Any,
    feature_engineer: Any,
    target_column: str | None = None,
) -> Any:
    """Decode numeric class predictions to original labels if possible.

    Looks for the target LabelEncoder, first under encoders['__target__'], then
    under encoders[target_column] (for pipelines where encoding happened before
    the Feature/Target Split).
    """

    target_encoder = extract_target_label_encoder(feature_engineer, target_column=target_column)
    if target_encoder is None:
        return predictions

    try:
        import numpy as np

        preds = np.asarray(predictions)
        # Best-effort: many sklearn classifiers output ints; decode expects int-like.
        if preds.dtype.kind in {"i", "u", "b"}:
            return target_encoder.inverse_transform(preds.astype(int))

        # If dtype isn't integer but values might still be numeric strings/floats,
        # try converting.
        return target_encoder.inverse_transform(preds.astype(int))
    except Exception as e:  # noqa: BLE001 - decode failure returns raw predictions
        logger.debug(f"Could not decode predictions via LabelEncoder: {e}")
        return predictions


class DeploymentService:
    @staticmethod
    def _validate_job_for_deployment(db_job: Any, job_id: str) -> None:
        """Raises ValueError if the job doesn't exist or hasn't completed successfully."""
        if not db_job:
            raise ValueError(f"Job {job_id} not found")

        if db_job.status not in ["completed", "succeeded"]:
            raise ValueError(f"Job {job_id} is not completed successfully")

    @staticmethod
    def _resolve_final_deployment_uri(
        artifact_uri: str | None, job_id: str, pipeline_id: Any
    ) -> str | None:
        """Resolves the deployment's stored artifact_uri, pointing at the specific bundled artifact file.

        If artifact_uri is a directory (doesn't end with .joblib or similar), we need to point to the
        specific file. The PipelineEngine saves the FULL bundled artifact (model + transformers) using
        the job_id as the key.
        """
        if not artifact_uri:
            return artifact_uri

        if artifact_uri.startswith("s3://"):
            # Handle S3 URI
            if not artifact_uri.endswith(".joblib") and not artifact_uri.endswith(".pkl"):
                # Assume it's a prefix, append job_id.joblib
                return f"{artifact_uri.rstrip('/')}/{job_id}.joblib"
            return artifact_uri
        # Check if it's a directory on disk
        elif Path(artifact_uri).is_dir():
            # The artifact is likely named {job_id}.joblib inside it.
            # We construct the full path to the file so predict() can parse it correctly.
            return str(Path(artifact_uri) / f"{job_id}.joblib")
        elif not artifact_uri.endswith(".joblib") and not artifact_uri.endswith(".pkl"):
            # Not a directory, and no extension. Assume it's a node_id or job_id.
            # Construct the abstract URI for exports/models
            return f"{pipeline_id}/{job_id}"
        else:
            # It's a file path (relative or absolute)
            return artifact_uri

    @staticmethod
    async def deploy_model(
        session: AsyncSession, job_id: str, user_id: int | None = None
    ) -> Deployment:
        # 1. Get Job Entity
        db_job = await JobService.get_job_by_id(session, job_id)
        DeploymentService._validate_job_for_deployment(db_job, job_id)
        if db_job is None:
            # Unreachable: _validate_job_for_deployment raises ValueError on a falsy db_job.
            raise ValueError(f"Job {job_id} not found")

        # 2. Get Artifact URI
        artifact_uri = db_job.artifact_uri
        pipeline_id = db_job.pipeline_id

        if not artifact_uri:
            # Fallback: use node_id if artifact_uri is missing (legacy jobs)
            artifact_uri = str(db_job.node_id)
            # node_id reaches the DB from the client-submitted graph via the
            # unvalidated NodeConfig dataclass, so it is tainted like job_id.
            logger.warning(
                f"No artifact URI found for job {sanitize_for_log(job_id)}, "
                f"falling back to node_id: {sanitize_for_log(artifact_uri)}"
            )

        # 3. Record the currently active deployment (if any) as the one this
        # new deployment replaces, then deactivate it. Capturing the id before
        # the UPDATE keeps the replacement chain traceable across the deploy.
        previous_deployment = await DeploymentService.get_active_deployment(session)
        await session.execute(
            update(Deployment).where(Deployment.is_active).values(is_active=False)
        )

        # 4. Create Deployment — the artifact URI must encode pipeline_id so it can
        # be resolved back to the export path (exports/models/<pipeline_id>/...).
        final_uri = DeploymentService._resolve_final_deployment_uri(
            artifact_uri, job_id, pipeline_id
        )

        deployment = Deployment(
            job_id=job_id,
            model_type=db_job.model_type or "unknown",
            artifact_uri=final_uri,
            is_active=True,
            deployed_by=user_id,
            previous_deployment_id=previous_deployment.id if previous_deployment else None,
        )
        session.add(deployment)
        await session.commit()
        await session.refresh(deployment)

        return deployment

    @staticmethod
    async def get_active_deployment(session: AsyncSession) -> Deployment | None:
        stmt = select(Deployment).where(Deployment.is_active).order_by(Deployment.created_at.desc())
        result = await session.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def list_deployments(
        session: AsyncSession, limit: int | None = None, skip: int = 0
    ) -> Sequence[Deployment]:
        """Lists deployment history."""
        effective_limit = limit if limit is not None else get_settings().DEFAULT_PAGE_SIZE
        stmt = (
            select(Deployment)
            .order_by(Deployment.created_at.desc())
            .limit(effective_limit)
            .offset(skip)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def deactivate_current_deployment(session: AsyncSession):
        """Deactivates the currently active deployment."""
        await session.execute(
            update(Deployment).where(Deployment.is_active).values(is_active=False)
        )
        await session.commit()

    @staticmethod
    def _resolve_predict_store_and_key_s3(uri: str) -> tuple[str, str]:
        """Resolves an s3:// artifact URI into (store_uri, artifact_key)."""
        if uri.endswith(".joblib"):
            store_uri = uri.rsplit("/", 1)[0]
            artifact_key = uri.rsplit("/", 1)[1].replace(".joblib", "")
        else:
            parts = uri.replace("s3://", "").split("/")
            bucket = parts[0]
            artifact_key = "/".join(parts[1:])
            store_uri = f"s3://{bucket}"
        return store_uri, artifact_key

    @staticmethod
    def _resolve_pipeline_node_path(pipeline_id: str, node_id: str) -> tuple[str, str]:
        """Builds the default exports/models store path for a pipeline_id/node_id pair."""
        store_uri = str(Path.cwd() / "exports" / "models" / pipeline_id)
        return store_uri, node_id

    @staticmethod
    def _resolve_predict_store_and_key_local(uri: str) -> tuple[str, str]:
        """Resolves a local artifact path (absolute, relative, or bare "pipeline_id/node_id") into (store_uri, artifact_key)."""
        if Path(uri).is_absolute():
            return str(Path(uri).parent), Path(uri).name

        if "/" in uri or "\\" in uri:
            if not Path(uri).exists() and not Path(uri).parent.exists():
                parts = uri.replace("\\", "/").split("/")
                if len(parts) == 2:
                    return DeploymentService._resolve_pipeline_node_path(parts[0], parts[1])
                return str(Path(uri).parent), Path(uri).name
            return str(Path(uri).parent), Path(uri).name

        parts = uri.split("/")
        if len(parts) >= 2:
            return DeploymentService._resolve_pipeline_node_path(parts[0], parts[1])
        raise ValueError(f"Invalid artifact URI format: {uri}")

    @staticmethod
    def _resolve_predict_store_and_key(uri: str) -> tuple[str, str]:
        """Resolves an artifact URI into (store_uri, artifact_key) for the predict artifact loader.

        Handles S3 URIs (with or without a .joblib/.pkl suffix) and local paths
        (absolute, relative with a separator, or bare "pipeline_id/node_id" strings).
        """
        if uri.startswith("s3://"):
            return DeploymentService._resolve_predict_store_and_key_s3(uri)
        return DeploymentService._resolve_predict_store_and_key_local(uri)

    @staticmethod
    def _load_predict_artifact(deployment: Deployment) -> Any:
        """Loads and unwraps the deployed artifact used by predict(), wrapping load failures in ValueError."""
        try:
            from backend.ml_pipeline.artifacts.factory import ArtifactFactory

            store_uri, artifact_key = DeploymentService._resolve_predict_store_and_key(
                deployment.artifact_uri
            )
            store = ArtifactFactory.get_artifact_store(store_uri)
            artifact = store.load(artifact_key)
        except Exception as e:
            logger.error(f"Failed to load artifact: {e}")
            raise ValueError(f"Could not load model artifact: {deployment.artifact_uri}") from e

        # Handle tuple artifact (model, metadata/tuning_result) from TunerCalculator
        if isinstance(artifact, tuple) and len(artifact) >= 1:
            logger.info("Artifact is a tuple, using the first element as the model.")
            artifact = artifact[0]

        return artifact

    @staticmethod
    def _drop_target_and_dropped_columns(
        df: pd.DataFrame, target_col: str | None, dropped_cols: Any
    ) -> pd.DataFrame:
        """Drops the target column and any explicitly dropped columns from the inference DataFrame."""
        if target_col and target_col in df.columns:
            logger.info(f"Dropping target column '{target_col}' from inference data")
            df = df.drop(columns=[target_col])

        if dropped_cols:
            # Ensure dropped_cols is a list of strings
            if isinstance(dropped_cols, str):
                dropped_cols = [dropped_cols]

            existing_dropped = [c for c in dropped_cols if c in df.columns]
            if existing_dropped:
                logger.info(
                    f"Dropping explicitly dropped columns {existing_dropped} from inference data"
                )
                df = df.drop(columns=existing_dropped)

        return df

    @staticmethod
    def _unwrap_tuple_estimator(estimator: Any) -> Any:
        """Unwraps a tuple estimator (e.g. from TunerCalculator) to its first element."""
        if isinstance(estimator, tuple) and len(estimator) >= 1:
            logger.info("Estimator inside artifact is a tuple, using the first element.")
            return estimator[0]
        return estimator

    @staticmethod
    def _transform_bundled_features(feature_engineer: Any, df: pd.DataFrame) -> Any:
        """Transforms the inference DataFrame via the bundled feature engineer, wrapping failures in ValueError."""
        try:
            # Use config_context so sklearn returns DataFrames with feature names during inference only
            with sklearn.config_context(transform_output="pandas"):
                return feature_engineer.transform(df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise ValueError(f"Feature engineering failed: {str(e)}") from e

    @staticmethod
    def _validate_override_thresholds(
        override_thresholds: dict[str, float], estimator_classes: Any
    ) -> None:
        """Raise OverrideThresholdMismatch unless the override keys match the model's classes exactly."""
        expected = {str(c) for c in (estimator_classes if estimator_classes is not None else [])}
        provided = set(override_thresholds.keys())
        if provided != expected:
            raise OverrideThresholdMismatch(
                f"override_thresholds keys {sorted(provided)} do not match "
                f"model classes {sorted(expected)}"
            )

    @staticmethod
    def _resolve_thresholds_for_predict(
        override_thresholds: dict[str, float] | None,
        job: TrainingJob | None,
        estimator_classes: Any,
    ) -> dict[str, float] | None:
        """Resolve which per-class thresholds to apply: override > saved+enabled > None.

        Returns a str-keyed dict (matching the JSON/response shape) or None.
        """
        if override_thresholds is not None:
            DeploymentService._validate_override_thresholds(override_thresholds, estimator_classes)
            return dict(override_thresholds)

        if (
            job is not None
            and job.tuned_thresholds_enabled
            and job.tuned_thresholds
            and estimator_classes is not None
        ):
            saved = job.tuned_thresholds.get("thresholds", {})
            resolved = {str(c): saved[str(c)] for c in estimator_classes if str(c) in saved}
            if len(resolved) == len(list(estimator_classes)):
                return resolved
            logger.warning("Saved tuned thresholds do not cover every model class; skipping them.")
        return None

    @staticmethod
    def _predict_and_decode(
        estimator: Any,
        X_transformed: Any,
        feature_engineer: Any,
        target_col: str | None,
        thresholds: dict[str, float] | None = None,
    ) -> tuple[list, dict[str, float] | None]:
        """Runs the estimator's predict (or threshold-applied predict) and decodes labels.

        When ``thresholds`` is provided, uses ``predict_proba`` +
        ``apply_thresholds`` instead of the estimator's default decision rule.
        Returns ``(predictions, thresholds_applied)``.
        """
        try:
            if thresholds is not None:
                from skyulf.modeling import apply_thresholds

                y_proba = estimator.predict_proba(X_transformed)
                # apply_thresholds requires dict keys that exactly match
                # estimator.classes_ (dtype included); saved/override thresholds
                # arrive with str keys, so reconcile by string identity.
                reconciled = {c: float(thresholds[str(c)]) for c in estimator.classes_}
                predictions = apply_thresholds(y_proba, reconciled, classes=estimator.classes_)
            else:
                predictions = estimator.predict(X_transformed)

            predictions = _maybe_decode_predictions(
                predictions, feature_engineer, target_column=target_col
            )
            if hasattr(predictions, "tolist"):
                return cast(list[Any], predictions.tolist()), thresholds
            return list(predictions), thresholds
        except OverrideThresholdMismatch:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {str(e)}") from e

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame, feature_columns: Any) -> None:
        """Raise a clear, actionable error if inference data is missing required columns.

        Without this check, a missing column surfaces later as a cryptic
        sklearn ``X has N features, but <Model> is expecting M features``
        error — this catches it earlier with the actual column names.
        """
        if not feature_columns:
            return
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s) for prediction: {missing}. "
                f"Expected columns: {list(feature_columns)}"
            )

    @staticmethod
    def _warn_on_engine_mismatch(artifact: dict) -> None:
        """Warn when a bundle trained on one engine is served on another (F-25).

        The deployment bundle records the DataFrame engine the model was
        trained on (``engine``). Prediction input is always served as pandas,
        so a polars-trained bundle is the mismatch case: dual-engine nodes
        keep their artifacts engine-neutral and this is normally safe, but
        the warning makes the one place to look obvious when predictions
        ever come out wrong after an engine change.
        """
        train_engine = artifact.get("engine")
        if not train_engine or train_engine == "pandas":
            return
        logger.warning(
            "Deployed model was trained on the '%s' engine but prediction input "
            "is served as pandas. Dual-engine artifacts are engine-neutral, so "
            "this is normally safe; check here first if predictions look wrong.",
            train_engine,
        )

    @staticmethod
    def _predict_with_bundled_artifact(
        artifact: dict, df: pd.DataFrame, thresholds: dict[str, float] | None = None
    ) -> tuple[list, dict[str, float] | None]:
        """Predicts using the new SDK bundled artifact format: {"feature_engineer": ..., "model": ...}."""
        DeploymentService._warn_on_engine_mismatch(artifact)
        feature_engineer = artifact["feature_engineer"]
        estimator = artifact["model"]

        # Clean Data
        target_col = artifact.get("target_column")
        dropped_cols = artifact.get("dropped_columns", [])
        df = DeploymentService._drop_target_and_dropped_columns(df, target_col, dropped_cols)

        # Validate against the feature engineer's expected input columns
        # (pre-transform), not the model's feature_columns (post-transform).
        # F-03: feature_columns was recorded from the training frame *after*
        # feature engineering, so validating the raw request against it would
        # reject any pipeline with column-adding transformers.
        input_columns = DeploymentService._extract_features_from_engineer(feature_engineer)
        DeploymentService._validate_required_columns(df, input_columns)

        estimator = DeploymentService._unwrap_tuple_estimator(estimator)

        X_transformed = DeploymentService._transform_bundled_features(feature_engineer, df)

        # F-02: Reindex to the recorded training feature order so that
        # positional consumers (sklearn on bare numpy, no feature_names_in_)
        # receive columns in the order they were trained on. The legacy path
        # already does this; the bundled path was missing it.
        feature_columns = artifact.get("feature_columns")
        if feature_columns and hasattr(X_transformed, "columns"):
            missing = [c for c in feature_columns if c not in X_transformed.columns]
            if not missing:
                X_transformed = X_transformed[feature_columns]

        return DeploymentService._predict_and_decode(
            estimator, X_transformed, feature_engineer, target_col, thresholds=thresholds
        )

    @staticmethod
    def _predict_with_legacy_artifact(artifact: Any, df: pd.DataFrame) -> list:
        """Predicts using a legacy artifact that is directly a fitted predictor (no bundled feature engineer)."""
        # Log columns for debugging
        if isinstance(df, pd.DataFrame):
            logger.info(f"Predicting with columns: {df.columns.tolist()}")
            # Check if model has feature names and if they match
            if hasattr(artifact, "feature_names_in_"):
                model_cols = artifact.feature_names_in_.tolist()
                missing_in_df = set(model_cols) - set(df.columns)
                if missing_in_df:
                    logger.warning(f"Missing columns in input DataFrame: {missing_in_df}")
                    for c in missing_in_df:
                        df[c] = 0
                # Reorder columns to match model
                df = df[model_cols]

        predictions = artifact.predict(df)
        if hasattr(predictions, "tolist"):
            return cast(list[Any], predictions.tolist())
        return list(predictions)

    @staticmethod
    async def predict(
        session: AsyncSession,
        data: list[dict],
        override_thresholds: dict[str, float] | None = None,
    ) -> tuple[list, dict[str, float] | None]:
        # 1. Get active deployment
        deployment = await DeploymentService.get_active_deployment(session)
        if not deployment:
            raise ValueError("No active model deployed")

        # 2. Load Artifact
        artifact = DeploymentService._load_predict_artifact(deployment)

        # 3. Prepare Data
        df = pd.DataFrame(data)

        # 4. Predict
        # Check for new SDK format: {"feature_engineer": ..., "model": ...}
        if isinstance(artifact, dict) and "feature_engineer" in artifact and "model" in artifact:
            # Threshold application is only supported on the bundled artifact
            # path. Resolve which thresholds (if any) to apply, honouring the
            # priority: explicit override > saved+enabled tuned thresholds.
            estimator = DeploymentService._unwrap_tuple_estimator(artifact["model"])
            job = await DeploymentService._get_job_for_deployment(session, deployment.job_id)
            thresholds = DeploymentService._resolve_thresholds_for_predict(
                override_thresholds, job, getattr(estimator, "classes_", None)
            )
            return DeploymentService._predict_with_bundled_artifact(
                artifact, df, thresholds=thresholds
            )
        # Legacy support or direct model loading (if artifact is just the model)
        elif hasattr(artifact, "predict"):
            if override_thresholds is not None:
                raise OverrideThresholdMismatch(
                    "override_thresholds is not supported for this deployed model "
                    "(legacy artifact without probability outputs)."
                )
            predictions = DeploymentService._predict_with_legacy_artifact(artifact, df)
            return predictions, None
        else:
            raise ValueError(
                "Loaded artifact is not a valid predictor or recognized pipeline format"
            )

    @staticmethod
    async def _get_job_for_deployment(session: AsyncSession, job_id: str) -> TrainingJob | None:
        """Fetches the TrainingJob backing a deployment (for tuned-threshold lookup)."""
        result = await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        return result.scalar_one_or_none()

    @staticmethod
    def _load_artifact_from_s3_for_details(artifact_uri: str) -> Any:
        """Loads an artifact from S3 for schema inspection, building storage options from settings."""
        # Parse bucket and key: s3://bucket/key
        parts = artifact_uri.replace("s3://", "").split("/")
        bucket_name = parts[0]
        key = "/".join(parts[1:])

        settings = get_settings()
        storage_options = {
            "key": settings.AWS_ACCESS_KEY_ID,
            "secret": settings.AWS_SECRET_ACCESS_KEY,
            "endpoint_url": settings.AWS_ENDPOINT_URL,
            "region_name": settings.AWS_DEFAULT_REGION,
        }
        # Filter None values
        storage_options = {k: v for k, v in storage_options.items() if v is not None}

        store = S3ArtifactStore(bucket_name=bucket_name, storage_options=storage_options)
        return store.load(key)

    @staticmethod
    def _resolve_local_base_and_key_for_details(artifact_uri: str) -> tuple[str, str]:
        """Resolves a local artifact URI into (base_path, node_id) for schema-inspection loading."""
        if Path(artifact_uri).is_absolute():
            return str(Path(artifact_uri).parent), Path(artifact_uri).name
        elif "/" in artifact_uri or "\\" in artifact_uri:
            if not Path(artifact_uri).exists() and not Path(artifact_uri).parent.exists():
                parts = artifact_uri.replace("\\", "/").split("/")
                if len(parts) == 2:
                    pipeline_id = parts[0]
                    node_id = parts[1]
                    base_path = str(Path.cwd() / "exports" / "models" / pipeline_id)
                    return base_path, node_id
            return str(Path(artifact_uri).parent), Path(artifact_uri).name
        else:
            # Fallback
            return str(Path.cwd()), artifact_uri

    @staticmethod
    def _load_artifact_for_details(artifact_uri: str) -> Any:
        """Loads the deployed artifact for schema inspection, mirroring predict()'s URI resolution.

        Unlike predict(), this instantiates S3ArtifactStore/LocalArtifactStore directly and
        returns None (rather than raising) when the artifact does not exist locally.
        """
        if artifact_uri.startswith("s3://"):
            return DeploymentService._load_artifact_from_s3_for_details(artifact_uri)

        base_path, node_id = DeploymentService._resolve_local_base_and_key_for_details(artifact_uri)
        store = LocalArtifactStore(base_path)
        return store.load(node_id) if store.exists(node_id) else None

    @staticmethod
    def _extract_features_from_engineer(fe: Any) -> Any:
        """Best-effort extraction of feature names from a feature engineer or its first pipeline step."""
        if hasattr(fe, "feature_names_in_"):
            return fe.feature_names_in_

        if hasattr(fe, "steps") and fe.steps:
            # Try first step
            first_step = fe.steps[0]
            # If it's a tuple (name, transformer)
            if isinstance(first_step, tuple) and len(first_step) > 1:
                transformer = first_step[1]
                if hasattr(transformer, "feature_names_in_"):
                    return transformer.feature_names_in_

        return []

    @staticmethod
    def _extract_features_from_bundled_artifact(artifact: dict) -> Any:
        """Best-effort extraction of input feature names from a bundled artifact.

        Returns the columns the *caller* must send — i.e. the feature engineer's
        expected input, not the post-transform columns the model sees internally.
        Falls back to the persisted ``feature_columns`` only when there is no
        feature engineer (e.g. a bare model without preprocessing).
        """
        fe = artifact.get("feature_engineer")
        if fe is not None:
            input_features = DeploymentService._extract_features_from_engineer(fe)
            if len(input_features) > 0:
                return input_features

        # No feature engineer (or it yielded nothing) — fall back to the
        # persisted feature_columns, which are the model's expected columns.
        feature_columns = artifact.get("feature_columns")
        if feature_columns:
            return feature_columns

        # Last resort: try the model's feature_names_in_
        if "model" in artifact:
            model = artifact["model"]
            if isinstance(model, tuple):
                model = model[0]
            if hasattr(model, "feature_names_in_"):
                return model.feature_names_in_

        return []

    @staticmethod
    def _extract_input_features(artifact: Any) -> Any:
        """Best-effort extraction of the input feature name list from an artifact's feature engineer or model."""
        input_features = []

        # Check dict format
        if isinstance(artifact, dict) and "feature_engineer" in artifact:
            input_features = DeploymentService._extract_features_from_bundled_artifact(artifact)
        # Check direct model
        elif hasattr(artifact, "feature_names_in_"):
            input_features = artifact.feature_names_in_

        if hasattr(input_features, "tolist"):
            input_features = cast(Any, input_features).tolist()

        return input_features

    @staticmethod
    def _extract_target_column_from_graph(graph: dict) -> str | None:
        """Finds the first `target_column` param among a job graph's nodes, or None if not found."""
        nodes = graph.get("nodes", [])
        for node in nodes:
            # Handle both dict and object (though graph is usually dict from DB)
            if isinstance(node, dict):
                params = node.get("params", {})
                if "target_column" in params:
                    return cast(str, params["target_column"])
            elif hasattr(node, "params"):
                params = getattr(node, "params", {})
                if "target_column" in params:
                    return cast(str, params["target_column"])
        return None

    @staticmethod
    def _pretty_dtype(raw: Any) -> str:
        """Map a pandas/numpy or Polars dtype string to a short label users can act on.

        Nullable extension dtypes ("Int32", "Float32", "boolean") normalise to the
        same label as their numpy counterparts — the distinction is an
        implementation detail no one filling in a prediction form cares about.
        Polars dtype names (``Date``, ``Duration('us')``, ``Categorical``, ``Enum``)
        are recognised too, since the configured engine's schema strings are
        captured verbatim at training time (F-30).
        """
        if not raw:
            return "unknown"

        name = str(raw).lower()
        if name.startswith(("datetime", "period")):
            return "datetime"
        if name == "date":
            return "date"
        if name.startswith(("timedelta", "duration")):
            return "duration"
        if name.startswith(("category", "categorical", "enum")):
            return "category"
        if name in {"bool", "boolean"}:
            return "boolean"
        if name.startswith(("int", "uint")):
            return "integer"
        if name.startswith("float"):
            return "float"
        if name in {"object", "string", "str"}:
            return "text"
        return "unknown"

    @staticmethod
    def _input_schema_entries(
        feature_names: Any, feature_dtypes: dict[str, Any] | None
    ) -> list[dict[str, str]]:
        """Build the API's input-schema list, preserving the model's column order.

        ``feature_dtypes`` is absent on artifacts bundled before dtype capture
        existed, so every column there degrades to "unknown" rather than failing.
        """
        dtypes = feature_dtypes or {}
        return [
            {"name": str(f), "type": DeploymentService._pretty_dtype(dtypes.get(str(f)))}
            for f in feature_names
        ]

    @staticmethod
    def _dtypes_for_columns(df: pd.DataFrame, columns: Any) -> dict[str, str]:
        """Record each requested column's dtype as a string, skipping absent columns."""
        return {str(c): str(df[c].dtype) for c in columns if c in df.columns}

    @staticmethod
    def _build_input_schema_from_artifact(artifact_uri: str) -> list[dict[str, str]] | None:
        """Loads the artifact and extracts its input schema, unwrapping tuple artifacts first."""
        artifact = DeploymentService._load_artifact_for_details(artifact_uri)
        if not artifact:
            return None

        if isinstance(artifact, tuple) and len(artifact) >= 1:
            artifact = artifact[0]

        input_features = DeploymentService._extract_input_features(artifact)
        if not input_features:
            return None

        feature_dtypes = artifact.get("feature_dtypes") if isinstance(artifact, dict) else None
        return DeploymentService._input_schema_entries(input_features, feature_dtypes)

    @staticmethod
    async def _get_jobs_by_ids(
        session: AsyncSession, job_ids: Sequence[str] | set[str]
    ) -> dict[str, TrainingJob]:
        """Batch-fetches the TrainingJob rows for the given ids in a single query."""
        if not job_ids:
            return {}
        result = await session.execute(select(TrainingJob).where(TrainingJob.id.in_(job_ids)))
        return {job.id: job for job in result.scalars().all()}

    @staticmethod
    def _lineage_fields_from_job(job: TrainingJob | None) -> dict[str, Any]:
        """Builds the cheap dataset/version/target-column lineage fields from an
        already-fetched TrainingJob, without touching the deployed artifact."""
        if job is None:
            return {"dataset_id": None, "version": None, "target_column": None}
        target_column = (
            DeploymentService._extract_target_column_from_graph(job.graph) if job.graph else None
        )
        return {
            "dataset_id": job.dataset_source_id,
            "version": job.version,
            "target_column": target_column,
        }

    @staticmethod
    async def get_deployment_details(
        session: AsyncSession, deployment: Deployment
    ) -> dict[str, Any]:
        """
        Returns deployment info enriched with input/output schema from the artifact.
        """
        info = deployment.to_dict()
        info["input_schema"] = None
        info["output_schema"] = None
        info["dataset_id"] = None
        info["version"] = None

        try:
            artifact_uri = str(deployment.artifact_uri)
            input_schema = DeploymentService._build_input_schema_from_artifact(artifact_uri)
            if input_schema:
                info["input_schema"] = input_schema

            # The backing TrainingJob carries the dataset/version identity and the
            # target column, so a single fetch backs both instead of one query per
            # concern (this used to be two separate job lookups).
            source_job = await DeploymentService._get_job_for_deployment(session, deployment.job_id)
            lineage = DeploymentService._lineage_fields_from_job(source_job)
            if lineage["target_column"] is not None:
                info["target_column"] = lineage["target_column"]
            info["dataset_id"] = lineage["dataset_id"]
            info["version"] = lineage["version"]

        except Exception as e:  # noqa: BLE001 - enrichment best-effort; info still returned
            logger.warning(f"Failed to extract schema for deployment {deployment.id}: {e}")

        return cast(dict[str, Any], info)

    @staticmethod
    async def list_deployment_details(
        session: AsyncSession, limit: int | None = None, skip: int = 0
    ) -> list[dict[str, Any]]:
        """Lists deployment history enriched with cheap lineage fields only.

        Unlike `get_deployment_details`, this never loads the deployed artifact:
        it batches a single TrainingJob query across the whole page instead of
        fetching one job (and deserializing one artifact) per row, so paging
        through history stays O(1) artifact loads regardless of page size.
        """
        deployments = await DeploymentService.list_deployments(session, limit, skip)
        job_ids = {d.job_id for d in deployments}
        jobs_by_id = await DeploymentService._get_jobs_by_ids(session, job_ids)

        results = []
        for deployment in deployments:
            info = deployment.to_dict()
            info["input_schema"] = None
            info["output_schema"] = None
            info.update(
                DeploymentService._lineage_fields_from_job(jobs_by_id.get(deployment.job_id))
            )
            results.append(info)
        return results
