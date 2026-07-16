"""`POST /preview` — temporary-store pipeline preview (E9 phase 2).

The largest single endpoint in the package. Owns:

- temp `LocalArtifactStore` lifecycle,
- pipeline partitioning into one tab per (training × branch) experiment,
- artifact extraction (pandas/polars/SplitDataset/(X,y) shapes),
- per-branch labelling with `_branch_label` and #N model-type
  disambiguation,
- recommendation generation from the first branch's frame,
- merge-warning de-duplication across parallel branches.

All ML execution still runs through `PipelineEngine`.
"""

import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from typing import Any, cast

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

import backend.database.engine as db_engine
from backend.data.catalog import create_catalog_from_options
from backend.data_ingestion.service import DataIngestionService
from backend.database.engine import get_async_session
from backend.exceptions.core import SkyulfException
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import (
    NodeConfig,
    PipelineConfig,
    coerce_step_type,
)
from backend.ml_pipeline._internal._advisor import (
    AdvisorEngine,
    DataProfiler,
    Recommendation,
)
from backend.ml_pipeline._internal._helpers import branch_label as _branch_label
from backend.ml_pipeline._internal._schemas import (
    PipelineConfigModel,
    PreviewResponse,
)
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline.resolution import resolve_pipeline_nodes
from skyulf.data.dataset import SplitDataset

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])


def _to_records(df: Any) -> list[dict[str, Any]]:
    """Convert the first 50 rows of a pandas/polars frame or series to JSON records."""
    if isinstance(df, pd.DataFrame):
        return json.loads(df.head(50).to_json(orient="records"))
    if isinstance(df, pd.Series):
        return json.loads(df.to_frame().head(50).to_json(orient="records"))
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return json.loads(df.head(50).to_pandas().to_json(orient="records"))
        if isinstance(df, pl.Series):
            return json.loads(df.head(50).to_pandas().to_frame().to_json(orient="records"))
    except ImportError:
        pass
    return []


def _count_rows(df: Any) -> int:
    """Return the true total row count for an artifact, ignoring the 50-row preview cap.

    Used so the UI can show the real dataset size in tab badges.
    """
    if df is None:
        return 0
    if isinstance(df, pd.DataFrame):
        return int(df.shape[0])
    if isinstance(df, pd.Series):
        return int(df.shape[0])
    try:
        import polars as pl

        if isinstance(df, (pl.DataFrame, pl.Series)):
            return int(df.shape[0])
    except ImportError:
        pass
    try:
        return int(len(df))
    except Exception:
        return 0


def _process_xy(xy_tuple: tuple[Any, Any], prefix: str) -> dict[str, Any]:
    """Build preview records for an (X, y) tuple, keyed as ``{prefix}_X``/``{prefix}_y``."""
    X, y = xy_tuple
    return {f"{prefix}_X": _to_records(X), f"{prefix}_y": _to_records(y)}


def _process_xy_totals(xy_tuple: tuple[Any, Any], prefix: str) -> dict[str, int]:
    """Build row-count totals for an (X, y) tuple, keyed as ``{prefix}_X``/``{prefix}_y``."""
    X, y = xy_tuple
    return {f"{prefix}_X": _count_rows(X), f"{prefix}_y": _count_rows(y)}


def _to_pandas_safe(df: Any) -> pd.DataFrame | None:
    """Best-effort conversion of a pandas/polars DataFrame to pandas; ``None`` otherwise."""
    if isinstance(df, pd.DataFrame):
        return df
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
    except ImportError:
        pass
    return None


def _pick_target_node_id(node_list: list[NodeConfig]) -> str | None:
    """Pick the last data-bearing node to preview.

    Training/tuning nodes are stripped before execution, so the last
    node in a runnable sub-pipeline is already the right preview
    target. Falls back to the input of a training node for callers
    that pass an unstripped node list.
    """
    if not node_list:
        return None
    target = node_list[-1]
    target_id = target.node_id
    if (
        target.step_type
        in [
            StepType.BASIC_TRAINING,
            StepType.ADVANCED_TUNING,
        ]
        and target.inputs
    ):
        target_id = target.inputs[0]
    return target_id


def _is_polars_dataframe(artifact: Any) -> bool:
    """Return True if `artifact` is a polars DataFrame, tolerating a missing polars install."""
    try:
        import polars as pl

        return isinstance(artifact, pl.DataFrame)
    except ImportError:
        return False


def _extract_preview_frame(
    artifact: Any, is_polars: bool
) -> tuple[Any, dict[str, int], pd.DataFrame | None]:
    """Build preview output for a single polars/pandas DataFrame artifact."""
    if is_polars:
        preview_data = _to_records(artifact)
        df_for_analysis = _to_pandas_safe(artifact)
    else:
        preview_data = json.loads(artifact.head(50).to_json(orient="records"))
        df_for_analysis = artifact
    totals = {"_total": _count_rows(artifact)}
    return preview_data, totals, df_for_analysis


def _extract_preview_split(
    artifact: SplitDataset,
) -> tuple[Any, dict[str, int], pd.DataFrame | None]:
    """Build preview output for a `SplitDataset` artifact (train/test/validation splits)."""
    preview_data: dict[str, Any] = {}
    totals: dict[str, int] = {}
    df_for_analysis = None

    train = artifact.train
    if isinstance(train, tuple):
        xy_train = cast(tuple[Any, Any], train)
        preview_data.update(_process_xy(xy_train, "train"))
        totals.update(_process_xy_totals(xy_train, "train"))
        df_for_analysis = _to_pandas_safe(xy_train[0])
    else:
        preview_data["train"] = _to_records(train)
        totals["train"] = _count_rows(train)
        df_for_analysis = _to_pandas_safe(train)

    test = artifact.test
    if isinstance(test, tuple):
        xy_test = cast(tuple[Any, Any], test)
        preview_data.update(_process_xy(xy_test, "test"))
        totals.update(_process_xy_totals(xy_test, "test"))
    else:
        preview_data["test"] = _to_records(test)
        totals["test"] = _count_rows(test)

    validation = artifact.validation
    if validation is not None:
        if isinstance(validation, tuple):
            xy_validation = cast(tuple[Any, Any], validation)
            preview_data.update(_process_xy(xy_validation, "validation"))
            totals.update(_process_xy_totals(xy_validation, "validation"))
        else:
            preview_data["validation"] = _to_records(validation)
            totals["validation"] = _count_rows(validation)

    return preview_data, totals, df_for_analysis


def _extract_preview_xy_tuple(
    artifact: tuple[Any, Any],
) -> tuple[Any, dict[str, int], pd.DataFrame | None]:
    """Build preview output for a raw (X, y) tuple artifact."""
    X, y = artifact
    preview_data = {"X": _to_records(X), "y": _to_records(y)}
    totals = {"X": _count_rows(X), "y": _count_rows(y)}
    df_for_analysis = _to_pandas_safe(X)
    return preview_data, totals, df_for_analysis


def _extract_preview_dict_train(
    artifact: dict[str, Any],
) -> tuple[Any, dict[str, int], pd.DataFrame | None]:
    """Build preview output for a dict artifact keyed by split name with (X, y) tuples."""
    preview_data: dict[str, Any] = {}
    totals: dict[str, int] = {}

    preview_data.update(_process_xy(artifact["train"], "train"))
    totals.update(_process_xy_totals(artifact["train"], "train"))
    df_for_analysis = _to_pandas_safe(artifact["train"][0])

    if "test" in artifact:
        preview_data.update(_process_xy(artifact["test"], "test"))
        totals.update(_process_xy_totals(artifact["test"], "test"))
    if "validation" in artifact:
        preview_data.update(_process_xy(artifact["validation"], "validation"))
        totals.update(_process_xy_totals(artifact["validation"], "validation"))

    return preview_data, totals, df_for_analysis


def _extract_preview(
    artifact_store: LocalArtifactStore, target_node_id: str | None
) -> tuple[Any, dict[str, int], pd.DataFrame | None]:
    """Return (preview_data, totals, df_for_analysis) for a single artifact.

    `totals` mirrors the shape of `preview_data`:
    - dict of split-name → row count when `preview_data` is a dict;
    - a single int wrapped under the `_total` key when `preview_data`
      is a list (single-frame case).
    """
    preview_data: Any = {}
    totals: dict[str, int] = {}
    df_for_analysis = None
    if not target_node_id or not artifact_store.exists(target_node_id):
        return preview_data, totals, df_for_analysis

    artifact = artifact_store.load(target_node_id)
    logger.debug(f"Loaded artifact for node {target_node_id}. Type: {type(artifact)}")

    is_polars = _is_polars_dataframe(artifact)

    if is_polars or isinstance(artifact, pd.DataFrame):
        return _extract_preview_frame(artifact, is_polars)
    if isinstance(artifact, SplitDataset):
        return _extract_preview_split(artifact)
    if isinstance(artifact, tuple) and len(artifact) == 2:
        return _extract_preview_xy_tuple(artifact)
    if isinstance(artifact, dict) and "train" in artifact and isinstance(artifact["train"], tuple):
        return _extract_preview_dict_train(artifact)

    return preview_data, totals, df_for_analysis


def _build_preview_nodes(config_nodes: list[Any]) -> list[NodeConfig]:
    """Adapt raw request nodes for preview: force sampling on Data Loader nodes."""
    nodes = []
    for node in config_nodes:
        params = node.params.copy()

        # Force sampling for Data Loader
        if node.step_type == StepType.DATA_LOADER:
            params["sample"] = True
            params["limit"] = 1000  # Default preview limit

        nodes.append(
            NodeConfig(
                node_id=node.node_id,
                step_type=coerce_step_type(node.step_type),
                params=params,
                inputs=node.inputs,
            )
        )
    return nodes


def _is_data_preview_sub(sub: PipelineConfig) -> bool:
    """True when a sub-pipeline's terminal node is a Data Preview node."""
    return bool(sub.nodes) and sub.nodes[-1].step_type == "data_preview"


def _strip_non_preview_nodes(sub: PipelineConfig, training_types: set[StepType]) -> PipelineConfig:
    """Return a copy of ``sub`` with training/tuning and data_preview nodes removed."""
    stripped = [
        n for n in sub.nodes if n.step_type not in training_types and n.step_type != "data_preview"
    ]
    return PipelineConfig(
        pipeline_id=sub.pipeline_id,
        nodes=stripped,
        metadata=sub.metadata,
    )


def _pair_training_subs(
    training_subs: list[PipelineConfig], training_types: set[StepType]
) -> list[tuple[PipelineConfig, PipelineConfig]]:
    """Pair each non-data-preview training sub-pipeline with its stripped runnable form."""
    return [
        (orig, _strip_non_preview_nodes(orig, training_types))
        for orig in training_subs
        if not _is_data_preview_sub(orig)
    ]


def _append_uncovered_preview_branches(
    paired_subs: list[tuple[PipelineConfig, PipelineConfig]],
    pipeline_config: PipelineConfig,
    partition_for_preview: Any,
) -> None:
    """Append preview-only branches (data leaves not fed into any training terminal).

    Without this, a canvas mixing a training pipeline with one or more
    dangling preprocessing chains would silently drop the dangling chains
    from Run Preview — users only saw the training branch's data and
    assumed the others vanished. Mutates ``paired_subs`` in place.
    """
    covered_node_ids: set[str] = set()
    for orig, _runnable in paired_subs:
        for n in orig.nodes:
            covered_node_ids.add(n.node_id)
    preview_only_subs = partition_for_preview(pipeline_config)
    for sub in preview_only_subs:
        if not sub.nodes:
            continue
        if _is_data_preview_sub(sub):
            continue
        leaf = sub.nodes[-1]
        # Skip sub-pipelines whose leaf is already produced by a
        # training branch (avoids duplicate tabs for the same data).
        if leaf.node_id in covered_node_ids:
            continue
        paired_subs.append((sub, sub))


def _partition_preview_pipeline(
    pipeline_config: PipelineConfig, nodes: list[NodeConfig]
) -> list[tuple[PipelineConfig, PipelineConfig]]:
    """Partition the pipeline into (original_sub, runnable_sub) pairs for preview.

    We want one preview tab per logical experiment, mirroring "Run All
    Experiments". Strategy:
      1. Use partition_parallel_pipeline() — gives one sub per
         (training_node × input_branch) combination. This handles the
         case where one data node feeds multiple training nodes.
      2. If no training nodes exist, fall back to partition_for_preview()
         which splits by data leaves so parallel preprocessing chains
         still get separate tabs.
    Either way, training/tuning nodes are stripped before execution
    because preview never fits models.

    DataPreview is a separate evaluation node with its own background
    job — it must NOT appear as a Preview Results tab. Filter it out
    at every stage so neither partition_parallel_pipeline (which treats
    it as a parallel terminal) nor partition_for_preview (which treats
    it as a data leaf) can leak it into the tab list.
    """
    from backend.ml_pipeline._execution.graph_utils import (
        partition_for_preview,
        partition_parallel_pipeline,
    )

    training_types = {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}
    has_training = any(n.step_type in training_types for n in nodes)

    if not has_training:
        preview_subs = partition_for_preview(pipeline_config)
        # No training → no separate label source; reuse the runnable sub.
        return [(sub, sub) for sub in preview_subs if not _is_data_preview_sub(sub)]

    training_subs = partition_parallel_pipeline(pipeline_config)
    paired_subs = _pair_training_subs(training_subs, training_types)
    _append_uncovered_preview_branches(paired_subs, pipeline_config, partition_for_preview)
    return paired_subs


def _branch_terminal_group_key(leaf: Any) -> tuple[str, str]:
    """Return the (model/step grouping key, terminal node_id) for a branch's leaf node.

    Training terminals group by model_type/algorithm; other terminals
    (e.g. data_preview) group by step_type. Returns empty strings when the
    leaf shouldn't participate in dup-suffix grouping.
    """
    if leaf.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}:
        # Training terminal: group by model_type.
        mt = str(leaf.params.get("model_type") or leaf.params.get("algorithm") or "")
    else:
        # Non-training terminal (e.g. data_preview): group by
        # step_type so multiple Data Preview nodes get #1/#2/#3,
        # matching the canvas dup-suffix logic in useBranchColors.
        mt = str(leaf.step_type)
    return mt, leaf.node_id


def _collect_terminal_ids_by_group(
    sub_results: list[tuple[Any, Any, Any]],
) -> tuple[dict[str, list[str]], dict[int, str]]:
    """Group branch terminal node_ids by their model/step key, per branch index."""
    terminals_by_model: dict[str, list[str]] = {}
    terminal_id_per_branch: dict[int, str] = {}
    for i, (orig_sub, _runnable, _res) in enumerate(sub_results):
        mt = ""
        term_id = ""
        if orig_sub.nodes:
            mt, term_id = _branch_terminal_group_key(orig_sub.nodes[-1])
        if mt and term_id:
            terminal_id_per_branch[i] = term_id
            ids = terminals_by_model.setdefault(mt, [])
            if term_id not in ids:
                ids.append(term_id)
    return terminals_by_model, terminal_id_per_branch


def _compute_branch_dup_suffixes(sub_results: list[tuple[Any, Any, Any]]) -> dict[int, str]:
    """Compute "#N" disambiguators for training terminals sharing the same model_type.

    Mirrors the canvas edge label scheme in useBranchColors. The suffix is
    keyed on the *terminal node_id*, not the branch index — so two branches
    feeding the same XGBoost terminal both render as "#1", and a second
    XGBoost terminal (with any number of branches) becomes "#2". Without
    this, parallel branches of the same terminal incorrectly got
    "#1" / "#2" / "#3" despite originating from the same training node on
    the canvas. Preview-only branches (no model_type) skip this and stay
    unsuffixed.
    """
    terminals_by_model, terminal_id_per_branch = _collect_terminal_ids_by_group(sub_results)
    terminal_suffix: dict[str, str] = {}
    for ids in terminals_by_model.values():
        if len(ids) < 2:
            continue
        for n_, term_id in enumerate(ids, start=1):
            terminal_suffix[term_id] = f"#{n_}"
    return {
        branch_idx: terminal_suffix[term_id]
        for branch_idx, term_id in terminal_id_per_branch.items()
        if term_id in terminal_suffix
    }


def _aggregate_branch_previews(
    sub_results: list[tuple[Any, Any, Any]],
    artifact_store: LocalArtifactStore,
    dup_suffix_by_branch: dict[int, str],
) -> tuple[
    Any,
    dict[str, int],
    dict[str, Any] | None,
    dict[str, dict[str, int]] | None,
    dict[str, list[str]] | None,
    dict[str, Any],
    str,
    pd.DataFrame | None,
]:
    """Aggregate per-branch preview data, totals, node results and status across sub-pipelines.

    Returns (preview_data, preview_totals, branch_previews, branch_preview_totals,
    branch_node_ids, combined_node_results, agg_status, first_pdf).
    """
    preview_data: Any = {}
    preview_totals: dict[str, int] = {}
    branch_previews: dict[str, Any] | None = None
    branch_preview_totals: dict[str, dict[str, int]] | None = None
    branch_node_ids: dict[str, list[str]] | None = None
    combined_node_results: dict[str, Any] = {}
    # Aggregate status: failed > running > success
    agg_status = "success"
    first_pdf = None

    is_multi = len(sub_results) > 1
    if is_multi:
        branch_previews = {}
        branch_preview_totals = {}
        branch_node_ids = {}

    for idx, (orig_sub, runnable_sub, sub_result) in enumerate(sub_results):
        for k, v in sub_result.node_results.items():
            combined_node_results[k] = v.__dict__
        if sub_result.status != "success":
            agg_status = sub_result.status
        if sub_result.status == "success" and runnable_sub.nodes:
            target_id = _pick_target_node_id(runnable_sub.nodes)
            pdata, ptotals, pdf = _extract_preview(artifact_store, target_id)
            if idx == 0:
                preview_data = pdata
                preview_totals = ptotals
                first_pdf = pdf
            if (
                is_multi
                and branch_previews is not None
                and branch_node_ids is not None
                and branch_preview_totals is not None
            ):
                # Label uses the ORIGINAL sub (training node included) so
                # the model name shows up in the tab.
                label = _branch_label(idx, orig_sub, dup_suffix_by_branch.get(idx, ""))
                branch_previews[label] = pdata
                branch_preview_totals[label] = ptotals
                branch_node_ids[label] = [n.node_id for n in runnable_sub.nodes]

    return (
        preview_data,
        preview_totals,
        branch_previews,
        branch_preview_totals,
        branch_node_ids,
        combined_node_results,
        agg_status,
        first_pdf,
    )


def _generate_recommendations(first_pdf: pd.DataFrame | None) -> list[Recommendation]:
    """Generate advisor recommendations from the first branch's data only (avoid heavy work)."""
    if first_pdf is None:
        return []
    try:
        profile = DataProfiler.generate_profile(first_pdf)
        advisor = AdvisorEngine()
        return advisor.analyze(profile)
    except Exception as e:
        logger.warning("Error generating recommendations: %s", e)
        return []


def _dedupe_by_key(
    items: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], tuple]
) -> list[dict[str, Any]]:
    """Deduplicate a list of dicts by a computed key, preserving first-seen order."""
    seen: set = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_preview_warnings(
    sub_results: list[tuple[Any, Any, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Dedupe merge_warnings and node_warnings across parallel preview branches.

    When N parallel sub-pipelines run independently (Path A / Path B), each
    engine emits the same advisory for the shared upstream merge node (or the
    same per-node soft warning). Collapse duplicates so the UI shows one row
    per logically distinct warning.
    """
    all_warnings = [
        w for _orig, _runnable, res in sub_results for w in getattr(res, "merge_warnings", [])
    ]
    deduped_warnings = _dedupe_by_key(
        all_warnings,
        lambda w: (
            w.get("node_id"),
            w.get("kind"),
            tuple(sorted(w.get("inputs") or ())),
            tuple(w.get("overlap_columns") or ()),
            tuple(w.get("dropped_columns") or ()),
            w.get("part"),
        ),
    )

    all_node_warnings = [
        w for _orig, _runnable, res in sub_results for w in getattr(res, "node_warnings", [])
    ]
    deduped_node_warnings = _dedupe_by_key(
        all_node_warnings, lambda w: (w.get("node_id"), w.get("message"))
    )
    return deduped_warnings, deduped_node_warnings


def _run_preview_sub_pipelines(
    pipeline_config: PipelineConfig,
    nodes: list[NodeConfig],
    config_nodes: list[Any],
    resolved_s3_options: Any,
    sync_session: Any,
    artifact_store: LocalArtifactStore,
) -> list[tuple[PipelineConfig, PipelineConfig, Any]]:
    """Partition the pipeline into branches, run each through the engine, and return results.

    Branches are sorted by BFS position of their terminal node so branch
    letters (A, B, C, …) match the canvas edge colors from useBranchColors.
    A single shared artifact store across branches deduplicates work for
    ancestor nodes that appear in multiple sub-pipelines.
    """
    paired_subs = _partition_preview_pipeline(pipeline_config, nodes)

    catalog = create_catalog_from_options(resolved_s3_options, config_nodes, session=sync_session)
    engine = PipelineEngine(artifact_store, catalog=catalog)

    node_bfs_pos = {n.node_id: i for i, n in enumerate(pipeline_config.nodes)}
    paired_subs.sort(
        key=lambda pair: node_bfs_pos.get(
            pair[0].nodes[-1].node_id if pair[0].nodes else "",
            len(pipeline_config.nodes),
        )
    )
    return [(orig, runnable, engine.run(runnable)) for orig, runnable in paired_subs]


@router.post("/preview", response_model=PreviewResponse)
async def preview_pipeline(
    config: PipelineConfigModel, session: AsyncSession = Depends(get_async_session)
):
    """Run the pipeline in Preview Mode.

    - Uses a temporary artifact store (cleaned up after request).
    - Resolves dataset paths from IDs.
    """
    # 1. Create Temporary Artifact Store
    temp_dir = tempfile.mkdtemp(prefix="skyulf_preview_")
    artifact_store = LocalArtifactStore(temp_dir)

    # Resolve paths and credentials for Preview (Async)
    ingestion_service = DataIngestionService(session)
    resolved_s3_options = await resolve_pipeline_nodes(config.nodes, ingestion_service)

    sync_session = None
    try:
        logger.debug(f"Preview request received with {len(config.nodes)} nodes")
        for n in config.nodes:
            logger.debug(f"Node {n.node_id} - Type: {n.step_type}")

        # 2. Adapt Config for Preview
        nodes = _build_preview_nodes(config.nodes)

        pipeline_config = PipelineConfig(
            pipeline_id=config.pipeline_id, nodes=nodes, metadata=config.metadata
        )

        # 3. Run Engine
        # SmartCatalog uses sync ORM (`self.session.query()`), so we must
        # hand it a sync session — the request-scoped session here is async.
        if db_engine.sync_session_factory is None:
            raise SkyulfException(message="Database not initialized")

        sync_session = db_engine.sync_session_factory()

        sub_results = _run_preview_sub_pipelines(
            pipeline_config, nodes, config.nodes, resolved_s3_options, sync_session, artifact_store
        )

        # 4. Aggregate per-branch previews
        dup_suffix_by_branch = _compute_branch_dup_suffixes(sub_results)
        (
            preview_data,
            preview_totals,
            branch_previews,
            branch_preview_totals,
            branch_node_ids,
            combined_node_results,
            agg_status,
            first_pdf,
        ) = _aggregate_branch_previews(sub_results, artifact_store, dup_suffix_by_branch)

        # 5. Recommendations from first branch's data only (avoid heavy work).
        recommendations = _generate_recommendations(first_pdf)

        deduped_warnings, deduped_node_warnings = _dedupe_preview_warnings(sub_results)

        return PreviewResponse(
            pipeline_id=pipeline_config.pipeline_id,
            status=agg_status,
            node_results=combined_node_results,
            preview_data=preview_data,
            preview_totals=preview_totals,
            branch_previews=branch_previews,
            branch_preview_totals=branch_preview_totals,
            branch_node_ids=branch_node_ids,
            recommendations=recommendations,
            merge_warnings=deduped_warnings,
            node_warnings=deduped_node_warnings,
        )

    except Exception:
        logger.exception("Pipeline preview failed")
        raise SkyulfException(message="Pipeline preview failed") from None
    finally:
        # 5. Cleanup — close sync session before removing temp artefacts.
        if sync_session is not None:
            sync_session.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = ["router"]
