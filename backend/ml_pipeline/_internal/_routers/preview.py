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
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends
from skyulf.data.dataset import SplitDataset
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])


@router.post("/preview", response_model=PreviewResponse)
async def preview_pipeline(  # noqa: C901
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

    try:
        logger.debug(f"Preview request received with {len(config.nodes)} nodes")
        for n in config.nodes:
            logger.debug(f"Node {n.node_id} - Type: {n.step_type}")

        # 2. Adapt Config for Preview
        nodes = []
        for node in config.nodes:
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

        pipeline_config = PipelineConfig(
            pipeline_id=config.pipeline_id, nodes=nodes, metadata=config.metadata
        )

        # 3. Run Engine
        # SmartCatalog uses sync ORM (`self.session.query()`), so we must
        # hand it a sync session — the request-scoped session here is async.
        if db_engine.sync_session_factory is None:
            raise SkyulfException(message="Database not initialized")

        sync_session = db_engine.sync_session_factory()

        # Partition the pipeline for preview. We want one preview tab per
        # logical experiment, mirroring "Run All Experiments". Strategy:
        #   1. Use partition_parallel_pipeline() — gives one sub per
        #      (training_node × input_branch) combination. This handles the
        #      case where one data node feeds multiple training nodes.
        #   2. If no training nodes exist, fall back to partition_for_preview()
        #      which splits by data leaves so parallel preprocessing chains
        #      still get separate tabs.
        # Either way, training/tuning nodes are stripped before execution
        # because preview never fits models.
        from backend.ml_pipeline._execution.graph_utils import (
            partition_for_preview,
            partition_parallel_pipeline,
        )

        training_types = {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}
        has_training = any(n.step_type in training_types for n in nodes)

        # DataPreview is a separate evaluation node with its own background
        # job — it must NOT appear as a Preview Results tab. Filter it out
        # at every stage so neither partition_parallel_pipeline (which treats
        # it as a parallel terminal) nor partition_for_preview (which treats
        # it as a data leaf) can leak it into the tab list.
        def _is_data_preview_sub(sub: PipelineConfig) -> bool:
            return bool(sub.nodes) and sub.nodes[-1].step_type == "data_preview"

        if has_training:
            training_subs = partition_parallel_pipeline(pipeline_config)

            def _strip(sub: PipelineConfig) -> PipelineConfig:
                stripped = [
                    n
                    for n in sub.nodes
                    if n.step_type not in training_types and n.step_type != "data_preview"
                ]
                return PipelineConfig(
                    pipeline_id=sub.pipeline_id,
                    nodes=stripped,
                    metadata=sub.metadata,
                )

            paired_subs = [
                (orig, _strip(orig)) for orig in training_subs if not _is_data_preview_sub(orig)
            ]

            # Also include preview-only branches (data leaves that don't feed
            # any training terminal). Without this, a canvas mixing a training
            # pipeline with one or more dangling preprocessing chains would
            # silently drop the dangling chains from Run Preview — users only
            # saw the training branch's data and assumed the others vanished.
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
        else:
            preview_subs = partition_for_preview(pipeline_config)
            # No training → no separate label source; reuse the runnable sub.
            paired_subs = [(sub, sub) for sub in preview_subs if not _is_data_preview_sub(sub)]

        try:
            catalog = create_catalog_from_options(
                resolved_s3_options, config.nodes, session=sync_session
            )
            engine = PipelineEngine(artifact_store, catalog=catalog)
            # Sort paired_subs by BFS position of their terminal node so branch
            # letters (A, B, C, …) match the canvas edge colors from useBranchColors.
            node_bfs_pos = {n.node_id: i for i, n in enumerate(pipeline_config.nodes)}
            paired_subs.sort(
                key=lambda pair: node_bfs_pos.get(
                    pair[0].nodes[-1].node_id if pair[0].nodes else "",
                    len(pipeline_config.nodes),
                )
            )
            # Single shared artifact store across branches deduplicates work
            # for ancestor nodes that appear in multiple sub-pipelines.
            sub_results = [(orig, runnable, engine.run(runnable)) for orig, runnable in paired_subs]
        finally:
            sync_session.close()

        # ---- Preview extraction helpers ----

        def to_records(df):
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

        def count_rows(df) -> int:
            # True total row count for the underlying artifact, regardless of
            # the 50-row preview cap applied by `to_records`. Used so the UI
            # can show the real dataset size in tab badges.
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

        def process_xy(xy_tuple, prefix):
            X, y = xy_tuple
            return {f"{prefix}_X": to_records(X), f"{prefix}_y": to_records(y)}

        def process_xy_totals(xy_tuple, prefix):
            X, y = xy_tuple
            return {f"{prefix}_X": count_rows(X), f"{prefix}_y": count_rows(y)}

        def to_pandas_safe(df):
            if isinstance(df, pd.DataFrame):
                return df
            try:
                import polars as pl

                if isinstance(df, pl.DataFrame):
                    return df.to_pandas()
            except ImportError:
                pass
            return None

        def pick_target_node_id(node_list) -> Optional[str]:
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
            if target.step_type in [
                StepType.BASIC_TRAINING,
                StepType.ADVANCED_TUNING,
            ]:
                if target.inputs:
                    target_id = target.inputs[0]
            return target_id

        def extract_preview(target_node_id: Optional[str]):
            """Return (preview_data, totals, df_for_analysis) for a single artifact.

            `totals` mirrors the shape of `preview_data`:
            - dict of split-name → row count when `preview_data` is a dict;
            - a single int wrapped under the `_total` key when `preview_data`
              is a list (single-frame case).
            """
            preview_data: Any = {}
            totals: Dict[str, int] = {}
            df_for_analysis = None
            if not target_node_id or not artifact_store.exists(target_node_id):
                return preview_data, totals, df_for_analysis

            artifact = artifact_store.load(target_node_id)
            logger.debug(f"Loaded artifact for node {target_node_id}. Type: {type(artifact)}")

            is_polars = False
            try:
                import polars as pl

                if isinstance(artifact, pl.DataFrame):
                    is_polars = True
            except ImportError:
                pass

            if is_polars:
                preview_data = to_records(artifact)
                totals = {"_total": count_rows(artifact)}
                df_for_analysis = to_pandas_safe(artifact)
            elif isinstance(artifact, pd.DataFrame):
                preview_data = json.loads(artifact.head(50).to_json(orient="records"))
                totals = {"_total": count_rows(artifact)}
                df_for_analysis = artifact
            elif isinstance(artifact, SplitDataset):
                preview_data = {}
                if isinstance(artifact.train, tuple):
                    preview_data.update(process_xy(artifact.train, "train"))
                    totals.update(process_xy_totals(artifact.train, "train"))
                    df_for_analysis = to_pandas_safe(artifact.train[0])
                else:
                    preview_data["train"] = to_records(artifact.train)
                    totals["train"] = count_rows(artifact.train)
                    df_for_analysis = to_pandas_safe(artifact.train)
                if isinstance(artifact.test, tuple):
                    preview_data.update(process_xy(artifact.test, "test"))
                    totals.update(process_xy_totals(artifact.test, "test"))
                else:
                    preview_data["test"] = to_records(artifact.test)
                    totals["test"] = count_rows(artifact.test)
                if artifact.validation is not None:
                    if isinstance(artifact.validation, tuple):
                        preview_data.update(process_xy(artifact.validation, "validation"))
                        totals.update(process_xy_totals(artifact.validation, "validation"))
                    else:
                        preview_data["validation"] = to_records(artifact.validation)
                        totals["validation"] = count_rows(artifact.validation)
            elif isinstance(artifact, tuple) and len(artifact) == 2:
                X, y = artifact
                preview_data = {"X": to_records(X), "y": to_records(y)}
                totals = {"X": count_rows(X), "y": count_rows(y)}
                df_for_analysis = to_pandas_safe(X)
            elif (
                isinstance(artifact, dict)
                and "train" in artifact
                and isinstance(artifact["train"], tuple)
            ):
                preview_data = {}
                preview_data.update(process_xy(artifact["train"], "train"))
                totals.update(process_xy_totals(artifact["train"], "train"))
                df_for_analysis = to_pandas_safe(artifact["train"][0])
                if "test" in artifact:
                    preview_data.update(process_xy(artifact["test"], "test"))
                    totals.update(process_xy_totals(artifact["test"], "test"))
                if "validation" in artifact:
                    preview_data.update(process_xy(artifact["validation"], "validation"))
                    totals.update(process_xy_totals(artifact["validation"], "validation"))

            return preview_data, totals, df_for_analysis

        # 4. Aggregate per-branch previews
        preview_data: Any = {}
        preview_totals: Dict[str, int] = {}
        branch_previews: Optional[Dict[str, Any]] = None
        branch_preview_totals: Optional[Dict[str, Dict[str, int]]] = None
        branch_node_ids: Optional[Dict[str, List[str]]] = None
        recommendations: List[Recommendation] = []
        combined_node_results: Dict[str, Any] = {}
        # Aggregate status: failed > running > success
        agg_status = "success"
        first_pdf = None

        is_multi = len(sub_results) > 1
        if is_multi:
            branch_previews = {}
            branch_preview_totals = {}
            branch_node_ids = {}

        # Pre-compute "#N" disambiguators for training terminals that share
        # the same model_type (mirrors the canvas edge label scheme in
        # useBranchColors). The suffix is keyed on the *terminal node_id*,
        # not the branch index — so two branches feeding the same XGBoost
        # terminal both render as "#1", and a second XGBoost terminal (with
        # any number of branches) becomes "#2". Without this, parallel
        # branches of the same terminal incorrectly got "#1" / "#2" / "#3"
        # despite originating from the same training node on the canvas.
        # Preview-only branches (no model_type) skip this and stay unsuffixed.
        terminals_by_model: Dict[str, List[str]] = {}
        terminal_id_per_branch: Dict[int, str] = {}
        for i, (orig_sub, _runnable, _res) in enumerate(sub_results):
            mt = ""
            term_id = ""
            if orig_sub.nodes:
                leaf = orig_sub.nodes[-1]
                if leaf.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}:
                    # Training terminal: group by model_type.
                    mt = str(leaf.params.get("model_type") or leaf.params.get("algorithm") or "")
                    term_id = leaf.node_id
                else:
                    # Non-training terminal (e.g. data_preview): group by
                    # step_type so multiple Data Preview nodes get #1/#2/#3,
                    # matching the canvas dup-suffix logic in useBranchColors.
                    mt = str(leaf.step_type)
                    term_id = leaf.node_id
            if mt and term_id:
                terminal_id_per_branch[i] = term_id
                ids = terminals_by_model.setdefault(mt, [])
                if term_id not in ids:
                    ids.append(term_id)
        terminal_suffix: Dict[str, str] = {}
        for _mt, ids in terminals_by_model.items():
            if len(ids) < 2:
                continue
            for n_, term_id in enumerate(ids, start=1):
                terminal_suffix[term_id] = f"#{n_}"
        dup_suffix_by_branch: Dict[int, str] = {
            branch_idx: terminal_suffix[term_id]
            for branch_idx, term_id in terminal_id_per_branch.items()
            if term_id in terminal_suffix
        }

        for idx, (orig_sub, runnable_sub, sub_result) in enumerate(sub_results):
            for k, v in sub_result.node_results.items():
                combined_node_results[k] = v.__dict__
            if sub_result.status != "success":
                agg_status = sub_result.status
            if sub_result.status == "success" and runnable_sub.nodes:
                target_id = pick_target_node_id(runnable_sub.nodes)
                pdata, ptotals, pdf = extract_preview(target_id)
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

        # 5. Recommendations from first branch's data only (avoid heavy work).
        if first_pdf is not None:
            try:
                profile = DataProfiler.generate_profile(first_pdf)
                advisor = AdvisorEngine()
                recommendations = advisor.analyze(profile)
            except Exception as e:
                logger.warning("Error generating recommendations: %s", e)

        # Dedup merge warnings across branches: when N parallel sub-pipelines
        # run independently (Path A / Path B), each engine emits the same
        # advisory for the shared upstream merge node. Collapse duplicates so
        # the UI shows one row per logically distinct merge.
        all_warnings = [
            w for _orig, _runnable, res in sub_results for w in getattr(res, "merge_warnings", [])
        ]
        seen_warning_keys: set = set()
        deduped_warnings: List[Dict[str, Any]] = []
        for w in all_warnings:
            key = (
                w.get("node_id"),
                w.get("kind"),
                tuple(sorted(w.get("inputs") or ())),
                tuple(w.get("overlap_columns") or ()),
                tuple(w.get("dropped_columns") or ()),
                w.get("part"),
            )
            if key in seen_warning_keys:
                continue
            seen_warning_keys.add(key)
            deduped_warnings.append(w)

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
        )

    except Exception:
        logger.exception("Pipeline preview failed")
        raise SkyulfException(message="Pipeline preview failed")
    finally:
        # 5. Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = ["router"]
