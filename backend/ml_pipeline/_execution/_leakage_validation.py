"""Pre-execution validation guarding against preprocessing-before-split data leakage.

A pipeline is a user-built DAG (arbitrary node order), so nothing stops a
stateful preprocessing node (e.g. a ``StandardScaler`` or ``SimpleImputer``)
from being wired *upstream* of a ``TrainTestSplitter``/``Split`` node. When
that happens, ``StatefulTransformer._fit_transform_inner`` (see
``skyulf.preprocessing.base``) fits the transformer's statistics (mean/std,
learned categories, medians, variance thresholds, etc.) on the *entire*
dataset — train and test combined — before the split even happens. The
resulting test-set evaluation is then contaminated: it's no longer a fair
estimate of generalization to unseen data.

This module walks the pipeline's node graph and raises a clear,
actionable ``ValueError`` before execution starts if any data-dependent
preprocessing node can reach a train/test splitter downstream (i.e. it is
an ancestor of a splitter, meaning it necessarily runs and fits *before*
the split).

The node classification is **single-sourced from skyulf-core**: every node
declares ``learns_from_data`` on its ``@node_meta``, and
``skyulf.leakage.data_dependent_transformers`` /
``train_test_splitters`` derive both sets from the registry. There is no
second hand-maintained list here to drift (audit finding F-16 / gap G2).
Nodes deliberately classified ``learns_from_data=False`` include the
fixed-map replacements (ValueReplacement/AliasReplacement/
InvalidValueReplacement), Casting, user-fixed edges/bounds
(CustomBinning/ManualBounds), per-row rules (DropMissingRows, GeoDistance,
H3Index, DateFeatures, LagFeatures, RollingAggregate, feature math /
polynomial / interaction generation, TextCleaning, tokenizer,
sentence_embedder, hashing_vectorizer) and the inspection nodes.
"""

import logging
from typing import Any

from skyulf.leakage import (
    OnLeakage,
    data_dependent_transformers,
    is_constant_imputation,
    is_explicit_column_drop,
    is_explicit_hash_encoding,
    is_explicit_missing_indicator,
    train_test_splitters,
)

from .schemas import NodeConfig

logger = logging.getLogger(__name__)

_ON_LEAKAGE_MODES = frozenset({"raise", "warn", "ignore"})

NO_SPLIT_DIAGNOSTIC = (
    "No train/test split is defined in this pipeline graph, so the leakage "
    "guarantee does not apply: every fit sees the whole dataset. Add a "
    "TrainTestSplitter (or rely on cross-validation) to restore the guarantee."
)


def data_dependent_step_types() -> frozenset[str]:
    """Step types whose ``.fit()`` learns parameters from the data it's given
    (means/std, learned categories, medians, variance/correlation,
    quantile-based thresholds, target statistics, vocabulary/IDF, missingness
    structure, duplicate sets, etc.). Fitting one of these on data that still
    includes the test/validation portion leaks that portion's information
    into the fitted parameters, even though the transformer is only ever
    *applied* to train afterward. Derived from the skyulf-core registry."""
    return data_dependent_transformers()


def train_test_split_step_types() -> frozenset[str]:
    """Step types that partition rows into train/test (the leakage boundary).
    ``feature_target_split`` is deliberately not one — it only separates
    features (X) from the target (y) and creates no train/test boundary, so
    preprocessing before it is not a leakage concern."""
    return train_test_splitters()


# Encoder step types that can operate purely on the target column (y)
# instead of feature columns, depending on their config.
TARGET_CAPABLE_ENCODER_STEP_TYPES: frozenset[str] = frozenset({"LabelEncoder", "OrdinalEncoder"})


# Step types whose params carry the pipeline's target column name (see
# ``graph_utils.extract_job_details``, which reads the same set of step
# types to resolve ``target_column`` for training).
_TARGET_COLUMN_SOURCE_STEP_TYPES: frozenset[str] = frozenset(
    {"train_test_split", "TrainTestSplitter", "Split", "feature_target_split", "training"}
)


def _find_target_column(nodes: list[NodeConfig]) -> str | None:
    """Finds the pipeline's configured target column name, if any node declares one."""
    for n in nodes:
        if n.step_type in _TARGET_COLUMN_SOURCE_STEP_TYPES:
            target_column = n.params.get("target_column")
            if target_column:
                return target_column
    return None


def _is_target_only_encoding(step_type: str, params: dict, target_column: str | None) -> bool:
    """True if a Label/Ordinal encoder node is configured to encode *only* the
    target column (y), with no feature columns.

    Per ``skyulf-core``'s ``LabelEncoderCalculator``/``OrdinalEncoderCalculator``
    (see ``_maybe_fit_target``/``_should_encode_target``), the node fits
    *only* on ``y`` — never touching feature columns — when its ``columns``
    param is empty/missing, OR when ``columns`` names exactly the target
    column (users commonly pick the target explicitly from the column
    picker rather than leaving it blank). Encoding the target this way is
    not a leakage risk even when it runs before the train/test split: it's
    a deterministic category-label -> integer mapping (sklearn's
    ``LabelEncoder``/``OrdinalEncoder`` assign ids from sorted class order,
    not from any train/test-dependent statistic), and every downstream
    consumer needs the target already numeric/consistent before a split can
    even be stratified on it. This is standard practice, not test-set
    contamination.

    If ``columns`` names the target *plus* other (feature) columns, or
    names feature columns only, the node also encodes those feature columns
    by learning a vocabulary from whichever rows it sees — that part
    remains a genuine leakage risk, so this returns ``False`` and the node
    is still flagged.
    """
    if step_type not in TARGET_CAPABLE_ENCODER_STEP_TYPES:
        return False
    columns = params.get("columns")
    if not columns:
        return True
    return target_column is not None and set(columns) == {target_column}


def _build_descendant_map(nodes: list[NodeConfig]) -> dict[str, set[str]]:
    """Returns ``{node_id: {ids reachable by following outgoing/forward edges}}``.

    Built with a single reverse-topological accumulation pass (each node's
    descendant set is the union of its direct children's descendant sets,
    plus the children themselves) rather than a BFS/DFS per node, so the
    whole map is O(nodes + edges) instead of O(n^2) in the worst case.
    """
    children: dict[str, list[str]] = {n.node_id: [] for n in nodes}
    for n in nodes:
        for parent_id in n.inputs:
            if parent_id in children:
                children[parent_id].append(n.node_id)

    descendants: dict[str, set[str]] = {}

    def _collect(node_id: str, visiting: set[str]) -> set[str]:
        if node_id in descendants:
            return descendants[node_id]
        if node_id in visiting:
            # Cycle — bail out gracefully; cycles are an unrelated
            # validation concern handled elsewhere (topological sort).
            return set()
        visiting.add(node_id)
        result: set[str] = set()
        for child_id in children.get(node_id, []):
            result.add(child_id)
            result |= _collect(child_id, visiting)
        visiting.discard(node_id)
        descendants[node_id] = result
        return result

    for n in nodes:
        _collect(n.node_id, set())
    return descendants


def validate_no_preprocessing_before_split(
    nodes: list[NodeConfig], on_leakage: OnLeakage = "raise"
) -> dict[str, Any]:
    """Raises ``ValueError`` if a data-dependent preprocessing node precedes a splitter.

    A node "precedes" a splitter here if the splitter is reachable by
    following the graph's forward (input->output) edges from that node —
    i.e. the preprocessing node is a topological ancestor of the splitter,
    so it necessarily executes (and fits) before the split happens.

    ``on_leakage`` selects the verdict for definite violations: ``"raise"``
    (default) blocks execution, ``"warn"`` logs the same message without
    blocking, ``"ignore"`` stays silent. A graph with no train/test splitter
    at all (e.g. inference-only pipelines) gets an explicit advisory warning
    instead of silence — the leakage guarantee simply does not apply there —
    unless ``on_leakage="ignore"``.

    Returns the gate verdict so the engine can persist it on the job record
    (Job Details shows it as factual per-job information):
    ``{"status": "passed" | "no_split" | "warnings", "messages": [...]}``.
    The verdict reflects the graph analysis regardless of ``on_leakage``;
    the mode only controls whether violations raise or log. Under ``"raise"``
    a violating graph never returns — it raises first.

    Step types unknown to the skyulf-core registry (backend infrastructure
    such as data loaders, trainers and evaluators) are skipped: every real
    preprocessing node is registered there, and the required
    ``learns_from_data`` declaration makes it impossible for one to be
    silently omitted from the data-dependent set.
    """
    if on_leakage not in _ON_LEAKAGE_MODES:
        raise ValueError(
            f"on_leakage must be one of {sorted(_ON_LEAKAGE_MODES)}, got {on_leakage!r}"
        )

    splitter_ids = {n.node_id for n in nodes if n.step_type in train_test_split_step_types()}
    if not splitter_ids:
        if on_leakage != "ignore":
            logger.warning(NO_SPLIT_DIAGNOSTIC)
        return {"status": "no_split", "messages": [NO_SPLIT_DIAGNOSTIC]}

    descendants = _build_descendant_map(nodes)
    target_column = _find_target_column(nodes)
    data_dependent = data_dependent_step_types()

    messages: list[str] = []
    for n in nodes:
        if n.step_type not in data_dependent:
            continue
        if _is_target_only_encoding(n.step_type, n.params, target_column):
            continue
        if is_explicit_column_drop(n.step_type, n.params):
            continue
        if is_constant_imputation(n.step_type, n.params):
            continue
        if is_explicit_missing_indicator(n.step_type, n.params):
            continue
        if is_explicit_hash_encoding(n.step_type, n.params):
            continue
        leaking_splitters = descendants.get(n.node_id, set()) & splitter_ids
        if leaking_splitters:
            splitter_name = sorted(leaking_splitters)[0]
            message = (
                f"Data leakage risk: node '{n.node_id}' ({n.step_type}) fits on "
                f"the whole dataset before the '{splitter_name}' train/test split "
                "downstream, so its learned statistics (e.g. mean/std, learned "
                "categories, medians, thresholds) are computed using test data "
                "too. Move this node so it runs AFTER the train/test splitter "
                "(e.g. Splitter -> Preprocessing -> Model), or use a "
                "FeatureTargetSplitter before it if you only need to separate "
                "the target column (that does not create a train/test boundary)."
            )
            if on_leakage == "raise":
                raise ValueError(message)
            if on_leakage == "warn":
                logger.warning(message)
            messages.append(message)

    if messages:
        return {"status": "warnings", "messages": messages}
    return {"status": "passed", "messages": []}
