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
the split). It also checks each training leaf independently, so a splitter
on a different branch cannot make an unprotected branch appear safe.

The node classification is **single-sourced from skyulf-core**: every node
declares ``learns_from_data`` on its ``@node_meta``, and
``skyulf.leakage.data_dependent_transformers`` /
``train_test_splitters`` derive both sets from the registry. There is no
second hand-maintained list here to drift (audit finding F-16 / gap G2).
The shared operation classifier refines conservative metadata using actual
parameters, so fixed modes and explicit empty selections are not confused
with value-dependent auto-selection. Composite operations are inspected in
execution order, and target-only exemptions use the relevant branch target.
"""

import logging
from typing import Any

from skyulf.leakage import (
    OnLeakage,
    data_dependent_transformers,
    is_target_only_encoding,
    leakage_exemption_reason,
    step_learns_from_data,
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
    """Step types whose ``.fit()`` learns parameters from the data it's given.

    Covers means/std, learned categories, medians, variance/correlation,
    quantile-based thresholds, target statistics, vocabulary/IDF, missingness
    structure, duplicate sets, etc. Fitting one of these on data that still
    includes the test/validation portion leaks that portion's information
    into the fitted parameters, even though the transformer is only ever
    *applied* to train afterward. Derived from the skyulf-core registry.
    """
    return data_dependent_transformers()


def train_test_split_step_types() -> frozenset[str]:
    """Step types that partition rows into train/test (the leakage boundary).

    ``feature_target_split`` is deliberately not one — it only separates
    features (X) from the target (y) and creates no train/test boundary, so
    preprocessing before it is not a leakage concern.
    """
    return train_test_splitters()


# Step types whose params carry the pipeline's target column name (see
# ``graph_utils.extract_job_details``, which reads the same set of step
# types to resolve ``target_column`` for training).
_TARGET_COLUMN_SOURCE_STEP_TYPES: frozenset[str] = frozenset(
    {"train_test_split", "TrainTestSplitter", "Split", "feature_target_split", "training", "tuning"}
)
_TRAINING_LEAF_STEP_TYPES: frozenset[str] = frozenset({"training", "tuning"})


def _find_target_column(nodes: list[NodeConfig]) -> str | None:
    """Resolve a target only when the relevant graph declares one unambiguous name."""
    targets = {
        n.params["target_column"]
        for n in nodes
        if n.step_type in _TARGET_COLUMN_SOURCE_STEP_TYPES and n.params.get("target_column")
    }
    return next(iter(targets)) if len(targets) == 1 else None


def _expand_composite_nodes(nodes: list[NodeConfig]) -> list[NodeConfig]:
    """Expose ordered composite operations without changing the caller's graph."""
    expanded: list[NodeConfig] = []
    used_ids = {node.node_id for node in nodes}
    for node in nodes:
        if node.step_type != "feature_engineering":
            expanded.append(node)
            continue
        inputs = list(node.inputs)
        for index, step in enumerate(node.params.get("steps", [])):
            step_id = f"{node.node_id}::step[{index}]"
            while step_id in used_ids:
                step_id += ":operation"
            used_ids.add(step_id)
            operation = NodeConfig(
                node_id=step_id,
                step_type=step.get("transformer", ""),
                params=step.get("params") or {},
                inputs=inputs,
            )
            expanded.extend(_expand_composite_nodes([operation]))
            inputs = [step_id]
        expanded.append(NodeConfig(node.node_id, "feature_engineering", params={}, inputs=inputs))
    return expanded


def _target_column_for_node(nodes: list[NodeConfig], node_id: str) -> str | None:
    """Ignore unrelated sibling targets when resolving a node's execution context."""
    descendants = _build_descendant_map(nodes)
    related = descendants.get(node_id, set()) | {node_id}
    related.update(n.node_id for n in nodes if node_id in descendants.get(n.node_id, set()))
    return _find_target_column([node for node in nodes if node.node_id in related])


def execution_target_column(nodes: list[NodeConfig], node_id: str) -> str | None:
    """Resolve the unambiguous target for a transformer, including composite steps."""
    return _target_column_for_node(_expand_composite_nodes(nodes), node_id)


def _has_splitter_on_every_input_path(
    node_id: str,
    nodes_by_id: dict[str, NodeConfig],
    splitter_ids: set[str],
    visiting: set[str] | None = None,
) -> bool:
    """Return whether every upstream path already crosses a real row partition."""
    if node_id in splitter_ids:
        return True
    visiting = set() if visiting is None else visiting
    node = nodes_by_id.get(node_id)
    if node_id in visiting or node is None or not node.inputs:
        return False
    visiting.add(node_id)
    protected = all(
        _has_splitter_on_every_input_path(parent, nodes_by_id, splitter_ids, visiting)
        for parent in node.inputs
    )
    visiting.remove(node_id)
    return protected


def _is_target_only_encoding(step_type: str, params: dict, target_column: str | None) -> bool:
    """True if a Label/Ordinal encoder node is configured to encode *only* the target.

    Encodes the target column (y), with no feature columns. Per ``skyulf-core``'s
    ``LabelEncoderCalculator``/``OrdinalEncoderCalculator``
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
    return is_target_only_encoding(step_type, params, target_column)


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


def _find_unprotected_learners(
    node: NodeConfig,
    nodes_by_id: dict[str, NodeConfig],
    splitter_ids: set[str],
    data_dependent: frozenset[str],
    target_column: str | None,
) -> list[NodeConfig]:
    """Find data-dependent ancestors reached without crossing a splitter."""
    learners: dict[str, NodeConfig] = {}
    protection: dict[str, bool] = {}

    def _has_splitter_on_every_input_path(node_id: str, visiting: set[str]) -> bool:
        """Return whether every path into a node crosses a train/test splitter."""
        if node_id in splitter_ids:
            return True
        if node_id in protection:
            return protection[node_id]
        if node_id in visiting:
            return False
        node_config = nodes_by_id.get(node_id)
        if node_config is None or not node_config.inputs:
            protection[node_id] = False
            return False
        visiting.add(node_id)
        result = all(
            _has_splitter_on_every_input_path(parent_id, visiting)
            for parent_id in node_config.inputs
        )
        visiting.remove(node_id)
        protection[node_id] = result
        return result

    pending = list(node.inputs)
    visited: set[str] = set()
    while pending:
        ancestor_id = pending.pop()
        if ancestor_id in visited:
            continue
        visited.add(ancestor_id)
        ancestor = nodes_by_id.get(ancestor_id)
        if ancestor is None:
            continue
        if (
            ancestor.step_type in data_dependent
            and step_learns_from_data(
                ancestor.step_type, ancestor.params, target_column=target_column
            )
            and not _has_splitter_on_every_input_path(ancestor_id, set())
        ):
            learners[ancestor.node_id] = ancestor
        if ancestor_id in splitter_ids:
            continue
        pending.extend(ancestor.inputs)
    return [candidate for candidate in nodes_by_id.values() if candidate.node_id in learners]


def _has_explicit_cross_validation(node: NodeConfig) -> bool:
    """Return whether a training node explicitly enables cross-validation."""
    if node.params.get("run_mode") == "tuned":
        return True
    if node.params.get("cv_enabled") is True:
        return True
    tuning_config = node.params.get("tuning_config")
    return isinstance(tuning_config, dict) and tuning_config.get("cv_enabled") is True


def _supports_unsplit_cv_refit(node: NodeConfig, nodes_by_id: dict[str, NodeConfig]) -> bool:
    """Check whether an unsplit branch supports per-fold preprocessing refitting.

    The engine can refit a linear transformer chain from its raw loader.
    Its fork/join adapter requires a shared splitter, so an unprotected
    learner behind a merge cannot use CV as a substitute for that boundary.
    Duplicate handles from one source still form a single input path.
    """
    visited: set[str] = set()
    current = node
    while current.node_id not in visited:
        visited.add(current.node_id)
        inputs = list(dict.fromkeys(current.inputs))
        if not inputs:
            return current.step_type in {"data_loader", "DataLoader"}
        if len(inputs) != 1:
            return False
        parent = nodes_by_id.get(inputs[0])
        if parent is None or parent.step_type in {"training", "tuning", "data_preview"}:
            return False
        current = parent
    return False


def _exemption_reason(step_type: str, params: dict, target_column: str | None) -> str | None:
    """Human-readable reason when a data-dependent node type is exempted, else None.

    Exemption comes from its params: a stateless configuration of a stateful node.
    """
    return leakage_exemption_reason(step_type, params, target_column=target_column)


def validate_no_preprocessing_before_split(
    nodes: list[NodeConfig],
    on_leakage: OnLeakage = "raise",
    *,
    target_node_id: str | None = None,
) -> dict[str, Any]:
    """Raise ``ValueError`` when a data-dependent node is used unsafely.

    A node "precedes" a splitter here if the splitter is reachable by
    following the graph's forward (input->output) edges from that node —
    i.e. the preprocessing node is a topological ancestor of the splitter,
    so it necessarily executes (and fits) before the split happens.

    In addition to checking preprocessing ancestors of every splitter, each
    leaf ``training``/``tuning`` node must have a splitter on its own input
    branch unless it explicitly runs cross-validation with a linear input
    chain that supports per-fold preprocessing refitting. This prevents an
    unrelated splitter elsewhere in a multi-branch graph from masking a
    full-dataset fit. ``target_node_id`` limits checks to the selected node's
    ancestors while retaining the full graph's split/no-split context.

    ``on_leakage`` selects the verdict for definite violations: ``"raise"``
    (default) blocks execution, ``"warn"`` logs the same message without
    blocking, ``"ignore"`` stays silent. A graph with no train/test splitter
    at all (e.g. inference-only pipelines) gets an explicit advisory warning
    instead of silence — the leakage guarantee simply does not apply there —
    unless ``on_leakage="ignore"``.

    Returns the gate verdict so the engine can persist it on the job record
    (Job Details shows it as factual per-job information):
    ``{"status": "passed" | "no_split" | "warnings", "messages": [...],
    "splitters": [...], "checked": [{"node_id", "step_type", "before_split",
    "violation"}, ...], "exempted": [{"node_id", "step_type", "reason"},
    ...]}``. ``checked`` lists every data-dependent node the gate examined
    and ``exempted`` the ones allowed before the split via the param-aware
    exemptions (with why) — the detail powers the Job Details verdict modal.
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

    nodes = _expand_composite_nodes(nodes)
    splitter_ids = {n.node_id for n in nodes if n.step_type in train_test_split_step_types()}
    descendants = _build_descendant_map(nodes)
    nodes_by_id = {n.node_id: n for n in nodes}
    execution_ids = (
        {
            n.node_id
            for n in nodes
            if n.node_id == target_node_id or target_node_id in descendants.get(n.node_id, set())
        }
        if target_node_id in nodes_by_id
        else set(nodes_by_id)
    )
    execution_splitters = splitter_ids & execution_ids
    execution_nodes = [n for n in nodes if n.node_id in execution_ids]
    data_dependent = data_dependent_step_types()

    checked: list[dict[str, Any]] = []
    exempted: list[dict[str, Any]] = []
    messages: list[str] = []
    for n in nodes:
        if n.node_id not in execution_ids or n.step_type not in data_dependent:
            continue
        target_column = _target_column_for_node(execution_nodes, n.node_id)
        reason = _exemption_reason(n.step_type, n.params, target_column)
        if reason:
            exempted.append({"node_id": n.node_id, "step_type": n.step_type, "reason": reason})
            continue
        leaking_splitters = descendants.get(n.node_id, set()) & execution_splitters
        violation = bool(leaking_splitters) and not _has_splitter_on_every_input_path(
            n.node_id, nodes_by_id, splitter_ids
        )
        checked.append(
            {
                "node_id": n.node_id,
                "step_type": n.step_type,
                "before_split": violation,
                "violation": violation,
            }
        )
        if violation:
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

    detail = {"splitters": sorted(execution_splitters), "checked": checked, "exempted": exempted}
    if not splitter_ids:
        if on_leakage != "ignore":
            logger.warning(NO_SPLIT_DIAGNOSTIC)
        return {"status": "no_split", "messages": [NO_SPLIT_DIAGNOSTIC], **detail}
    training_leaves = [
        n
        for n in nodes
        if n.node_id in execution_ids
        and n.step_type in _TRAINING_LEAF_STEP_TYPES
        and not any(n.node_id in other.inputs for other in nodes if other.node_id in execution_ids)
    ]
    for training_node in training_leaves:
        branch_learners = _find_unprotected_learners(
            training_node,
            nodes_by_id,
            splitter_ids,
            data_dependent,
            _target_column_for_node(execution_nodes, training_node.node_id),
        )
        if not branch_learners:
            continue
        if _has_explicit_cross_validation(training_node) and _supports_unsplit_cv_refit(
            training_node, nodes_by_id
        ):
            continue
        learner_names = ", ".join(f"'{n.node_id}' ({n.step_type})" for n in branch_learners)
        message = (
            f"Data leakage risk: training node '{training_node.node_id}' has no "
            f"train/test splitter on its input branch, so data-dependent node(s) "
            f"{learner_names} fit on the full dataset. Add a splitter before "
            "these nodes, or enable cross-validation with a linear preprocessing "
            "path that can be refitted per fold."
        )
        for learner in branch_learners:
            existing = next((item for item in checked if item["node_id"] == learner.node_id), None)
            if existing is None:
                checked.append(
                    {
                        "node_id": learner.node_id,
                        "step_type": learner.step_type,
                        "before_split": False,
                        "violation": True,
                    }
                )
            else:
                existing["violation"] = True
        if on_leakage == "raise":
            raise ValueError(message)
        if on_leakage == "warn":
            logger.warning(message)
        messages.append(message)

    if messages:
        return {"status": "warnings", "messages": messages, **detail}
    return {"status": "passed", "messages": [], **detail}
