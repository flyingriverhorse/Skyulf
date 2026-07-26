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
"""

from .schemas import NodeConfig

# Step types whose ``.fit()`` learns parameters from the statistical
# distribution of the data it's given (means/std, learned categories,
# medians, variance/correlation, quantile-based thresholds, target
# statistics, vocabulary/IDF, etc.). Fitting one of these on data that
# still includes the test/validation portion leaks that portion's
# information into the fitted parameters, even though the transformer is
# only ever *applied* to train afterward.
#
# Deliberately excluded (stateless / rule-based — safe regardless of
# graph position): HashEncoder (deterministic hash, no learned vocabulary),
# CustomBinning/ManualBounds (user-fixed edges/bounds, not data-derived),
# ValueReplacement/AliasReplacement/InvalidValueReplacement (fixed
# replacement maps), Casting, DropMissingColumns/Rows, Deduplicate,
# MissingIndicator, GeoDistance, H3Index, DateFeatures, RollingAggregate,
# LagFeatures, FeatureMath/FeatureGenerationNode/FeatureInteraction,
# PolynomialFeatures, TextCleaning, tokenizer, sentence_embedder,
# hashing_vectorizer.
DATA_DEPENDENT_FIT_STEP_TYPES: frozenset[str] = frozenset(
    {
        # Imputation — mean/median/most-frequent/KNN-neighbor/iterative-model
        # statistics learned from the data.
        "SimpleImputer",
        "KNNImputer",
        "IterativeImputer",
        # Scaling — mean/std/min/max/median/IQR learned from the data.
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "MaxAbsScaler",
        # Encoding — category vocabulary, frequency, or target statistics
        # learned from the data.
        "OneHotEncoder",
        "LabelEncoder",
        "OrdinalEncoder",
        "DummyEncoder",
        "TargetEncoder",
        "WOEEncoder",
        # Outlier detection — thresholds/covariance learned from the data's
        # own distribution.
        "IQR",
        "ZScore",
        "Winsorize",
        "EllipticEnvelope",
        # Feature selection — selects/drops features using data statistics
        # (variance, correlation, univariate score, model importance).
        "VarianceThreshold",
        "CorrelationThreshold",
        "UnivariateSelection",
        "ModelBasedSelection",
        "feature_selection",
        # Bucketing/binning — bin edges learned from the data's own
        # quantiles/distribution (not user-fixed).
        "GeneralBinning",
        "EqualWidthBinning",
        "EqualFrequencyBinning",
        "KBinsDiscretizer",
        # Distribution transforms — optimal lambda learned from the data.
        "PowerTransformer",
        # Text vectorization — vocabulary/IDF learned from the training
        # corpus.
        "count_vectorizer",
        "tfidf_vectorizer",
    }
)

# Step types that partition rows into train/test (the leakage boundary).
# "feature_target_split" is deliberately excluded — it only separates
# features (X) from the target (y) and creates no train/test boundary, so
# preprocessing before it is not a leakage concern.
TRAIN_TEST_SPLIT_STEP_TYPES: frozenset[str] = frozenset({"TrainTestSplitter", "Split"})

# Encoder step types that can operate purely on the target column (y)
# instead of feature columns, depending on their config.
TARGET_CAPABLE_ENCODER_STEP_TYPES: frozenset[str] = frozenset({"LabelEncoder", "OrdinalEncoder"})


def _is_target_only_encoding(step_type: str, params: dict) -> bool:
    """True if a Label/Ordinal encoder node is configured to encode *only* the
    target column (y), with no feature ``columns`` selected.

    Per ``skyulf-core``'s ``LabelEncoderCalculator``/``OrdinalEncoderCalculator``
    (see ``_maybe_fit_target``/``_ordinal_fit_no_columns``), an empty/missing
    ``columns`` param means the node fits *only* on ``y`` — it never touches
    feature columns at all. Encoding the target this way is not a leakage risk
    even when it runs before the train/test split: it's a deterministic
    category-label -> integer mapping (sklearn's ``LabelEncoder``/
    ``OrdinalEncoder`` assign ids from sorted class order, not from any
    train/test-dependent statistic), and every downstream consumer needs the
    target already numeric/consistent before a split can even be stratified
    on it. This is standard practice, not test-set contamination.

    If ``columns`` IS populated, the node also (or instead) encodes those
    feature columns by learning a vocabulary from whichever rows it sees —
    that part remains a genuine leakage risk and is not exempted here.
    """
    if step_type not in TARGET_CAPABLE_ENCODER_STEP_TYPES:
        return False
    return not params.get("columns")


def _build_descendant_map(nodes: list[NodeConfig]) -> dict[str, set[str]]:
    """Returns ``{node_id: {ids reachable by following outgoing/forward edges}}``.

    Built with a single reverse-topological accumulation pass (each node's
    descendant set is the union of its direct children's descendant sets,
    plus the children themselves) rather than a BFS/DFS per node, so the
    whole map is O(nodes + edges) instead of O(nodes^2) in the worst case.
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


def validate_no_preprocessing_before_split(nodes: list[NodeConfig]) -> None:
    """Raises ``ValueError`` if a data-dependent preprocessing node precedes a splitter.

    A node "precedes" a splitter here if the splitter is reachable by
    following the graph's forward (input->output) edges from that node —
    i.e. the preprocessing node is a topological ancestor of the splitter,
    so it necessarily executes (and fits) before the split happens.

    No-op if the graph has no train/test splitter node at all (e.g.
    inference-only pipelines, or pipelines that never split), since there's
    no train/test boundary to leak across.
    """
    splitter_ids = {n.node_id for n in nodes if n.step_type in TRAIN_TEST_SPLIT_STEP_TYPES}
    if not splitter_ids:
        return

    descendants = _build_descendant_map(nodes)

    for n in nodes:
        if n.step_type not in DATA_DEPENDENT_FIT_STEP_TYPES:
            continue
        if _is_target_only_encoding(n.step_type, n.params):
            continue
        leaking_splitters = descendants.get(n.node_id, set()) & splitter_ids
        if leaking_splitters:
            splitter_name = sorted(leaking_splitters)[0]
            raise ValueError(
                f"Data leakage risk: node '{n.node_id}' ({n.step_type}) fits on "
                f"the whole dataset before the '{splitter_name}' train/test split "
                "downstream, so its learned statistics (e.g. mean/std, learned "
                "categories, medians, thresholds) are computed using test data "
                "too. Move this node so it runs AFTER the train/test splitter "
                "(e.g. Splitter -> Preprocessing -> Model), or use a "
                "FeatureTargetSplitter before it if you only need to separate "
                "the target column (that does not create a train/test boundary)."
            )
