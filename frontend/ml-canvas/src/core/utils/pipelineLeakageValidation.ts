import { NodeConfigModel, PipelineConfigModel } from '../api/client';
import { toast } from '../toast';

/**
 * Client-side mirror of the backend's pre-execution leakage guard
 * (`backend/ml_pipeline/_execution/_leakage_validation.py`).
 *
 * A canvas pipeline is a user-built DAG with no enforced node order, so
 * nothing stops a data-dependent preprocessing node (e.g. a
 * `StandardScaler` or `SimpleImputer`) from being wired *upstream* of a
 * `TrainTestSplitter` node. When that happens, the transformer fits its
 * statistics (mean/std, learned categories, medians, thresholds, ...) on
 * the *entire* dataset — train and test combined — before the split even
 * happens, contaminating the test-set evaluation.
 *
 * The backend already hard-blocks this at execution time, but surfacing
 * the same check here means the user gets instant feedback on the canvas
 * instead of waiting for a round trip + job failure.
 *
 * Single source of truth: at startup the app fetches the node registry
 * (`GET /api/pipeline/registry`, which carries each node's
 * `learns_from_data` / `is_splitter` flags straight from the skyulf-core
 * `@node_meta` declarations) and calls `applyRegistryLeakageFlags` to
 * replace the gate lists below. The hardcoded lists are only a bundled
 * fallback used until that fetch lands (or if it fails), so they should
 * stay a reasonable snapshot of the registry rather than being curated.
 */
const BUNDLED_DATA_DEPENDENT_FIT_STEP_TYPES: readonly string[] = [
  // Imputation
  'SimpleImputer',
  'KNNImputer',
  'IterativeImputer',
  // Scaling
  'StandardScaler',
  'MinMaxScaler',
  'RobustScaler',
  'MaxAbsScaler',
  // Encoding (category vocabulary / frequency / target statistics;
  // HashEncoder only when it auto-detects its columns — an explicit
  // column list makes it stateless, see isExplicitHashEncoding)
  'OneHotEncoder',
  'LabelEncoder',
  'OrdinalEncoder',
  'DummyEncoder',
  'TargetEncoder',
  'WOEEncoder',
  'HashEncoder',
  // Outlier detection
  'IQR',
  'ZScore',
  'Winsorize',
  'EllipticEnvelope',
  // Feature selection
  'VarianceThreshold',
  'CorrelationThreshold',
  'UnivariateSelection',
  'ModelBasedSelection',
  'feature_selection',
  // Bucketing/binning (data-derived edges)
  'GeneralBinning',
  'EqualWidthBinning',
  'EqualFrequencyBinning',
  'KBinsDiscretizer',
  'CustomBinning',
  // Distribution transforms
  'PowerTransformer',
  'GeneralTransformation',
  // Operation-sensitive nodes: fixed rules are exempted below.
  'FeatureGeneration',
  'FeatureMath',
  'FeatureGenerationNode',
  'Casting',
  // Text vectorization (vocabulary/IDF learned from the corpus)
  'count_vectorizer',
  'tfidf_vectorizer',
  // Missingness / dedup / resampling — each learns from the fitted rows:
  // which columns carry missing values, which columns to drop, the
  // duplicate set, and the resampled row distribution respectively.
  'MissingIndicator',
  'DropMissingColumns',
  'Deduplicate',
  'Oversampling',
  'Undersampling',
];

// `feature_target_split` is deliberately excluded — it only separates
// features (X) from the target (y) and creates no train/test boundary.
const BUNDLED_TRAIN_TEST_SPLIT_STEP_TYPES: readonly string[] = ['TrainTestSplitter', 'Split'];

// Live gate lists, seeded from the bundled fallback above. Set identity is
// stable — `applyRegistryLeakageFlags` mutates them in place so every
// consumer sees the backend-provided flags once they arrive.
export const DATA_DEPENDENT_FIT_STEP_TYPES = new Set<string>(
  BUNDLED_DATA_DEPENDENT_FIT_STEP_TYPES,
);
export const TRAIN_TEST_SPLIT_STEP_TYPES = new Set<string>(
  BUNDLED_TRAIN_TEST_SPLIT_STEP_TYPES,
);

let leakageFlagsRevision = 0;
const leakageFlagListeners = new Set<() => void>();
export const getLeakageFlagsRevision = (): number => leakageFlagsRevision;
export function subscribeLeakageFlags(listener: () => void): () => void {
  leakageFlagListeners.add(listener);
  return () => { leakageFlagListeners.delete(listener); };
}

function notifyLeakageFlagsChanged(): void {
  leakageFlagsRevision += 1;
  leakageFlagListeners.forEach(listener => listener());
}

export interface RegistryLeakageFlags {
  id: string;
  learns_from_data?: boolean;
  is_splitter?: boolean;
  aliases?: string[];
}

/**
 * Replace the gate lists with the flags served by the backend node
 * registry (`GET /api/pipeline/registry`), the single source of truth —
 * each node declares `learns_from_data` / `is_splitter` on its
 * `@node_meta` in skyulf-core, so a reclassified node reaches the canvas
 * without any code change here. Aliases (extra registration names for the
 * same node, e.g. 'Split' for 'TrainTestSplitter') are gated under every
 * spelling, since saved graphs may use any of them. An empty payload keeps
 * the bundled fallback rather than silently disabling the gate.
 */
export function applyRegistryLeakageFlags(items: readonly RegistryLeakageFlags[]): void {
  if (items.length === 0) return;
  DATA_DEPENDENT_FIT_STEP_TYPES.clear();
  TRAIN_TEST_SPLIT_STEP_TYPES.clear();
  for (const item of items) {
    const names = [item.id, ...(item.aliases ?? [])];
    if (item.learns_from_data) names.forEach((n) => DATA_DEPENDENT_FIT_STEP_TYPES.add(n));
    if (item.is_splitter) names.forEach((n) => TRAIN_TEST_SPLIT_STEP_TYPES.add(n));
  }
  notifyLeakageFlagsChanged();
}

/** Restore the bundled fallback gate lists (e.g. after a failed fetch). */
export function resetLeakageFlags(): void {
  DATA_DEPENDENT_FIT_STEP_TYPES.clear();
  TRAIN_TEST_SPLIT_STEP_TYPES.clear();
  for (const id of BUNDLED_DATA_DEPENDENT_FIT_STEP_TYPES) DATA_DEPENDENT_FIT_STEP_TYPES.add(id);
  for (const id of BUNDLED_TRAIN_TEST_SPLIT_STEP_TYPES) TRAIN_TEST_SPLIT_STEP_TYPES.add(id);
  notifyLeakageFlagsChanged();
}

// Encoder step types that can operate purely on the target column (y)
// instead of feature columns, depending on their config.
export const TARGET_CAPABLE_ENCODER_STEP_TYPES = new Set<string>(['LabelEncoder', 'OrdinalEncoder']);

// Step types whose params carry the pipeline's target column name.
const TARGET_COLUMN_SOURCE_STEP_TYPES = new Set<string>([
  'train_test_split',
  'TrainTestSplitter',
  'Split',
  'feature_target_split',
  'training',
]);

/** Use only unambiguous target context from this node's execution lineage. */
function findTargetColumn(nodes: NodeConfigModel[], relatedIds: ReadonlySet<string>): string | undefined {
  const targets = new Set<string>();
  for (const node of nodes) {
    if (!relatedIds.has(node.node_id) || !TARGET_COLUMN_SOURCE_STEP_TYPES.has(node.step_type)) continue;
    const value = node.params.target_column;
    if (typeof value === 'string' && value) targets.add(value);
  }
  return targets.size === 1 ? [...targets][0] : undefined;
}

/**
 * True if a Label/Ordinal encoder node is configured to encode *only* the
 * target column (y), with no feature columns. Mirrors the backend's
 * `_is_target_only_encoding` (see
 * `backend/ml_pipeline/_execution/_leakage_validation.py`) — the node fits
 * only on `y` (a deterministic category->integer mapping, not a leakage
 * risk before the train/test split) when `columns` is explicitly empty or
 * names exactly the target. Only LabelEncoder treats omitted/null columns
 * as target-only; OrdinalEncoder auto-detects feature categories in that
 * mode. Keep in sync with the backend check.
 */
export function isTargetOnlyEncoding(
  stepType: string,
  params: Record<string, unknown>,
  targetColumn: string | undefined,
): boolean {
  if (!TARGET_CAPABLE_ENCODER_STEP_TYPES.has(stepType)) return false;
  const columns = params.columns;
  if (Array.isArray(columns) && columns.length === 0) return true;
  if (stepType === 'LabelEncoder' && columns == null) return true;
  return (
    !!targetColumn && Array.isArray(columns) && columns.length === 1 && columns[0] === targetColumn
  );
}

/**
 * True if a `DropMissingColumns` node is configured to drop only explicitly
 * named columns — a fixed user decision ("exclude this column from the
 * model"), not a learned statistic, so it is safe before the train/test
 * split. With a positive `missing_threshold` the node's fit decides WHICH
 * columns to drop from the rows it sees, and that must stay after the
 * split. Mirrors `skyulf.leakage.is_explicit_column_drop` (and the node's
 * own two-mode split in `infer_output_schema`). Keep in sync.
 */
export function isExplicitColumnDrop(stepType: string, params: Record<string, unknown>): boolean {
  if (stepType !== 'DropMissingColumns') return false;
  const raw = params.missing_threshold;
  const threshold =
    typeof raw === 'number' ? raw : typeof raw === 'string' && raw !== '' ? Number(raw) : Number.NaN;
  return !(threshold > 0);
}

/**
 * True if a `SimpleImputer` node fills with a user-fixed constant
 * (`strategy: 'constant'`) — the fill value comes from the config, not
 * from the fitted rows, so nothing is learned and it is safe before the
 * split. `mean`/`median`/`most_frequent` compute statistics from the data
 * and stay gated. Mirrors `skyulf.leakage.is_constant_imputation`.
 * Keep in sync.
 */
export function isConstantImputation(stepType: string, params: Record<string, unknown>): boolean {
  return stepType === 'SimpleImputer' && params.strategy === 'constant';
}

/**
 * True if a `MissingIndicator` node flags explicitly named columns — the
 * column list comes from the config, so nothing is learned from the rows
 * and it is safe before the split. With no explicit list the fit discovers
 * WHICH columns contain missing values from the data it sees, and that
 * must stay after the split. Mirrors
 * `skyulf.leakage.is_explicit_missing_indicator` (and the node's own
 * two-mode split in `infer_output_schema`). Keep in sync.
 */
export function isExplicitMissingIndicator(
  stepType: string,
  params: Record<string, unknown>,
): boolean {
  if (stepType !== 'MissingIndicator') return false;
  const columns = params.columns;
  return Array.isArray(columns) && columns.length > 0;
}

/**
 * True if a `HashEncoder` node operates on a user-chosen column list — the
 * hashing itself is deterministic (fixed `n_features` from the config), so
 * fit learns nothing and it is safe before the split. An explicit empty
 * list is the "nothing selected" no-op, equally safe. Only when `columns`
 * is absent does fit auto-detect WHICH columns are categorical from the
 * rows it sees, and that must stay after the split. Mirrors
 * `skyulf.leakage.is_explicit_hash_encoding` (and the node's own
 * `user_picked_no_columns` short-circuit). Keep in sync.
 */
export function isExplicitHashEncoding(stepType: string, params: Record<string, unknown>): boolean {
  return stepType === 'HashEncoder' && Array.isArray(params.columns);
}

const EMPTY_SELECTION_NOOP_STEP_TYPES = new Set([
  'OneHotEncoder', 'DummyEncoder', 'TargetEncoder', 'WOEEncoder', 'PowerTransformer',
  'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'SimpleImputer',
  'KNNImputer', 'IterativeImputer', 'GeneralBinning', 'KBinsDiscretizer', 'CustomBinning',
  'IQR', 'ZScore', 'Winsorize', 'EllipticEnvelope',
]);
const FEATURE_GENERATION_STEP_TYPES = new Set([
  'FeatureGeneration', 'FeatureMath', 'FeatureGenerationNode',
]);
const FIXED_TRANSFORMATION_METHODS = new Set([
  'log', 'sqrt', 'square_root', 'cube_root', 'reciprocal', 'square', 'exp', 'exponential',
]);
const FIXED_FEATURE_OPERATIONS = new Set(['arithmetic', 'ratio', 'similarity', 'datetime_extract']);

/** Narrow untrusted node parameters without treating arrays as operation objects. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Resolve the same casting overrides as the calculator before checking for learned categories. */
function isFixedCasting(params: Record<string, unknown>): boolean {
  const columnTypes = new Map<string, unknown>(
    isRecord(params.column_types) ? Object.entries(params.column_types) : [],
  );
  const targetType = params.target_type;
  if (Array.isArray(params.columns) && typeof targetType === 'string' && targetType) {
    for (const column of params.columns) {
      if (typeof column === 'string') columnTypes.set(column, targetType);
    }
  }
  return ![...columnTypes.values()].some(type =>
    typeof type === 'string' && ['category', 'categorical'].includes(type.toLowerCase()),
  );
}

/** Exempt parameter modes whose actual calculator only records fixed rules or performs no work. */
function isFixedOperation(
  stepType: string,
  params: Record<string, unknown>,
  targetColumn: string | undefined,
): boolean {
  if (EMPTY_SELECTION_NOOP_STEP_TYPES.has(stepType)
    && Array.isArray(params.columns) && params.columns.length === 0) return true;
  if (stepType === 'CustomBinning') return Array.isArray(params.columns);
  if (stepType === 'Casting') return isFixedCasting(params);
  if (stepType === 'count_vectorizer' || stepType === 'tfidf_vectorizer') {
    const columns = params.columns;
    // Only graph target context can grant an exemption, never a node's own hint.
    return columns == null || (Array.isArray(columns)
      && (columns.length === 0 || (!!targetColumn && columns.every(column => column === targetColumn))));
  }
  if (stepType === 'GeneralTransformation') {
    const rules = params.transformations ?? [];
    return Array.isArray(rules) && rules.every(rule =>
      isRecord(rule) && typeof rule.method === 'string' && FIXED_TRANSFORMATION_METHODS.has(rule.method),
    );
  }
  if (FEATURE_GENERATION_STEP_TYPES.has(stepType)) {
    const operations = params.operations ?? [];
    return Array.isArray(operations) && operations.every(operation => {
      if (!isRecord(operation)) return false;
      const type = operation.operation_type === undefined ? 'arithmetic' : operation.operation_type;
      return typeof type === 'string' && FIXED_FEATURE_OPERATIONS.has(type);
    });
  }
  return false;
}

export interface LeakageIssue {
  nodeId: string;
  stepType: string;
  splitterNodeId: string;
}

/**
 * Returns every data-dependent preprocessing node that can reach a
 * train/test splitter downstream (i.e. necessarily fits *before* the
 * split), or `[]` if the graph is safe (including graphs with no
 * splitter at all, e.g. inference-only pipelines).
 */
export function findPreprocessingBeforeSplitIssues(nodes: NodeConfigModel[]): LeakageIssue[] {
  const splitterIds = new Set(
    nodes.filter((n) => TRAIN_TEST_SPLIT_STEP_TYPES.has(n.step_type)).map((n) => n.node_id),
  );
  if (splitterIds.size === 0) return [];

  // Forward adjacency: `inputs` point upstream, so invert to get children.
  const children = new Map<string, string[]>();
  for (const n of nodes) children.set(n.node_id, []);
  for (const n of nodes) {
    for (const parentId of n.inputs) {
      children.get(parentId)?.push(n.node_id);
    }
  }

  const descendants = new Map<string, Set<string>>();
  const visiting = new Set<string>();

  function collect(nodeId: string): Set<string> {
    const cached = descendants.get(nodeId);
    if (cached) return cached;
    if (visiting.has(nodeId)) return new Set(); // cycle guard
    visiting.add(nodeId);
    const result = new Set<string>();
    for (const childId of children.get(nodeId) ?? []) {
      result.add(childId);
      for (const d of collect(childId)) result.add(d);
    }
    visiting.delete(nodeId);
    descendants.set(nodeId, result);
    return result;
  }

  const nodesById = new Map(nodes.map(node => [node.node_id, node]));
  const protection = new Map<string, boolean>();
  function protectedBySplit(nodeId: string, active = new Set<string>()): boolean {
    if (splitterIds.has(nodeId)) return true;
    const cached = protection.get(nodeId);
    if (cached !== undefined) return cached;
    const node = nodesById.get(nodeId);
    if (!node?.inputs.length || active.has(nodeId)) return false;
    active.add(nodeId);
    const protectedInput = node.inputs.every(parentId => protectedBySplit(parentId, active));
    active.delete(nodeId);
    protection.set(nodeId, protectedInput);
    return protectedInput;
  }

  const issues: LeakageIssue[] = [];
  for (const n of nodes) {
    if (!DATA_DEPENDENT_FIT_STEP_TYPES.has(n.step_type)) continue;
    if (protectedBySplit(n.node_id)) continue;
    const relatedIds = new Set([n.node_id, ...collect(n.node_id)]);
    const upstream = [...n.inputs];
    const visited = new Set<string>();
    while (upstream.length) {
      const parentId = upstream.pop()!;
      if (visited.has(parentId)) continue;
      visited.add(parentId);
      relatedIds.add(parentId);
      upstream.push(...(nodesById.get(parentId)?.inputs ?? []));
    }
    const targetColumn = findTargetColumn(nodes, relatedIds);
    if (isTargetOnlyEncoding(n.step_type, n.params, targetColumn)) continue;
    if (isExplicitColumnDrop(n.step_type, n.params)) continue;
    if (isConstantImputation(n.step_type, n.params)) continue;
    if (isExplicitMissingIndicator(n.step_type, n.params)) continue;
    if (isExplicitHashEncoding(n.step_type, n.params)) continue;
    if (isFixedOperation(n.step_type, n.params, targetColumn)) continue;
    const reachable = collect(n.node_id);
    const hitSplitter = [...splitterIds].find((id) => reachable.has(id));
    if (hitSplitter) {
      issues.push({ nodeId: n.node_id, stepType: n.step_type, splitterNodeId: hitSplitter });
    }
  }
  return issues;
}

/** Human-readable message matching the backend's `ValueError` wording. */
export function formatLeakageIssueMessage(issue: LeakageIssue): string {
  return (
    `Data leakage risk: node '${issue.nodeId}' (${issue.stepType}) fits on the whole ` +
    `dataset before the '${issue.splitterNodeId}' train/test split downstream. Move it ` +
    'so it runs AFTER the train/test splitter (Splitter -> Preprocessing -> Model).'
  );
}

/**
 * Shared pre-flight gate: call this before submitting ANY pipeline run
 * (preview, per-node train/tune, segmentation, etc.) so every submission
 * path gives the same instant canvas feedback instead of only the
 * backend's server-side job failure. Shows a toast and returns `true`
 * (caller should abort) if a leakage issue was found; returns `false`
 * (safe to proceed) otherwise.
 */
export function warnAndBlockOnLeakage(pipelineConfig: Pick<PipelineConfigModel, 'nodes'>): boolean {
  const issues = findPreprocessingBeforeSplitIssues(pipelineConfig.nodes);
  if (issues.length === 0) return false;
  toast.error('Data leakage risk detected', formatLeakageIssueMessage(issues[0]!));
  return true;
}
