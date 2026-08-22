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
  // HashEncoder's bucket occupancy also depends on the fitted rows)
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
  // Distribution transforms
  'PowerTransformer',
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
}

/** Restore the bundled fallback gate lists (e.g. after a failed fetch). */
export function resetLeakageFlags(): void {
  DATA_DEPENDENT_FIT_STEP_TYPES.clear();
  TRAIN_TEST_SPLIT_STEP_TYPES.clear();
  for (const id of BUNDLED_DATA_DEPENDENT_FIT_STEP_TYPES) DATA_DEPENDENT_FIT_STEP_TYPES.add(id);
  for (const id of BUNDLED_TRAIN_TEST_SPLIT_STEP_TYPES) TRAIN_TEST_SPLIT_STEP_TYPES.add(id);
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

/** Finds the pipeline's configured target column name, if any node declares one. */
function findTargetColumn(nodes: NodeConfigModel[]): string | undefined {
  for (const n of nodes) {
    if (TARGET_COLUMN_SOURCE_STEP_TYPES.has(n.step_type)) {
      const targetColumn = n.params.target_column;
      if (typeof targetColumn === 'string' && targetColumn) return targetColumn;
    }
  }
  return undefined;
}

/**
 * True if a Label/Ordinal encoder node is configured to encode *only* the
 * target column (y), with no feature columns. Mirrors the backend's
 * `_is_target_only_encoding` (see
 * `backend/ml_pipeline/_execution/_leakage_validation.py`) — the node fits
 * only on `y` (a deterministic category->integer mapping, not a leakage
 * risk before the train/test split) when its `columns` param is
 * empty/missing, OR when `columns` names exactly the target column (users
 * commonly pick the target explicitly from the column picker rather than
 * leaving it blank). Keep in sync with the backend check.
 */
export function isTargetOnlyEncoding(
  stepType: string,
  params: Record<string, unknown>,
  targetColumn: string | undefined,
): boolean {
  if (!TARGET_CAPABLE_ENCODER_STEP_TYPES.has(stepType)) return false;
  const columns = params.columns;
  if (!columns || (Array.isArray(columns) && columns.length === 0)) return true;
  return (
    !!targetColumn && Array.isArray(columns) && columns.length === 1 && columns[0] === targetColumn
  );
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

  const issues: LeakageIssue[] = [];
  const targetColumn = findTargetColumn(nodes);
  for (const n of nodes) {
    if (!DATA_DEPENDENT_FIT_STEP_TYPES.has(n.step_type)) continue;
    if (isTargetOnlyEncoding(n.step_type, n.params, targetColumn)) continue;
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
