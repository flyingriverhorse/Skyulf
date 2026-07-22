import { NodeConfigModel } from '../api/client';

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
 * instead of waiting for a round trip + job failure. Keep this list in
 * sync with `DATA_DEPENDENT_FIT_STEP_TYPES` in the backend module above.
 */
export const DATA_DEPENDENT_FIT_STEP_TYPES = new Set<string>([
  // Imputation
  'SimpleImputer',
  'KNNImputer',
  'IterativeImputer',
  // Scaling
  'StandardScaler',
  'MinMaxScaler',
  'RobustScaler',
  'MaxAbsScaler',
  // Encoding (category vocabulary / frequency / target statistics)
  'OneHotEncoder',
  'LabelEncoder',
  'OrdinalEncoder',
  'DummyEncoder',
  'TargetEncoder',
  'WOEEncoder',
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
]);

// `feature_target_split` is deliberately excluded — it only separates
// features (X) from the target (y) and creates no train/test boundary.
export const TRAIN_TEST_SPLIT_STEP_TYPES = new Set<string>(['TrainTestSplitter', 'Split']);

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
  for (const n of nodes) {
    if (!DATA_DEPENDENT_FIT_STEP_TYPES.has(n.step_type)) continue;
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
