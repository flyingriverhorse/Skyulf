import type { NodeDefinition } from '../types/nodes';

type SearchableNode = Pick<NodeDefinition, 'type' | 'label' | 'category' | 'description' | 'hidden'>;

// Discovery vocabulary only: keep node IDs, saved configs, and execution unchanged.
const taskTerms: Readonly<Record<string, string>> = {
  scale_numeric_features: 'normalize normalise normalization standardize standardise standardization min max z score',
  imputation_node: 'fill blanks null nan impute missing data',
  MissingIndicator: 'flag missing data null nan blanks',
  drop_missing_rows: 'remove missing data null nan blanks',
  deduplicate: 'remove duplicates repeated rows',
  encoding: 'one hot categorical numbers dummy ordinal label encoder',
  ResamplingNode: 'imbalanced classes class imbalance smote balance classes',
  TrainTestSplitter: 'holdout validation split train test',
  feature_target_split: 'separate features target x y',
  classification: 'predict classes categories labels classify',
  regression: 'predict numbers numeric continuous values price',
  text_classification: 'classify text sentiment predict text labels',
  SegmentationNode: 'cluster clustering group similar records',
  data_preview: 'inspect data view sample rows table',
};

/** Normalize punctuation and spacing so task phrases and catalog IDs are both searchable. */
function normalize(text: string): string {
  return text.toLowerCase().replace(/[-_\s]+/g, ' ').trim();
}

/** Rank visible nodes by name, task vocabulary, and descriptive context in both pickers. */
export function searchNodes<T extends SearchableNode>(nodes: readonly T[], query: string): T[] {
  const visible = nodes.filter(node => !node.hidden);
  const q = normalize(query);
  if (!q) return visible;
  const words = q.split(' ');
  return visible.map(node => {
    const label = normalize(node.label);
    const description = normalize(node.description);
    const category = normalize(node.category);
    const type = normalize(node.type);
    const terms = taskTerms[node.type] ?? '';
    const text = `${label} ${description} ${category} ${type} ${terms}`;
    // All words must match; a second task word narrows rather than floods results.
    if (!words.every(word => text.includes(word))) return { node, score: 0 };
    const score = label === q ? 200 : label.startsWith(q) ? 150 : label.includes(q) ? 100
      : terms.includes(q) ? 60 : category.includes(q) ? 40 : type.includes(q) ? 30
      : description.includes(q) ? 20 : 10;
    return { node, score };
  }).filter(match => match.score > 0)
    .sort((a, b) => b.score - a.score)
    .map(match => match.node);
}
