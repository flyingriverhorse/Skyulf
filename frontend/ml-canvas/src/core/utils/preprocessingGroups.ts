import type { NodeDefinition } from '../types/nodes';

// Library organization only; execution categories and saved node types stay stable.
const groups = [
  { id: 'clean', label: 'Data cleaning', types: [
    'drop_missing_columns', 'imputation_node', 'drop_missing_rows', 'deduplicate',
    'MissingIndicator', 'outlier', 'value_replacement', 'AliasReplacement', 'InvalidValueReplacement',
  ] },
  { id: 'numeric', label: 'Numeric & categorical', types: [
    'casting', 'scale_numeric_features', 'encoding', 'TransformationNode', 'BinningNode',
  ] },
  { id: 'features', label: 'Feature engineering', types: [
    'FeatureGenerationNode', 'PolynomialFeaturesNode', 'FeatureInteractionNode', 'feature_selection', 'TimeSeriesNode',
  ] },
  { id: 'text', label: 'Text processing', types: [
    'TextCleaning', 'count_vectorizer', 'tfidf_vectorizer', 'hashing_vectorizer', 'tokenizer', 'sentence_embedder',
  ] },
  { id: 'split', label: 'Splitting & sampling', types: [
    'TrainTestSplitter', 'feature_target_split', 'ResamplingNode',
  ] },
];
const assignedTypes = new Set(groups.flatMap(group => group.types));

/** Organize visible preprocessing nodes, keeping unclassified additions reachable in a fallback group. */
export function groupPreprocessingNodes<T extends Pick<NodeDefinition, 'type' | 'category' | 'hidden'>>(nodes: readonly T[]) {
  const visible = nodes.filter(node => node.category === 'Preprocessing' && !node.hidden);
  return [
    ...groups.map(group => ({
      id: group.id, label: group.label, nodes: visible.filter(node => group.types.includes(node.type)),
    })),
    { id: 'other', label: 'Other preprocessing', nodes: visible.filter(node => !assignedTypes.has(node.type)) },
  ].filter(group => group.nodes.length > 0);
}
