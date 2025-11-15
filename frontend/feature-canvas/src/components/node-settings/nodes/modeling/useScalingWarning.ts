import { useMemo } from 'react';
import type { FeatureGraph, FeatureNodeParameterOption } from '../../../../api';

export type ScalingWarning = {
  headline: string;
  summary: string;
  details: string[];
};

const SCALING_NODE_TYPES = new Set([
  'scale_numeric_features',
  'scaling',
  'scale_numeric',
]);

const SCALING_CONVERGENCE_PATTERNS = [
  /convergencewarning/i,
  /failed to converge/i,
  /did not converge/i,
  /optimizer failed to converge/i,
  /increase the number of iterations/i,
  /scale your features/i,
  /standardize your features/i,
  /consider scaling your (data|features)/i,
  /feature scaling is (recommended|required)/i,
];

const treeInsensitiveKeywords = ['forest', 'tree', 'boost', 'xgboost', 'lightgbm', 'catboost', 'gradient', 'histgradient', 'bagging'];
const bayesInsensitiveKeywords = ['bayes', 'naive_bayes', 'gaussian_nb'];

const requiresScaling = (modelKey: string): { required: boolean; reason: 'logistic' | 'svm' | 'knn' | 'sgd' | 'linear' | null } => {
  const key = modelKey.trim().toLowerCase();
  if (!key) {
    return { required: false, reason: null };
  }

  if (treeInsensitiveKeywords.some((token) => key.includes(token))) {
    return { required: false, reason: null };
  }

  if (bayesInsensitiveKeywords.some((token) => key.includes(token))) {
    return { required: false, reason: null };
  }

  if (key.includes('logistic')) {
    return { required: true, reason: 'logistic' };
  }

  if (key.includes('svm') || key.includes('svc') || key.includes('svr')) {
    return { required: true, reason: 'svm' };
  }

  if (key.includes('knn') || key.includes('nearest')) {
    return { required: true, reason: 'knn' };
  }

  if (key.includes('sgd') || key.includes('perceptron')) {
    return { required: true, reason: 'sgd' };
  }

  const linearKeywords = ['linear', 'ridge', 'lasso', 'elastic', 'regression'];
  if (linearKeywords.some((token) => key.includes(token))) {
    return { required: true, reason: 'linear' };
  }

  return { required: false, reason: null };
};

const collectUpstreamNodeIds = (graph: FeatureGraph | null, nodeId: string): Set<string> => {
  const upstreamIds = new Set<string>();
  if (!graph || !nodeId) {
    return upstreamIds;
  }

  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const stack: string[] = [nodeId];

  while (stack.length) {
    const current = stack.pop();
    if (!current) {
      continue;
    }

    for (const edge of edges) {
      if (!edge) {
        continue;
      }
      const sourceRaw = typeof edge.source === 'string' ? edge.source.trim() : '';
      const targetRaw = typeof edge.target === 'string' ? edge.target.trim() : '';
      if (!sourceRaw || !targetRaw) {
        continue;
      }
      if (targetRaw === current && !upstreamIds.has(sourceRaw)) {
        upstreamIds.add(sourceRaw);
        stack.push(sourceRaw);
      }
    }
  }

  return upstreamIds;
};

const hasScalingUpstream = (graph: FeatureGraph | null, nodeId: string): boolean => {
  if (!graph || !nodeId) {
    return false;
  }
  const upstreamIds = collectUpstreamNodeIds(graph, nodeId);
  if (!upstreamIds.size) {
    return false;
  }

  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  return nodes.some((node: any) => {
    if (!node || typeof node.id !== 'string') {
      return false;
    }
    if (!upstreamIds.has(node.id)) {
      return false;
    }
    const catalogType = String(node?.data?.catalogType ?? node?.type ?? '').toLowerCase();
    if (!catalogType) {
      return false;
    }
    if (SCALING_NODE_TYPES.has(catalogType)) {
      return true;
    }
    return catalogType.includes('scale') && catalogType.includes('numeric');
  });
};

const resolveModelLabel = (
  modelType: string,
  modelTypeOptions: FeatureNodeParameterOption[],
): string => {
  const match = modelTypeOptions.find((option) => option?.value === modelType);
  if (match?.label) {
    return match.label;
  }
  return modelType.replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
};

const gatherStringCandidates = (
  input: unknown,
  seen: Set<unknown> = new Set(),
  depth = 0,
): string[] => {
  if (input === null || input === undefined) {
    return [];
  }
  if (typeof input === 'string') {
    return [input];
  }
  if (typeof input === 'number' || typeof input === 'boolean') {
    return [];
  }
  if (depth > 3) {
    return [];
  }
  if (Array.isArray(input)) {
    if (!input.length) {
      return [];
    }
    const limit = 32;
    const results: string[] = [];
    for (let index = 0; index < input.length && index < limit; index += 1) {
      results.push(...gatherStringCandidates(input[index], seen, depth + 1));
    }
    return results;
  }
  if (typeof input === 'object') {
    if (seen.has(input)) {
      return [];
    }
    seen.add(input);
    const values = Object.values(input as Record<string, unknown>);
    if (!values.length) {
      return [];
    }
    const limit = 32;
    const results: string[] = [];
    for (let index = 0; index < values.length && index < limit; index += 1) {
      results.push(...gatherStringCandidates(values[index], seen, depth + 1));
    }
    return results;
  }
  return [];
};

export const hasScalingConvergenceMessage = (payload: unknown): boolean => {
  if (!payload) {
    return false;
  }
  const candidates = gatherStringCandidates(payload);
  if (!candidates.length) {
    return false;
  }
  return candidates.some((message) =>
    SCALING_CONVERGENCE_PATTERNS.some((pattern) => pattern.test(message)),
  );
};

export const detectScalingConvergenceFromJob = (job: any): boolean => {
  if (!job) {
    return false;
  }

  const candidateFields: unknown[] = [
    job.error_message,
    job.warning_message,
    job.message,
    job.details,
    job.notes,
    job.status_reason,
    job.status_message,
    job.error,
    job.metadata?.warnings,
    job.metadata?.warning,
    job.metadata?.notes,
    job.metadata?.messages,
    job.metadata?.message,
    job.metadata?.errors,
    job.metrics?.warnings,
    job.metrics?.warning,
    job.metrics?.messages,
    job.metrics?.message,
    job.job_metadata?.warnings,
    job.job_metadata?.warning,
    job.job_metadata?.messages,
    job.job_metadata?.notes,
  ];

  for (const field of candidateFields) {
    if (hasScalingConvergenceMessage(field)) {
      return true;
    }
  }

  if (hasScalingConvergenceMessage(job.metadata)) {
    return true;
  }
  if (hasScalingConvergenceMessage(job.metrics)) {
    return true;
  }

  return false;
};

const buildWarning = (
  reason: 'logistic' | 'svm' | 'knn' | 'sgd' | 'linear',
  modelLabel: string,
  problemType: 'classification' | 'regression',
): ScalingWarning => {
  const baseLabel = modelLabel || 'This model';
  switch (reason) {
    case 'logistic':
      return {
        headline: `${baseLabel} needs scaled numeric features`,
        summary:
          'Standardize numeric inputs before launching jobs to keep the logistic solver stable and avoid repeated "scale your features" warnings.',
        details: [
          'Insert a “Scale numeric features” step before this node or reuse an upstream scaler.',
          'StandardScaler works for most datasets; RobustScaler can help when outliers dominate.',
          'Once scaling is in place, re-run tuning or training so the solver converges cleanly.',
        ],
      };
    case 'svm':
      return {
        headline: `${baseLabel} is sensitive to feature scale`,
        summary:
          'Support Vector Machines depend on distance calculations. Keep numeric features on comparable scales to preserve margin geometry.',
        details: [
          'Add the “Scale numeric features” step upstream of this node.',
          'Try StandardScaler or MinMaxScaler depending on the kernel and expected ranges.',
          'Retrain after scaling to refresh support vectors and hyperparameters.',
        ],
      };
    case 'knn':
      return {
        headline: `${baseLabel} uses distance-based comparisons`,
        summary:
          'k-NN models weigh distances directly, so unscaled columns dominate neighbors. Scale numeric fields to keep votes balanced.',
        details: [
          'Drop in a “Scale numeric features” step before launching jobs.',
          'MinMaxScaler keeps features in [0, 1]; StandardScaler works when values are roughly normal.',
          'Requeue jobs after scaling so neighbor search reflects the new feature space.',
        ],
      };
    case 'sgd':
      return {
        headline: `${baseLabel} benefits from normalized inputs`,
        summary:
          'Stochastic gradient models converge faster and produce steadier coefficients when numeric inputs share a common scale.',
        details: [
          'Insert a “Scale numeric features” node prior to this step.',
          'StandardScaler plus optional RobustScaler for heavy-tailed features keeps gradients stable.',
          'Re-run jobs after scaling to avoid exploding or vanishing updates.',
        ],
      };
    default:
      return {
        headline: `${baseLabel} prefers standardized numeric features`,
        summary:
          problemType === 'regression'
            ? 'Linear regression families rely on comparable feature magnitudes. Scaling keeps coefficients well-conditioned and improves R² stability.'
            : 'Linear classification models respond better when numeric inputs share a common scale. Scaling prevents any single column from dominating the loss.',
        details: [
          'Add the “Scale numeric features” step ahead of this node or reuse an existing scaler.',
          'StandardScaler is a safe default; RobustScaler helps when outliers skew distributions.',
          'Re-launch jobs once scaling is applied so coefficients represent the standardized space.',
        ],
      };
  }
};

type UseScalingWarningArgs = {
  graph: FeatureGraph | null;
  nodeId: string;
  modelType: string | null | undefined;
  problemType: 'classification' | 'regression';
  modelTypeOptions: FeatureNodeParameterOption[];
  enabled: boolean;
};

export const useScalingWarning = ({
  graph,
  nodeId,
  modelType,
  problemType,
  modelTypeOptions,
  enabled,
}: UseScalingWarningArgs): ScalingWarning | null => {
  return useMemo(() => {
    if (!enabled) {
      return null;
    }

    if (!modelType || !modelType.trim()) {
      return null;
    }

    if (hasScalingUpstream(graph, nodeId)) {
      return null;
    }

    const normalized = modelType.trim().toLowerCase();
    const { required, reason } = requiresScaling(normalized);
    if (!required || !reason) {
      return null;
    }

    const modelLabel = resolveModelLabel(modelType, modelTypeOptions);
    return buildWarning(reason, modelLabel, problemType);
  }, [enabled, graph, modelType, modelTypeOptions, nodeId, problemType]);
};
