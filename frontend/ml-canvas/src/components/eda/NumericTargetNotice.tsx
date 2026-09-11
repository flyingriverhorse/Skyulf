import type { CausalTargetExclusionReason } from '../../core/types/edaProfile';

interface NumericTargetNoticeProps {
  target?: string | null;
  reason?: CausalTargetExclusionReason | null;
}

/** Explain the shared numeric-only boundary without guessing from stored column dtypes. */
export function NumericTargetNotice({ target, reason }: NumericTargetNoticeProps) {
  if (!target || !reason) return null;

  const explanations: Record<CausalTargetExclusionReason, string> = {
    categorical: `The selected target “${target}” is treated as categorical and is omitted from numeric-only causal discovery and Pearson correlations. Use Target Analysis for categorical associations.`,
    excluded: `The selected target “${target}” is excluded from this report's analysis.`,
    unsupported: `The selected target “${target}” is not supported by numeric-only causal discovery or Pearson correlations.`,
  };

  return <p className="mb-4 text-sm text-gray-600 dark:text-gray-400">{explanations[reason]}</p>;
}
