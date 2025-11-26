import { useMemo } from 'react';

interface ResetPermissionOptions {
  isResetAvailable?: boolean;
  isDataset: boolean;
  defaultConfigTemplate?: Record<string, any> | null;
  footerResetFlags: boolean[];
  isScalingNode: boolean;
  isBinningNode: boolean;
  isBinnedDistributionNode: boolean;
  isSkewnessDistributionNode: boolean;
}

interface ResetPermissionResult {
  canResetNode: boolean;
  headerCanResetNode: boolean;
  footerCanResetNode: boolean;
}

/**
 * Centralizes the logic for deciding whether a node can be reset from the header/footer.
 * Keeping this here keeps NodeSettingsModal leaner as we continue modularization.
 */
export const useResetPermissions = ({
  isResetAvailable = false,
  isDataset,
  defaultConfigTemplate,
  footerResetFlags,
  isScalingNode,
  isBinningNode,
  isBinnedDistributionNode,
  isSkewnessDistributionNode,
}: ResetPermissionOptions): ResetPermissionResult => {
  const canResetNode = useMemo(
    () => Boolean(isResetAvailable && !isDataset && defaultConfigTemplate),
    [defaultConfigTemplate, isDataset, isResetAvailable],
  );

  const footerOnlyResetNodes = footerResetFlags.some(Boolean);
  const disableAllGlobalResets = isBinnedDistributionNode || isSkewnessDistributionNode;
  const headerResetDisabled = disableAllGlobalResets || isScalingNode || isBinningNode || footerOnlyResetNodes;

  const headerCanResetNode = canResetNode && !headerResetDisabled;
  const footerCanResetNode = canResetNode && !disableAllGlobalResets;

  return {
    canResetNode,
    headerCanResetNode,
    footerCanResetNode,
  };
};
