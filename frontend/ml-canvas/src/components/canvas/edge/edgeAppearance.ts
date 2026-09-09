import type { CSSProperties } from 'react';
import type { EdgeProps } from '@xyflow/react';

/** Branch metadata supplied by the graph's edge annotation pass. */
export interface EdgeBranch {
  branchColor?: string;
  branchLabel?: string | null;
  branchShared?: boolean;
  isMergeWinner?: boolean;
}

/** Shared branches remain dashed, and the merge winner keeps its wider stroke. */
function getBranchStyle(branch: EdgeBranch, style: CSSProperties): CSSProperties {
  return {
    ...style,
    stroke: branch.branchColor,
    strokeDasharray: branch.branchShared ? '6 4' : undefined,
    filter: undefined,
    strokeWidth: branch.isMergeWinner ? 4 : 2,
    opacity: branch.branchShared ? 0.7 : 1,
  };
}

/** Branch color takes priority over winner amber and caller stroke/filter values. */
export function getEdgeStyle(branch: EdgeBranch, style: CSSProperties = {}): CSSProperties {
  if (branch.branchColor) return getBranchStyle(branch, style);
  return {
    ...style,
    strokeWidth: branch.isMergeWinner ? 4 : 2,
    stroke: branch.isMergeWinner ? '#f59e0b' : style.stroke,
    filter: branch.isMergeWinner ? undefined : style.filter,
  };
}

/** Only string endpoint labels are exposed in accessible names. */
export function getEndpointLabels(data: EdgeProps['data']) {
  return {
    sourceLabel: typeof data?.sourceLabel === 'string' ? data.sourceLabel : 'Source node',
    targetLabel: typeof data?.targetLabel === 'string' ? data.targetLabel : 'Target node',
  };
}
