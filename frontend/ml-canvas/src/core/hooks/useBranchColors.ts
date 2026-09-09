import { useMemo } from 'react';
import type { Node, Edge } from '@xyflow/react';
import { getBranchColors } from './branchColors/colors';
import type { BranchEdgeInfo } from './branchColors/types';

export { generateBranchColors } from './branchColors/labels';
export type { BranchEdgeInfo } from './branchColors/types';

/** Assign stable path labels and colors to merge or parallel graph branches. */
export function useBranchColors(nodes: Node[], edges: Edge[]): Map<string, BranchEdgeInfo> {
  return useMemo(() => getBranchColors(nodes, edges), [nodes, edges]);
}
