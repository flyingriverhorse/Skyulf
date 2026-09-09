import type { Edge, Node } from '@xyflow/react';

export interface BranchEdgeInfo {
  color: string;
  /** Only a representative edge entering the terminal carries its label. */
  label: string | null;
  /** Shared upstream edges keep the first branch's color. */
  shared: boolean;
}

export interface BranchDef {
  terminal: Node;
  inputEdge: Edge | null;
  allTerminalEdges: Edge[];
  yHelperEdges: Edge[];
  localIndex: number;
}

export type IncomingEdges = Map<string, Edge[]>;
