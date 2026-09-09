import type { Node } from '@xyflow/react';

export interface ConvertedNode {
  stepType: string;
  params: Record<string, unknown>;
  inputs?: string[];
}

export type NodeConverter = (node: Node) => ConvertedNode;
