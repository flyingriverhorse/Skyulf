import type { Edge, Node } from 'react-flow-renderer';
import type { FeaturePipelineResponse } from '../../api';

export type PipelineHydrationContext = 'sample' | 'stored' | 'reset';

export interface PipelineHydrationPayload {
  nodes: Node[];
  edges: Edge[];
  pipeline?: FeaturePipelineResponse | null;
  context: PipelineHydrationContext;
}

export interface CanvasShellProps {
  sourceId?: string | null;
  datasetName?: string | null;
  onGraphChange?: (nodes: Node[], edges: Edge[]) => void;
  onPipelineHydrated?: (payload: PipelineHydrationPayload) => void;
  onPipelineError?: (error: Error) => void;
}

export type CanvasShellHandle = {
  openCatalog: () => void;
  closeCatalog: () => void;
  clearGraph: () => void;
};
