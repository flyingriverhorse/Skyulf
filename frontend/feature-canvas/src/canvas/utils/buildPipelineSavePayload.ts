import type { Edge, Node } from 'react-flow-renderer';
import type { DatasetSourceSummary, FeaturePipelinePayload } from '../../api';

type GraphSnapshot = {
  nodes: Node[];
  edges: Edge[];
};

type BuildPipelineSavePayloadArgs = {
  snapshot: GraphSnapshot;
  selectedDataset: DatasetSourceSummary | null;
  activeSourceId: string;
};

const normalizeEdgeType = (edge?: Edge): Edge['type'] => {
  const existingType = edge?.type;
  if (!existingType || existingType === 'smoothstep' || existingType === 'default') {
    return 'animatedEdge';
  }
  return existingType;
};

export const buildPipelineSavePayload = ({
  snapshot,
  selectedDataset,
  activeSourceId,
}: BuildPipelineSavePayloadArgs): FeaturePipelinePayload => {
  const nodes = snapshot.nodes ?? [];
  const edges = snapshot.edges ?? [];

  return {
    name: selectedDataset?.name
      ? `Draft pipeline for ${selectedDataset.name}`
      : `Draft pipeline for ${activeSourceId}`,
    graph: {
      nodes,
      edges: edges.map((edge) => ({
        ...edge,
        animated: edge?.animated ?? true,
        type: normalizeEdgeType(edge),
      })),
    },
    metadata: {
      lastClientSave: new Date().toISOString(),
      nodeCount: nodes.length,
      edgeCount: edges.length,
    },
  };
};
