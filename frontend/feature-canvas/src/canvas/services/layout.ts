import type { Edge, Node, ReactFlowInstance } from 'react-flow-renderer';
import type { FeaturePipelineResponse } from '../../api';

const initialNodes: Node[] = [
  {
    id: 'dataset-source',
    type: 'featureNode',
    position: { x: -200, y: 0 },
    data: {
      label: 'Dataset input',
      description: 'Start from the currently selected dataset',
      isDataset: true,
      isRemovable: false,
      isConfigured: true,
    },
    style: {
      background: 'linear-gradient(165deg, rgba(30, 64, 175, 0.9), rgba(59, 130, 246, 0.85))',
      border: '1px solid rgba(59, 130, 246, 0.55)',
      color: '#e0f2fe',
      fontWeight: 600,
    },
  },
];

const initialEdges: Edge[] = [];

const cloneNode = (node: Node): Node => ({
  ...node,
  data: {
    ...(node.data ?? {}),
  },
});

export const getDefaultNodes = () => initialNodes.map((node) => cloneNode(node));

export const getDefaultEdges = () => initialEdges.map((edge) => ({ ...edge }));

export const HISTORY_LIMIT = 12;

export const getSamplePipelineGraph = (datasetLabel?: string | null) => {
  const displayName = datasetLabel && datasetLabel.trim() ? datasetLabel.trim() : 'Demo dataset';
  const datasetNodeLabel = `Dataset input\n(${displayName})`;

  const nodes: Node[] = [
    {
      id: 'dataset-source',
      type: 'featureNode',
      position: { x: -200, y: 40 },
      data: {
        label: datasetNodeLabel,
        isDataset: true,
        isRemovable: false,
      },
      style: {
        whiteSpace: 'pre-line',
        background: '#eef2ff',
        border: '1px solid rgba(99, 102, 241, 0.35)',
        color: '#312e81',
        fontWeight: 600,
      },
    },
    {
      id: 'node-1',
      type: 'featureNode',
      position: { x: 260, y: 40 },
      data: {
        label: 'Profile columns\n(assess types & nulls)',
      },
      style: {
        whiteSpace: 'pre-line',
        background: 'linear-gradient(165deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.85))',
      },
    },
    {
      id: 'node-4',
      type: 'featureNode',
      position: { x: 600, y: 60 },
      data: {
        label: 'Feature preview\n(sample rows)',
      },
      style: {
        whiteSpace: 'pre-line',
        background: 'linear-gradient(165deg, rgba(8, 145, 178, 0.42), rgba(2, 132, 199, 0.32))',
      },
    },
  ];

  const edges: Edge[] = [
    {
      id: 'edge-dataset-profile',
      source: 'dataset-source',
      target: 'node-1',
      animated: true,
      type: 'animatedEdge',
      sourceHandle: 'dataset-source-source',
      targetHandle: 'node-1-target',
    },
    {
      id: 'edge-profile-preview',
      source: 'node-1',
      target: 'node-4',
      animated: true,
      type: 'animatedEdge',
      sourceHandle: 'node-1-source',
      targetHandle: 'node-4-target',
    },
  ];

  return { nodes, edges };
};

export const resolveDropPosition = (
  currentNodes: Node[],
  instance: ReactFlowInstance | null,
  viewportEl: HTMLDivElement | null
): { x: number; y: number } | undefined => {
  if (!instance || typeof instance.project !== 'function' || !viewportEl) {
    return undefined;
  }

  const bounds = viewportEl.getBoundingClientRect();
  const projectedCenter = instance.project({ x: bounds.width / 2, y: bounds.height / 2 });
  if (!projectedCenter) {
    return undefined;
  }

  const nonDatasetCount = currentNodes.filter((node) => node.id !== 'dataset-source').length;
  const pattern = [
    { x: 0, y: 0 },
    { x: 220, y: 0 },
    { x: -220, y: 0 },
    { x: 0, y: 180 },
    { x: 220, y: 180 },
    { x: -220, y: 180 },
    { x: 0, y: -180 },
    { x: 220, y: -180 },
    { x: -220, y: -180 },
  ];

  const patternIndex = nonDatasetCount % pattern.length;
  const ring = Math.floor(nonDatasetCount / pattern.length);
  const offset = pattern[patternIndex];
  const scale = ring + 1;

  return {
    x: projectedCenter.x + offset.x * scale,
    y: projectedCenter.y + offset.y * scale,
  };
};
