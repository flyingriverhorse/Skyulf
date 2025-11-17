// @ts-nocheck
import { FeatureGraph } from './types';

const sortGraphValue = (value: any): any => {
  if (Array.isArray(value)) {
    return value.map((item) => sortGraphValue(item));
  }
  if (value && typeof value === 'object') {
    const sortedKeys = Object.keys(value).sort();
    const result: Record<string, any> = {};
    sortedKeys.forEach((key) => {
      result[key] = sortGraphValue(value[key]);
    });
    return result;
  }
  return value;
};

const toHex = (buffer: ArrayBuffer): string => {
  return Array.from(new Uint8Array(buffer))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
};

const sha256Hex = async (value: string): Promise<string> => {
  if (!globalThis.crypto || !globalThis.crypto.subtle) {
    throw new Error('Secure hashing is not available in this environment');
  }
  const encoder = new TextEncoder();
  const digest = await globalThis.crypto.subtle.digest('SHA-256', encoder.encode(value));
  return toHex(digest);
};

const sanitizeGraphForHash = (graph: FeatureGraph | null | undefined): FeatureGraph => {
  const rawNodes = graph && Array.isArray(graph.nodes) ? graph.nodes : [];
  const rawEdges = graph && Array.isArray(graph.edges) ? graph.edges : [];

  const safeNodes = rawNodes
    .map((node: any) => {
      let config = null;
      if (node && node.data && typeof node.data === 'object') {
        const candidate = node.data.config;
        if (candidate !== undefined) {
          try {
            config = JSON.parse(JSON.stringify(candidate));
          } catch (error) {
            config = candidate;
          }
        }
      }
      return {
        id: node && Object.prototype.hasOwnProperty.call(node, 'id') ? node.id ?? null : null,
        type: node && Object.prototype.hasOwnProperty.call(node, 'type') ? node.type ?? null : null,
        catalogType:
          node && node.data && Object.prototype.hasOwnProperty.call(node.data, 'catalogType')
            ? node.data.catalogType ?? null
            : null,
        config: config,
      };
    })
    .sort((a, b) => {
      const first = typeof a.id === 'string' ? a.id : String(a.id ?? '');
      const second = typeof b.id === 'string' ? b.id : String(b.id ?? '');
      return first.localeCompare(second);
    });

  const safeEdges = rawEdges
    .map((edge: any) => ({
      source: edge && Object.prototype.hasOwnProperty.call(edge, 'source') ? edge.source ?? null : null,
      target: edge && Object.prototype.hasOwnProperty.call(edge, 'target') ? edge.target ?? null : null,
      sourceHandle:
        edge && Object.prototype.hasOwnProperty.call(edge, 'sourceHandle') ? edge.sourceHandle ?? null : null,
      targetHandle:
        edge && Object.prototype.hasOwnProperty.call(edge, 'targetHandle') ? edge.targetHandle ?? null : null,
    }))
    .sort((a, b) => {
      const sourceCompare = String(a.source ?? '').localeCompare(String(b.source ?? ''));
      if (sourceCompare !== 0) {
        return sourceCompare;
      }
      return String(a.target ?? '').localeCompare(String(b.target ?? ''));
    });

  return {
    nodes: safeNodes,
    edges: safeEdges,
  };
};

export async function generatePipelineId(
  datasetSourceId: string,
  graph: FeatureGraph | null | undefined
): Promise<string> {
  if (!datasetSourceId) {
    throw new Error('datasetSourceId is required to compute pipeline ID');
  }

  const safeGraph = sanitizeGraphForHash(graph);
  const graphJson = JSON.stringify(sortGraphValue(safeGraph));
  const hash = await sha256Hex(graphJson);
  return `${datasetSourceId}_${hash.slice(0, 8)}`;
}
