import type { Edge, Node } from '@xyflow/react';

/** Identify the submitted graph without transient canvas layout or selection state. */
export function graphSemanticSignature(nodes: readonly Node[], edges: readonly Edge[]): string {
  return JSON.stringify({
    nodes: nodes.map(node => ({ id: node.id, type: node.type, data: node.data })),
    edges: edges.map(edge => ({
      id: edge.id, source: edge.source, target: edge.target,
      sourceHandle: edge.sourceHandle, targetHandle: edge.targetHandle,
    })),
  });
}

/** Read the backend's explicit error fields without interpreting node ids in prose. */
function errorMessage(error: unknown): string | null {
  if (typeof error === 'string') return error || null;
  if (!error || typeof error !== 'object') return null;
  const record = error as Record<string, unknown>;
  const response = record.response as { data?: unknown } | undefined;
  return errorMessage(response?.data) ?? errorMessage(record.detail)
    ?? errorMessage(record.message) ?? errorMessage(record.error);
}

/** Recognize explicit backend leakage and fold-refit safety failures only. */
export function getLeakageErrorMessage(error: unknown): string | null {
  const isSafetyMessage = (message: string): boolean =>
    /\bdata leakage (?:risk|detected)\b|\bper-fold preprocessing refit skipped:/i.test(message);
  const message = errorMessage(error);
  if (message) return isSafetyMessage(message) ? message : null;
  if (!error || typeof error !== 'object') return null;
  const response = error as { status?: string; node_results?: Record<string, unknown> };
  if (response.status !== 'failed' || !response.node_results) return null;
  const messages = Object.values(response.node_results).map(errorMessage)
    .filter((value): value is string => value !== null);
  return messages.length > 0 && messages.every(isSafetyMessage)
    ? [...new Set(messages)].join('\n') : null;
}
