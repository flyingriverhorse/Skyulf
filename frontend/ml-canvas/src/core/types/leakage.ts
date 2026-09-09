/** Ephemeral canvas feedback, never part of a saved pipeline or node payload. */
export interface CanvasLeakageIssue {
  id: string;
  nodeId: string;
  edgeIds: string[];
  severity: 'error' | 'warning';
  message: string;
  suggestion: string;
}

export interface LeakageNotice {
  message: string;
  graphSignature: string;
}
