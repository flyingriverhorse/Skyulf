import type { OperationalRecordKind, OperationalRef } from '../operationalContext';

/** Non-blank string field, or `null` when absent/whitespace-only. */
export function readText(params: URLSearchParams, key: string): string | null {
  const raw = params.get('oc.' + key);
  if (raw === null) return null;
  const trimmed = raw.trim();
  return trimmed === '' ? null : trimmed;
}

/**
 * Integer field, or `null` when absent/malformed. Server ids are integers, so a
 * float or non-numeric value indicates a corrupted link, not a valid record.
 */
function readInt(params: URLSearchParams, key: string): number | null {
  const raw = readText(params, key);
  if (raw === null) return null;
  const parsed = Number(raw);
  return Number.isInteger(parsed) ? parsed : null;
}

// Each kind must parse its own exact member of the public discriminated union.
type RecordParsers = {
  [Kind in OperationalRecordKind]: (params: URLSearchParams) =>
    Extract<OperationalRef, { kind: Kind }> | null;
};

export const RECORD_PARSERS: RecordParsers = {
  job(params) {
    const jobId = readText(params, 'jobId');
    return jobId === null ? null : { kind: 'job', jobId };
  },
  pipeline(params) {
    const pipelineId = readText(params, 'pipelineId');
    return pipelineId === null ? null : { kind: 'pipeline', pipelineId };
  },
  node(params) {
    const nodeId = readText(params, 'nodeId');
    if (nodeId === null) return null;
    const pipelineId = readText(params, 'pipelineId');
    return pipelineId === null ? { kind: 'node', nodeId } : { kind: 'node', nodeId, pipelineId };
  },
  dataset(params) {
    const datasetId = readText(params, 'datasetId');
    return datasetId === null ? null : { kind: 'dataset', datasetId };
  },
  modelVersion(params) {
    const jobId = readText(params, 'jobId');
    const version = readText(params, 'version');
    return jobId === null || version === null ? null : { kind: 'modelVersion', jobId, version };
  },
  deployment(params) {
    const deploymentId = readInt(params, 'deploymentId');
    return deploymentId === null ? null : { kind: 'deployment', deploymentId };
  },
  driftCheck(params) {
    const checkId = readInt(params, 'checkId');
    if (checkId === null) return null;
    const jobId = readText(params, 'jobId');
    return jobId === null ? { kind: 'driftCheck', checkId } : { kind: 'driftCheck', checkId, jobId };
  },
  incident(params) {
    const incidentId = readInt(params, 'incidentId');
    return incidentId === null ? null : { kind: 'incident', incidentId };
  },
  auditEntry(params) {
    const auditId = readInt(params, 'auditId');
    if (auditId === null) return null;
    const datasetId = readText(params, 'datasetId');
    return datasetId === null ? { kind: 'auditEntry', auditId } : { kind: 'auditEntry', auditId, datasetId };
  },
  slowNode(params) {
    const stepType = readText(params, 'stepType');
    if (stepType === null) return null;
    const nodeId = readText(params, 'nodeId');
    return nodeId === null ? { kind: 'slowNode', stepType } : { kind: 'slowNode', stepType, nodeId };
  },
};
