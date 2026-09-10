import type { OperationalRef } from '../operationalContext';

/** Primary identity field, shared by URL serialization and accessible names. */
export function recordIdentity(ref: OperationalRef): [key: string, value: string | number] {
  switch (ref.kind) {
    case 'pipeline':
      return ['pipelineId', ref.pipelineId];
    case 'node':
      return ['nodeId', ref.nodeId];
    case 'dataset':
      return ['datasetId', ref.datasetId];
    case 'deployment':
      return ['deploymentId', ref.deploymentId];
    case 'driftCheck':
      return ['checkId', ref.checkId];
    case 'incident':
      return ['incidentId', ref.incidentId];
    case 'auditEntry':
      return ['auditId', ref.auditId];
    case 'slowNode':
      return ['stepType', ref.stepType];
    default:
      // Both a job and a model version are identified first by the owning job.
      return ['jobId', ref.jobId];
  }
}

/** Add the secondary identity only when present, retaining explicit blank text. */
function setOptionalText(params: URLSearchParams, key: string, value: string | undefined): void {
  if (value !== undefined) params.set(`oc.${key}`, value);
}

/** Write identity fields in canonical order, independent of object property order. */
export function serializeRecord(params: URLSearchParams, ref: OperationalRef): void {
  const [key, value] = recordIdentity(ref);
  params.set(`oc.${key}`, String(value));
  switch (ref.kind) {
    case 'modelVersion':
      params.set('oc.version', ref.version);
      break;
    case 'node':
      setOptionalText(params, 'pipelineId', ref.pipelineId);
      break;
    case 'driftCheck':
      setOptionalText(params, 'jobId', ref.jobId);
      break;
    case 'auditEntry':
      setOptionalText(params, 'datasetId', ref.datasetId);
      break;
    case 'slowNode':
      setOptionalText(params, 'nodeId', ref.nodeId);
      break;
  }
}
