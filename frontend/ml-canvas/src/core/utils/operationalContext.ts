/**
 * Typed operational-context contract shared by every Operations surface
 * (Jobs, Registry, Deployments, Drift, Errors, Slow Nodes, Audit Log).
 *
 * Operations investigations are cross-page by nature: a drift alert leads to a
 * job, which leads to a model version, which leads to a deployment. Without one
 * owner for identity + origin + time/filter scope, each page would hand-roll its
 * own query keys and parsing, so a copied link could silently point at the wrong
 * record or lose the time window it was read in.
 *
 * Parsing never invents a target: anything unknown, blank, malformed, or
 * incomplete degrades to `null` (no reference) rather than a plausible guess.
 */

import { RECORD_PARSERS, readText } from './operationalContext/recordParsers';
import { recordIdentity, serializeRecord } from './operationalContext/recordIdentity';

const PREFIX = 'oc.';
const FILTER_PREFIX = 'oc.f.';

/** Record types addressable by an operational link. */
export const OPERATIONAL_RECORD_KINDS = [
  'job',
  'pipeline',
  'node',
  'dataset',
  'modelVersion',
  'deployment',
  'driftCheck',
  'incident',
  'auditEntry',
  'slowNode',
] as const;

export type OperationalRecordKind = (typeof OPERATIONAL_RECORD_KINDS)[number];

/**
 * A reference to exactly one operational record.
 *
 * Field names and value types mirror the API contracts they come from — notably
 * `deploymentId`/`checkId`/`incidentId`/`auditId` are numeric server ids while
 * job/pipeline/node/dataset ids are opaque strings. Keeping that distinction
 * prevents a numeric id from round-tripping back as a string and silently
 * failing a strict equality check at the consumer.
 */
export type OperationalRef =
  | { kind: 'job'; jobId: string }
  | { kind: 'pipeline'; pipelineId: string }
  | { kind: 'node'; nodeId: string; pipelineId?: string }
  | { kind: 'dataset'; datasetId: string }
  | { kind: 'modelVersion'; jobId: string; version: string }
  | { kind: 'deployment'; deploymentId: number }
  | { kind: 'driftCheck'; checkId: number; jobId?: string }
  | { kind: 'incident'; incidentId: number }
  | { kind: 'auditEntry'; auditId: number; datasetId?: string }
  | { kind: 'slowNode'; stepType: string; nodeId?: string };

export const OPERATIONAL_TIME_RANGES = ['1h', '6h', '24h', '7d', '30d', 'all'] as const;
export type OperationalTimeRange = (typeof OPERATIONAL_TIME_RANGES)[number];

export interface OperationalContext {
  /** The record being addressed. */
  ref: OperationalRef;
  /** Route the user came from, so the target can offer an accurate return link. */
  origin?: string;
  /** Time window the originating view was scoped to. */
  timeRange?: OperationalTimeRange;
  /** Free-form view filters (status, type, ...) preserved across the handoff. */
  filters?: Record<string, string>;
}

/** Page that owns each record kind. */
const ROUTE_BY_KIND: Record<OperationalRecordKind, string> = {
  job: '/jobs',
  pipeline: '/canvas',
  node: '/canvas',
  dataset: '/data',
  modelVersion: '/registry',
  deployment: '/deployments',
  driftCheck: '/drift',
  incident: '/errors',
  auditEntry: '/audit',
  slowNode: '/slow-nodes',
};

const KIND_LABELS: Record<OperationalRecordKind, string> = {
  job: 'Job',
  pipeline: 'Pipeline',
  node: 'Node',
  dataset: 'Dataset',
  modelVersion: 'Model version',
  deployment: 'Deployment',
  driftCheck: 'Drift check',
  incident: 'Incident',
  auditEntry: 'Audit entry',
  slowNode: 'Slow node',
};

function isRecordKind(value: string): value is OperationalRecordKind {
  return (OPERATIONAL_RECORD_KINDS as readonly string[]).includes(value);
}

function isTimeRange(value: string): value is OperationalTimeRange {
  return (OPERATIONAL_TIME_RANGES as readonly string[]).includes(value);
}

/** Serializes a context to a `?`-prefixed query string. Absent optionals are omitted entirely. */
export function serializeOperationalContext(context: OperationalContext): string {
  const params = new URLSearchParams();
  const { ref } = context;
  params.set(`${PREFIX}kind`, ref.kind);

  serializeRecord(params, ref);

  if (context.origin !== undefined) params.set(`${PREFIX}origin`, context.origin);
  if (context.timeRange !== undefined) params.set(`${PREFIX}t`, context.timeRange);
  for (const [key, value] of Object.entries(context.filters ?? {})) {
    params.set(FILTER_PREFIX + key, value);
  }

  return `?${params.toString()}`;
}

/** Builds the reference alone; `null` when the kind or its required ids are unusable. */
function parseRef(params: URLSearchParams): OperationalRef | null {
  const kind = readText(params, 'kind');
  if (kind === null || !isRecordKind(kind)) return null;
  return RECORD_PARSERS[kind](params);
}

/**
 * Parses a query string (or `URLSearchParams`) back into a context.
 *
 * Returns `null` when no usable reference is present. Unknown params, an
 * unrecognised time range, and unrelated query keys are ignored rather than
 * failing the whole parse, so links stay forward/backward compatible as new
 * optional fields are added.
 */
export function parseOperationalContext(
  input: string | URLSearchParams,
): OperationalContext | null {
  const params =
    typeof input === 'string'
      ? new URLSearchParams(input.startsWith('?') ? input.slice(1) : input)
      : input;

  const ref = parseRef(params);
  if (ref === null) return null;

  const context: OperationalContext = { ref };

  const origin = readText(params, 'origin');
  if (origin !== null) context.origin = origin;

  const timeRange = readText(params, 't');
  if (timeRange !== null && isTimeRange(timeRange)) context.timeRange = timeRange;

  const filters: Record<string, string> = {};
  for (const [key, value] of params.entries()) {
    if (key.startsWith(FILTER_PREFIX)) filters[key.slice(FILTER_PREFIX.length)] = value;
  }
  if (Object.keys(filters).length > 0) context.filters = filters;

  return context;
}

/** Route + serialized context for the record a context points at. */
export function buildRecordHref(context: OperationalContext): string {
  return `${ROUTE_BY_KIND[context.ref.kind]}${serializeOperationalContext(context)}`;
}

/** Human-readable name for a reference, used as accessible link text. */
export function describeOperationalRef(ref: OperationalRef): string {
  const label = KIND_LABELS[ref.kind];
  if (ref.kind === 'modelVersion') return `${label} ${ref.version} (job ${ref.jobId})`;
  return `${label} ${recordIdentity(ref)[1]}`;
}
