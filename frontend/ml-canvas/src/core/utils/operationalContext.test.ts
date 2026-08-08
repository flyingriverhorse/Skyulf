import { describe, it, expect } from 'vitest';
import {
  serializeOperationalContext,
  parseOperationalContext,
  buildRecordHref,
  describeOperationalRef,
  type OperationalContext,
} from './operationalContext';

describe('operationalContext — serializer/parser round trips', () => {
  const cases: Array<[string, OperationalContext]> = [
    ['job', { ref: { kind: 'job', jobId: 'job-abc-123' } }],
    ['pipeline', { ref: { kind: 'pipeline', pipelineId: 'pipe_9' } }],
    ['node with pipeline', { ref: { kind: 'node', nodeId: 'n1', pipelineId: 'pipe_9' } }],
    ['node without pipeline', { ref: { kind: 'node', nodeId: 'n1' } }],
    ['dataset', { ref: { kind: 'dataset', datasetId: '7f3a' } }],
    ['model version', { ref: { kind: 'modelVersion', jobId: 'job-1', version: '3' } }],
    ['deployment', { ref: { kind: 'deployment', deploymentId: 42 } }],
    ['drift check', { ref: { kind: 'driftCheck', checkId: 5, jobId: 'job-1' } }],
    ['incident', { ref: { kind: 'incident', incidentId: 900 } }],
    ['audit entry', { ref: { kind: 'auditEntry', auditId: 12, datasetId: 'ds-1' } }],
    ['slow node', { ref: { kind: 'slowNode', stepType: 'StandardScaler' } }],
  ];

  it.each(cases)('round-trips a %s reference without loss', (_name, ctx) => {
    const parsed = parseOperationalContext(serializeOperationalContext(ctx));
    expect(parsed).toEqual(ctx);
  });

  it('round-trips origin, time range, and filters together', () => {
    const ctx: OperationalContext = {
      ref: { kind: 'job', jobId: 'job-1' },
      origin: '/jobs',
      timeRange: '24h',
      filters: { status: 'failed', type: 'training' },
    };
    expect(parseOperationalContext(serializeOperationalContext(ctx))).toEqual(ctx);
  });

  it('preserves identifiers containing URL-hostile characters', () => {
    const ctx: OperationalContext = {
      ref: { kind: 'dataset', datasetId: 'a b&c=d?e/f#g' },
    };
    expect(parseOperationalContext(serializeOperationalContext(ctx))).toEqual(ctx);
  });

  it('keeps numeric identifiers numeric rather than stringifying them', () => {
    const parsed = parseOperationalContext(
      serializeOperationalContext({ ref: { kind: 'deployment', deploymentId: 42 } }),
    );
    expect(parsed?.ref).toEqual({ kind: 'deployment', deploymentId: 42 });
  });

  it('omits absent optional fields instead of emitting empty values', () => {
    const query = serializeOperationalContext({ ref: { kind: 'job', jobId: 'job-1' } });
    expect(query).not.toContain('origin');
    expect(query).not.toContain('undefined');
  });
});

describe('operationalContext — safe degradation', () => {
  it('returns null when no context is present at all', () => {
    expect(parseOperationalContext('')).toBeNull();
    expect(parseOperationalContext('?unrelated=1')).toBeNull();
  });

  it('returns null for an unknown record kind rather than inventing a target', () => {
    expect(parseOperationalContext('?oc.kind=wormhole&oc.jobId=job-1')).toBeNull();
  });

  it('returns null when a required identifier is missing', () => {
    expect(parseOperationalContext('?oc.kind=job')).toBeNull();
  });

  it('returns null when a required identifier is blank or whitespace', () => {
    expect(parseOperationalContext('?oc.kind=job&oc.jobId=')).toBeNull();
    expect(parseOperationalContext('?oc.kind=job&oc.jobId=%20%20')).toBeNull();
  });

  it('returns null when a numeric identifier is not a valid number', () => {
    expect(parseOperationalContext('?oc.kind=deployment&oc.deploymentId=abc')).toBeNull();
    expect(parseOperationalContext('?oc.kind=deployment&oc.deploymentId=1.5')).toBeNull();
  });

  it('drops an unrecognised time range but keeps the valid reference', () => {
    const parsed = parseOperationalContext('?oc.kind=job&oc.jobId=job-1&oc.t=since-tuesday');
    expect(parsed).toEqual({ ref: { kind: 'job', jobId: 'job-1' } });
  });

  it('drops an unknown optional identifier but keeps the valid reference', () => {
    const parsed = parseOperationalContext('?oc.kind=job&oc.jobId=job-1&oc.somethingElse=x');
    expect(parsed).toEqual({ ref: { kind: 'job', jobId: 'job-1' } });
  });

  it('ignores unrelated query params surrounding a valid context', () => {
    const parsed = parseOperationalContext('?page=2&oc.kind=job&oc.jobId=job-1&sort=desc');
    expect(parsed).toEqual({ ref: { kind: 'job', jobId: 'job-1' } });
  });

  it('accepts a URLSearchParams instance as well as a string', () => {
    const params = new URLSearchParams('oc.kind=job&oc.jobId=job-1');
    expect(parseOperationalContext(params)).toEqual({ ref: { kind: 'job', jobId: 'job-1' } });
  });
});

describe('operationalContext — href building', () => {
  it('routes each record kind to its owning page', () => {
    const routes: Array<[OperationalContext, string]> = [
      [{ ref: { kind: 'job', jobId: 'j' } }, '/jobs'],
      [{ ref: { kind: 'pipeline', pipelineId: 'p' } }, '/canvas'],
      [{ ref: { kind: 'node', nodeId: 'n' } }, '/canvas'],
      [{ ref: { kind: 'dataset', datasetId: 'd' } }, '/data'],
      [{ ref: { kind: 'modelVersion', jobId: 'j', version: '1' } }, '/registry'],
      [{ ref: { kind: 'deployment', deploymentId: 1 } }, '/deployments'],
      [{ ref: { kind: 'driftCheck', checkId: 1 } }, '/drift'],
      [{ ref: { kind: 'incident', incidentId: 1 } }, '/errors'],
      [{ ref: { kind: 'auditEntry', auditId: 1 } }, '/audit'],
      [{ ref: { kind: 'slowNode', stepType: 's' } }, '/slow-nodes'],
    ];
    for (const [ctx, expectedPath] of routes) {
      expect(buildRecordHref(ctx).split('?')[0]).toBe(expectedPath);
    }
  });

  it('produces an href whose query parses back to the original context', () => {
    const ctx: OperationalContext = {
      ref: { kind: 'incident', incidentId: 7 },
      origin: '/jobs',
      timeRange: '7d',
    };
    const href = buildRecordHref(ctx);
    expect(parseOperationalContext(href.slice(href.indexOf('?')))).toEqual(ctx);
  });
});

describe('operationalContext — accessible descriptions', () => {
  it('names the record type and identifier for each kind', () => {
    expect(describeOperationalRef({ kind: 'job', jobId: 'job-abc' })).toBe('Job job-abc');
    expect(describeOperationalRef({ kind: 'deployment', deploymentId: 42 })).toBe('Deployment 42');
    expect(
      describeOperationalRef({ kind: 'modelVersion', jobId: 'job-1', version: '3' }),
    ).toBe('Model version 3 (job job-1)');
    expect(describeOperationalRef({ kind: 'slowNode', stepType: 'StandardScaler' })).toBe(
      'Slow node StandardScaler',
    );
  });
});
