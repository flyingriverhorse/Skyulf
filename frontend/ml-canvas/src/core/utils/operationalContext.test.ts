import { describe, it, expect } from 'vitest';
import {
  serializeOperationalContext,
  parseOperationalContext,
  buildRecordHref,
  describeOperationalRef,
  type OperationalContext,
  type OperationalRef,
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

describe('operationalContext — exact public URL contract', () => {
  const records: Array<[OperationalRef, string, string, string]> = [
    [{ kind: 'job', jobId: '001' }, 'oc.kind=job&oc.jobId=001', '/jobs', 'Job 001'],
    [{ kind: 'pipeline', pipelineId: '001' }, 'oc.kind=pipeline&oc.pipelineId=001', '/canvas', 'Pipeline 001'],
    [{ kind: 'node', nodeId: 'n' }, 'oc.kind=node&oc.nodeId=n', '/canvas', 'Node n'],
    [{ kind: 'node', nodeId: 'n', pipelineId: 'p' }, 'oc.kind=node&oc.nodeId=n&oc.pipelineId=p', '/canvas', 'Node n'],
    [{ kind: 'dataset', datasetId: '001' }, 'oc.kind=dataset&oc.datasetId=001', '/data', 'Dataset 001'],
    [{ kind: 'modelVersion', version: '03', jobId: '001' }, 'oc.kind=modelVersion&oc.jobId=001&oc.version=03', '/registry', 'Model version 03 (job 001)'],
    [{ kind: 'deployment', deploymentId: 42 }, 'oc.kind=deployment&oc.deploymentId=42', '/deployments', 'Deployment 42'],
    [{ kind: 'driftCheck', checkId: 5 }, 'oc.kind=driftCheck&oc.checkId=5', '/drift', 'Drift check 5'],
    [{ kind: 'driftCheck', jobId: 'j', checkId: 5 }, 'oc.kind=driftCheck&oc.checkId=5&oc.jobId=j', '/drift', 'Drift check 5'],
    [{ kind: 'incident', incidentId: 7 }, 'oc.kind=incident&oc.incidentId=7', '/errors', 'Incident 7'],
    [{ kind: 'auditEntry', auditId: 9 }, 'oc.kind=auditEntry&oc.auditId=9', '/audit', 'Audit entry 9'],
    [{ kind: 'auditEntry', datasetId: 'd', auditId: 9 }, 'oc.kind=auditEntry&oc.auditId=9&oc.datasetId=d', '/audit', 'Audit entry 9'],
    [{ kind: 'slowNode', stepType: 'Scaler' }, 'oc.kind=slowNode&oc.stepType=Scaler', '/slow-nodes', 'Slow node Scaler'],
    [{ kind: 'slowNode', nodeId: 'n', stepType: 'Scaler' }, 'oc.kind=slowNode&oc.stepType=Scaler&oc.nodeId=n', '/slow-nodes', 'Slow node Scaler'],
  ];

  it.each(records)('preserves fields, routes, and accessible descriptions for %j', (ref, query, route, label) => {
    // Copied links must retain canonical key order regardless of object insertion order.
    expect(serializeOperationalContext({ ref })).toBe(`?${query}`);
    expect(parseOperationalContext(query)).toEqual({ ref });
    expect(buildRecordHref({ ref })).toBe(`${route}?${query}`);
    expect(describeOperationalRef(ref)).toBe(label);
  });

  const requiredFields = [
    ['job', 'jobId', ''], ['pipeline', 'pipelineId', ''], ['node', 'nodeId', ''],
    ['dataset', 'datasetId', ''], ['modelVersion', 'jobId', '&oc.version=1'],
    ['modelVersion', 'version', '&oc.jobId=j'], ['deployment', 'deploymentId', ''],
    ['driftCheck', 'checkId', ''], ['incident', 'incidentId', ''],
    ['auditEntry', 'auditId', ''], ['slowNode', 'stepType', ''],
  ];
  it.each(requiredFields)('rejects absent and blank required %s.%s', (kind, key, other) => {
    // Every required identity component must be usable before a link can select a record.
    const base = `oc.kind=${kind}${other}`;
    expect(parseOperationalContext(base)).toBeNull();
    expect(parseOperationalContext(`${base}&oc.${key}=`)).toBeNull();
    expect(parseOperationalContext(`${base}&oc.${key}=+%09+`)).toBeNull();
  });

  const optionals: Array<[OperationalRef, string, string]> = [
    [{ kind: 'node', nodeId: 'n' }, 'oc.kind=node&oc.nodeId=n', 'pipelineId'],
    [{ kind: 'driftCheck', checkId: 5 }, 'oc.kind=driftCheck&oc.checkId=5', 'jobId'],
    [{ kind: 'auditEntry', auditId: 9 }, 'oc.kind=auditEntry&oc.auditId=9', 'datasetId'],
    [{ kind: 'slowNode', stepType: 'Scaler' }, 'oc.kind=slowNode&oc.stepType=Scaler', 'nodeId'],
  ];
  it.each(optionals)('preserves optional blank serialization and parse omission for %j', (ref, query, key) => {
    // Optional opaque ids accept text; only absent or blank values disappear on parsing.
    expect(serializeOperationalContext({ ref: { ...ref, [key]: '' } })).toBe(`?${query}&oc.${key}=`);
    expect(parseOperationalContext(`${query}&oc.${key}=+%09+`)).toEqual({ ref });
    expect(parseOperationalContext(`${query}&oc.${key}=`)).toEqual({ ref });
    expect(parseOperationalContext(`${query}&oc.${key}=+NaN+`)).toEqual({ ref: { ...ref, [key]: 'NaN' } });
  });

  const numericFields = [
    ['deployment', 'deploymentId'], ['driftCheck', 'checkId'],
    ['incident', 'incidentId'], ['auditEntry', 'auditId'],
  ];
  it.each(numericFields)('retains Number and integer semantics for %s.%s', (kind, key) => {
    // Existing numeric URL syntax includes nondecimal and negative integer spellings.
    const accepted: Array<[string, number]> = [
      ['0', 0], ['-0', -0], ['-7', -7], ['+08', 8], ['1.0', 1], ['1e2', 100],
      ['0x10', 16], ['0b10', 2], ['0o10', 8], [' 12 ', 12], ['9007199254740993', 9007199254740992],
    ];
    for (const [raw, value] of accepted) {
      expect(parseOperationalContext(`oc.kind=${kind}&oc.${key}=${encodeURIComponent(raw)}`))
        .toEqual({ ref: { kind, [key]: value } });
    }
    for (const raw of ['NaN', 'Infinity', '-Infinity', '1.5', '1_000', '1x', '', ' ']) {
      expect(parseOperationalContext(`oc.kind=${kind}&oc.${key}=${encodeURIComponent(raw)}`)).toBeNull();
    }
  });

  it('encodes hostile text and preserves context and filter insertion order without mutation', () => {
    // Sharing must encode identifiers and filters without normalizing the source objects.
    const ref = Object.freeze({ kind: 'dataset' as const, datasetId: ' a+&=?/#%é ' });
    const filters = Object.freeze({ z: '  a+b&c  ', 'a.b': '', '': 'x/y' });
    const context = Object.freeze({ ref, origin: '/jobs?q=a&b', timeRange: '6h' as const, filters });
    const query = '?oc.kind=dataset&oc.datasetId=+a%2B%26%3D%3F%2F%23%25%C3%A9+&oc.origin=%2Fjobs%3Fq%3Da%26b&oc.t=6h&oc.f.z=++a%2Bb%26c++&oc.f.a.b=&oc.f.=x%2Fy';
    expect(serializeOperationalContext(context)).toBe(query);
    expect(parseOperationalContext(query)).toEqual({ ...context, ref: { kind: 'dataset', datasetId: 'a+&=?/#%é' } });
    expect(context).toEqual({ ref: { kind: 'dataset', datasetId: ' a+&=?/#%é ' }, origin: '/jobs?q=a&b', timeRange: '6h', filters: { z: '  a+b&c  ', 'a.b': '', '': 'x/y' } });
  });

  it('uses first duplicate identity/context values and last duplicate filters without moving keys', () => {
    // URLSearchParams.get and the filter iteration intentionally resolve duplicates differently.
    const query = 'oc.kind=+job+&oc.kind=dataset&oc.jobId=+001+&oc.jobId=second&oc.origin=+%2Fjobs+&oc.origin=%2Fdata&oc.t=+7d+&oc.t=1h&oc.f.z=first&oc.f.a=middle&oc.f.z=last';
    const params = new URLSearchParams(query);
    expect(parseOperationalContext(params)).toEqual({ ref: { kind: 'job', jobId: '001' }, origin: '/jobs', timeRange: '7d', filters: { z: 'last', a: 'middle' } });
    expect(Object.entries(parseOperationalContext(params)?.filters ?? {})).toEqual([['z', 'last'], ['a', 'middle']]);
    expect(params.toString()).toBe(query);
    expect(parseOperationalContext('oc.kind=job&oc.jobId=&oc.jobId=valid')).toBeNull();
  });

  it.each(['1h', '6h', '24h', '7d', '30d', 'all'])('accepts the exact time range %s after trimming', (timeRange) => {
    // All supported view windows survive navigation.
    expect(parseOperationalContext(`?oc.kind=job&oc.jobId=j&oc.t=+${timeRange}+`))
      .toEqual({ ref: { kind: 'job', jobId: 'j' }, timeRange });
  });

  it.each(['', '+', 'future', '24H'])('omits unusable optional context values %s', (value) => {
    // Optional context cannot invalidate a usable identity.
    expect(parseOperationalContext(`oc.kind=job&oc.jobId=j&oc.origin=+&oc.t=${value}&oc.future=x&page=2`))
      .toEqual({ ref: { kind: 'job', jobId: 'j' } });
  });

  it('preserves blank serialization, case-sensitive prefixes, and query-only input handling', () => {
    // URL syntax remains limited to queries, with unrelated or differently cased prefixes ignored.
    expect(serializeOperationalContext({ ref: { kind: 'job', jobId: '' }, origin: '', filters: {} }))
      .toBe('?oc.kind=job&oc.jobId=&oc.origin=');
    expect(parseOperationalContext('??oc.kind=job&oc.jobId=j')).toEqual({ ref: { kind: 'job', jobId: 'j' } });
    expect(parseOperationalContext('https://host/jobs?oc.kind=job&oc.jobId=j')).toBeNull();
    expect(parseOperationalContext('OC.kind=job&oc.jobId=j')).toBeNull();
    expect(parseOperationalContext('oc.kind=Job&oc.jobId=j')).toBeNull();
    expect(parseOperationalContext('kind=job&jobId=j')).toBeNull();
    expect(parseOperationalContext('oc.kind=job&oc.jobId=j&oc.faux=x&OC.f.z=y&oc.future=x'))
      .toEqual({ ref: { kind: 'job', jobId: 'j' } });
  });
});

describe('operationalContext — safe degradation', () => {
  it.each(['__proto__', 'constructor', 'toString', 'hasOwnProperty'])(
    'rejects an inherited object property as a record kind: %s', kind => {
      // URL input may only select explicitly registered record parsers.
      expect(parseOperationalContext(`?oc.kind=${kind}&oc.jobId=job-1`)).toBeNull();
    },
  );

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
