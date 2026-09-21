import { afterEach, describe, expect, it, vi } from 'vitest';
import type { PipelineRunLog } from '../../core/api/monitoring';
import { mergeTimeline } from './errorLogAggregation';

const HOUR = 3_600_000;

/** Preserve the existing local-naive pipeline timestamp contract in fixtures. */
function localTimestamp(epoch: number) {
  const date = new Date(epoch);
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${String(date.getMilliseconds()).padStart(3, '0')}`;
}

/** Create one event without coupling timeline tests to unrelated API fields. */
function log(run_at: string | null): PipelineRunLog {
  return { id: 1, message: 'failure', level: 'error', run_at };
}

afterEach(() => vi.useRealTimers());

describe('rolling error timeline', () => {
  it('retains both partial edge hours and rejects pipeline timestamps outside the exact window', () => {
    // Dropping an edge bucket or filtering only by hour loses or admits events.
    const now = Date.parse('2026-09-20T12:30:00Z');
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(now);
    const cutoff = now - 24 * HOUR;
    const result = mergeTimeline([
      { hour: '2026-09-19T12:00', count: 2 },
      { hour: '2026-09-20T12:00', count: 3 },
      { hour: '2026-09-20T13:00', count: 99 },
      { hour: 'invalid', count: 99 },
    ], [
      log(localTimestamp(cutoff - 1)), log(localTimestamp(cutoff)),
      log(localTimestamp(now)), log(localTimestamp(now + 1)),
      log('invalid'), log(null),
    ]);
    expect(result).toHaveLength(25);
    expect(result[0]?.count).toBe(3);
    expect(result.at(-1)?.count).toBe(4);
    expect(result.reduce((total, bucket) => total + bucket.count, 0)).toBe(7);
  });

  it('keeps UTC hour identities through fractional offsets and daylight saving transitions', () => {
    // UTC arithmetic must not collapse repeated local hours or shift fractional offsets.
    const now = Date.parse('2026-11-01T07:30:00Z');
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(now);
    const starts = Array.from({ length: 25 }, (_, index) => Math.floor(now / HOUR) * HOUR - (24 - index) * HOUR);
    const result = mergeTimeline(starts.map((epoch, index) => ({
      hour: new Date(epoch).toISOString().slice(0, 16), count: index + 1,
    })), []);
    expect(result.map(bucket => bucket.count)).toEqual(starts.map((_, index) => index + 1));
    expect(new Set(result.map(bucket => bucket._key)).size).toBe(25);
    expect(result.map(bucket => bucket.hour)).toEqual(starts.map(epoch => localTimestamp(epoch).slice(0, 19)));
  });
});
