import type { PipelineRunLog } from '../../core/api/monitoring';
import { localHourPrefix } from './errorLogFormatting';

export interface PipelineIssue {
  node_type: string;
  count: number;
  last_seen: string;
  first_seen: string;
}

/** Group only pipeline failures, keeping first/last timestamps and frequency order. */
export function groupPipelineIssues(pipelineLogs: PipelineRunLog[]): PipelineIssue[] {
  const map = new Map<string, PipelineIssue>();
  pipelineLogs.filter(l => l.level === 'error').forEach(l => {
    const key = l.node_type ?? 'unknown';
    const entry = map.get(key);
    if (!entry) {
      map.set(key, { node_type: key, count: 1, last_seen: l.run_at ?? '', first_seen: l.run_at ?? '' });
    } else {
      entry.count++;
      if (l.run_at) {
        if (l.run_at > entry.last_seen) entry.last_seen = l.run_at;
        if (l.run_at < entry.first_seen) entry.first_seen = l.run_at;
      }
    }
  });
  return Array.from(map.values()).sort((a, b) => b.count - a.count);
}

/** Merge UTC HTTP hours and naive pipeline times into the last 24 local hours. */
export function mergeTimeline(timeline: { hour: string; count: number }[], pipelineLogs: PipelineRunLog[]) {
  // Always build 24 local-time buckets; merge BOTH the HTTP timeline (UTC strings,
  // converted to local) and pipeline logs (local naive strings) into them.
  const buckets = Array.from({ length: 24 }, (_, i) => {
    const d = new Date();
    d.setMinutes(0, 0, 0);
    d.setHours(d.getHours() - (23 - i));
    return { hour: localHourPrefix(d) + ':00:00', _key: localHourPrefix(d), count: 0 };
  });
  const byKey = new Map(buckets.map(b => [b._key, b]));
  // HTTP timeline: hour is "YYYY-MM-DDTHH:00" in UTC.
  timeline.forEach(t => {
    const utcStr = t.hour.length >= 13 ? t.hour.slice(0, 13) + ':00:00Z' : t.hour;
    const d = new Date(utcStr);
    if (Number.isNaN(d.getTime())) return;
    const key = localHourPrefix(d);
    const b = byKey.get(key);
    if (b) b.count += t.count;
  });
  // Pipeline logs: run_at is "YYYY-MM-DDTHH:MM:SS" (server local time, no Z).
  pipelineLogs.forEach(l => {
    if (!l.run_at) return;
    const key = l.run_at.replace(' ', 'T').slice(0, 13);
    const b = byKey.get(key);
    if (b) b.count += 1;
  });
  return buckets;
}
