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

/** Merge the last 24 hours, including both partial edge hours, by UTC hour identity. */
export function mergeTimeline(timeline: { hour: string; count: number }[], pipelineLogs: PipelineRunLog[]) {
  const hourMs = 3_600_000;
  const now = Date.now();
  const cutoff = now - 24 * hourMs;
  const firstHour = Math.floor(cutoff / hourMs);
  const buckets = Array.from({ length: 25 }, (_, i) => {
    const key = firstHour + i;
    const d = new Date(key * hourMs);
    // Labels remain local; identity stays UTC so DST repeats cannot merge bins.
    const minute = String(d.getMinutes()).padStart(2, '0');
    return { hour: `${localHourPrefix(d)}:${minute}:00`, _key: key, count: 0 };
  });
  const byKey = new Map(buckets.map(b => [b._key, b]));
  // HTTP timeline: hour is "YYYY-MM-DDTHH:00" in UTC.
  timeline.forEach(t => {
    const d = new Date(`${t.hour}Z`);
    if (Number.isNaN(d.getTime())) return;
    const key = Math.floor(d.getTime() / hourMs);
    const b = byKey.get(key);
    if (b) b.count += t.count;
  });
  // Pipeline logs: run_at is "YYYY-MM-DDTHH:MM:SS" (server local time, no Z).
  pipelineLogs.forEach(l => {
    if (!l.run_at) return;
    const timestamp = new Date(l.run_at.replace(' ', 'T')).getTime();
    if (!Number.isFinite(timestamp) || timestamp < cutoff || timestamp > now) return;
    const key = Math.floor(timestamp / hourMs);
    const b = byKey.get(key);
    if (b) b.count += 1;
  });
  return buckets;
}
