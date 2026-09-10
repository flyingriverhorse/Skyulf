import type { ErrorEvent, ErrorSeverity } from '../../core/api/monitoring';

export type TimeRange = '1h' | '6h' | '24h' | '7d' | 'all';

export const TIME_RANGES: { label: string; value: TimeRange }[] = [
  { label: '1h', value: '1h' },
  { label: '6h', value: '6h' },
  { label: '24h', value: '24h' },
  { label: '7d', value: '7d' },
  { label: 'All', value: 'all' },
];

export function sinceIso(range: TimeRange): string | undefined {
  if (range === 'all') return undefined;
  const ms = { '1h': 3600_000, '6h': 21_600_000, '24h': 86_400_000, '7d': 604_800_000 }[range];
  return new Date(Date.now() - ms).toISOString();
}

/** Maps the unified severity facet to the pipeline log's own `error`/`warning`/`info` taxonomy. */
export const SEVERITY_TO_PIPELINE_LEVEL: Record<ErrorSeverity, string> = {
  critical: 'error',
  warning: 'warning',
  info: 'info',
};

export const SEVERITY_LABELS: Record<ErrorSeverity, string> = {
  critical: 'Critical',
  warning: 'Warning',
  info: 'Info',
};

export function severityBadgeClass(severity: ErrorSeverity): string {
  if (severity === 'critical') return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
  if (severity === 'warning') return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
  return 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300';
}

export function exportCsv(rows: ErrorEvent[]): void {
  const header = ['id', 'severity', 'status_code', 'error_type', 'message', 'route', 'job_id', 'created_at'];
  const escape = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [header.join(','), ...rows.map(r =>
    [r.id, r.severity, r.status_code, r.error_type, r.message, r.route, r.job_id ?? '', r.created_at].map(escape).join(',')
  )];
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `error-log-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function statusColor(code: number): string {
  if (code >= 500) return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
  if (code >= 400) return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
  return 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300';
}

export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/** Format a naive server datetime string ("YYYY-MM-DDTHH:MM:SS") to HH:MM. */
export function clockTime(iso: string): string {
  return iso.slice(11, 16);
}

/** Local-time ISO prefix "YYYY-MM-DDTHH" for bucket matching. */
export function localHourPrefix(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}`;
}
