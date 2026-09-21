import type { CorrelationMatrix } from '../../core/types/edaProfile';

export const CORRELATION_MAX_COLUMNS = 20;
export const CORRELATION_SCALE = [-1, -0.4, -0.02, 0, 0.02, 0.4, 1];

/** Opaque diverging colors keep screen, dark mode and PNG numerically identical. */
export function correlationColor(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return 'rgb(229, 231, 235)';
  const strength = Math.sqrt(Math.min(1, Math.abs(value)));
  const endpoint = value < 0 ? [59, 130, 246] : [239, 68, 68];
  return `rgb(${endpoint.map(channel => Math.round(255 + (channel - 255) * strength)).join(', ')})`;
}

/** Both analysis omissions and the independent display cap belong with every rendered matrix. */
export function correlationScopeNotes(data: CorrelationMatrix): string[] {
  const notes: string[] = [];
  const backendOmissions = data.omitted_columns ?? [];
  if (backendOmissions.length) {
    const total = data.total_columns ?? data.columns.length + backendOmissions.length;
    notes.push(`Correlation analysis used the first ${total - backendOmissions.length} of ${total} numeric columns by column order (${backendOmissions.length} omitted: ${backendOmissions.slice(0, 5).join(', ')}${backendOmissions.length > 5 ? ', …' : ''}). These omitted columns are not included in the data table.`);
  }
  const hidden = data.columns.slice(CORRELATION_MAX_COLUMNS);
  if (hidden.length) {
    notes.push(`Showing the first ${CORRELATION_MAX_COLUMNS} of ${data.columns.length} columns (${hidden.length} omitted from the heatmap: ${hidden.slice(0, 5).join(', ')}${hidden.length > 5 ? ', …' : ''}). The full computed matrix is available in the data table and CSV.`);
  }
  return notes;
}
