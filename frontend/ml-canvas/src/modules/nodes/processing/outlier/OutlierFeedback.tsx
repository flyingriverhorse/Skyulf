import type { OutlierConfig } from './types';

function getTargetExplanation(config: OutlierConfig) {
  switch (config.method) {
    case 'iqr': return "Removes statistical outliers (IQR)";
    case 'zscore': return "Removes deviations > 3σ";
    case 'winsorize': return "Clips extreme values";
    case 'elliptic_envelope': return "Detects multivariate anomalies";
    default: return "";
  }
}

function hasFeedback(metrics: Record<string, unknown>): boolean {
  const { rows_removed, bounds, stats, contamination, warnings, values_clipped } = metrics;
  return !(rows_removed === undefined && !bounds && !stats && contamination === undefined && !warnings && values_clipped === undefined);
}

function OutlierWarnings({ warnings }: { warnings: string[] | undefined }) {
  return <>
    {warnings && warnings.length > 0 && (
      <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400 rounded border border-yellow-200 dark:border-yellow-800">
        <div className="font-medium mb-1">Warnings</div>
        <ul className="list-disc list-inside space-y-0.5">
          {warnings.map((w, i) => (
            <li key={i} className="truncate" title={w}>{w}</li>
          ))}
        </ul>
      </div>
    )}
  </>;
}

function OutlierRowSummary({ config, metrics }: { config: OutlierConfig; metrics: Record<string, unknown> }) {
  // Check for rows removed (common for IQR, ZScore, Elliptic)
  const rowsRemoved = metrics.rows_removed as number | undefined;
  const rowsRemaining = metrics.rows_remaining as number | undefined;
  const rowsTotal = metrics.rows_total as number | undefined ?? ((rowsRemoved ?? 0) + (rowsRemaining ?? 0));

  const valuesClipped = metrics.values_clipped as number | undefined;
  return <>
    {config.method === 'winsorize' ? (
      <div className="flex justify-between items-center p-2 bg-background rounded border">
        <span className="text-muted-foreground">Values Clipped</span>
        <div className="text-right">
          <span className="font-mono font-medium text-blue-600">{valuesClipped ?? 0}</span>
          <span className="text-[10px] text-muted-foreground block">
            (Replaced with bounds)
          </span>
        </div>
      </div>
    ) : (
      <div className="flex justify-between items-center p-2 bg-background rounded border">
        <span className="text-muted-foreground">Rows Removed</span>
        <div className="text-right">
          <div className="flex items-center justify-end gap-1">
            <span className="font-mono font-medium text-destructive">{rowsRemoved ?? 0}</span>
            {rowsTotal > 0 && (
              <span className="text-[10px] text-muted-foreground">
                ({(((rowsRemoved ?? 0) / rowsTotal) * 100).toFixed(1)}%)
              </span>
            )}
          </div>
          {rowsRemaining !== undefined && (
            <span className="text-[10px] text-muted-foreground block">
              {rowsRemaining} remaining
            </span>
          )}
        </div>
      </div>
    )}
  </>;
}

function OutlierColumnStatistics({ metrics }: { metrics: Record<string, unknown> }) {
  const bounds = metrics.bounds as Record<string, { lower: number; upper: number }> | undefined;
  const stats = metrics.stats as Record<string, { mean: number; std: number }> | undefined;
  return <>
    {bounds && (
      <div>
        <div className="font-medium text-muted-foreground mb-1">Calculated Bounds</div>
        <div className="max-h-32 overflow-y-auto bg-background p-2 rounded border space-y-1">
          {Object.entries(bounds).map(([col, bound]) => (
            <div key={col} className="flex justify-between items-center border-b border-border/50 last:border-0 pb-1 last:pb-0">
              <span className="truncate max-w-[100px] font-medium" title={col}>{col}</span>
              <span className="font-mono text-[10px] text-muted-foreground">
                [{bound.lower.toFixed(2)}, {bound.upper.toFixed(2)}]
              </span>
            </div>
          ))}
        </div>
      </div>
    )}

    {stats && (
      <div>
        <div className="font-medium text-muted-foreground mb-1">Z-Score Stats</div>
        <div className="max-h-32 overflow-y-auto bg-background p-2 rounded border space-y-1">
          {Object.entries(stats).map(([col, stat]) => (
            <div key={col} className="flex justify-between items-center border-b border-border/50 last:border-0 pb-1 last:pb-0">
              <span className="truncate max-w-[100px] font-medium" title={col}>{col}</span>
              <span className="font-mono text-[10px] text-muted-foreground">
                μ={stat.mean.toFixed(2)}, σ={stat.std.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    )}
  </>;
}

export function OutlierFeedback({ config, metrics }: { config: OutlierConfig; metrics: Record<string, unknown> | null }) {
  if (!metrics || !hasFeedback(metrics)) return null;
  const warnings = metrics.warnings as string[] | undefined;
  const contamination = metrics.contamination as number | undefined;





  return (
    <div className="mt-4 p-3 bg-muted/50 rounded border text-xs space-y-3">
      <div className="flex justify-between items-center">
        <div className="font-medium text-muted-foreground">Execution Feedback</div>
        <div className="text-[10px] text-primary/80 bg-primary/5 px-1.5 py-0.5 rounded border border-primary/10" title="Goal of the selected method">
          {getTargetExplanation(config)}
        </div>
      </div>

      <OutlierWarnings warnings={warnings} />

      <OutlierRowSummary config={config} metrics={metrics} />

      {contamination !== undefined && (
        <div className="flex justify-between items-center p-2 bg-background rounded border">
          <span className="text-muted-foreground">Contamination</span>
          <span className="font-mono font-medium">{(contamination * 100).toFixed(1)}%</span>
        </div>
      )}

      <OutlierColumnStatistics metrics={metrics} />
    </div>
  );
}
