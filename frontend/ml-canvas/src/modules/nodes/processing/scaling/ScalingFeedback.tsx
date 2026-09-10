import type { ScalingConfig } from './types';

function getTargetExplanation(config: ScalingConfig) {
  switch (config.method) {
    case 'standard': return "Target: μ ≈ 0, σ ≈ 1";
    case 'minmax': return `Target: ${config.feature_range_min ?? 0} to ${config.feature_range_max ?? 1}`;
    case 'robust': return "Target: Median ≈ 0, IQR ≈ 1";
    case 'maxabs': return "Target: MaxAbs = 1";
    default: return "";
  }
}

function formatPair(metrics: Record<string, unknown>, idx: number, firstKey: string, secondKey: string, firstLabel: string, secondLabel: string): string {
  const first = metrics[firstKey] ? (metrics[firstKey] as number[])[idx] : undefined;
  const second = metrics[secondKey] ? (metrics[secondKey] as number[])[idx] : undefined;
  if (typeof first !== 'number') return '';
  return `${firstLabel}=${first.toFixed(2)}, ${secondLabel}=${typeof second === 'number' ? second.toFixed(2) : '-'}`;
}

function getColumnDetails(config: ScalingConfig, metrics: Record<string, unknown>, idx: number): string {
  switch (config.method) {
    case 'standard': return formatPair(metrics, idx, 'mean', 'scale', '\u03bc', '\u03c3');
    case 'minmax': return formatPair(metrics, idx, 'data_min', 'data_max', 'Min', 'Max');
    case 'robust': return formatPair(metrics, idx, 'center', 'scale', 'Med', 'IQR');
    case 'maxabs': {
      const maxAbs = metrics.max_abs ? (metrics.max_abs as number[])[idx] : undefined;
      return typeof maxAbs === 'number' ? `MaxAbs=${maxAbs.toFixed(2)}` : '';
    }
    default: return '';
  }
}

export function ScalingFeedback({ config, metrics }: { config: ScalingConfig; metrics: Record<string, unknown> | null }) {
  if (!metrics) return null;
  const cols = metrics.columns as string[] | undefined;

  if (!cols) return null;



  return (
    <div className="mt-4 p-3 bg-muted/50 rounded border text-xs space-y-2">
      <div className="flex justify-between items-center">
        <div className="font-medium text-muted-foreground">Scaling Statistics</div>
        <div className="text-[10px] text-primary/80 bg-primary/5 px-1.5 py-0.5 rounded border border-primary/10" title="Values close to these targets indicate successful scaling.">
          {getTargetExplanation(config)}
        </div>
      </div>
      <div className="max-h-40 overflow-y-auto bg-background p-2 rounded border space-y-1">
        {cols.map((col, idx) => {
          const details = getColumnDetails(config, metrics, idx);

          return (
            <div key={col} className="flex justify-between items-center border-b border-border/50 last:border-0 pb-1 last:pb-0">
              <span className="truncate max-w-[100px] font-medium" title={col}>{col}</span>
              <span className="font-mono text-[10px] text-muted-foreground">{details}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
