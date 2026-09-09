import type { NodeExecutionResult } from '../../../../core/api/client';
import { getNodeMetricDetails, hasWrappedNodeMetrics } from '../../../../core/utils/preprocessingMetrics';
import type { FeatureSelectionConfig } from './types';

interface SelectionMetrics {
  dropped_columns?: string[];
  feature_scores?: Record<string, number>;
  p_values?: Record<string, number>;
  feature_importances?: Record<string, number>;
  variances?: Record<string, number>;
  ranking?: Record<string, number>;
}

function getDropFeedbackState(result: NodeExecutionResult, metrics: SelectionMetrics | null) {
  if (metrics?.dropped_columns?.length) return 'dropped';
  if (result.status !== 'success') return 'hidden';
  if (!metrics) return hasWrappedNodeMetrics(result.metrics) ? 'unavailable' : 'hidden';
  return 'none';
}

function DroppedFeatureMetrics({ column: col, metrics }: { column: string; metrics: SelectionMetrics }) {
  const score = metrics.feature_scores?.[col];
  const pval = metrics.p_values?.[col];
  const imp = metrics.feature_importances?.[col];
  const variance = metrics.variances?.[col];
  const rank = metrics.ranking?.[col];


  return (
    <div className="flex flex-col items-end text-[10px] text-muted-foreground">
      {score !== undefined && <span>Score: {score.toFixed(4)}</span>}
      {pval !== undefined && <span>p-val: {pval.toExponential(2)}</span>}
      {imp !== undefined && <span>Imp: {imp.toFixed(4)}</span>}
      {variance !== undefined && <span>Var: {variance.toFixed(4)}</span>}
      {rank !== undefined && <span>Rank: {rank}</span>}
    </div>

  );
}

function DroppedFeatures({ metrics }: { metrics: SelectionMetrics }) {
  const dropped = metrics.dropped_columns!;
  return (
    <div>
      <div className="font-medium text-muted-foreground mb-1 flex justify-between items-center">
        <span>Dropped Columns ({dropped.length})</span>
      </div>
      <div className="max-h-40 overflow-y-auto bg-background p-2 rounded border space-y-1">
        {dropped.map(col => (
          <div key={col} className="flex justify-between items-center border-b border-border/50 last:border-0 pb-1 last:pb-0">
            <span className="truncate max-w-[120px] font-medium text-destructive" title={col}>{col}</span>
            <DroppedFeatureMetrics column={col} metrics={metrics} />
          </div>
        ))}
      </div>
    </div>

  );
}

function NoDropsFeedback({ config }: { config: FeatureSelectionConfig }) {
  return (
    <div className="text-green-600 text-[10px] space-y-1">
      <div>No columns were dropped.</div>
      {config.method === 'variance_threshold' && (
        <div className="text-muted-foreground italic">
          All features have variance &gt; {config.threshold ?? 0}. Try increasing the threshold.
        </div>
      )}
      {config.method === 'select_k_best' && (
        <div className="text-muted-foreground italic">
          K ({config.k ?? 10}) is likely larger than or equal to the number of features. Try reducing K.
        </div>
      )}
      {config.method === 'correlation_threshold' && (
        <div className="text-muted-foreground italic">
          No feature pairs found with correlation &gt; {config.threshold ?? 0.95}.
        </div>
      )}
    </div>

  );
}

function DropFeedback({ state, metrics, config }: {
  state: ReturnType<typeof getDropFeedbackState>;
  metrics: SelectionMetrics | null;
  config: FeatureSelectionConfig;
}) {
  switch (state) {
    case 'dropped': return <DroppedFeatures metrics={metrics!} />;
    case 'unavailable': return (
      <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400 rounded border border-yellow-200 dark:border-yellow-800">
        Feature-selection drop details were unavailable because this wrapped metrics payload could not be resolved to a single step.
      </div>

    );
    case 'none': return <NoDropsFeedback config={config} />;
    default: return null;
  }
}

function TopFeatures({ values }: { values: Record<string, number> | undefined }) {
  if (!values) return null;
  return (
    <div>
      <div className="font-medium text-muted-foreground mb-1">Top 5 Features</div>
      <div className="max-h-32 overflow-y-auto bg-background p-2 rounded border space-y-1">
        {Object.entries(values)
          .sort(([, a], [, b]) => (b as number) - (a as number))
          .slice(0, 5)
          .map(([col, val]) => (
            <div key={col} className="flex justify-between items-center">
              <span className="truncate max-w-[120px]" title={col}>{col}</span>
              <span className="font-mono text-[10px]">{(val as number).toFixed(4)}</span>
            </div>
          ))
        }
      </div>
    </div>

  );
}

export function FeatureSelectionFeedback({ result, config }: { result: NodeExecutionResult | undefined; config: FeatureSelectionConfig }) {
  if (!result) return null;
  const metrics = getNodeMetricDetails(result.metrics) as SelectionMetrics | null;
  const state = getDropFeedbackState(result, metrics);
  const topFeatures = metrics?.feature_scores || metrics?.feature_importances;
  if (!result.error && state === 'hidden' && !topFeatures) return null;

  return (
    <div className="mt-4 p-3 bg-muted/50 rounded border text-xs space-y-3">
      <div className="font-medium text-muted-foreground">Execution Feedback</div>
      {result.error && (
        <div className="p-2 bg-destructive/10 text-destructive rounded border border-destructive/20">
          {result.error}
        </div>
      )}


      <DropFeedback state={state} metrics={metrics} config={config} />
      <TopFeatures values={topFeatures} />
    </div>
  );
}
