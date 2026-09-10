import { Activity, Info } from 'lucide-react';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { getNodeMetricDetails } from '../../../../core/utils/preprocessingMetrics';

export function FeatureExecutionFeedback({ nodeId }: { nodeId: string | undefined }) {
  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = getNodeMetricDetails(nodeResult?.metrics);
  const generatedFeatures: string[] =
    metrics && Array.isArray(metrics.generated_features)
      ? (metrics.generated_features as unknown[]).map((v) => String(v))
      : [];

  return <>
    {metrics && (
      <div className="border rounded-lg bg-muted/10 p-4">
        <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-primary">
          <Activity size={14} />
          <span>Last Run Results</span>
        </div>

        <div className="space-y-2 text-xs">
          {generatedFeatures.length > 0 ? (
            <>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">New Features Created:</span>
                <span className="font-medium text-primary bg-primary/10 px-2 py-0.5 rounded-full">
                  {generatedFeatures.length}
                </span>
              </div>

              <div className="pt-2">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold mb-1.5 block">Generated Columns</span>
                <div className="flex flex-wrap gap-1.5">
                  {generatedFeatures.map((f) => (
                    <span key={f} className="px-2 py-1 bg-background border rounded text-[10px] font-mono text-foreground shadow-sm">
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="flex items-center gap-2 text-muted-foreground italic">
              <Info size={12} />
              <span>No features generated in last run.</span>
            </div>
          )}
        </div>
      </div>
    )}
  </>;
}
