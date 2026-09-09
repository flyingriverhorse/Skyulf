import { Activity } from 'lucide-react';

function metricCounts(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

export function EncodingFeedback({ metrics }: { metrics: Record<string, unknown> | null }) {
  if (!metrics) return null;
  const categoriesCount = metricCounts(metrics.categories_count);
  const classesCount = metricCounts(metrics.classes_count);
  return (
    <div className="mt-4 p-3 bg-muted/30 rounded-md border border-border">
      <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-primary">
        <Activity size={14} />
        <span>Last Run Results</span>
      </div>
      <div className="space-y-1 text-xs">
        {metrics.encoded_columns_count !== undefined && (
          <div className="flex justify-between">
            <span className="text-muted-foreground">Columns Encoded:</span>
            <span className="font-medium">{String(metrics.encoded_columns_count)}</span>
          </div>
        )}
        {metrics.new_features_count !== undefined && (
          <div className="flex justify-between">
            <span className="text-muted-foreground">New Features Created:</span>
            <span className="font-medium text-primary">{String(metrics.new_features_count)}</span>
          </div>
        )}

        {/* Detailed Counts */}
        {categoriesCount && (
          <div className="mt-2 border-t pt-2">
            <span className="text-muted-foreground block mb-1 font-medium">Categories Found:</span>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {Object.entries(categoriesCount).map(([col, count]) => (
                <div key={col} className="flex justify-between text-[10px]">
                  <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                  <span className="font-mono">{String(count)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {classesCount && (
          <div className="mt-2 border-t pt-2">
            <span className="text-muted-foreground block mb-1 font-medium">Classes Found:</span>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {Object.entries(classesCount).map(([col, count]) => (
                <div key={col} className="flex justify-between text-[10px]">
                  <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                  <span className="font-mono">{String(count)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
