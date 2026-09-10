import { ValidationField } from '../../../../components/shared/ValidationField';
import { ColumnMultiSelect } from '../../shared/ColumnMultiSelect';
import type { ImputationSettingsProps } from './types';
import type { ImputationData } from './useImputationData';

/** Connection and schema loading notices are independent of the selected method. */
export function ImputationStatus({ datasetId, isLoading }: Pick<ImputationData, 'datasetId' | 'isLoading'>) {
  return (
    <div className="shrink-0 p-4 pb-0 space-y-2">
      {!datasetId && (
        <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400 text-xs rounded border border-yellow-200 dark:border-yellow-800">
          Connect a dataset node to see available columns.
        </div>
      )}

      {isLoading && !!datasetId && (
        <div className="text-xs text-muted-foreground animate-pulse">
          Loading schema...
        </div>
      )}
    </div>
  );
}

/** Keep column search mounted and show matching missing-count badges. */
export function ImputationColumns({
  config, onChange, availableColumns, missingCounts, isLoading, isWide,
}: ImputationSettingsProps & Pick<ImputationData, 'availableColumns' | 'missingCounts' | 'isLoading'> & { isWide: boolean }) {
  return (
    <div className={`flex flex-col overflow-hidden ${isWide ? 'min-h-0 flex-1' : 'shrink-0'}`}>
      <ValidationField field="columns" className={isWide ? "flex min-h-0 flex-1 flex-col" : ""}>
        <ColumnMultiSelect
          columns={availableColumns}
          selected={config.columns}
          onChange={(newCols) => { onChange({ ...config, columns: newCols }); }}
          label="Target Columns"
          variant="panel"
          isLoading={isLoading}
          fillHeight={isWide}
          renderItemBadge={(col) =>
            missingCounts && missingCounts[col] !== undefined ? (
              <span
                className="text-[10px] text-muted-foreground font-mono shrink-0 bg-muted px-1.5 py-0.5 rounded"
                title={`${String(missingCounts[col])} missing values filled`}
              >
                {String(missingCounts[col])}
              </span>
            ) : null
          }
        />
      </ValidationField>
    </div>


  );
}

/** Show specific results once, with generic success only when no detail mappings exist. */
export function ImputationFeedback({ metrics, fillValues, missingCounts, nodeResult }: Pick<ImputationData, 'metrics' | 'fillValues' | 'missingCounts' | 'nodeResult'>) {
  if (!metrics) return null;
  return (
    <div className="mt-4 p-3 bg-muted/50 rounded border text-xs">
      <div className="font-medium text-muted-foreground mb-2">Execution Feedback</div>

      {fillValues && (
        <div className="space-y-1">
          <span className="text-muted-foreground block mb-1 font-medium">Imputed Values:</span>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 max-h-32 overflow-y-auto">
            {Object.entries(fillValues).map(([col, val]) => (
              <div key={col} className="flex justify-between text-[10px]">
                <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                <span className="font-mono">{typeof val === 'number' ? val.toFixed(4) : String(val)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {missingCounts && (
        <div className="space-y-1 mt-2">
          <span className="text-muted-foreground block mb-1 font-medium">Missing Values Filled:</span>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 max-h-32 overflow-y-auto">
            {Object.entries(missingCounts).map(([col, count]) => (
              <div key={col} className="flex justify-between text-[10px]">
                <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                <span className="font-mono">{String(count)}</span>
              </div>
            ))}
          </div>
          {metrics.total_missing !== undefined && (
            <div className="text-[10px] text-muted-foreground mt-1 pt-1 border-t">
              Total Filled: <span className="font-mono font-medium">{String(metrics.total_missing)}</span>
            </div>
          )}
        </div>
      )}

      {/* Generic success message if no specific metrics but successful */}
      {!fillValues && !missingCounts && nodeResult?.status === 'success' && (
        <div className="text-green-600">Imputation completed successfully.</div>
      )}
    </div>

  );
}
