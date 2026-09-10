import { useId } from 'react';
import { AlertTriangle, BarChart3, ChevronRight } from 'lucide-react';
import type { ColumnProfile } from '../../../../core/api/client';
import type { EnsembleConfig } from '../EnsembleSettings';
import { HelpTooltip } from '../components/HelpTooltip';
import type { UpdateFn } from './modelOptions';

/** Collapsible outer evaluation cross-validation (same shape as the training nodes). */
export function CrossValidationSection({ config, update, showCV, setShowCV, columns }: {
  config: EnsembleConfig;
  update: UpdateFn;
  showCV: boolean;
  setShowCV: (v: boolean) => void;
  columns: ColumnProfile[];
}) {
  const fieldId = useId();
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        type="button"
        aria-expanded={showCV}
        onClick={() => { setShowCV(!showCV); }}
        className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-purple-500" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-200">Cross Validation</span>
        </div>
        <ChevronRight className={`w-4 h-4 text-gray-400 transition-transform ${showCV ? 'rotate-90' : ''}`} />
      </button>
      {showCV && (
        <div className="p-3 space-y-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
          <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
            <input
              type="checkbox"
              checked={config.cv_enabled !== false}
              onChange={(e) => { update({ cv_enabled: e.target.checked }); }}
              className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
            />
            Enable Cross-Validation
          </label>
          <p className="text-xs text-gray-500 dark:text-gray-400 pl-6">
            {config.run_mode === 'advanced'
              ? 'Candidates are already scored by CV during the search. This re-evaluates the winning ensemble after tuning (full metric panel + fold-to-fold variance); it never changes the selected hyperparameters.'
              : 'Runs a k-fold evaluation of the trained ensemble after training. Evaluation only — it measures generalization and never changes the model or its configuration.'}
          </p>
          {config.cv_enabled !== false && (
            <CrossValidationFields config={config} update={update} columns={columns} fieldId={fieldId} />
          )}
        </div>
      )}
    </div>
  );
}

/** Fold method, time ordering and shuffle settings appear only when CV is enabled. */
function CrossValidationFields({ config, update, columns, fieldId }: {
  config: EnsembleConfig; update: UpdateFn; columns: ColumnProfile[]; fieldId: string;
}) {
  return (
    <div className="space-y-3 pl-6 border-l-2 border-gray-100 dark:border-gray-800">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label htmlFor={`${fieldId}-cv_folds`} className="block text-xs text-gray-500 mb-1">Folds</label>
          <input
            id={`${fieldId}-cv_folds`}
            type="number"
            min={2}
            value={config.cv_folds ?? 5}
            onChange={(e) => { update({ cv_folds: Number(e.target.value) }); }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
          />
        </div>
        <div>
          <label htmlFor={`${fieldId}-cv_type`} className="block text-xs text-gray-500 mb-1">Method</label>
          <select
            id={`${fieldId}-cv_type`}
            value={config.cv_type ?? 'k_fold'}
            onChange={(e) => { update({ cv_type: e.target.value }); }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
          >
            <option value="k_fold">K-Fold</option>
            <option value="stratified_k_fold">Stratified</option>
            <option value="time_series_split">Time Series</option>
            <option value="shuffle_split">Shuffle Split</option>
            <option value="nested_cv">Nested CV</option>
          </select>
        </div>
      </div>

      {config.cv_type === 'time_series_split' && (
        <div className="space-y-2">
          <div className="flex items-start gap-1.5 p-2 bg-amber-50 dark:bg-amber-900/20 rounded text-xs text-amber-700 dark:text-amber-400">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            <span>Data must be sorted by time. Select a date column below or ensure your data is pre-sorted.</span>
          </div>
          <div>
            <label htmlFor={`${fieldId}-cv_time_column`} className="block text-xs text-gray-500 mb-1">Time Column (optional)</label>
            <select
              id={`${fieldId}-cv_time_column`}
              value={config.cv_time_column ?? ''}
              onChange={(e) => { update({ cv_time_column: e.target.value }); }}
              className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
            >
              <option value="">Auto-detect</option>
              {columns
                .filter((col) => {
                  const dt = String(col.dtype).toLowerCase();
                  return dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp');
                })
                .map((col) => (
                  <option key={col.name} value={col.name}>{col.name}</option>
                ))
              }
              {columns
                .filter((col) => {
                  const dt = String(col.dtype).toLowerCase();
                  return !(dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp'));
                })
                .map((col) => (
                  <option key={col.name} value={col.name}>{col.name}</option>
                ))
              }
            </select>
          </div>
        </div>
      )}

      <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
        <input
          type="checkbox"
          checked={config.cv_shuffle !== false}
          onChange={(e) => { update({ cv_shuffle: e.target.checked }); }}
          className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
        />
        Shuffle Data
      </label>
      {config.cv_shuffle !== false && config.cv_type !== 'time_series_split' && (
        <div>
          <span className="flex items-center gap-1 text-xs text-gray-500 mb-1">
            <label htmlFor={`${fieldId}-cv_random_state`}>Fold Split Seed</label>
            <HelpTooltip text="Seed controlling how rows are dealt to folds — same seed = identical fold splits, so CV scores stay comparable across runs." />
          </span>
          <input
            id={`${fieldId}-cv_random_state`}
            type="number"
            min={0}
            value={config.cv_random_state ?? 42}
            onChange={(e) => { update({ cv_random_state: Number(e.target.value) }); }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
          />
        </div>
      )}
    </div>
  );
}
