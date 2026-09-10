import type { TaskType } from '../../../core/types/taskType';
import type { EnsembleSubFilter } from './history';

/** Job History tabs, in display order (plan §0.5: task type, not engine). */
const TASK_TABS: { task: TaskType; label: string }[] = [
  { task: 'classification', label: 'Classification' },
  { task: 'regression', label: 'Regression' },
  { task: 'text_classification', label: 'Text Classification' },
  { task: 'segmentation', label: 'Segmentation' },
  { task: 'ensemble', label: 'Ensemble' },
];

/** Sub-filter options shown only while the Ensemble tab is active. */
const ENSEMBLE_SUB_FILTERS: { value: 'all' | 'classification' | 'regression'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
];

interface HistoryTabsProps {
  activeTab: TaskType;
  setTab: (tab: TaskType) => void;
  ensembleSubFilter: EnsembleSubFilter;
  setEnsembleSubFilter: (filter: EnsembleSubFilter) => void;
}

/** Task and ensemble controls remain controlled by the drawer's persistent state. */
export function HistoryTabs({ activeTab, setTab, ensembleSubFilter, setEnsembleSubFilter }: HistoryTabsProps) {
  return <>
    <div className="flex border-b border-gray-200 dark:border-gray-700">
      {TASK_TABS.map(({ task, label }) => (
        <button
          key={task}
          className={`flex-1 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === task
              ? 'border-blue-500 text-blue-600 dark:text-blue-400 bg-blue-50/50 dark:bg-blue-900/20'
              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            }`}
          onClick={() => setTab(task)}
        >
          {label}
        </button>
      ))}
    </div>

    {/* Ensemble sub-filter pill (only shown on the Ensemble tab) */}
    {activeTab === 'ensemble' && (
      <div className="flex items-center gap-1.5 px-4 py-2 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        {ENSEMBLE_SUB_FILTERS.map(({ value, label }) => (
          <button
            key={value}
            onClick={() => setEnsembleSubFilter(value)}
            className={`px-2.5 py-1 text-xs rounded-full border transition-colors ${ensembleSubFilter === value
                ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-600 dark:text-blue-400'
                : 'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
          >
            {label}
          </button>
        ))}
      </div>
    )}

  </>;
}
