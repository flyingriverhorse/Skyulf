import type { ReactNode } from 'react';
import type { TrainingSettingsState } from './useTrainingSettings';

type TrainingLayoutProps = Pick<TrainingSettingsState, 'isWide' | 'activeTab' | 'setActiveTab' | 'isAdvanced'> & {
  modelPanel: ReactNode;
  secondaryPanel: ReactNode;
};

export function TrainingLayout({
  isWide, activeTab, setActiveTab, isAdvanced, modelPanel, secondaryPanel,
}: TrainingLayoutProps) {
  const secondaryTabLabel = isAdvanced ? 'Search Space' : 'Hyperparameters';
  return <>
      {!isWide && (
        <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'model'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            aria-pressed={activeTab === 'model'}
            onClick={() => { setActiveTab('model'); }}
          >
            Configuration
          </button>
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'params'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            aria-pressed={activeTab === 'params'}
            onClick={() => { setActiveTab('params'); }}
          >
            {secondaryTabLabel}
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-1 pb-4 custom-scrollbar">
        {isWide ? (
            <div className="grid grid-cols-2 gap-6 h-full">
                <div className="overflow-y-auto pr-2">{modelPanel}</div>
                <div className="overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-800">{secondaryPanel}</div>
            </div>
        ) : (
            <>
                {activeTab === 'model' && modelPanel}
                {activeTab === 'params' && secondaryPanel}
            </>
        )}
      </div>
</>;
}
