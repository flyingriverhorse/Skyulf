import { X } from 'lucide-react';
import type { TrainingSettingsState } from './useTrainingSettings';

type TrainingInformationProps = Pick<TrainingSettingsState, 'showInfo' | 'setShowInfo'>;

export function TrainingInformation({
  showInfo,
  setShowInfo,
}: TrainingInformationProps) {
  return (
    <>
          {showInfo && (
            <div className="mb-4 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded text-xs text-blue-700 dark:text-blue-300 flex justify-between items-start gap-2">
              <span>Train a model with fixed parameters, or switch to Advanced to automatically tune it.</span>
              <button
                aria-label="Dismiss training information"
                onClick={() => {
                    setShowInfo(false);
                    sessionStorage.setItem('hide_info_training_node', 'true');
                }}
                className="text-blue-400 hover:text-blue-600 dark:hover:text-blue-200"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
    </>
  );
}
