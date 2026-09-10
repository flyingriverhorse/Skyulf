import { ThresholdSliderControls } from './ThresholdSliderControls';
import type { ChartEvaluationProps } from './types';
import type { EvaluationPresentation } from './useEvaluationPresentation';

/** The sticky control bar keeps split visibility shared between both threshold tabs. */
export function EvaluationControls(props: ChartEvaluationProps & EvaluationPresentation) {
  const {
    evaluationData,
    regressionSplitTabs,
    regressionSplitLabels,
    activeRegressionSplit,
    setSelectedRegressionSplit,
    activeTab,
    setActiveTab,
  } = props;
  return (
    <div className="sticky top-0 z-10 flex flex-wrap items-center gap-x-6 gap-y-2 bg-white dark:bg-gray-800 px-4 py-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      {/* Regression: split tabs inline in the control bar */}
      {evaluationData.problem_type === 'regression' && (
        <div className="flex items-center gap-0.5">
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mr-1">Split:</span>
          {regressionSplitTabs.map(tab => (
            <button
              key={tab}
              onClick={() => setSelectedRegressionSplit(tab)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${activeRegressionSplit === tab
                ? 'bg-blue-500 text-white'
                : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
            >{regressionSplitLabels[tab] ?? tab}</button>
          ))}
        </div>
      )}
      {/* Classification: Threshold Slider / Threshold Tuning tab switch —
      decides which control panel + confusion-matrix view below is
      shown. Independent of the "Splits:" checkboxes just below,
      which stay shared across both tabs. */}
      {evaluationData.problem_type === 'classification' && (
        <ThresholdModeTabs activeTab={activeTab} setActiveTab={setActiveTab} />
      )}
      {/* Split visibility toggles — classification only, shared by both tabs */}
      {evaluationData.problem_type !== 'regression' && <SplitVisibilityControls {...props} />}

      {/* Classification controls — Tab 1 (Threshold Slider) only */}
      {activeTab === 'slider' && evaluationData.problem_type === 'classification' && evaluationData.splits.train?.y_proba && (
        <ThresholdSliderControls {...props} proba={evaluationData.splits.train.y_proba} />
      )}
    </div>
  );
}

/** Select which threshold workflow supplies the controls below. */
function ThresholdModeTabs({ activeTab, setActiveTab }: Pick<ChartEvaluationProps, 'activeTab' | 'setActiveTab'>) {
  return (
    <div className="flex items-center rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 text-xs font-medium">
      <button
        onClick={() => setActiveTab('slider')}
        className={`px-3 py-1.5 transition-colors ${activeTab === 'slider' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}
      >
        Threshold Slider
      </button>
      <button
        onClick={() => setActiveTab('tuning')}
        className={`px-3 py-1.5 transition-colors border-l border-gray-200 dark:border-gray-700 ${activeTab === 'tuning' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}
      >
        Threshold Tuning
      </button>
    </div>
  );
}

/** Split checkboxes stay independent of the chosen threshold workflow. */
function SplitVisibilityControls({
  hasTrainSplit,
  hasTestSplit,
  hasValidationSplit,
  showTrainMetrics,
  setShowTrainMetrics,
  showTestMetrics,
  setShowTestMetrics,
  showValMetrics,
  setShowValMetrics,
}: ChartEvaluationProps & EvaluationPresentation) {
  return <>

    <div className="flex items-center gap-1 text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Splits:</div>
    {hasTrainSplit && (
      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
        <input type="checkbox" checked={showTrainMetrics} onChange={e => { setShowTrainMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700" />
        <span className="text-gray-700 dark:text-gray-300">Train</span>
      </label>
    )}
    {hasTestSplit && (
      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
        <input type="checkbox" checked={showTestMetrics} onChange={e => { setShowTestMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700" />
        <span className="text-gray-700 dark:text-gray-300">Test</span>
      </label>
    )}
    {hasValidationSplit && (
      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
        <input type="checkbox" checked={showValMetrics} onChange={e => { setShowValMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700" />
        <span className="text-gray-700 dark:text-gray-300">Validation</span>
      </label>
    )}

  </>;
}
