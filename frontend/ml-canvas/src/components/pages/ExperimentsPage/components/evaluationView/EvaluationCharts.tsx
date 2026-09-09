import { RegressionChartsForSplit } from '../RegressionChartsForSplit';
import { ClassificationChartsForSplit } from '../ClassificationChartsForSplit';
import { PerClassConfusionMatrix } from '../PerClassConfusionMatrix';
import type { EvaluationSplit } from '../../types';
import type { ChartEvaluationProps } from './types';

/** Overall slider charts retain the binary train-probability shortcut. */
function showSplitCharts({ evaluationData, activeTab, cmView }: ChartEvaluationProps) {
  return evaluationData.problem_type === 'regression' || (activeTab === 'slider' && (cmView === 'overall' || evaluationData.splits.train?.y_proba?.classes.length === 2));
}

/** Regression shows one active split; classification honors its independent toggles. */
function visibleSplits({ evaluationData, showTrainMetrics, showTestMetrics, showValMetrics }: ChartEvaluationProps, activeRegressionSplit: string | undefined) {
  return Object.entries(evaluationData.splits).filter(([splitName]) => {
    if (evaluationData.problem_type === 'regression') return splitName === activeRegressionSplit;
    if (splitName === 'train' && !showTrainMetrics) return false;
    if (splitName === 'test' && !showTestMetrics) return false;
    if (splitName === 'validation' && !showValMetrics) return false;
    return true;
  });
}

/** Route visible splits to the existing chart components without changing their props. */
export function EvaluationCharts(props: ChartEvaluationProps & { activeRegressionSplit: string | undefined }) {
  const {
    evaluationData,
    activeTab,
    cmView,
    activeRegressionSplit,
    selectedRocClass,
    threshold,
    showTrainMetrics,
    showTestMetrics,
    showValMetrics,
    handleDownload,
    downloadingChart,
    doneChart,
    tuningPreview,
  } = props;
  return <>
    {showSplitCharts(props) && (
      <div className="flex flex-col gap-6">
        {/* Charts per split */}
        {visibleSplits(props, activeRegressionSplit)
          .map(([splitName, splitData]: [string, EvaluationSplit]) => (
            <div key={splitName} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              {evaluationData.problem_type !== 'regression' && (
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4 capitalize">{splitName} Set</h4>
              )}

              {evaluationData.problem_type === 'regression' ? (
                <RegressionChartsForSplit
                  splitName={splitName}
                  splitData={splitData}
                  handleDownload={handleDownload}
                  downloadingChart={downloadingChart}
                  doneChart={doneChart}
                />
              ) : (
                <ClassificationChartsForSplit
                  splitName={splitName}
                  splitData={splitData}
                  selectedRocClass={selectedRocClass}
                  threshold={threshold}
                  handleDownload={handleDownload}
                  downloadingChart={downloadingChart}
                  doneChart={doneChart}
                />
              )}
            </div>
          ))}
      </div>
    )}
    {activeTab === 'slider' && evaluationData.problem_type === 'classification' && cmView === 'per-class' && (
      <PerClassConfusionMatrix
        evaluationData={evaluationData}
        selectedRocClass={selectedRocClass}
        threshold={threshold}
        showTrainMetrics={showTrainMetrics}
        showTestMetrics={showTestMetrics}
        showValMetrics={showValMetrics}
        handleDownload={handleDownload}
        downloadingChart={downloadingChart}
        doneChart={doneChart}
        tunedThresholds={null}
        useTunedThresholds={false}
      />
    )}
    {activeTab === 'tuning' && evaluationData.problem_type === 'classification' && (
      tuningPreview ? (
        <PerClassConfusionMatrix
          evaluationData={evaluationData}
          selectedRocClass={selectedRocClass}
          threshold={threshold}
          showTrainMetrics={showTrainMetrics}
          showTestMetrics={showTestMetrics}
          showValMetrics={showValMetrics}
          handleDownload={handleDownload}
          downloadingChart={downloadingChart}
          doneChart={doneChart}
          tunedThresholds={tuningPreview.thresholds}
          useTunedThresholds
        />
      ) : (
        <div className="text-xs text-gray-500 dark:text-gray-400 italic text-center py-8 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          Click Preview above to see tuned thresholds applied to your confusion matrix.
        </div>
      )
    )}
  </>;
}
