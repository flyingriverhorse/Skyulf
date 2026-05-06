import React from 'react';
import { LoadingState, ErrorState } from '../../../shared';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import type { EvaluationData, EvaluationSplit } from '../types';
import { RegressionChartsForSplit } from './RegressionChartsForSplit';
import { ClassificationChartsForSplit } from './ClassificationChartsForSplit';
import { PerClassConfusionMatrix } from './PerClassConfusionMatrix';

interface BestF1Info {
  threshold: number;
  f1: number;
  splitLabel: string;
  metricName: string;
}

interface Props {
  selectedJobIds: string[];
  evalJobId: string | null;
  fetchEvaluationData: (jobId: string) => void | Promise<void>;
  isEvalLoading: boolean;
  evalError: string | null;
  evaluationData: EvaluationData | null;
  selectedRegressionSplit: string | null;
  setSelectedRegressionSplit: (v: string | null) => void;
  showTrainMetrics: boolean;
  setShowTrainMetrics: (v: boolean) => void;
  showTestMetrics: boolean;
  setShowTestMetrics: (v: boolean) => void;
  showValMetrics: boolean;
  setShowValMetrics: (v: boolean) => void;
  threshold: number;
  setThreshold: (v: number) => void;
  selectedRocClass: string | null;
  setSelectedRocClass: (v: string) => void;
  cmView: 'overall' | 'per-class';
  setCmView: (v: 'overall' | 'per-class') => void;
  bestF1Info: BestF1Info | null;
  handleDownload: (elementId: string, fileName: string) => Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

export const EvaluationView: React.FC<Props> = ({
  selectedJobIds,
  evalJobId,
  fetchEvaluationData,
  isEvalLoading,
  evalError,
  evaluationData,
  selectedRegressionSplit,
  setSelectedRegressionSplit,
  showTrainMetrics,
  setShowTrainMetrics,
  showTestMetrics,
  setShowTestMetrics,
  showValMetrics,
  setShowValMetrics,
  threshold,
  setThreshold,
  selectedRocClass,
  setSelectedRocClass,
  cmView,
  setCmView,
  bestF1Info,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  return (
    <div className="space-y-6">
      {/* Job Selector if multiple */}
      {selectedJobIds.length > 1 && (
        <div
          className="flex gap-2 overflow-x-auto pb-2"
          role="tablist"
          aria-label="Select run for evaluation"
        >
          {selectedJobIds.map(id => {
            const isActive = evalJobId === id;
            return (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={isActive}
                title={isActive ? `Active run: ${id}` : `Switch to run ${id}`}
                onClick={() => { void fetchEvaluationData(id); }}
                className={`px-3 py-1 text-xs font-mono rounded border whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                  isActive
                    ? 'bg-blue-100 border-blue-300 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300'
                    : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700'
                }`}
              >
                {id.slice(0, 8)}
              </button>
            );
          })}
        </div>
      )}

      {/* Stale-while-revalidate render gate:
       *   - error  -> show error inline
       *   - no data yet -> show placeholder
       *   - have data (possibly stale) -> render charts; the small
       *     spinner badge above the data acts as the only loading
       *     indicator so the panel doesn't unmount/remount on every
       *     job switch (the "blink" the user reported). */}
      {evalError ? (
        <div className="h-64 flex items-center justify-center">
          <ErrorState error={evalError} />
        </div>
      ) : !evaluationData ? (
        isEvalLoading ? (
          <div className="h-64 flex items-center justify-center">
            <LoadingState message="Loading evaluation data..." />
          </div>
        ) : (
          <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
            <p>Select a completed job to view evaluation details.</p>
            <p className="text-xs mt-2 opacity-70">(Note: Only jobs run after this update have evaluation artifacts)</p>
          </div>
        )
      ) : (
        <div className={`space-y-6 transition-opacity ${isEvalLoading ? 'opacity-60' : ''}`}>
          {isEvalLoading && (
            <div className="text-xs text-gray-500 dark:text-gray-400 italic">
              Loading evaluation data…
            </div>
          )}
          {/* Controls bar — sticky so it stays visible while scrolling splits */}
          <div className="sticky top-0 z-10 flex flex-wrap items-center gap-x-6 gap-y-2 bg-white dark:bg-gray-800 px-4 py-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            {/* Regression: split tabs inline in the control bar */}
            {evaluationData.problem_type === 'regression' && (() => {
              const _cs = (Object.keys(evaluationData.splits) as string[]).filter(s => evaluationData.splits[s as keyof typeof evaluationData.splits] != null);
              const _ct = ['train', 'test', 'val'].filter(s => _cs.includes(s));
              const _cl: Record<string, string> = { train: 'Train', test: 'Test', val: 'Validation' };
              const _ca: string = (selectedRegressionSplit != null && _cs.includes(selectedRegressionSplit))
                ? selectedRegressionSplit
                : (_ct.find(t => t === 'val') ?? _ct.find(t => t === 'test') ?? _ct[0] ?? _cs[0]!);
              return (
                <div className="flex items-center gap-0.5">
                  <span className="text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide mr-1">Split:</span>
                  {_ct.map(tab => (
                    <button
                      key={tab}
                      onClick={() => setSelectedRegressionSplit(tab)}
                      className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                        _ca === tab
                          ? 'bg-blue-500 text-white'
                          : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                      }`}
                    >{_cl[tab] ?? tab}</button>
                  ))}
                </div>
              );
            })()}
            {/* Split visibility toggles — classification only */}
            {evaluationData.problem_type !== 'regression' && (<>
              <div className="flex items-center gap-1 text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide">Splits:</div>
              <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                <input type="checkbox" checked={showTrainMetrics} onChange={e => { setShowTrainMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                <span className="text-gray-700 dark:text-gray-300">Train</span>
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                <input type="checkbox" checked={showTestMetrics} onChange={e => { setShowTestMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                <span className="text-gray-700 dark:text-gray-300">Test</span>
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                <input type="checkbox" checked={showValMetrics} onChange={e => { setShowValMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                <span className="text-gray-700 dark:text-gray-300">Validation</span>
              </label>
            </>)}

            {/* Classification controls */}
            {evaluationData.problem_type === 'classification' && evaluationData.splits.train?.y_proba && (() => {
              const proba = evaluationData.splits.train.y_proba!;
              const isBinary = proba.classes.length === 2;
              return (
                <>
                  <div className="w-px h-5 bg-gray-200 dark:bg-gray-600" />
                  {/* Class selector — hidden for binary: both classes always shown inline */}
                  {!isBinary && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Class:</span>
                      <select
                        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-1.5"
                        value={selectedRocClass || ''}
                        onChange={(e) => { setSelectedRocClass(e.target.value); }}
                      >
                        {proba.classes.map((c: string | number, idx: number) => {
                          const label = proba.labels?.[idx] ?? c;
                          return <option key={String(c)} value={String(label)}>{String(label)}</option>;
                        })}
                      </select>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Threshold:</span>
                    <InfoTooltip
                      text={`Threshold (t): a sample is predicted as the selected class when P(class) ≥ t.\n\n↑ Raise t → fewer positives predicted → lower recall, higher precision (fewer false alarms, more misses).\n↓ Lower t → more positives predicted → higher recall, lower precision (fewer misses, more false alarms).\n\nDefault 0.5 works well for balanced classes. Adjust for imbalanced data or when the cost of false positives ≠ false negatives.`}
                      align="center"
                    />
                    <input
                      type="range" min={0.01} max={0.99} step={0.01}
                      value={threshold}
                      onChange={(e) => { setThreshold(parseFloat(e.target.value)); }}
                      className="w-28 accent-blue-500"
                    />
                    <span className="text-sm font-mono font-semibold text-blue-600 dark:text-blue-400 w-9">{threshold.toFixed(2)}</span>
                    {bestF1Info && (
                      <button
                        onClick={() => { setThreshold(bestF1Info.threshold); }}
                        className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-700 hover:bg-emerald-100 dark:hover:bg-emerald-900/50 transition-colors whitespace-nowrap"
                        title={`Best ${bestF1Info.metricName}=${bestF1Info.f1.toFixed(3)} on ${bestF1Info.splitLabel} split — click to apply`}
                      >
                        ★ best {bestF1Info.metricName}: {bestF1Info.threshold.toFixed(2)}
                        <span className="opacity-50 text-[10px]">({bestF1Info.splitLabel})</span>
                      </button>
                    )}
                    {bestF1Info && (
                      <InfoTooltip
                        text={`"Best ${bestF1Info.metricName}" is the threshold that maximises ${bestF1Info.metricName} for the selected class on the ${bestF1Info.splitLabel} split.\n\nIt is found by scanning every unique prediction score as a candidate — the same method sklearn uses internally.\n\nThe metric shown matches the scoring metric chosen when the job was trained (${bestF1Info.metricName}). Click the badge to snap the slider to this value.`}
                        align="center"
                      />
                    )}
                  </div>
                  {proba.labels && proba.labels.length === proba.classes.length && (
                    <div className="text-xs text-gray-400 dark:text-gray-500 whitespace-nowrap">
                      ({proba.classes.map((c, idx) => `${String(c)}→${String(proba.labels?.[idx] ?? c)}`).join(', ')})
                    </div>
                  )}
                  {/* Overall / Per Class toggle — hidden for binary */}
                  {!isBinary && (
                    <>
                      <div className="w-px h-5 bg-gray-200 dark:bg-gray-600" />
                      <div className="flex items-center rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 text-xs font-medium">
                        <button onClick={() => setCmView('overall')} className={`px-3 py-1.5 transition-colors ${cmView === 'overall' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}>Overall</button>
                        <button onClick={() => setCmView('per-class')} className={`px-3 py-1.5 transition-colors border-l border-gray-200 dark:border-gray-700 ${cmView === 'per-class' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}>Per Class</button>
                      </div>
                    </>
                  )}
                </>
              );
            })()}
          </div>

          {(evaluationData.problem_type === 'regression' || cmView === 'overall' || evaluationData.splits.train?.y_proba?.classes.length === 2) && (
            <div className="flex flex-col gap-6">
              {/* Charts per split */}
              {Object.entries(evaluationData.splits)
                .filter(([splitName]) => {
                  if (evaluationData.problem_type === 'regression') {
                    // Only show the active tab split
                    const _avail2 = (Object.keys(evaluationData.splits) as string[]).filter(s => evaluationData.splits[s as keyof typeof evaluationData.splits] != null);
                    const _tabs2 = ['train', 'test', 'val'].filter(s => _avail2.includes(s));
                    const _act2: string = (selectedRegressionSplit != null && _avail2.includes(selectedRegressionSplit))
                      ? selectedRegressionSplit
                      : (_tabs2.find(t => t === 'val') ?? _tabs2.find(t => t === 'test') ?? _tabs2[0] ?? _avail2[0]!);
                    return splitName === _act2;
                  }
                  if (splitName === 'train' && !showTrainMetrics) return false;
                  if (splitName === 'test' && !showTestMetrics) return false;
                  if (splitName === 'validation' && !showValMetrics) return false;
                  return true;
                })
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
          {evaluationData.problem_type === 'classification' && cmView === 'per-class' && (
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
            />
          )}
        </div>
      )}
    </div>
  );
};
