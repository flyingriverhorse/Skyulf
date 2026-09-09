import { InfoTooltip } from '../../../../ui/InfoTooltip';
import { thresholdMetricOptions, metricLabel, normalizeThresholdMetric } from '../../utils/classificationCharts';
import type { ThresholdMetric } from '../../utils/jobMeta';
import type { YProba } from '../../types';
import type { EvaluationViewProps } from './types';

/** Manual threshold controls only change local exploration, never saved predictions. */
export function ThresholdSliderControls({
  proba,
  selectedRocClass,
  setSelectedRocClass,
  selectedMetric,
  setSelectedMetric,
  threshold,
  setThreshold,
  bestMetricInfos,
  cmView,
  setCmView,
}: EvaluationViewProps & { proba: YProba }) {
  const isBinary = proba.classes.length === 2;
  return <>
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
      <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Metric:</span>
      <select
        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-1.5"
        value={normalizeThresholdMetric(selectedMetric, isBinary)}
        onChange={(e) => { setSelectedMetric(normalizeThresholdMetric(e.target.value as ThresholdMetric, isBinary)); }}
      >
        {thresholdMetricOptions(isBinary).map(m => (
          <option key={m} value={m}>{metricLabel(m, isBinary)}</option>
        ))}
      </select>
      <InfoTooltip
        text={`Which metric the best-threshold badges below and ROC/PR-based scan optimize for. Precision/Recall/F1 use the selected class as positive for binary jobs, and a support-weighted average across all classes for multiclass jobs (accuracy is unaffected either way).`}
        align="center"
      />
    </div>
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Threshold:</span>
      <InfoTooltip
        text={`Threshold (t): a sample is predicted as the selected class when P(class) ≥ t.\n\n↑ Raise t → fewer positives predicted → lower recall, higher precision (fewer false alarms, more misses).\n↓ Lower t → more positives predicted → higher recall, lower precision (fewer misses, more false alarms).\n\nDefault 0.5 works well for balanced classes. Adjust for imbalanced data or when the cost of false positives ≠ false negatives.`}
        align="center"
      />
      <input
        type="range" min={0.01} max={0.99} step={0.01}
        value={threshold}
        onChange={(e) => { setThreshold(Number.parseFloat(e.target.value)); }}
        className="w-28 accent-blue-500"
      />
      <span className="text-sm font-mono font-semibold text-blue-600 dark:text-blue-400 w-9">{threshold.toFixed(2)}</span>
      {bestMetricInfos.map(info => {
        const colors: Record<string, string> = {
          train: 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-900/50',
          test: 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-700 hover:bg-emerald-100 dark:hover:bg-emerald-900/50',
          validation: 'bg-orange-50 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 border-orange-200 dark:border-orange-700 hover:bg-orange-100 dark:hover:bg-orange-900/50',
        };
        const isActive = Math.abs(threshold - info.threshold) < 0.001;
        const badgeMetricLabel = metricLabel(info.metricName as ThresholdMetric, isBinary);
        return (
          <button
            key={info.splitLabel}
            onClick={() => { setThreshold(info.threshold); }}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border transition-colors whitespace-nowrap ${colors[info.splitLabel] ?? colors.test} ${isActive ? 'ring-2 ring-offset-1 ring-current' : ''}`}
            title={`Best ${badgeMetricLabel}=${info.value.toFixed(3)} on ${info.splitLabel} split — click to apply`}
          >
            ★ {info.splitLabel} {badgeMetricLabel}: {info.threshold.toFixed(2)}
          </button>
        );
      })}
      {bestMetricInfos.length > 0 && (
        <InfoTooltip
          text={`Each badge shows the threshold that maximises ${metricLabel(bestMetricInfos[0]!.metricName as ThresholdMetric, isBinary)} for the selected class on that split (found by scanning every unique prediction score, same method sklearn uses internally) — one per split currently checked in "Splits:" above. Click a badge to snap the slider to that split's optimal value.`}
          align="center"
        />
      )}
    </div>
    <ClassDisplayControls proba={proba} isBinary={isBinary} cmView={cmView} setCmView={setCmView} />
  </>;
}

/** Explain encoded class labels and choose the multiclass matrix presentation. */
function ClassDisplayControls({ proba, isBinary, cmView, setCmView }: Pick<EvaluationViewProps, 'cmView' | 'setCmView'> & { proba: YProba; isBinary: boolean }) {
  return <>
    {proba.labels && proba.labels.length === proba.classes.length && (
      <div className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
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

  </>;
}
