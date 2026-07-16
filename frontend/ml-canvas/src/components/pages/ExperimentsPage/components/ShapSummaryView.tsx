import React, { useMemo } from 'react';
import { Loader2, Check, Download } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import { useChartTheme } from '../../../../core/hooks/useChartTheme';

export interface ShapSummaryEntry {
  jobId: string;
  modelType: string;
  shapSummary: Record<string, number> | null;
}

interface Props {
  shapSummaryByJob: ShapSummaryEntry[];
  handleDownload: (elementId: string, fileName: string) => void | Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

const BAR_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#ff6b6b', '#4ecdc4'];

// Normalize a single job's SHAP summary so the largest value = 1.0. Mean
// |SHAP value| scales with the model's output magnitude (e.g. log-odds for
// classifiers vs raw target units for regressors), so without per-job
// normalisation one run's bars can dwarf another's.
const normalizePerJob = (raw: Record<string, number>): Record<string, number> => {
  const max = Math.max(...Object.values(raw).map(v => Math.abs(v)));
  if (!isFinite(max) || max === 0) return raw;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) {
    out[k] = v / max;
  }
  return out;
};

export const ShapSummaryView: React.FC<Props> = ({
  shapSummaryByJob,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartTheme = useChartTheme();
  // Mirrors FeatureImportanceView's memoisation: the scan over every job and
  // feature is only worth recomputing when the underlying data changes.
  const { chartData, barKeys, topFeatures, allFeatures, jobsWithDataCount } = useMemo(() => {
    const jobsWithData = shapSummaryByJob.filter(j => j.shapSummary !== null);

    const normalized = jobsWithData.map(j => ({
      ...j,
      shapSummary: normalizePerJob(j.shapSummary ?? {}),
    }));

    const allFeatures = new Set<string>();
    normalized.forEach(j => {
      Object.keys(j.shapSummary).forEach(f => allFeatures.add(f));
    });

    const featureAvg = Array.from(allFeatures).map(f => {
      let sum = 0;
      let count = 0;
      normalized.forEach(j => {
        const val = j.shapSummary[f];
        if (val !== undefined) { sum += Math.abs(val); count++; }
      });
      return { feature: f, avg: count > 0 ? sum / count : 0 };
    });
    featureAvg.sort((a, b) => b.avg - a.avg);
    const topFeatures = featureAvg.slice(0, 15).map(f => f.feature);

    const chartData = topFeatures.map(feature => {
      const row: Record<string, string | number> = { feature };
      normalized.forEach(j => {
        const shortId = j.jobId.slice(0, 8);
        const label = j.modelType !== 'unknown' ? `${j.modelType} (${shortId})` : shortId;
        row[label] = j.shapSummary[feature] ?? 0;
      });
      return row;
    });

    const barKeys = normalized.map((j) => {
      const shortId = j.jobId.slice(0, 8);
      return j.modelType !== 'unknown' ? `${j.modelType} (${shortId})` : shortId;
    });

    return { chartData, barKeys, topFeatures, allFeatures, jobsWithDataCount: jobsWithData.length };
  }, [shapSummaryByJob]);

  if (jobsWithDataCount === 0) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">SHAP Summary — All Runs</h3>
        <InfoTooltip
          text="Top 15 features by average mean(|SHAP value|) across the selected runs. Values are normalised per-run (each run's largest feature = 1.0) so runs with different output scales can be compared. Higher bar = stronger average impact on the model's prediction within that run."
          align="center"
        />
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id="shap-summary-chart">
        <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
          <button
            onClick={() => void handleDownload('shap-summary-chart', 'shap_summary_comparison')}
            disabled={downloadingChart === 'shap-summary-chart'}
            className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
            title="Download Graph"
          >
            {downloadingChart === 'shap-summary-chart' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === 'shap-summary-chart' ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke={chartTheme.gridColor} />
              <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 12, fill: chartTheme.axisColor }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: chartTheme.axisColor }} width={110} />
              <Tooltip
                contentStyle={chartTheme.tooltipContentStyle}
                formatter={(value: number) => value.toFixed(3)}
              />
              <Legend />
              {barKeys.map((key, i) => (
                <Bar key={key} dataKey={key} fill={BAR_COLORS[i % BAR_COLORS.length]} radius={[0, 4, 4, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-2 text-center">
          {topFeatures.length < allFeatures.size
            ? `Showing top ${topFeatures.length} of ${allFeatures.size} features · `
            : ''}
          values normalised per-run (max = 1.0)
        </p>
      </div>
    </div>
  );
};
