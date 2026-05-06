import React from 'react';
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

export interface FeatureImportanceEntry {
  jobId: string;
  modelType: string;
  importances: Record<string, number> | null;
}

interface Props {
  featureImportancesByJob: FeatureImportanceEntry[];
  handleDownload: (elementId: string, fileName: string) => void | Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

const BAR_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#ff6b6b', '#4ecdc4'];

// Normalize a single job's importances so the largest value = 1.0. Different
// model families produce importances on completely different scales (gini
// sums to ~1, gain can reach hundreds, permutation is unbounded). Without
// per-job normalisation the smaller-scale job's bars collapse to invisible
// slivers next to the larger-scale job — which is what users see as "the
// other run's bars disappeared".
const normalizePerJob = (raw: Record<string, number>): Record<string, number> => {
  const max = Math.max(...Object.values(raw).map(v => Math.abs(v)));
  if (!isFinite(max) || max === 0) return raw;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) {
    out[k] = v / max;
  }
  return out;
};

export const FeatureImportanceView: React.FC<Props> = ({
  featureImportancesByJob,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const jobsWithData = featureImportancesByJob.filter(j => j.importances !== null);
  if (jobsWithData.length === 0) return null;

  // Pre-compute normalised importances once per job
  const normalized = jobsWithData.map(j => ({
    ...j,
    importances: normalizePerJob(j.importances ?? {}),
  }));

  // Collect all unique features across selected jobs
  const allFeatures = new Set<string>();
  normalized.forEach(j => {
    Object.keys(j.importances).forEach(f => allFeatures.add(f));
  });

  // Rank features by AVERAGE normalised importance across jobs that report
  // them. With normalisation, every job contributes on the same [0,1] scale
  // so a feature that ranks high in either run still surfaces in the top 15.
  const featureAvg = Array.from(allFeatures).map(f => {
    let sum = 0;
    let count = 0;
    normalized.forEach(j => {
      const val = j.importances[f];
      if (val !== undefined) { sum += Math.abs(val); count++; }
    });
    return { feature: f, avg: count > 0 ? sum / count : 0 };
  });
  featureAvg.sort((a, b) => b.avg - a.avg);
  const topFeatures = featureAvg.slice(0, 15).map(f => f.feature);

  // Build chart data: each feature row carries one numeric column per job
  const chartData = topFeatures.map(feature => {
    const row: Record<string, string | number> = { feature };
    normalized.forEach(j => {
      const shortId = j.jobId.slice(0, 8);
      const label = j.modelType !== 'unknown' ? `${j.modelType} (${shortId})` : shortId;
      row[label] = j.importances[feature] ?? 0;
    });
    return row;
  });

  const barKeys = normalized.map((j) => {
    const shortId = j.jobId.slice(0, 8);
    return j.modelType !== 'unknown' ? `${j.modelType} (${shortId})` : shortId;
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Feature Importance Comparison</h3>
        <InfoTooltip
          text="Top 15 features by average importance across the selected runs. Values are normalised per-run (each run's largest feature = 1.0) so different model families (gini vs gain vs permutation) can be compared on the same scale. Higher bar = stronger relative influence within that run."
          align="center"
        />
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id="feature-importance-chart">
        <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
          <button
            onClick={() => void handleDownload('feature-importance-chart', 'feature_importance_comparison')}
            disabled={downloadingChart === 'feature-importance-chart'}
            className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
            title="Download Graph"
          >
            {downloadingChart === 'feature-importance-chart' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === 'feature-importance-chart' ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
              <Tooltip
                contentStyle={{ backgroundColor: 'rgba(0,0,0,0.85)', border: 'none', borderRadius: '8px', color: '#fff' }}
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
