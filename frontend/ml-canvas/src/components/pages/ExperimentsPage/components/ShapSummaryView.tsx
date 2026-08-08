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
import { ChartDataTable, type ChartDataTableColumn } from '../../../eda/ChartDataTable';
import { useChartTheme } from '../../../../core/hooks/useChartTheme';
import { buildFeatureBarChartData, reportedFlagKey } from '../utils/featureBarChart';
import { getArtifactCoverage, type ArtifactCoverageInput } from '../utils/artifactCoverage';
import { ArtifactCoverageList, type ArtifactCoverageEntry } from './ArtifactCoverageList';

export interface ShapSummaryEntry {
  jobId: string;
  pipeline_id: string;
  parent_pipeline_id?: string | null;
  modelType: string;
  shapSummary: Record<string, number> | null;
}

/** Per-run task/lifecycle context used to resolve artifact availability copy. */
export interface ShapSummaryCoverageInput extends ArtifactCoverageInput {
  jobId: string;
  label: string;
}

interface Props {
  shapSummaryByJob: ShapSummaryEntry[];
  coverageInputs: ShapSummaryCoverageInput[];
  handleDownload: (elementId: string, fileName: string) => void | Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

const BAR_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#ff6b6b', '#4ecdc4'];
const HATCH_PATTERN_ID = 'shap-summary-not-reported-hatch';

interface NotReportedBarShapeProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  fill?: string;
  payload?: Record<string, unknown>;
  dataKey?: string;
}

// Custom bar shape: a feature a run doesn't report renders as a hatched
// outline instead of a solid fill, so "not reported" never looks identical
// to "genuinely scored near zero" (UX finding EXP-003).
const NotReportedAwareBar: React.FC<NotReportedBarShapeProps> = ({ x = 0, y = 0, width = 0, height = 0, fill, payload, dataKey }) => {
  const reported = dataKey ? payload?.[reportedFlagKey(dataKey)] !== false : true;
  return (
    <rect
      x={x}
      y={y}
      width={width}
      height={height}
      fill={reported ? fill : `url(#${HATCH_PATTERN_ID})`}
      stroke={reported ? 'none' : '#9ca3af'}
      strokeWidth={reported ? 0 : 1}
    />
  );
};

export const ShapSummaryView: React.FC<Props> = ({
  shapSummaryByJob,
  coverageInputs,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartTheme = useChartTheme();
  // Mirrors FeatureImportanceView's memoisation: the scan over every job and
  // feature is only worth recomputing when the underlying data changes.
  const { chartData, barKeys, topFeatures, allFeatures, jobsWithDataCount, rawByBarKey } = useMemo(
    () => buildFeatureBarChartData(shapSummaryByJob.map((j) => ({ ...j, values: j.shapSummary }))),
    [shapSummaryByJob],
  );

  const coverageEntries: ArtifactCoverageEntry[] = useMemo(
    () => coverageInputs.map((input) => ({
      jobId: input.jobId,
      label: input.label,
      ...getArtifactCoverage('shap', input),
    })),
    [coverageInputs],
  );

  const tableColumns: ChartDataTableColumn[] = useMemo(() => {
    const cols: ChartDataTableColumn[] = [{ key: 'feature', label: 'Feature' }];
    barKeys.forEach((key) => {
      cols.push({ key: `${key}__normalized`, label: `${key} — normalized (run max = 1.0)` });
      cols.push({ key: `${key}__raw`, label: `${key} — raw mean(|SHAP|)` });
    });
    return cols;
  }, [barKeys]);

  const tableRows = useMemo(() => topFeatures.map((feature) => {
    const row = chartData.find((r) => r.feature === feature);
    const out: Record<string, string | number | null> = { feature };
    barKeys.forEach((key) => {
      const reported = row?.[reportedFlagKey(key)] !== false;
      const normalizedValue = row?.[key];
      const rawValue = rawByBarKey[key]?.[feature];
      out[`${key}__normalized`] = reported && typeof normalizedValue === 'number' ? Number(normalizedValue.toFixed(4)) : 'not reported';
      out[`${key}__raw`] = reported && rawValue !== undefined ? rawValue : 'not reported';
    });
    return out;
  }), [topFeatures, barKeys, chartData, rawByBarKey]);

  if (jobsWithDataCount === 0 && coverageEntries.length === 0) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">SHAP Summary — All Runs</h3>
        <InfoTooltip
          text="Top 15 features by average mean(|SHAP value|) across the selected runs. Values are normalised per-run (each run's largest feature = 1.0) so runs with different output scales can be compared. Higher bar = stronger average impact on the model's prediction within that run."
          align="center"
        />
      </div>

      <ArtifactCoverageList entries={coverageEntries} />

      {jobsWithDataCount > 0 && (
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
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
            Values are min–max normalized per run: 1.0 is that run&apos;s single largest mean(|SHAP|) feature, 0 is its smallest. Bars are not comparable to raw SHAP units across runs — use the data table below for raw magnitudes.
          </p>
          <div className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 120 }}>
                <defs>
                  <pattern id={HATCH_PATTERN_ID} width={6} height={6} patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
                    <line x1="0" y1="0" x2="0" y2="6" stroke="#9ca3af" strokeWidth={2} />
                  </pattern>
                </defs>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke={chartTheme.gridColor} />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 12, fill: chartTheme.axisColor }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: chartTheme.axisColor }} width={110} />
                <Tooltip
                  contentStyle={chartTheme.tooltipContentStyle}
                  itemStyle={chartTheme.tooltipItemStyle}
                  labelStyle={chartTheme.tooltipLabelStyle}
                  formatter={(value, name, item) => {
                    const reported = (item?.payload as Record<string, unknown> | undefined)?.[reportedFlagKey(String(name))] !== false;
                    return reported ? Number(value).toFixed(3) : 'not reported (shown as 0)';
                  }}
                />
                <Legend />
                {barKeys.map((key, i) => (
                  <Bar
                    key={key}
                    dataKey={key}
                    fill={BAR_COLORS[i % BAR_COLORS.length]}
                    radius={[0, 4, 4, 0]}
                    shape={(props: unknown) => <NotReportedAwareBar {...(props as NotReportedBarShapeProps)} />}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-2 text-center">
            {topFeatures.length < allFeatures.size
              ? `Showing top ${topFeatures.length} of ${allFeatures.size} features · `
              : ''}
            values normalised per-run (max = 1.0) · hatched bars mean the run did not report that feature (not a measured zero)
          </p>
          <ChartDataTable
            columns={tableColumns}
            rows={tableRows}
            filename="shap_summary_comparison"
            caption="SHAP summary comparison data, with normalized and raw values per run"
          />
        </div>
      )}
    </div>
  );
};
